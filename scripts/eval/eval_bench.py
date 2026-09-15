from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = Path("data/processed/bench_v1")

PROMPTS = {
    "mcq": ("Answer the following multiple-choice question step by step. The last line of "
            "your response should be of the form: 'The final answer is: $\\boxed{X}$' "
            "(without quotes) where X is the letter of the correct option.\n\n{q}"),
    "math": ("Solve the following math problem step by step. The last line of your response "
             "should be of the form: 'The final answer is: $\\boxed{ANSWER}$' (without "
             "quotes) where ANSWER is your final answer.\n\n{q}"),
    "numeric": ("Solve the following science problem step by step. Give the final answer in "
                "the unit: {unit}. The last line of your response should be of the form: "
                "'The final answer is: $\\boxed{ANSWER}$' (without quotes) where ANSWER is "
                "the final numeric value only, without the unit.\n\n{q}"),
    "rubric": ("Answer the following science question step by step with a complete, "
               "well-justified explanation.\n\n{q}"),
}


def build_prompt(row) -> str:
    tpl = PROMPTS[row.scoring]
    if row.scoring == "numeric":
        unit = json.loads(row.extra).get("unit") or "(dimensionless)"
        return tpl.replace("{unit}", unit).replace("{q}", row.question)
    return tpl.replace("{q}", row.question)


def load_bench(bench: str, subsample: int) -> pd.DataFrame:
    df = pd.read_parquet(BASE / f"{bench}.parquet")
    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=17).sort_index()
    return df


def generate(model: str, name: str, benches: list[str], subsample: int, max_new: int):
    todo: dict[str, pd.DataFrame] = {}
    caches: dict[str, Path] = {}
    for b in benches:
        df = load_bench(b, subsample)
        tag = f"{b}_sub{subsample}" if subsample and subsample < 10**9 else b
        cache = BASE / "gen" / name / f"{tag}.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        caches[b] = cache
        done = set()
        if cache.exists():
            done = {json.loads(line)["id"] for line in open(cache)}
        rest = df[~df["id"].isin(done)]
        if len(rest):
            todo[b] = rest
        print(f"[gen] {b}: total {len(df)}, cached {len(done)}, todo {len(rest)}")
    if not todo:
        return caches
    import os as _os
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(model)
    tpl_kw = {"enable_thinking": False} if _os.environ.get("CROPD_BENCH_NOTHINK") else {}
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    for b, df in todo.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r)}],
                                           tokenize=False, add_generation_prompt=True, **tpl_kw)
                   for r in df.itertuples()]
        outs = llm.generate(prompts, sp)
        with open(caches[b], "a") as f:
            for r, o in zip(df.itertuples(), outs):
                f.write(json.dumps({"id": r.id, "text": o.outputs[0].text}) + "\n")
        print(f"[gen] {b}: +{len(df)} done")
    return caches


BOX_RE = re.compile(r"\\boxed\{")


def extract_boxed_last(text: str) -> str | None:
    starts = [m.end() for m in BOX_RE.finditer(text)]
    if not starts:
        return None
    s = starts[-1]
    depth, i = 1, s
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[s:i - 1].strip()


def score_mcq(text: str, answer: str) -> tuple[bool, bool]:
    b = extract_boxed_last(text)
    if b:
        m = re.search(r"[A-J]", b.upper())
        if m:
            return m.group(0) == answer, True
    m = list(re.finditer(r"final answer is[:\s]*\(?\$?([A-J])\b", text[-500:], re.I))
    if m:
        return m[-1].group(1).upper() == answer, True
    return False, False


def score_math(text: str, answer: str) -> tuple[bool, bool]:
    b = extract_boxed_last(text)
    if b is None:
        return False, False
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${answer}$"), parse(f"${b}$"))), True
    except Exception:
        return False, True


NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("\\%", "").replace("%", "").replace("$", "")
    s = re.sub(r"\\times\s*10\^\{?(-?\d+)\}?", r"e\1", s)
    s = re.sub(r"10\^\{?(-?\d+)\}?", r"1e\1", s)
    m = NUM_RE.findall(s)
    if not m:
        return None
    try:
        return float(m[0]) if len(m) == 1 or "e" in m[0].lower() else float(m[0])
    except ValueError:
        return None


def score_numeric(text: str, answer: str) -> tuple[bool, bool]:
    b = extract_boxed_last(text)
    if b is None:
        return False, False
    got, ref = _to_float(b), _to_float(answer)
    if got is None or ref is None:
        return False, True
    tol = max(abs(ref) * 0.05, 1e-9)
    return abs(got - ref) <= tol, True


def score_rubric_batch(rows: list[tuple[str, str, list[dict]]], workers: int = 32) -> dict[str, float]:
    from cropd.verifiers import Unknown, run_verifier

    def one_crit(args):
        rid, text, k, crit = args
        try:
            v = run_verifier("llm_judge", text, {
                "criterion": str(crit.get("description") or ""),
                "cid": f"{rid}-r{k}"})
            return rid, k, float(crit.get("weight") or 0), float(v.score)
        except Unknown:
            return rid, k, None, None
        except Exception:
            return rid, k, None, None

    jobs = [(rid, text, k, c) for rid, text, rubric in rows for k, c in enumerate(rubric)]
    per: dict[str, list[tuple[float, float]]] = {rid: [] for rid, _, _ in rows}
    with ThreadPoolExecutor(workers) as ex:
        for rid, _k, w, s in ex.map(one_crit, jobs):
            if w is not None:
                per[rid].append((w, s))
    out = {}
    for rid, pairs in per.items():
        pos = sum(w for w, _ in pairs if w > 0)
        out[rid] = min(1.0, max(0.0, sum(w * s for w, s in pairs) / pos)) if pos > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="all")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--judge-workers", type=int, default=32)
    args = ap.parse_args()
    all_benches = ["gpqa_diamond", "mmlu_pro", "math500", "scibench", "rar_science", "aime26", "chembench", "rar_med"]
    benches = all_benches if args.bench == "all" else args.bench.split(",")
    caches = generate(args.model, args.name, benches, args.subsample, args.max_new)

    res_path = BASE / "results" / f"{args.name}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(res_path.read_text()) if res_path.exists() else {"name": args.name, "model": args.model}
    from cropd.verifiers import strip_think
    for b in benches:
        df = load_bench(b, args.subsample)
        gen = {json.loads(line)["id"]: strip_think(json.loads(line)["text"])
               for line in open(caches[b])}
        n, n_ok, n_ext = 0, 0, 0
        if b.startswith("rar_"):
            rows = [(r.id, gen[r.id], json.loads(r.extra)["rubric"]) for r in df.itertuples() if r.id in gen]
            scores = score_rubric_batch(rows, args.judge_workers)
            n = len(rows)
            score = sum(scores.values()) / n if n else 0.0
            entry = {"score": round(score, 4), "n": n}
        else:
            fns = {"mcq": score_mcq, "math": score_math, "numeric": score_numeric}
            for r in df.itertuples():
                if r.id not in gen:
                    continue
                ok, ext = fns[r.scoring](gen[r.id], r.answer)
                n += 1
                n_ok += ok
                n_ext += ext
            entry = {"score": round(n_ok / n, 4) if n else 0.0, "n": n,
                     "extract_rate": round(n_ext / n, 4) if n else 0.0}
        sub = f"_sub{args.subsample}" if args.subsample else ""
        results[b + sub] = entry
        print(f"[score] {b}{sub}: {entry}")
    res_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))
    print("->", res_path)


if __name__ == "__main__":
    main()
