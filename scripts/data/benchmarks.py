from __future__ import annotations

import argparse
import asyncio
import glob
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_rubrics import build_user, parse

RAW = ROOT / "data/raw/bench_v1"
BENCH = ROOT / "data/processed/bench_v1"
V1 = ROOT / "data/processed/verl_merge_v1"
MIX_DIR = ROOT / "data/processed/verl_8b_v1"
PARTS = MIX_DIR / "parts"
GATED = ROOT / "data/processed/criteria_8b/rar_gated.jsonl"
LETTERS = "ABCDEFGHIJ"
NUM_TOL = 0.05
AXES = ("mechanistic", "evidence")
MAX_PER_AXIS = 6
SEED = 20260905
RUBRIC_PROMPT = ("Answer the following science question step by step with a complete, "
                 "well-justified explanation.\n\n{q}")
SOURCES = {
    "rar_science/train": ("rar_science/data/train-*.parquet", "science"),
    "rar_science/val": ("rar_science/data/val-*.parquet", "science"),
    "rar_science/test": ("rar_science/data/test-*.parquet", "science"),
    "rar_med/test": ("rar_med/test.parquet", "medicine"),
}
RAR_PROMPT_VER = "rar_gate_v1"
RAR_SYSTEM_PROMPT = (
    "You are auditing grading rubrics for a science QA dataset. Given a question, an optional reference "
    "answer (may be empty), and a list of rubric items, output STRICT JSON only:\n"
    '{"usable": true|false, "items": [{"i": <idx>, "valid": true|false, '
    '"axis": "mechanistic"|"evidence"|"other"}]}\n'
    "valid = the rubric item states a specific, objectively checkable requirement on the answer's content "
    "(a fact, quantity, step, mechanism, or justification), not vague style or length advice. "
    "axis: mechanistic = causal/process/entity-relation reasoning about how or why something happens; "
    "evidence = stating or using specific facts, data, definitions, diagnoses, or claim-evidence support; "
    "other = formatting, style, or anything that fits neither. "
    "usable = the question is answerable as posed and at least one rubric item is valid. No prose."
)
BOXED_RE = re.compile(r"\\boxed\{")


def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def training_questions() -> tuple[set[str], set[str]]:
    train_qs: set[str] = set()
    files = glob.glob(str(ROOT / "data/processed/verl_open_v1/train_*.parquet")) + glob.glob(
        str(ROOT / "data/processed/verl_merge_v1/train.parquet"))
    for pth in files:
        t = pd.read_parquet(pth)
        for pr in t["prompt"]:
            msgs = pr.tolist() if hasattr(pr, "tolist") else pr
            c = str(msgs[-1].get("content", ""))
            q = c.split("\n\n", 1)[1] if "\n\n" in c else c
            train_qs.add(norm_q(q))
    train_qs.discard("")
    return train_qs, {q[:200] for q in train_qs if q}


def stage_gpqa() -> pd.DataFrame:
    df = pd.read_csv(RAW / "gpqa/gpqa_diamond.csv")
    rows = []
    for i, r in df.iterrows():
        opts = [str(r["Correct Answer"]).strip(), str(r["Incorrect Answer 1"]).strip(),
                str(r["Incorrect Answer 2"]).strip(), str(r["Incorrect Answer 3"]).strip()]
        rng = random.Random(f"gpqa-{i}")
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        ans = LETTERS[order.index(0)]
        q = str(r["Question"]).strip() + "\n\n" + "\n".join(
            f"{LETTERS[k]}) {o}" for k, o in enumerate(shuffled))
        rows.append(dict(id=f"gpqa-{i}", bench="gpqa_diamond", question=q, answer=ans,
                         scoring="mcq", subject=str(r.get("High-level domain", "") or ""),
                         extra=json.dumps({"options": shuffled})))
    return pd.DataFrame(rows)


def stage_mmlu_pro() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "mmlu_pro/data/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        opts = [str(o) for o in r["options"]]
        q = str(r["question"]).strip() + "\n\n" + "\n".join(
            f"{LETTERS[k]}) {o}" for k, o in enumerate(opts))
        rows.append(dict(id=f"mmlupro-{r['question_id']}", bench="mmlu_pro", question=q,
                         answer=str(r["answer"]), scoring="mcq", subject=str(r["category"]),
                         extra=json.dumps({"n_options": len(opts)})))
    return pd.DataFrame(rows)


def stage_math500() -> pd.DataFrame:
    rows = []
    for i, line in enumerate(open(RAW / "math500/test.jsonl")):
        r = json.loads(line)
        rows.append(dict(id=f"math500-{i}", bench="math500", question=str(r["problem"]).strip(),
                         answer=str(r["answer"]), scoring="math", subject=str(r["subject"]),
                         extra=json.dumps({"level": r["level"]})))
    return pd.DataFrame(rows)


def stage_scibench() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "scibench/*.json"))):
        book = Path(f).stem
        if book.endswith("_sol"):
            continue
        for k, r in enumerate(json.load(open(f))):
            unit = str(r.get("unit") or "").strip()
            rows.append(dict(
                id=f"scibench-{book}-{k}", bench="scibench",
                question=str(r["problem_text"]).strip(), answer=str(r["answer_number"]).strip(),
                scoring="numeric", subject=book,
                extra=json.dumps({"unit": unit, "answer_latex": str(r.get("answer_latex") or ""),
                                  "problemid": str(r["problemid"]).strip()})))
    return pd.DataFrame(rows)


def stage_chembench() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "chembench/*.parquet"))):
        topic = Path(f).stem
        if topic == "chemical_preference":
            continue
        d = pd.read_parquet(f)
        for r in d.itertuples():
            ex = r.examples[0]
            q0 = str(ex["input"]).strip()
            ts = ex.get("target_scores")
            if ts is not None and isinstance(ts, str) and ts.strip():
                s = json.loads(ts)
                opts = [str(o) for o in s]
                correct = [o for o in opts if s[o] == 1]
                if len(correct) != 1 or len(opts) > 10:
                    continue
                rng = random.Random(f"chembench-{r.uuid}")
                rng.shuffle(opts)
                q = q0 + "\n\n" + "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(opts))
                rows.append(dict(id=f"chembench-{r.uuid}", bench="chembench", question=q,
                                 answer=LETTERS[opts.index(correct[0])], scoring="mcq", subject=topic,
                                 extra=json.dumps({"n_options": len(opts)})))
            elif ex.get("target") is not None:
                rows.append(dict(id=f"chembench-{r.uuid}", bench="chembench", question=q0,
                                 answer=str(ex["target"]).strip(), scoring="numeric", subject=topic,
                                 extra=json.dumps({"preferred_score": str(r.preferred_score)})))
    return pd.DataFrame(rows)


def stage_rubric_bench(path: Path, bench: str, prefix: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    train_qs, train_pre = training_questions()
    rows, n_exact, n_near = [], 0, 0
    for i, r in df.iterrows():
        nq = norm_q(r["question"])
        if nq in train_qs:
            n_exact += 1
            continue
        if nq[:200] in train_pre:
            n_near += 1
            continue
        rubric = [dict(x) for x in r["rubric"]]
        rows.append(dict(id=f"{prefix}-{i}", bench=bench, question=str(r["question"]).strip(),
                         answer=str(r["reference_answer"] or ""), scoring="rubric",
                         subject=str(r["question_source"]), extra=json.dumps({"rubric": rubric})))
    print(f"[{bench}] test {len(df)} -> kept {len(rows)} (exact dup {n_exact}, near dup {n_near})")
    return pd.DataFrame(rows)


def stage_rar_med() -> pd.DataFrame:
    return stage_rubric_bench(RAW / "rar_med/test.parquet", "rar_med", "rarmed")


def stage_rar() -> pd.DataFrame:
    return stage_rubric_bench(RAW / "rar_science/data/test-00000-of-00001.parquet", "rar_science", "rar")


def stage_aime26() -> pd.DataFrame:
    rows = []
    df = pd.read_parquet(RAW / "aime26/aime_2026.parquet")
    for r in df.itertuples():
        rows.append(dict(id=f"aime26-{int(r.problem_idx)}", bench="aime26", question=str(r.problem).strip(),
                         answer=str(r.answer), scoring="math", subject="math", extra=json.dumps({})))
    return pd.DataFrame(rows)


def cmd_stage(args) -> None:
    BENCH.mkdir(parents=True, exist_ok=True)
    for name, fn in [("gpqa_diamond", stage_gpqa), ("mmlu_pro", stage_mmlu_pro),
                     ("math500", stage_math500), ("scibench", stage_scibench),
                     ("rar_science", stage_rar), ("aime26", stage_aime26), ("chembench", stage_chembench), ("rar_med", stage_rar_med)]:
        df = fn()
        assert len(df) > 0 and df["id"].is_unique, name
        df.to_parquet(BENCH / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows -> {BENCH}/{name}.parquet")


def eval_prompts() -> dict:
    spec = importlib.util.spec_from_file_location("eval_bench", ROOT / "scripts/eval/eval_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def make_prompt(prompts: dict, scoring: str, question: str, unit: str | None = None) -> str:
    tpl = prompts["math" if scoring == "math" else scoring]
    if scoring == "numeric":
        tpl = tpl.replace("{unit}", unit or "(dimensionless)")
    return tpl.replace("{q}", question)


def make_row(rid: str, prompt: str, ty: str, value, axis: str, domain: str, source: str,
             rel_tol: float | None = None) -> dict:
    from cropd.criteria.schema import build_criteria

    fa = {"type": ty, "value": value}
    if rel_tol is not None:
        fa["rel_tol"] = rel_tol
    crit = build_criteria({"final_answer": fa, "primary_axis": axis, "intermediate_criteria": []})
    return {
        "id": rid,
        "prompt": [{"content": prompt, "role": "user"}],
        "data_source": f"cropd/{axis}",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(
            {"type": ty, "value": value, "unit": None, "rel_tol": rel_tol})},
        "extra_info": {"id": rid, "criteria": json.dumps(crit, ensure_ascii=False), "domain": str(domain).lower(),
                       "capability": axis, "final_type": ty, "orig_source": source, "difficulty": "na"},
    }


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def staged(bench: str) -> pd.DataFrame:
    return pd.read_parquet(BENCH / f"{bench}.parquet")


def build_closed_rows(prompts: dict) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    skip = Counter()
    for bench in ("gpqa_diamond", "mmlu_pro", "chembench"):
        df = staged(bench)
        for i, r in enumerate(df.itertuples(index=False)):
            ex = json.loads(r.extra) if isinstance(r.extra, str) else dict(r.extra or {})
            if r.scoring == "mcq":
                ans = str(r.answer).strip().upper()
                if ans not in LETTERS:
                    skip[f"{bench}:bad_letter"] += 1
                    continue
                rows.append(make_row(f"bench-{bench}-{i}", make_prompt(prompts, "mcq", r.question), "mcq", ans,
                                     "quantitative", r.subject, bench))
            elif r.scoring == "numeric":
                v = _num(r.answer)
                if v is None:
                    skip[f"{bench}:bad_numeric"] += 1
                    continue
                rows.append(make_row(f"bench-{bench}-{i}", make_prompt(prompts, "numeric", r.question, ex.get("unit")),
                                     "numeric", v, "quantitative", r.subject, bench, rel_tol=NUM_TOL))
            else:
                skip[f"{bench}:scoring_{r.scoring}"] += 1
    df = staged("scibench")
    for i, r in enumerate(df.itertuples(index=False)):
        ex = json.loads(r.extra) if isinstance(r.extra, str) else dict(r.extra or {})
        v = _num(r.answer)
        if v is None:
            skip["scibench:bad_numeric"] += 1
            continue
        rows.append(make_row(f"bench-scibench-{i}", make_prompt(prompts, "numeric", r.question, ex.get("unit")),
                             "numeric", v, "quantitative", r.subject, "scibench", rel_tol=NUM_TOL))
    df = staged("math500")
    for i, r in enumerate(df.itertuples(index=False)):
        rows.append(make_row(f"bench-math500-{i}", make_prompt(prompts, "math", r.question), "expression",
                             str(r.answer), "symbolic", "math", "math500"))
    vfiles = sorted(glob.glob(str(RAW / "mmlu_pro/data/validation-*.parquet")))
    if vfiles:
        dv = pd.concat([pd.read_parquet(f) for f in vfiles], ignore_index=True)
        for i, r in enumerate(dv.itertuples(index=False)):
            opts = list(r.options)
            q = str(r.question).strip() + "\n\n" + "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(opts))
            ans = str(r.answer).strip().upper()
            if ans not in LETTERS[: len(opts)]:
                skip["mmlu_val:bad_letter"] += 1
                continue
            rows.append(make_row(f"bench-mmlu_pro_val-{i}", make_prompt(prompts, "mcq", q), "mcq", ans,
                                 "quantitative", r.category, "mmlu_pro_val"))
    else:
        skip["mmlu_val:missing"] += 1
    return rows, skip


def last_boxed(s: str) -> str | None:
    idx = [m.start() for m in BOXED_RE.finditer(s)]
    if not idx:
        return None
    start = idx[-1] + len("\\boxed{")
    depth, j = 1, start
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[start:j - 1].strip() if depth == 0 else None


def build_math_train(prompts: dict, exclude_q: set[str]) -> tuple[list[dict], Counter]:
    files = sorted(glob.glob(str(RAW / "math_train/**/train-*.parquet"), recursive=True))
    rows, skip = [], Counter()
    if not files:
        skip["math_train:missing"] += 1
        return rows, skip
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for i, r in enumerate(df.itertuples(index=False)):
        q = str(r.problem).strip()
        if norm_q(q) in exclude_q:
            skip["math_train:in_math500"] += 1
            continue
        ans = last_boxed(str(r.solution))
        if not ans:
            skip["math_train:no_boxed"] += 1
            continue
        rows.append(make_row(f"bench-math_train-{i}", make_prompt(prompts, "math", q), "expression", ans,
                             "symbolic", "math", "math_train"))
    return rows, skip


def cmd_rows(args) -> None:
    prompts = eval_prompts()
    rows, skip = build_closed_rows(prompts)
    if not args.no_math_train:
        m500 = {norm_q(q) for q in staged("math500")["question"]}
        mrows, mskip = build_math_train(prompts, m500)
        rows += mrows
        skip.update(mskip)
    PARTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(PARTS / "closed_rows.parquet", index=False)
    stats = {"n": len(df), "by_source": df["extra_info"].apply(lambda e: e["orig_source"]).value_counts().to_dict(),
             "by_final_type": df["extra_info"].apply(lambda e: e["final_type"]).value_counts().to_dict(),
             "by_axis": df["data_source"].value_counts().to_dict(), "skipped": dict(skip)}
    (PARTS / "closed_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


def load_rar_rows(limit: int) -> list[dict]:
    rows = []
    for src, (pat, domain) in SOURCES.items():
        files = sorted(RAW.glob(pat))
        if not files:
            print(f"!! missing {src}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for idx, r in enumerate(df.itertuples(index=False)):
            rub = [dict(x) for x in (r.rubric if r.rubric is not None else [])]
            if not rub:
                continue
            rows.append({"key": f"{src}#{idx}", "src": src, "idx": idx, "domain": domain,
                         "question": str(r.question or ""), "ref": str(r.reference_answer or ""), "rubric": rub})
    return rows[:limit] if limit else rows


async def gate_rar_run(args) -> None:
    from openai import AsyncOpenAI

    rows = load_rar_rows(args.limit)
    GATED.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if GATED.exists():
        with open(GATED) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [r for r in rows if r["key"] not in done]
    print(f"rows {len(rows)} done {len(done)} todo {len(todo)}", flush=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("CROPD_JUDGE_API_KEY", "x"), timeout=180)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stat = {"ok": 0, "fail": 0}
    t0 = time.time()
    out_f = open(GATED, "a")

    async def one(row: dict) -> None:
        async with sem:
            err = None
            try:
                resp = await client.chat.completions.create(
                    model="judge", temperature=0.0, max_tokens=1200,
                    messages=[{"role": "system", "content": RAR_SYSTEM_PROMPT},
                              {"role": "user", "content": build_user(row["question"], row["ref"], row["rubric"])}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                verdict = parse(resp.choices[0].message.content, len(row["rubric"]))
            except Exception as e:
                verdict, err = None, type(e).__name__
            rec = {"key": row["key"], "src": row["src"], "idx": row["idx"], "domain": row["domain"],
                   "prompt_ver": RAR_PROMPT_VER, "n_rubric": len(row["rubric"])}
            if verdict is None:
                rec["gate"], rec["err"] = "unknown", err or "parse_fail"
                stat["fail"] += 1
            else:
                rec["gate"] = "pass" if verdict["usable"] else "unusable"
                rec["items"] = verdict["items"]
                stat["ok"] += 1
            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()

    async def reporter() -> None:
        while True:
            await asyncio.sleep(60)
            n = stat["ok"] + stat["fail"]
            rate = n / max(time.time() - t0, 1)
            print(f"[{time.strftime('%H:%M')}] {n}/{len(todo)} ok={stat['ok']} fail={stat['fail']} "
                  f"{rate:.1f}/s eta {((len(todo) - n) / max(rate, 1e-6)) / 60:.0f} min", flush=True)

    rep = asyncio.create_task(reporter())
    await asyncio.gather(*(one(r) for r in todo))
    rep.cancel()
    out_f.close()
    print(f"DONE ok={stat['ok']} fail={stat['fail']} in {(time.time() - t0) / 60:.1f} min -> {GATED}", flush=True)


def cmd_gate_rar(args) -> None:
    asyncio.run(gate_rar_run(args))


def question_of(row: dict) -> str:
    c = str(row["prompt"][-1]["content"])
    return c.split("\n\n", 1)[1] if "\n\n" in c else c


def load_v1() -> list[dict]:
    df = pd.read_parquet(V1 / "train.parquet")
    rows = []
    for r in df.to_dict("records"):
        r["prompt"] = [dict(m) for m in r["prompt"]]
        r["reward_model"] = dict(r["reward_model"])
        r["extra_info"] = {k: ("" if v is None else str(v)) for k, v in dict(r["extra_info"]).items()}
        rows.append(r)
    return rows


def load_closed() -> list[dict]:
    df = pd.read_parquet(PARTS / "closed_rows.parquet")
    rows = []
    for r in df.to_dict("records"):
        r["prompt"] = [dict(m) for m in r["prompt"]]
        r["reward_model"] = dict(r["reward_model"])
        r["extra_info"] = {k: str(v) for k, v in dict(r["extra_info"]).items()}
        rows.append(r)
    return rows


def build_open_entry(rec: dict, rubric: list[dict], rid: str, diag: Counter) -> list | None:
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for it in rec.get("items", []):
        if not it["valid"]:
            diag["item_invalid"] += 1
            continue
        ax = it["axis"]
        if ax not in AXES:
            diag["item_other_axis"] += 1
            continue
        r = rubric[it["i"]]
        text = str(r.get("description", "")).strip()
        w = r.get("weight")
        if not text or not isinstance(w, (int, float)) or w == 0:
            diag["item_bad_fields"] += 1
            continue
        by_axis[ax].append({"text": text, "weight": float(w)})
    for ax, items in by_axis.items():
        items.sort(key=lambda c: -abs(c["weight"]))
        diag[f"trunc_{ax}"] += max(0, len(items) - MAX_PER_AXIS)
        by_axis[ax] = items[:MAX_PER_AXIS]
    score = {ax: sum(abs(c["weight"]) for c in by_axis.get(ax, [])) for ax in AXES}
    primary = max(AXES, key=lambda ax: (score[ax], ax == "mechanistic"))
    if score[primary] == 0 or not any(c["weight"] > 0 for c in by_axis[primary]):
        diag["row_no_positive_primary"] += 1
        return None
    criteria = []
    for ax in sorted(AXES, key=lambda a: a != primary):
        for c in by_axis.get(ax, []):
            cid = f"c{len(criteria) + 1}"
            criteria.append({"cid": cid, "text": c["text"], "verifier": "llm_judge",
                             "verifier_args": {"criterion": c["text"], "cid": f"{rid}:{cid}"},
                             "axis": ax, "weight": c["weight"]})
    return [primary, criteria]


def load_open() -> tuple[list[dict], Counter]:
    diag = Counter()
    recs = [json.loads(l) for l in open(GATED) if l.strip()]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_src[r["src"]].append(r)
    rows = []
    for src, (pat, domain) in SOURCES.items():
        files = sorted(RAW.glob(pat))
        if not files or src not in by_src:
            diag[f"missing_{src}"] += 1
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for rec in by_src[src]:
            diag[f"gate_{rec['gate']}"] += 1
            if rec["gate"] != "pass":
                continue
            raw = df.iloc[rec["idx"]]
            rubric = [dict(x) for x in (raw["rubric"] if raw["rubric"] is not None else [])]
            rid = f"rar-{src.replace('/', '-')}-{rec['idx']}"
            ent = build_open_entry(rec, rubric, rid, diag)
            if ent is None:
                continue
            primary, criteria = ent
            q = str(raw["question"]).strip()
            rows.append({
                "id": rid,
                "prompt": [{"content": RUBRIC_PROMPT.replace("{q}", q), "role": "user"}],
                "data_source": f"cropd/{primary}",
                "reward_model": {"style": "rule", "ground_truth": json.dumps({"type": "open"})},
                "extra_info": {"id": rid, "criteria": json.dumps(criteria, ensure_ascii=False), "domain": domain,
                               "capability": primary, "final_type": "open", "orig_source": src, "difficulty": "na"},
            })
    return rows, diag


def dedup(groups: list[tuple[str, list[dict]]]) -> tuple[list[dict], Counter]:
    seen_exact: set[str] = set()
    seen_pref: set[str] = set()
    out, drops = [], Counter()
    for name, rows in groups:
        for r in rows:
            n = norm_q(question_of(r))
            if not n:
                drops[f"{name}:empty"] += 1
                continue
            if n in seen_exact or n[:200] in seen_pref:
                drops[f"{name}:dup"] += 1
                continue
            seen_exact.add(n)
            seen_pref.add(n[:200])
            out.append(r)
    return out, drops


def reward_smoke(rows: list[dict], n_per_type: int) -> dict:
    from cropd.rewards.criteria_reward import compute_score
    rng = random.Random(1)
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["extra_info"]["final_type"] != "open" and r["extra_info"]["orig_source"] not in ("MegaScience", "natural_reasoning", "WebInstruct-Verified", "Dr. SCI", ""):
            by_t[r["extra_info"]["final_type"]].append(r)
    res = {}
    for ty, lst in by_t.items():
        ok = tot = 0
        for r in rng.sample(lst, min(n_per_type, len(lst))):
            gt = json.loads(r["reward_model"]["ground_truth"])
            v = gt["value"]
            ans = f"Reasoning omitted.\n\nThe final answer is: $\\boxed{{{v}}}$"
            s = compute_score(r["data_source"], ans, r["reward_model"]["ground_truth"], r["extra_info"])
            tot += 1
            ok += int(s["r_final"] == 1.0 and s["score"] > 0)
        res[ty] = f"{ok}/{tot}"
    return res


def cmd_mix(args) -> None:
    v1 = load_v1()
    closed = load_closed()
    opened, diag = load_open()
    rows, drops = dedup([("v1", v1), ("closed", closed), ("open", opened)])
    aime = {norm_q(q) for q in pd.read_parquet(BENCH / "aime26.parquet")["question"]}
    leak = [r["id"] for r in rows if norm_q(question_of(r)) in aime]
    assert not leak, f"AIME26 leak: {leak[:5]}"
    random.Random(SEED).shuffle(rows)
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(MIX_DIR / "train.parquet", index=False)
    shutil.copy(V1 / "dev.parquet", MIX_DIR / "dev.parquet")
    src = Counter(r["extra_info"]["orig_source"] for r in rows)
    axis = Counter(r["data_source"] for r in rows)
    ftype = Counter(r["extra_info"]["final_type"] for r in rows)
    smoke = reward_smoke(rows, args.smoke_n)
    stats = {"n_train": len(rows), "n_dev": 2229, "inputs": {"v1": len(v1), "closed": len(closed), "open_gated": len(opened)},
             "by_source": dict(src), "by_axis": dict(axis), "by_final_type": dict(ftype), "dedup_drops": dict(drops),
             "open_gate_diag": dict(diag), "reward_smoke_r_final": smoke, "seed": SEED}
    (MIX_DIR / "MANIFEST.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    md = ["# verl_8b_v1 - student training mix with the full benchmark families included (release track, not the paper's held-out protocol)\n",
          f"- train {len(rows)} rows / dev 2229 rows (verl_merge_v1 dev, unchanged)",
          f"- inputs: v1 {len(v1)} + closed-form bench {len(closed)} + gated RaR open-ended {len(opened)}; dedup drops {dict(drops)}",
          f"- by source: {dict(src)}", f"- by axis: {dict(axis)}", f"- by final_type: {dict(ftype)}",
          f"- open-ended gate diagnostics: {dict(diag)}", f"- reward smoke test (reference answer -> r_final=1): {smoke}",
          "- AIME26 (30 problems): zero overlap; the seven benchmark columns must be reported as in-distribution (trained on bench data)"]
    (MIX_DIR / "STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description="External benchmarks: stage sources into one schema, build closed-form training rows, judge-gate RaR rubrics, and export the benchmark-augmented student mix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", help="normalize the raw benchmark sources into data/processed/bench_v1/<bench>.parquet").set_defaults(fn=cmd_stage)
    p = sub.add_parser("rows", help="closed-form benchmark rows with a single final-answer criterion")
    p.add_argument("--no-math-train", action="store_true")
    p.set_defaults(fn=cmd_rows)
    p = sub.add_parser("gate-rar", help="judge-gate RaR-Science / RaR-Medicine rubric items (resumable)")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--base-url", default=os.environ.get("CROPD_JUDGE_BASE_URL", "http://localhost:8000/v1"))
    p.set_defaults(fn=cmd_gate_rar)
    p = sub.add_parser("mix", help="merge the student mix with benchmark rows, dedup, check the AIME26 leak and export")
    p.add_argument("--smoke-n", type=int, default=30)
    p.set_defaults(fn=cmd_mix)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
