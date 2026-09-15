from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/eval_a4"
DEVS = [ROOT / "data/processed/verl_v1/dev.parquet", ROOT / "data/processed/verl_open_v1/dev.parquet"]
MAX_TOKENS = int(__import__("os").environ.get("CROPD_EVAL_MAX_TOKENS", "3072"))
JUDGE_THREADS = 24


def check_eos(model_dir: str) -> None:
    gc = json.loads((Path(model_dir) / "generation_config.json").read_text())
    eos = gc.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    assert 151645 in eos, f"{model_dir}: eos_token_id must include 151645 (im_end), got {eos}; rollouts would never stop"


def load_rows() -> list[dict]:
    rows = []
    for f in DEVS:
        t = pq.read_table(f)
        for i in range(len(t)):
            rows.append(
                {
                    "id": t["id"][i].as_py(),
                    "prompt": t["prompt"][i].as_py(),
                    "data_source": t["data_source"][i].as_py(),
                    "ground_truth": t["reward_model"][i].as_py()["ground_truth"],
                    "extra_info": t["extra_info"][i].as_py(),
                }
            )
    return rows


def generate(model: str, rows: list[dict], gen_path: Path) -> dict[str, str]:
    if gen_path.exists():
        cached = {r["id"]: r["response"] for r in map(json.loads, gen_path.open())}
        if len(cached) == len(rows):
            print(f"gen cache hit: {len(cached)}")
            return cached
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    tpl_kw = {"enable_thinking": False} if __import__("os").environ.get("CROPD_EVAL_NOTHINK") else {}
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, **tpl_kw) for r in rows]
    llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=1024 + MAX_TOKENS, dtype="bfloat16")
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    gens = {r["id"]: o.outputs[0].text for r, o in zip(rows, outs)}
    with gen_path.open("w") as f:
        for rid, resp in gens.items():
            f.write(json.dumps({"id": rid, "response": resp}, ensure_ascii=False) + "\n")
    return gens


def main() -> None:
    global DEVS
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rescore", action="store_true", help="ignore an existing <name>.json and re-score")
    ap.add_argument("--closed-only", action="store_true",
                    help="closed-form dev only (verl_v1, rule verifiers, no judge; output suffixed _closed); used for offline expert selection")
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="test = final held-out evaluation (output suffixed _test)")
    args = ap.parse_args()
    if args.closed_only:
        DEVS = [ROOT / "data/processed/verl_v1/dev.parquet"]
        args.name = f"{args.name}_closed"
    if args.split == "test":
        DEVS = [ROOT / "data/processed/verl_v1/test.parquet", ROOT / "data/processed/verl_open_v1/test.parquet"]
        args.name = f"{args.name}_test"

    check_eos(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = OUT_DIR / f"{args.name}.json"
    if res_path.exists() and not args.rescore:
        print(f"exists: {res_path} (use --rescore to re-score)")
        return

    rows = load_rows()
    gens = generate(args.model, rows, OUT_DIR / f"{args.name}_gen.jsonl")

    os.environ.setdefault("CROPD_REWARD_TIME_BUDGET", "60")
    os.environ.setdefault("CROPD_JUDGE_TIMEOUT", "90")
    from cropd.rewards.criteria_reward import compute_score

    def score(r: dict) -> tuple[str, dict]:
        return r["data_source"], compute_score(r["data_source"], gens[r["id"]], r["ground_truth"], r["extra_info"])

    with ThreadPoolExecutor(JUDGE_THREADS) as ex:
        scored = list(ex.map(score, rows))

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ds, s in scored:
        for k, v in s.items():
            if isinstance(v, (int, float)):
                agg[ds][k].append(float(v))
    result = {"name": args.name, "model": args.model, "n": len(rows)}
    for ds in sorted(agg):
        result[ds] = {k: round(sum(v) / len(v), 4) for k, v in sorted(agg[ds].items())}
    res_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
