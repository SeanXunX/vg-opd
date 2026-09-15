from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/drsci/Dr_SCI_open-ended.parquet"
CRIT_DIR = ROOT / "data/processed/criteria_open_v1"
GATED = CRIT_DIR / "gated_r1.jsonl"
BANK = CRIT_DIR / "criteria_bank_open.jsonl"
SPLITDIR = ROOT / "data/splits/open_v1"
VERL_DIR = ROOT / "data/processed/verl_open_v1"
SUBJECTS = ["physics", "chemistry", "biology", "medicine"]
SEED = 42
PROMPT_VER = "open_gate_v0"
AXES = ("mechanistic", "evidence")
MAX_PER_AXIS = 6
N_TEST = 1000
N_DEV = 1000
SYSTEM_PROMPT = (
    "You are auditing grading rubrics for a science QA dataset. Given a question, its reference "
    "answer, and a list of rubric items, output STRICT JSON only:\n"
    '{"usable": true|false, "items": [{"i": <idx>, "valid": true|false, '
    '"axis": "mechanistic"|"evidence"|"other"}]}\n'
    "valid = the rubric item can be judged consistently using the reference answer alone. "
    "axis: mechanistic = causal/process/entity-relation reasoning about how or why; "
    "evidence = citing sources, claim-evidence support, factual attribution; other = neither. "
    "usable = reference answer is substantive enough to grade against."
)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user(q: str, ref: str, rubric: list[dict]) -> str:
    items = "\n".join(f"[{i}] (w={r.get('weight')}) {r.get('description', '')[:400]}" for i, r in enumerate(rubric))
    return f"Question:\n{q[:2000]}\n\nReference answer:\n{ref[:3000]}\n\nRubric items:\n{items}\n\nJSON:"


def parse(text: str, n_items: int) -> dict | None:
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group())
        assert isinstance(d.get("usable"), bool) and isinstance(d.get("items"), list)
        ok_items = [
            {"i": int(it["i"]), "valid": bool(it["valid"]), "axis": str(it.get("axis", "other"))}
            for it in d["items"] if isinstance(it, dict) and 0 <= int(it.get("i", -1)) < n_items
        ]
        return {"usable": d["usable"], "items": ok_items}
    except Exception:
        return None


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(RAW)
    df = df[df["extra_info"].apply(lambda x: x.get("subject") in SUBJECTS)].reset_index(drop=True)
    return df


def select_rows() -> pd.DataFrame:
    df = load_raw()
    df["__subj"] = df["extra_info"].apply(lambda x: x.get("subject"))
    df["__diff"] = df["extra_info"].apply(lambda x: str(x.get("difficulty", "?")))
    return df


async def gate_run(args) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180, max_retries=1)

    CRIT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if GATED.exists():
        with GATED.open() as f:
            done = {json.loads(line)["orig_row"] for line in f if line.strip()}

    df = select_rows().reset_index().rename(columns={"index": "orig_row"})
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    df = df[df["__subj"].isin(subjects)]
    if args.exclude_done:
        df = df[~df["orig_row"].isin(done)]
    df = (
        df.groupby(["__subj", "__diff"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, round(args.n * len(g) / len(df)))), random_state=SEED))
        .reset_index(drop=True)
    )
    if args.limit:
        df = df.head(args.limit)

    todo = df[~df["orig_row"].isin(done)]
    print(f"target={len(df)} done={len(done)} todo={len(todo)}", flush=True)

    out_f = GATED.open("a")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    stat = {"ok": 0, "fail": 0}
    t0 = time.time()

    async def one(row) -> None:
        async with sem:
            ei = row.extra_info
            rubric = [dict(r) for r in row.reward_model["rubric"]]
            err = None
            try:
                resp = await client.chat.completions.create(
                    model="judge", temperature=0.0, max_tokens=1200,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": build_user(ei["question"], ei["reference_answer"], rubric)}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                verdict = parse(resp.choices[0].message.content, len(rubric))
            except Exception as e:
                verdict = None
                err = type(e).__name__
            rec = {
                "orig_row": int(row.orig_row), "subject": ei["subject"], "difficulty": ei.get("difficulty"),
                "prompt_ver": PROMPT_VER, "n_rubric": len(rubric),
            }
            if verdict is None:
                rec["gate"] = "unknown"
                rec["err"] = err or "parse_fail"
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
            el = (time.time() - t0) / 60
            rate = (stat["ok"] + stat["fail"]) / max(el * 60, 1)
            eta = (len(todo) - stat["ok"] - stat["fail"]) / max(rate, 1e-6) / 3600
            print(f"[{el:6.1f}min] ok={stat['ok']} fail={stat['fail']} ({rate:.1f}/s, ETA {eta:.1f}h)", flush=True)

    rep = asyncio.create_task(reporter())
    await asyncio.gather(*(one(r) for r in todo.itertuples()))
    rep.cancel()
    out_f.close()
    print(f"DONE ok={stat['ok']} fail={stat['fail']}", flush=True)


def cmd_gate(args) -> None:
    asyncio.run(gate_run(args))


def build_entry(rec: dict, rubric: list[dict]) -> tuple[dict | None, dict]:
    diag = Counter()
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
        return None, diag

    rid = f"open-{rec['orig_row']}"
    criteria = []
    for ax in sorted(AXES, key=lambda a: a != primary):
        for c in by_axis.get(ax, []):
            cid = f"c{len(criteria) + 1}"
            criteria.append(
                {
                    "cid": cid, "text": c["text"], "verifier": "llm_judge",
                    "verifier_args": {"criterion": c["text"], "cid": f"{rid}:{cid}"},
                    "axis": ax, "weight": c["weight"],
                }
            )
    entry = {
        "id": rid,
        "orig_row": rec["orig_row"],
        "primary_axis": primary,
        "criteria": criteria,
        "gen_meta": {
            "model": "judge", "prompt_ver": rec.get("prompt_ver", "open_gate_v0"),
            "gate": "ref_selfcheck", "source": "drsci_rubric_import",
        },
    }
    return entry, diag


def stratified_split(entries: list[dict], subj_of: dict[int, str]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        buckets[(subj_of[e["orig_row"]], e["primary_axis"])].append(e)
    total = len(entries)
    splits: dict[str, list[dict]] = {"test": [], "dev": [], "train": []}
    for key in sorted(buckets):
        grp = sorted(buckets[key], key=lambda e: e["orig_row"])
        rng.shuffle(grp)
        n_t = round(N_TEST * len(grp) / total)
        n_d = round(N_DEV * len(grp) / total)
        splits["test"] += grp[:n_t]
        splits["dev"] += grp[n_t : n_t + n_d]
        splits["train"] += grp[n_t + n_d :]
    return splits


def cmd_export(args) -> None:
    raw = load_raw()
    gated = [json.loads(line) for line in GATED.open() if line.strip()]
    n_gate = Counter(r["gate"] for r in gated)
    print(f"gated rows: {len(gated)} {dict(n_gate)}")

    entries: list[dict] = []
    diag = Counter()
    subj_of: dict[int, str] = {}
    for rec in gated:
        if rec["gate"] != "pass":
            continue
        row = raw.iloc[rec["orig_row"]]
        assert row["extra_info"]["subject"] == rec["subject"], f"orig_row mapping drift @{rec['orig_row']}"
        entry, d = build_entry(rec, [dict(r) for r in row["reward_model"]["rubric"]])
        diag.update(d)
        if entry is None:
            continue
        subj_of[rec["orig_row"]] = rec["subject"]
        entries.append(entry)
    print(f"bank entries: {len(entries)}; diag: {dict(diag)}")

    BANK.parent.mkdir(parents=True, exist_ok=True)
    with BANK.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    splits = stratified_split(entries, subj_of)
    SPLITDIR.mkdir(parents=True, exist_ok=True)
    for name, sel in splits.items():
        (SPLITDIR / f"{name}.txt").write_text("\n".join(e["id"] for e in sel) + "\n")

    VERL_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {"bank": len(entries), "diag": dict(diag)}
    for name, sel in splits.items():
        rows = []
        for e in sel:
            r = raw.iloc[e["orig_row"]]
            ei = r["extra_info"]
            rows.append(
                {
                    "id": e["id"],
                    "prompt": [dict(m) for m in r["prompt"]],
                    "data_source": f"cropd/{e['primary_axis']}",
                    "reward_model": {"style": "rule", "ground_truth": json.dumps({"type": "open"})},
                    "extra_info": {
                        "id": e["id"],
                        "criteria": json.dumps(e["criteria"], ensure_ascii=False),
                        "domain": ei["subject"],
                        "capability": e["primary_axis"],
                        "final_type": "open",
                        "orig_source": str(ei.get("from") or r["data_source"] or ""),
                        "difficulty": str(ei.get("difficulty", "")),
                    },
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), VERL_DIR / f"{name}.parquet")
        stats[name] = {
            "rows": len(rows),
            "axis": dict(Counter(r["data_source"] for r in rows)),
            "subject": dict(Counter(r["extra_info"]["domain"] for r in rows)),
        }
        if name == "train":
            for ax, short in (("mechanistic", "mech"), ("evidence", "evidence")):
                sub = [r for r in rows if r["data_source"] == f"cropd/{ax}"]
                pq.write_table(pa.Table.from_pylist(sub), VERL_DIR / f"train_{short}.parquet")
                stats[f"train_{short}"] = {"rows": len(sub)}
    n_crit = [len(e["criteria"]) for e in entries]
    neg = sum(1 for e in entries for c in e["criteria"] if c["weight"] < 0)
    stats["criteria"] = {
        "mean_per_row": round(sum(n_crit) / max(len(n_crit), 1), 2),
        "total": sum(n_crit), "negative_weight": neg,
    }
    (VERL_DIR / "export_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    lines = [
        "# open_v1 split statistics (generated by open_rubrics.py export)", "",
        f"- bank: {len(entries)} rows; mean criteria per row {stats['criteria']['mean_per_row']}; negative-weight (penalty) items {neg}",
        f"- stratified by subject x primary_axis, seed={SEED}; gated input {len(gated)} rows {dict(n_gate)}", "",
        "| split | rows | mechanistic | evidence |", "|---|---|---|---|",
    ]
    for name in ("test", "dev", "train"):
        ax = stats[name]["axis"]
        lines.append(
            f"| {name} | {stats[name]['rows']} | {ax.get('cropd/mechanistic', 0)} | {ax.get('cropd/evidence', 0)} |"
        )
    lines += ["", f"expert slices: train_mech {stats['train_mech']['rows']} / train_evidence {stats['train_evidence']['rows']}", ""]
    (SPLITDIR / "STATS.md").write_text("\n".join(lines))
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-ended rubrics: judge-gate Dr.SCI rubric items per axis, then export the open-ended criteria bank, splits and verl parquet files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate", help="judge-gate rubric items and label each with an axis (resumable)")
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--limit", type=int, default=0, help="debug: only process the first N rows")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--subjects", default=",".join(SUBJECTS), help="comma-separated subjects restricting the sampling pool (for top-up rounds)")
    p.add_argument("--exclude-done", action="store_true", help="exclude already-processed rows before sampling (--n then counts new rows)")
    p.set_defaults(fn=cmd_gate)
    sub.add_parser("export", help="build the open-ended criteria bank, split it and export verl parquet files").set_defaults(fn=cmd_export)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
