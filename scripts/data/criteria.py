from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cropd.criteria.prompts import PROMPT_VER, messages
from cropd.criteria.schema import build_criteria, validate_annotation
from cropd.verifiers import run_verifier

POOL = ROOT / "data/processed/pool_v1.parquet"
SPLITDIR = ROOT / "data/splits/v1"
CRIT_DIR = ROOT / "data/processed/criteria_v1"
VERL_DIR = ROOT / "data/processed/verl_v1"
SEED = 20260810
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)
BOXED_HINT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_json(content: str) -> dict:
    c = THINK_RE.sub("", content)
    fences = FENCE_RE.findall(c)
    raw = fences[-1] if fences else None
    if raw is None:
        i, j = c.find("{"), c.rfind("}")
        raw = c[i : j + 1] if 0 <= i < j else None
    if raw is None:
        raise ValueError("no JSON found in response")
    return json.loads(raw)


def target_ids(args) -> list[str]:
    if args.ids_file:
        return Path(args.ids_file).read_text().split()
    order: list[str] = []
    train = None
    for name in args.splits.split(","):
        ids = (SPLITDIR / f"{name}.txt").read_text().split()
        if name == "train":
            train = ids
        else:
            order.extend(ids)
    if train is not None:
        random.Random(SEED).shuffle(train)
        order.extend(train)
    return order[: args.limit] if args.limit else order


def load_pool_rows(ids: list[str]) -> dict[str, dict]:
    t = pq.read_table(POOL, columns=["id", "reward_model", "extra_info"])
    t = t.filter(pc.is_in(t["id"], value_set=pa.array(ids)))
    ei = t["extra_info"]
    out = {}
    for i, q, r, g in zip(
        t["id"].to_pylist(),
        pc.struct_field(ei, "question").to_pylist(),
        pc.struct_field(ei, "reference_answer").to_pylist(),
        pc.struct_field(t["reward_model"], "ground_truth").to_pylist(),
    ):
        out[i] = {"id": i, "question": q or "", "reference": r or "", "gt": g or ""}
    return out


def done_ids(only_round: int | None = None) -> set[str]:
    files = (
        [CRIT_DIR / f"annotations_r{only_round}.jsonl"]
        if only_round is not None
        else list(CRIT_DIR.glob("annotations_r*.jsonl"))
    )
    done = set()
    for f in files:
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


async def annotate_one(client, sem, row, out_f, fail_f, stats, args):
    async with sem:
        last = "unknown"
        for attempt, strict in ((1, args.strict), (2, True)):
            msgs = messages(row["question"], row["reference"], row["gt"], strict=strict, regen=args.regen)
            max_tok = args.max_tokens if attempt == 1 else int(args.max_tokens * 1.6)
            resp = None
            for backoff in (2, 5, 15, 40, 90):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model, messages=msgs, max_tokens=max_tok,
                        temperature=0.6, top_p=0.95, timeout=900,
                    )
                    break
                except Exception as e:
                    name = type(e).__name__
                    if any(k in name for k in ("Connection", "Timeout", "APIStatus", "InternalServer", "RateLimit")):
                        last = f"conn {name}"
                        await asyncio.sleep(backoff)
                        continue
                    last = f"{name}: {str(e)[:200]}"
                    break
            if resp is None:
                continue
            try:
                ann = parse_json(resp.choices[0].message.content or "")
                errs = validate_annotation(ann)
                if errs:
                    raise ValueError("schema: " + "; ".join(errs[:3]))
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            rec = {
                "id": row["id"],
                "annotation": ann,
                "raw": (resp.choices[0].message.content or "")[:30000],
                "meta": {
                    "model": resp.model, "prompt_ver": PROMPT_VER,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "attempt": attempt,
                    "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens},
                },
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["ok"] += 1
            stats["out_toks"] += resp.usage.completion_tokens
            return
        fail_f.write(json.dumps({"id": row["id"], "err": last}) + "\n")
        fail_f.flush()
        stats["fail"] += 1


async def annotate_run(args) -> None:
    from openai import AsyncOpenAI

    ids = target_ids(args)
    skip = done_ids(args.round if args.ids_file else None)
    todo = [i for i in ids if i not in skip]
    rows = load_pool_rows(todo) if todo else {}
    todo = [i for i in todo if i in rows]
    print(f"target={len(ids)} done={len(ids) - len(todo)} todo={len(todo)}")
    if not todo:
        return

    CRIT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"ok": 0, "fail": 0, "out_toks": 0}
    t0 = time.monotonic()

    async def reporter():
        while True:
            await asyncio.sleep(60)
            el = time.monotonic() - t0
            done_n = stats["ok"] + stats["fail"]
            rate = done_n / el if el else 0
            eta_h = (len(todo) - done_n) / rate / 3600 if rate else float("inf")
            print(
                f"[{el/60:6.1f}min] ok={stats['ok']} fail={stats['fail']} "
                f"({rate:.1f}/s, {stats['out_toks']/el:.0f} out-tok/s, ETA {eta_h:.1f}h)",
                flush=True,
            )

    rep = asyncio.create_task(reporter())
    with (CRIT_DIR / f"annotations_r{args.round}.jsonl").open("a") as out_f, (
        CRIT_DIR / f"failures_r{args.round}.jsonl"
    ).open("a") as fail_f:
        await asyncio.gather(
            *(annotate_one(client, sem, rows[i], out_f, fail_f, stats, args) for i in todo)
        )
    rep.cancel()
    el = time.monotonic() - t0
    print(f"DONE ok={stats['ok']} fail={stats['fail']} in {el/60:.1f}min ({stats['out_toks']/el:.0f} out-tok/s)")


def cmd_annotate(args) -> None:
    asyncio.run(annotate_run(args))


def load_annotations() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for f in sorted(CRIT_DIR.glob("annotations_r*.jsonl")):
        with f.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    merged[rec["id"]] = rec
                except Exception:
                    continue
    return merged


def load_pool_refs(ids: list[str]) -> dict[str, dict]:
    t = pq.read_table(POOL, columns=["id", "reward_model", "extra_info", "subject"])
    t = t.filter(pc.is_in(t["id"], value_set=pa.array(ids)))
    ei = t["extra_info"]
    out = {}
    for i, gt, ref, subj in zip(
        t["id"].to_pylist(),
        pc.struct_field(t["reward_model"], "ground_truth").to_pylist(),
        pc.struct_field(ei, "reference_answer").to_pylist(),
        t["subject"].to_pylist(),
    ):
        out[i] = {"gt": gt or "", "reference": ref or "", "subject": subj}
    return out


def _is_restatement(final: dict, c: dict) -> bool:
    import math

    fv, cv = final["verifier"], c["verifier"]
    fa, ca = final["verifier_args"], c["verifier_args"]
    try:
        if fv == "numeric_unit" and cv == "numeric_unit":
            tol = max(float(fa.get("rel_tol", 0.02)), float(ca.get("rel_tol", 0.05)))
            return math.isclose(float(fa["value"]), float(str(ca["value"])), rel_tol=tol, abs_tol=1e-12)
        if fv == "sympy_equiv" and cv == "sympy_equiv":
            from cropd.verifiers.sympy_equiv import _equiv, _parse

            return _equiv(_parse(str(fa["target"])), _parse(str(ca["target"])))
        if cv == "rule":
            from cropd.verifiers.rule import _norm

            t1 = _norm(str(fa.get("target", fa.get("value", ""))))
            t2 = _norm(str(ca["target"]))
            return bool(t1) and bool(t2) and (t1 == t2 or t1 in t2 or t2 in t1)
    except Exception:
        return False
    return False


def gate_one(item: tuple) -> dict:
    rid, rec, gt, reference = item
    ann = rec["annotation"]
    errs = validate_annotation(ann)
    if errs:
        return {"id": rid, "status": "invalid", "errs": errs[:3]}
    crits = build_criteria(ann)

    final = crits[0]
    fres = run_verifier(final["verifier"], "\\boxed{" + gt + "}", final["verifier_args"])
    if fres.passed is not True:
        return {"id": rid, "status": "final_fail", "detail": fres.detail, "verifier": final["verifier"]}

    ref_substantive = len((reference or "").strip()) >= 200
    kept, dropped = [final], []
    n_restate = n_sanity = 0
    for c in crits[1:]:
        if _is_restatement(final, c):
            n_restate += 1
            continue
        dry = run_verifier(c["verifier"], "", c["verifier_args"])
        if dry.passed is None:
            n_sanity += 1
            continue
        if ref_substantive:
            r = run_verifier(c["verifier"], reference, c["verifier_args"])
            (kept if r.passed is True else dropped).append(
                {**c, "_gate": r.detail} if r.passed is not True else c
            )
        else:
            kept.append(c)
    return {
        "gate_mode": "ref_checked" if ref_substantive else "endpoint_only",
        "n_sanity": n_sanity,
        "id": rid, "status": "pass",
        "final_answer": ann["final_answer"], "primary_axis": ann["primary_axis"],
        "criteria": kept,
        "n_dropped": len(dropped),
        "n_restate": n_restate,
        "dropped_verifiers": [d["verifier"] for d in dropped],
        "meta": rec["meta"],
    }


def cmd_gate(args) -> None:
    ann = load_annotations()
    pool = load_pool_refs(list(ann))
    items = [(i, ann[i], pool[i]["gt"], pool[i]["reference"]) for i in sorted(ann) if i in pool]
    print(f"gating {len(items)} annotated samples ...")

    with Pool(args.procs) as pw:
        results = pw.map(gate_one, items, chunksize=64)

    passed = [r for r in results if r["status"] == "pass"]
    failed = [r for r in results if r["status"] != "pass"]

    with (CRIT_DIR / "criteria_bank.jsonl").open("w") as fh:
        for r in passed:
            fh.write(
                json.dumps(
                    {
                        "id": r["id"], "final_answer": r["final_answer"],
                        "primary_axis": r["primary_axis"], "criteria": r["criteria"],
                        "gen_meta": {
                            **r["meta"], "gate": r["gate_mode"],
                            "n_intermediate_dropped": r["n_dropped"],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    fail_ids = sorted(r["id"] for r in failed)
    if args.final:
        (CRIT_DIR / "unusable.txt").write_text("\n".join(fail_ids) + ("\n" if fail_ids else ""))
        (CRIT_DIR / "regen_queue.txt").write_text("")
    else:
        (CRIT_DIR / "regen_queue.txt").write_text("\n".join(fail_ids) + ("\n" if fail_ids else ""))

    n_int_kept = sum(len(r["criteria"]) - 1 for r in passed)
    n_int_drop = sum(r["n_dropped"] for r in passed)
    stats = {
        "annotated": len(items),
        "pass": len(passed),
        "fail": len(failed),
        "fail_reasons": dict(Counter(r["status"] for r in failed)),
        "fail_final_verifier": dict(Counter(r.get("verifier", "?") for r in failed if r["status"] == "final_fail")),
        "intermediate_kept": n_int_kept,
        "intermediate_dropped": n_int_drop,
        "intermediate_restatement_dropped": sum(r["n_restate"] for r in passed),
        "intermediate_sanity_dropped": sum(r["n_sanity"] for r in passed),
        "gate_modes": dict(Counter(r["gate_mode"] for r in passed)),
        "intermediate_drop_rate": round(n_int_drop / max(1, n_int_kept + n_int_drop), 4),
        "avg_criteria_per_pass": round(sum(len(r["criteria"]) for r in passed) / max(1, len(passed)), 2),
        "dropped_by_verifier": dict(Counter(v for r in passed for v in r["dropped_verifiers"])),
        "final_type_dist": dict(Counter(r["final_answer"]["type"] for r in passed)),
        "axis_dist": dict(Counter(r["primary_axis"] for r in passed)),
    }
    (CRIT_DIR / "gate_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def axis_of(entry: dict) -> str:
    ty = entry["final_answer"]["type"]
    if ty == "numeric":
        return "quantitative"
    if ty == "expression":
        return "symbolic"
    return entry["primary_axis"]


def cmd_export(args) -> None:
    bank: dict[str, dict] = {}
    with (CRIT_DIR / "criteria_bank.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            bank[rec["id"]] = rec
    print(f"criteria_bank: {len(bank)}")

    splits = {n: (SPLITDIR / f"{n}.txt").read_text().split() for n in ("test", "dev", "train")}
    t = pq.read_table(POOL)
    t = t.filter(pc.is_in(t["id"], value_set=pa.array(list(bank))))
    ei = t["extra_info"]
    cols = {
        "id": t["id"].to_pylist(),
        "prompt": t["prompt"].to_pylist(),
        "question": pc.struct_field(ei, "question").to_pylist(),
        "subject": t["subject"].to_pylist(),
        "difficulty": pc.struct_field(ei, "difficulty").to_pylist(),
        "from": pc.struct_field(ei, "from").to_pylist(),
        "orig_ds": t["data_source"].to_pylist(),
    }

    rows = []
    for k in range(len(cols["id"])):
        rid = cols["id"][k]
        entry = bank[rid]
        ax = axis_of(entry)
        prompt = cols["prompt"][k]
        if not any("boxed" in (m.get("content") or "") for m in prompt):
            prompt = [*prompt]
            prompt[-1] = {**prompt[-1], "content": prompt[-1]["content"] + "\n\n" + BOXED_HINT}
        rows.append(
            {
                "id": rid,
                "prompt": prompt,
                "data_source": f"cropd/{ax}",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(entry["final_answer"], ensure_ascii=False),
                },
                "extra_info": {
                    "id": rid,
                    "criteria": json.dumps(entry["criteria"], ensure_ascii=False),
                    "domain": cols["subject"][k],
                    "capability": ax,
                    "final_type": entry["final_answer"]["type"],
                    "orig_source": cols["from"][k] or cols["orig_ds"][k] or "",
                    "difficulty": cols["difficulty"][k],
                },
            }
        )
    by_id = {r["id"]: r for r in rows}

    VERL_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, ids in splits.items():
        sel = [by_id[i] for i in ids if i in by_id]
        if not sel:
            print(f"split {name}: 0 rows, skip")
            continue
        pq.write_table(pa.Table.from_pylist(sel), VERL_DIR / f"{name}.parquet")
        stats[name] = {
            "rows": len(sel),
            "axis": dict(Counter(r["data_source"] for r in sel)),
            "final_type": dict(Counter(r["extra_info"]["final_type"] for r in sel)),
        }
        if name == "train":
            for ax, ty in (("quant", "numeric"), ("symbolic", "expression")):
                sub = [r for r in sel if r["extra_info"]["final_type"] == ty]
                if sub:
                    pq.write_table(pa.Table.from_pylist(sub), VERL_DIR / f"train_{ax}.parquet")
                    stats[f"train_{ax}"] = {"rows": len(sub)}
    (VERL_DIR / "export_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Criteria for the verifiable pool: annotate with the judge, gate against the reference answer, export verl parquet files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("annotate", help="annotate criteria with the judge (async, resumable)")
    p.add_argument("--splits", default="test,dev,train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids-file", default=None)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--regen", action="store_true", help="regeneration round: append the hint that the previous normalization failed the ground-truth check")
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--model", default="judge")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.set_defaults(fn=cmd_annotate)
    p = sub.add_parser("gate", help="re-verify annotated criteria on the reference answer and build the criteria bank")
    p.add_argument("--final", action="store_true", help="after the regeneration round: mark still-failing ids as unusable")
    p.add_argument("--procs", type=int, default=32)
    p.set_defaults(fn=cmd_gate)
    sub.add_parser("export", help="export verl parquet files and per-axis expert slices").set_defaults(fn=cmd_export)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
