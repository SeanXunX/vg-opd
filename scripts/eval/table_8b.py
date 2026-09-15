from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data/processed/bench_v1/results"
EVAL = ROOT / "data/processed/eval_a4"
ARMS = [("Vanilla 8B", "qwen3-8b-nothink"), ("OPSD (standalone)", "8b-t11s-opsd-pure"), ("MOPD (standalone)", "8b-t7s-mopd-pure"),
        ("GRPO", "8b-t8q-grpo"), ("dGRPO (GRPO+OPSD)", "8b-t11-opsd"), ("CriPO", "8b-t9-cripo"), ("VG-OPD (ours)", "8b-t10-vgfusion")]
FIVE = ["gpqa_diamond", "mmlu_pro_sub2000", "math500", "scibench", "rar_science_sub500"]
COLLAPSED = {"8b-t11-opsd-s50", "8b-t11s-opsd-pure-s75", "8b-t7s-mopd-pure-s75"}


def row(name: str) -> dict | None:
    bp = BENCH / f"{name}.json"
    if not bp.exists():
        return None
    b = json.loads(bp.read_text())
    need = FIVE + ["chembench", "rar_med_sub150"]
    if any(k not in b for k in need):
        return None
    ep = EVAL / f"{name}.json"
    dev = None
    if ep.exists():
        e = json.loads(ep.read_text())
        ax = [v["score"] for k, v in e.items() if k.startswith("cropd/")]
        dev = sum(ax) / len(ax) if ax else None
    sci = (b["gpqa_diamond"]["score"] + b["rar_science_sub500"]["score"] + b["scibench"]["score"]) / 3
    dom = (b["chembench"]["score"] + b["rar_med_sub150"]["score"]) / 2
    gen = (b["mmlu_pro_sub2000"]["score"] + b["math500"]["score"]) / 2
    return {"name": name, "dev": dev, "sci": sci, "dom": dom, "gen": gen, "avg": (sci + dom + gen) / 3,
            "aime": b.get("aime26", {}).get("score"), "five": sum(b[k]["score"] for k in FIVE) / 5,
            "raw": {k: b[k]["score"] for k in need + (["aime26"] if "aime26" in b else [])}}


def fmt(x, mult=100.0, nd=1):
    return "-" if x is None else f"{x * mult:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true", help="print only the representative checkpoint (highest Avg) per arm")
    a = ap.parse_args()
    print("| Method | ckpt | Internal dev | SciR | DomSci | GenR | Avg | AIME26 (clean) | five-mean |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, prefix in ARMS:
        cands = sorted(glob.glob(str(BENCH / f"{prefix}*.json")))
        rows = [r for r in (row(Path(c).stem) for c in cands) if r]
        if not rows:
            print(f"| {label} | (pending) | | | | | | | |")
            continue
        rows.sort(key=lambda r: (r["name"] in COLLAPSED, -r["avg"]))
        show = rows[:1] if a.pick else rows
        for i, r in enumerate(show):
            ck = re.sub(rf"^{re.escape(prefix)}-?", "", r["name"]) or "-"
            if r["name"] in COLLAPSED:
                ck += " (collapsed)"
            mark = "**" if (i == 0 and len(rows) > 1) else ""
            print(f"| {label if i == 0 else ''} | {mark}{ck}{mark} | {fmt(r['dev'])} | {fmt(r['sci'])} | {fmt(r['dom'])} | {fmt(r['gen'])} | {fmt(r['avg'])} | {fmt(r['aime'])} | {fmt(r['five'])} |")
    print("\nNote: the seven benchmark families were included in training (in-distribution); AIME26 and the internal dev set were not. Representative checkpoint = highest Avg per arm (bold).")


if __name__ == "__main__":
    main()
