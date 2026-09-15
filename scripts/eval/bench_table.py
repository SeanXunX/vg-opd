from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path("data/processed/bench_v1/results")
COLS = [("gpqa_diamond", "GPQA-D"), ("math500", "MATH500"), ("mmlu_pro", "MMLU-Pro"), ("scibench", "SciBench"),
        ("rar_science", "RaR-Sci"), ("aime26", "AIME26"), ("chembench", "ChemBench"), ("rar_med", "RaR-Med")]


def score(d: dict, base: str):
    for k in (base, f"{base}_sub2000", f"{base}_sub500"):
        v = d.get(k)
        if isinstance(v, dict):
            return v.get("score")
    return None


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem for p in R.glob("*.json"))
    rows = []
    for n in names:
        p = R / f"{n}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        vals = [score(d, k) for k, _ in COLS]
        got = [v for v in vals if v is not None]
        mean4 = sum(vals[:4]) / 4 if all(v is not None for v in vals[:4]) else None
        m5v = [v for v in vals[:5] if v is not None]
        mean5 = sum(m5v) / 5 if len(m5v) == 5 else None
        rows.append((n, vals, mean4, mean5))
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or r[2] or 0)))
    print("| model | " + " | ".join(c for _, c in COLS) + " | mean4 | mean5 |")
    print("|---|" + "---|" * (len(COLS) + 2))
    for n, vals, m4, m5 in rows:
        f = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"| {n} | " + " | ".join(f(v) for v in vals) + f" | {f(m4)} | {f(m5)} |")


if __name__ == "__main__":
    main()
