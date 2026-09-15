from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260818


def main() -> None:
    ap = argparse.ArgumentParser(description="Axis-balanced student training mix of the verifiable and open-ended pools")
    ap.add_argument("--per-axis", type=int, default=6000)
    args = ap.parse_args()

    by_axis: dict[str, list[dict]] = defaultdict(list)
    for f in ["verl_v1/train.parquet", "verl_open_v1/train.parquet"]:
        t = pq.read_table(ROOT / "data/processed" / f)
        for i in range(len(t)):
            row = {c: t[c][i].as_py() for c in t.column_names}
            row["extra_info"] = {k: (str(v) if k == "difficulty" else v) for k, v in row["extra_info"].items()}
            by_axis[row["data_source"]].append(row)

    rng = random.Random(SEED)
    picked: list[dict] = []
    for ax, rows in sorted(by_axis.items()):
        take = rng.sample(rows, min(args.per_axis, len(rows)))
        picked += take
        print(f"{ax}: {len(take)}/{len(rows)}")
    rng.shuffle(picked)

    out = ROOT / "data/processed/verl_merge_v1"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(picked), out / "train.parquet")
    dev: list[dict] = []
    for f in ["verl_v1/dev.parquet", "verl_open_v1/dev.parquet"]:
        t = pq.read_table(ROOT / "data/processed" / f)
        for i in range(len(t)):
            row = {c: t[c][i].as_py() for c in t.column_names}
            row["extra_info"] = {k: (str(v) if k == "difficulty" else v) for k, v in row["extra_info"].items()}
            dev.append(row)
    rng.shuffle(dev)
    pq.write_table(pa.Table.from_pylist(dev), out / "dev.parquet")
    print(f"dev: {len(dev)} rows")
    man = ROOT / "data/splits/merge_v1"
    man.mkdir(parents=True, exist_ok=True)
    (man / "train.txt").write_text("\n".join(r["id"] for r in picked) + "\n")
    stats = {"total": len(picked), "axis": dict(Counter(r["data_source"] for r in picked)), "dev": len(dev), "seed": SEED}
    (man / "STATS.md").write_text("# merge_v1\n\n" + json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
