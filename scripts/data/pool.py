from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "MiniByte-666/Dr.SCI"
SNAPSHOT = "81c1fa29725dc5df3262ae1225f2289349531f16"
HF_CACHE = Path.home() / ".cache/huggingface/hub/datasets--MiniByte-666--Dr.SCI/snapshots" / SNAPSHOT
RAW_DIR = ROOT / "data/raw/drsci"
RAW_FILES = ["Dr_SCI_verifiable.parquet", "Dr_SCI_open-ended.parquet", "README.md"]
LICENSE_NOTE = (
    "The MIT label of the reproduction repository does not hold: upstream sources are NaturalReasoning (CC-BY-NC), "
    "MegaScience (CC-BY-NC-SA), RaR-Science (no explicit license) and WebInstruct-Verified (Apache-2.0); "
    "usable for research training, but released artifacts must declare the NC terms and must not be relicensed or redistributed"
)
POOL = ROOT / "data/processed/pool_v1.parquet"
SPLITDIR = ROOT / "data/splits/v1"
SUBJECTS = ["physics", "chemistry"]
SEED = 20260810
N_TEST, N_DEV = 1500, 1500
NGRAM = 13
MIRRORS = ["nmayorga7/gpqa_diamond", "fingertap/GPQA-Diamond", "hendrydong/gpqa_diamond"]
QCOLS = ["Question", "question", "problem", "prompt", "query"]
NUM_RE = re.compile(r"^\s*-?[\d.,]+([eE][+-]?\d+)?\s*(\\?[a-zA-Z\u03bc\u00b0%^{}/\s.*]*)?\s*$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_stage(args) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in RAW_FILES:
        src, dst = HF_CACHE / name, RAW_DIR / name
        if not src.exists():
            sys.exit(f"missing in HF cache: {src}")
        if not dst.exists():
            shutil.copyfile(src, dst)
        entries[name] = {"bytes": dst.stat().st_size, "sha256": sha256(dst)}
    source = {
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "snapshot": SNAPSHOT,
        "staged_date": date.today().isoformat(),
        "files": entries,
        "license_note": LICENSE_NOTE,
    }
    (RAW_DIR / "SOURCE.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(source, ensure_ascii=False, indent=2))


def rough_type(g: str) -> str:
    g = g.strip()
    if len(g) <= 3:
        return "short_mcq"
    if len(g) < 40 and NUM_RE.match(g):
        return "numeric"
    if any(s in g for s in ("\\", "^", "_", "=", "frac")):
        return "expression"
    return "other"


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def cmd_build(args) -> None:
    t = pq.read_table(RAW_DIR / "Dr_SCI_verifiable.parquet")
    t = t.append_column("orig_row", pa.array(range(t.num_rows), pa.int64()))
    mask = pc.is_in(pc.struct_field(t["extra_info"], "subject"), value_set=pa.array(SUBJECTS))
    t = t.filter(mask)

    ei = t["extra_info"]
    q = pc.struct_field(ei, "question").to_pylist()
    subj = pc.struct_field(ei, "subject").to_pylist()
    diff = pc.struct_field(ei, "difficulty").to_pylist()
    gt = pc.struct_field(t["reward_model"], "ground_truth").to_pylist()
    orig = t["orig_row"].to_pylist()

    seen: set[str] = set()
    keep = []
    for qq in q:
        h = hashlib.md5(norm_q(qq).encode()).hexdigest()
        keep.append(h not in seen)
        seen.add(h)

    t = (
        t.append_column("id", pa.array([f"drsci-v-{i:06d}" for i in orig]))
        .append_column("subject", pa.array(subj))
        .append_column("gt_rough_type", pa.array([rough_type(g) for g in gt]))
        .append_column(
            "difficulty_bucket",
            pa.array(["d0" if d == 0 else ("d1" if d <= 0.25 else "d2") for d in diff]),
        )
        .filter(pa.array(keep))
    )
    POOL.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, POOL)

    print(f"pool_v1: {t.num_rows} rows (dropped {len(keep) - sum(keep)} dups) -> {POOL}")
    for col in ("subject", "gt_rough_type", "difficulty_bucket"):
        print(col, {d["values"]: d["counts"] for d in pc.value_counts(t[col]).to_pylist()})


def allocate(sizes: dict, total: int) -> dict:
    n = sum(sizes.values())
    quota = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rest = total - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k)):
        if rest == 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            rest -= 1
    assert sum(alloc.values()) == total
    return alloc


def cmd_split(args) -> None:
    t = pq.read_table(POOL, columns=["id", "subject", "gt_rough_type", "difficulty_bucket"])
    strata: dict[tuple, list[str]] = defaultdict(list)
    for i, s, g, d in zip(*(t[c].to_pylist() for c in t.column_names)):
        strata[(s, g, d)].append(i)

    rng = random.Random(SEED)
    for k in sorted(strata):
        strata[k].sort()
        rng.shuffle(strata[k])

    sizes = {k: len(v) for k, v in strata.items()}
    picks = {"test": {}, "dev": {}}
    a_test = allocate(sizes, N_TEST)
    for k in sorted(strata):
        picks["test"][k] = strata[k][: a_test[k]]
    remaining = {k: sizes[k] - a_test[k] for k in sizes}
    a_dev = allocate(remaining, N_DEV)
    for k in sorted(strata):
        picks["dev"][k] = strata[k][a_test[k] : a_test[k] + a_dev[k]]

    splits = {
        "test": sorted(x for v in picks["test"].values() for x in v),
        "dev": sorted(x for v in picks["dev"].values() for x in v),
    }
    taken = set(splits["test"]) | set(splits["dev"])
    splits["train"] = sorted(x for v in strata.values() for x in v if x not in taken)

    SPLITDIR.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (SPLITDIR / f"{name}.txt").write_text("\n".join(ids) + "\n")

    lines = [
        f"# splits v1 (seed={SEED}, stratified by subject x gt_rough_type x difficulty_bucket)",
        "",
        f"pool={t.num_rows}  test={len(splits['test'])}  dev={len(splits['dev'])}  train={len(splits['train'])}",
        "",
        "| stratum | pool | test | dev |",
        "|---|---|---|---|",
    ]
    for k in sorted(strata):
        lines.append(f"| {'/'.join(k)} | {sizes[k]} | {a_test[k]} | {a_dev[k]} |")
    (SPLITDIR / "STATS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def ngrams(words: list[str]):
    return (tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1))


def load_gpqa():
    import datasets

    for repo in MIRRORS:
        try:
            ds = datasets.load_dataset(repo, split="train")
        except Exception as e:
            print(f"mirror {repo} failed: {type(e).__name__}")
            continue
        col = next((c for c in QCOLS if c in ds.column_names), None)
        if col is None or not (150 <= len(ds) <= 250):
            print(f"mirror {repo} unusable: cols={ds.column_names} n={len(ds)}")
            continue
        return repo, len(ds), [r[col] for r in ds]
    raise SystemExit("no usable GPQA-Diamond mirror")


def cmd_contam(args) -> None:
    repo, n_gpqa, questions = load_gpqa()
    bench_grams = {g for q in questions for g in ngrams(norm_words(q))}
    print(f"GPQA-Diamond via {repo}: {n_gpqa} questions, {len(bench_grams)} {NGRAM}-grams")

    t = pq.read_table(POOL, columns=["id", "extra_info"])
    ids = t["id"].to_pylist()
    qs = pc.struct_field(t["extra_info"], "question").to_pylist()
    hits = {i for i, q in zip(ids, qs) if any(g in bench_grams for g in ngrams(norm_words(q)))}
    print(f"pool hits: {len(hits)}")

    splits = {n: (SPLITDIR / f"{n}.txt").read_text().split() for n in ("test", "dev", "train")}
    hit_by = {n: sorted(set(v) & hits) for n, v in splits.items()}
    quarantined = hit_by["train"]
    if quarantined:
        keep = [x for x in splits["train"] if x not in set(quarantined)]
        (SPLITDIR / "train.txt").write_text("\n".join(keep) + "\n")
    (SPLITDIR / "quarantine.txt").write_text("\n".join(quarantined) + ("\n" if quarantined else ""))

    report = [
        f"# Contamination check (GPQA-Diamond, exact {NGRAM}-gram word overlap)",
        "",
        f"- benchmark source: `{repo}` (public mirror; the official Idavidrein/gpqa is gated), n={n_gpqa}",
        f"- pool hits {len(hits)}; train hits {len(quarantined)} -> moved to quarantine.txt and train.txt rewritten",
        f"- test hits {len(hit_by['test'])} / dev hits {len(hit_by['dev'])} (internal evaluation sets: reported only, not removed)",
        "- decontamination against the full external benchmark set is handled in the evaluation harness",
    ]
    if hit_by["test"] or hit_by["dev"]:
        report.append(f"- details: test={hit_by['test']} dev={hit_by['dev']}")
    (SPLITDIR / "CONTAM.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dr.SCI verifiable pool: stage raw files, build the pool, split it, check contamination")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", help="copy the Dr.SCI snapshot from the local Hugging Face cache and record checksums").set_defaults(fn=cmd_stage)
    sub.add_parser("build", help="build the physics/chemistry pool with stable ids and question dedup").set_defaults(fn=cmd_build)
    sub.add_parser("split", help="stratified test/dev/train split with a fixed seed").set_defaults(fn=cmd_split)
    sub.add_parser("contam", help="13-gram contamination check against GPQA-Diamond; quarantine train hits").set_defaults(fn=cmd_contam)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
