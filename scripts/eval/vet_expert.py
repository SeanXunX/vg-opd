from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = Path(os.environ.get("CROPD_MODELS", "models"))
CAP = {"quant": "quantitative", "symbolic": "symbolic", "mech": "mechanistic", "evidence": "evidence"}
ZERO = {"quantitative": 0.3821, "symbolic": 0.2032, "mechanistic": 0.6092, "evidence": 0.4961}


def eval_one(path: Path, closed: bool) -> dict:
    name = path.name + ("_closed" if closed else "")
    out = ROOT / f"data/processed/eval_a4/{name}.json"
    if not out.exists():
        cmd = [sys.executable, "scripts/eval/eval_matrix.py", "--name", path.name, "--model", str(path)]
        if closed:
            cmd.append("--closed-only")
        env = dict(os.environ, CROPD_EVAL_NOTHINK="1", CROPD_EVAL_MAX_TOKENS=os.environ.get("CROPD_EVAL_MAX_TOKENS", "4096"),
                   no_proxy="localhost,127.0.0.1")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(CAP))
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--no-link", action="store_true", help="evaluate only; do not create the winner symlink")
    a = ap.parse_args()
    cap = CAP[a.axis]
    closed = a.axis in ("quant", "symbolic")
    rows = []
    for c in a.cands:
        p = Path(c)
        d = eval_one(p, closed)
        s = d[f"cropd/{cap}"]["score"]
        rows.append((s, p, d))
        print(f"{p.name:32s} {cap}={s:.4f}  " + "  ".join(f"{k.split('/')[1][:4]}={v['score']:.3f}" for k, v in d.items() if k.startswith("cropd/")), flush=True)
    rows.sort(key=lambda r: -r[0])
    best_s, best_p, _ = rows[0]
    flag = "OK" if best_s > ZERO[cap] else "WARN: <= zero"
    print(f"WINNER {best_p.name} {cap}={best_s:.4f} vs zero {ZERO[cap]:.4f} -> {flag}")
    if a.no_link:
        return
    link = M / f"8b-expert-{a.axis}"
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(best_p)
        print(f"linked {link} -> {best_p}")
    else:
        print(f"!! {link} is a real dir (provisional fold); rename it manually then re-run to link", file=sys.stderr)


if __name__ == "__main__":
    main()
