from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACE = Path("outputs/trace")
OUT = Path("data/processed/analysis_a6")
FIGS = Path("figs")
EXPERTS = ["mech", "evidence", "quant", "symbolic"]
CAPS = ["mechanistic", "evidence", "quantitative", "symbolic"]


def load(exp: str) -> list[dict]:
    recs = []
    for f in sorted((TRACE / exp).glob("trace_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def _q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def analyze(recs: list[dict]) -> dict:
    n = len(recs)
    route = {c: Counter() for c in CAPS}
    crit_n, crit_g, delta_sum, delta_n, q_g = Counter(), Counter(), Counter(), Counter(), defaultdict(list)
    attr = Counter()
    pos = [0.0] * 10
    pos_n = 0
    frac, cov, wmax, R, ngated, rewards = [], [], [], [], [], []
    gate_by_reward = {"r=0": [0, 0], "0<r<0.5": [0, 0], "r>=0.5": [0, 0]}
    for r in recs:
        cap = r.get("capability") or "?"
        route.setdefault(cap, Counter())[r["teacher"]] += 1
        frac.append(r["w_nonzero_frac"]); cov.append(r["coverage"]); wmax.append(r["w_max"])
        R.append(r["R"]); ngated.append(r["n_gated"])
        rw = r.get("reward")
        if rw is not None:
            rewards.append(rw)
        bucket = None if rw is None else ("r=0" if rw <= 0 else ("0<r<0.5" if rw < 0.5 else "r>=0.5"))
        h = r.get("pos_hist") or []
        tot = sum(h)
        if tot:
            pos = [p + hi / tot for p, hi in zip(pos, h)]
            pos_n += 1
        for c in r.get("crit", []):
            a = c.get("attr")
            if a:
                attr[a] += 1
            for cand in c.get("cands", []):
                e = cand.get("expert", "?")
                crit_n[e] += 1
                if bucket:
                    gate_by_reward[bucket][1] += 1
                if cand.get("gated"):
                    crit_g[e] += 1
                    q_g[e].append(float(cand.get("q", 0.0)))
                    if bucket:
                        gate_by_reward[bucket][0] += 1
                if cand.get("delta") is not None:
                    delta_sum[e] += float(cand["delta"]); delta_n[e] += 1
    experts = [e for e in EXPERTS if crit_n[e]] + sorted(e for e in crit_n if e not in EXPERTS)
    return {
        "n_records": n, "n_questions": len({r["qh"] for r in recs}),
        "per_capability": {c: sum(route[c].values()) for c in route if sum(route[c].values())},
        "routing_matrix": {c: dict(route[c]) for c in route if sum(route[c].values())},
        "cross_axis_frac": (sum(v for c, cnt in route.items() for t, v in cnt.items()
                                if c in CAPS and t != EXPERTS[CAPS.index(c)]) / n) if n else None,
        "gate_by_expert": {e: {"n_crit": crit_n[e], "n_gated": crit_g[e],
                               "gate_rate": round(crit_g[e] / crit_n[e], 4),
                               "mean_delta": round(delta_sum[e] / delta_n[e], 4) if delta_n[e] else None,
                               "mean_q_gated": round(st.fmean(q_g[e]), 4) if q_g[e] else None}
                           for e in experts},
        "gate_by_reward": {k: {"gated": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None}
                           for k, v in gate_by_reward.items()},
        "attr_mode_counts": dict(attr),
        "n_gated_per_sample": dict(sorted(Counter(ngated).items())),
        "sparsity": {"w_nonzero_frac": {"mean": round(st.fmean(frac), 4), "median": round(_q(frac, .5), 4),
                                        "p10": round(_q(frac, .1), 4), "p90": round(_q(frac, .9), 4)},
                     "criterion_mask_coverage_mean": round(st.fmean(cov), 4),
                     "w_max_mean": round(st.fmean(wmax), 4), "R_mean": round(st.fmean(R), 1)} if n else {},
        "position_density": [round(p / pos_n, 4) for p in pos] if pos_n else [],
        "reward_mean_traced": round(st.fmean(rewards), 4) if rewards else None,
    }


def markdown(exp: str, s: dict) -> str:
    g = s["gate_by_expert"]
    lines = [f"**{exp}**: {s['n_records']} trace records / {s['n_questions']} questions; mean reward of traced samples "
             f"{s['reward_mean_traced']}; cross-axis teaching fraction {s['cross_axis_frac']:.3f}" if s["cross_axis_frac"] is not None
             else f"**{exp}**: {s['n_records']} trace records"]
    lines.append("| expert | failed criteria | gated | gate rate | mean delta' | mean q (gated) |")
    lines.append("|---|---|---|---|---|---|")
    for e, v in g.items():
        lines.append(f"| {e} | {v['n_crit']} | {v['n_gated']} | {v['gate_rate']:.3f} | {v['mean_delta']} | {v['mean_q_gated']} |")
    rm = s["routing_matrix"]
    cols = [e for e in EXPERTS if any(e in rm[c] for c in rm)]
    lines.append("| capability \\ teacher | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for c, cnt in rm.items():
        lines.append(f"| {c} | " + " | ".join(str(cnt.get(e, 0)) for e in cols) + " |")
    sp = s["sparsity"]
    lines.append(f"sparsity: non-zero w fraction mean {sp['w_nonzero_frac']['mean']} / median {sp['w_nonzero_frac']['median']} "
                 f"(p10-p90 {sp['w_nonzero_frac']['p10']}-{sp['w_nonzero_frac']['p90']}); criterion mask coverage "
                 f"{sp['criterion_mask_coverage_mean']}; mean w_max {sp['w_max_mean']}; mean R {sp['R_mean']}")
    lines.append("position density (deciles, uniform=0.1): " + " ".join(f"{p:.3f}" for p in s["position_density"]))
    lines.append("gate rate by reward: " + ", ".join(f"{k} {v['rate']} (n={v['n']})" for k, v in s["gate_by_reward"].items()))
    lines.append(f"attribution source: {s['attr_mode_counts']}; gated criteria per sample: {s['n_gated_per_sample']}")
    return "\n".join(lines)


def plot(exp: str, s: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    rm = s["routing_matrix"]
    caps = list(rm)
    cols = [e for e in EXPERTS if any(e in rm[c] for c in caps)]
    mat = [[rm[c].get(e, 0) for e in cols] for c in caps]
    im = ax[0, 0].imshow(mat, cmap="Blues")
    ax[0, 0].set_xticks(range(len(cols)), cols); ax[0, 0].set_yticks(range(len(caps)), caps)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax[0, 0].text(j, i, str(v), ha="center", va="center", color="black" if v < max(map(max, mat)) / 2 else "white")
    ax[0, 0].set_title("routing: sample capability x chosen teacher"); fig.colorbar(im, ax=ax[0, 0])
    g = s["gate_by_expert"]
    ax[0, 1].bar(list(g), [v["gate_rate"] for v in g.values()], color="tab:green")
    for i, (e, v) in enumerate(g.items()):
        ax[0, 1].text(i, v["gate_rate"] + 0.01, f"n={v['n_crit']}\ndelta'={v['mean_delta']}", ha="center", fontsize=8)
    ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_title("criterion-level gate acceptance by expert")
    pd = s["position_density"]
    ax[1, 0].bar(range(1, 11), pd, color="tab:blue"); ax[1, 0].axhline(0.1, color="gray", ls="--", lw=1)
    ax[1, 0].set_xlabel("response decile"); ax[1, 0].set_title("where non-zero distill weights fall (mean per-sample density)")
    ax[1, 1].hist([r for r in s["_frac"]], bins=20, color="tab:orange")
    ax[1, 1].set_xlabel("fraction of response tokens with w>0"); ax[1, 1].set_title("mask sparsity")
    fig.suptitle(f"{exp}: trace n={s['n_records']} ({s['n_questions']} questions)")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"trace_{exp}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    exps = sys.argv[1:] or sorted(p.name for p in TRACE.iterdir() if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    for exp in exps:
        recs = load(exp)
        if not recs:
            print(f"[skip] {exp}: no records"); continue
        s = analyze(recs)
        s["_frac"] = [r["w_nonzero_frac"] for r in recs]
        plot(exp, s)
        del s["_frac"]
        (OUT / f"trace_{exp}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
        print(markdown(exp, s)); print()


if __name__ == "__main__":
    main()
