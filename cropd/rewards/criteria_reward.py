from __future__ import annotations

import json
import os
import time

from cropd.verifiers import extract_boxed, run_verifier, strip_think

W_FINAL = 0.8
W_INT = 0.2
TIME_BUDGET = float(os.environ.get("CROPD_REWARD_TIME_BUDGET", "12.0"))

_AXIS_KEY = {"quantitative": "r_quant", "symbolic": "r_symbolic", "mechanistic": "r_mech", "evidence": "r_evidence"}


def _wmean(pairs: list[tuple[float, float]]) -> float:
    tot = sum(w for w, _ in pairs)
    return sum(w * s for w, s in pairs) / tot if tot > 0 else 0.0


def _signed_mean(pairs: list[tuple[float, float]]) -> float:
    pos = sum(w for w, _ in pairs if w > 0)
    if pos <= 0:
        return 0.0
    return min(1.0, max(0.0, sum(w * s for w, s in pairs) / pos))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    solution_str = strip_think(solution_str or "")
    out = {
        "score": 0.0, "r_final": 0.0, "r_quant": 0.0, "r_symbolic": 0.0,
        "r_mech": 0.0, "r_evidence": 0.0, "r_format": 0.0, "crit_detail": "[]",
    }
    try:
        gt = json.loads(ground_truth or "{}")
    except (TypeError, ValueError):
        gt = {}
    is_open = isinstance(gt, dict) and gt.get("type") == "open"

    if is_open:
        if not (solution_str or "").strip():
            return out
    else:
        boxed = extract_boxed(solution_str or "")
        if boxed is None or not boxed.strip():
            return out
    out["r_format"] = 1.0

    try:
        criteria = json.loads((extra_info or {}).get("criteria") or "[]")
    except (TypeError, ValueError):
        criteria = []
    if not criteria:
        out["score"] = out["r_format"]
        return out

    deadline = time.monotonic() + TIME_BUDGET
    axis_pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in _AXIS_KEY.values()}
    detail = []
    for c in criteria:
        remain = deadline - time.monotonic()
        if remain <= 0.05:
            detail.append({"cid": c.get("cid"), "pass": None, "note": "time_budget"})
            continue
        args = dict(c.get("verifier_args") or {})
        if c.get("verifier") == "sympy_equiv":
            args["timeout"] = min(float(args.get("timeout", 8.0)), remain)
        res = run_verifier(c.get("verifier", ""), solution_str, args)
        detail.append({"cid": c.get("cid"), "pass": res.passed})
        if c.get("cid") == "final":
            out["r_final"] = float(res.passed is True)
        elif res.passed is not None:
            key = _AXIS_KEY.get(c.get("axis", ""), None)
            if key:
                w = c.get("weight", 0.5)
                if is_open:
                    s = res.score if isinstance(res.score, (int, float)) else float(res.passed)
                    axis_pairs[key].append((float(w) if isinstance(w, (int, float)) and w != 0 else 0.5, float(s)))
                else:
                    axis_pairs[key].append((float(w) if isinstance(w, (int, float)) and w > 0 else 0.5, float(res.passed)))

    for key, pairs in axis_pairs.items():
        out[key] = _signed_mean(pairs) if is_open else (_wmean(pairs) if pairs else 0.0)
        if not pairs:
            detail.append({"axis": key, "note": "absent"})

    expert_axis = str(data_source or "").rsplit("/", 1)[-1]
    expert_key = _AXIS_KEY.get(expert_axis)
    if is_open:
        out["score"] = out["r_format"] * (out[expert_key] if expert_key else 0.0)
    else:
        int_term = _wmean(axis_pairs[expert_key]) if expert_key and axis_pairs[expert_key] else out["r_final"]
        out["score"] = out["r_format"] * (W_FINAL * out["r_final"] + W_INT * int_term)
    out["crit_detail"] = json.dumps(detail, ensure_ascii=False)
    return out
