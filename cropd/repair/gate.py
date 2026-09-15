from __future__ import annotations

import math
from dataclasses import dataclass, field

from cropd.verifiers import run_verifier


def criterion_score(text: str, criterion: dict) -> float | None:
    res = run_verifier(criterion["verifier"], text, criterion["verifier_args"])
    return None if res.passed is None else float(res.passed)


@dataclass
class RepairCandidate:
    expert: str
    y_tilde: str
    delta: float | None = None
    edit_ratio: float = 0.0
    gated: bool = False
    q: float = 0.0
    detail: dict = field(default_factory=dict)


def evaluate_candidate(y: str, cand: RepairCandidate, criterion: dict) -> RepairCandidate:
    s0 = criterion_score(y, criterion)
    s1 = criterion_score(cand.y_tilde, criterion)
    cand.delta = None if (s0 is None or s1 is None) else s1 - s0
    cand.detail.update({"s0": s0, "s1": s1})
    return cand


def gate_and_route(
    cands: list[RepairCandidate],
    delta_min: float = 0.0,
    tau: float = 0.5,
    max_edit_ratio: float = 0.35,
) -> list[RepairCandidate]:
    for c in cands:
        c.gated = c.delta is not None and c.delta > delta_min and c.edit_ratio <= max_edit_ratio
        c.q = 0.0
    gated = [c for c in cands if c.gated]
    if not gated:
        return cands
    mx = max(c.delta for c in gated)
    exps = [math.exp((c.delta - mx) / max(tau, 1e-6)) for c in gated]
    z = sum(exps)
    for c, e in zip(gated, exps):
        c.q = e / z
    return cands


def select_teacher(cands: list[RepairCandidate]) -> RepairCandidate | None:
    gated = [c for c in cands if c.gated]
    return max(gated, key=lambda c: c.delta) if gated else None
