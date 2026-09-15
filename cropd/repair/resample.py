from __future__ import annotations

from cropd.repair.client import continue_from
from cropd.repair.gate import RepairCandidate, criterion_score
from cropd.repair.edit_align import edit_ratio
from cropd.repair.steps import split_steps


def resample_repair(
    question: str,
    y: str,
    criterion: dict,
    expert: str,
    max_back: int = 3,
    max_tokens: int = 2048,
) -> RepairCandidate:
    s0 = criterion_score(y, criterion)
    spans = split_steps(y)
    cuts = [s for s, _ in reversed(spans[1:])][:max_back] or [len(y) // 2]
    tried = 0
    for cut in cuts:
        prefix = y[:cut]
        cont = continue_from(question, prefix, expert, max_tokens=max_tokens)
        if cont is None:
            continue
        tried += 1
        y_t = prefix + cont
        s1 = criterion_score(y_t, criterion)
        if s1 is not None and s0 is not None and s1 > s0:
            c = RepairCandidate(expert=expert, y_tilde=y_t, edit_ratio=edit_ratio(y, y_t))
            c.delta = s1 - s0
            c.detail = {"s0": s0, "s1": s1, "cut": cut, "mode": "resample", "tried": tried}
            return c
    c = RepairCandidate(expert=expert, y_tilde="", edit_ratio=1.0)
    c.delta = None if s0 is None else 0.0
    c.detail = {"s0": s0, "mode": "resample", "tried": tried, "no_flip": True}
    return c
