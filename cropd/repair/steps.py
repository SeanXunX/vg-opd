from __future__ import annotations

import re

_STEP_HEAD_RE = re.compile(r"^\s*(?:Step\s*\d+|\d+[.)]|\(\d+\)|[-*]\s|##)", re.IGNORECASE)


def split_steps(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    bounds = {0, len(text)}
    for m in re.finditer(r"\n\s*\n", text):
        bounds.add(m.end())
    pos = 0
    for line in text.splitlines(keepends=True):
        if _STEP_HEAD_RE.match(line):
            bounds.add(pos)
        pos += len(line)
    for m in re.finditer(r"\$\$.+?\$\$", text, re.DOTALL):
        bounds.add(m.start())
        bounds.add(m.end())
    cuts = sorted(b for b in bounds if 0 <= b <= len(text))
    spans = [(s, e) for s, e in zip(cuts, cuts[1:]) if text[s:e].strip()]
    return spans or [(0, len(text))]


def first_satisfying_step(text: str, criterion: dict) -> int | None:
    from cropd.repair.gate import criterion_score

    steps = split_steps(text)
    for i, (_s, e) in enumerate(steps):
        if criterion_score(text[:e], criterion) == 1.0:
            return i
    return None


def step_span_fallback(y: str, y_tilde: str, criterion: dict) -> tuple[int, int] | None:
    j = first_satisfying_step(y_tilde, criterion)
    if j is None:
        return None
    y_steps = split_steps(y)
    if not y_steps:
        return None
    return y_steps[min(j, len(y_steps) - 1)]
