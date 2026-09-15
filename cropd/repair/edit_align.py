from __future__ import annotations

import difflib


def char_edit_spans(y: str, y_tilde: str) -> list[tuple[int, int]]:
    sm = difflib.SequenceMatcher(None, y, y_tilde, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:
            spans.append((max(0, i1 - 1), min(len(y), i1 + 1)))
        else:
            spans.append((i1, i2))
    return merge_spans(spans)


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def edit_ratio(y: str, y_tilde: str) -> float:
    if not y:
        return 1.0
    sm = difflib.SequenceMatcher(None, y, y_tilde, autojunk=False)
    edited = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return edited / len(y)


def spans_to_token_mask(
    spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
    expand: int = 2,
) -> list[bool]:
    n = len(offsets)
    hit = [False] * n
    for t, (ts, te) in enumerate(offsets):
        if ts == te:
            continue
        for s, e in spans:
            if ts < e and s < te:
                hit[t] = True
                break
    if expand > 0:
        base = hit[:]
        for t, h in enumerate(base):
            if h:
                for d in range(1, expand + 1):
                    if t - d >= 0:
                        hit[t - d] = True
                    if t + d < n:
                        hit[t + d] = True
    return hit


def token_mask_for_repair(
    tokenizer,
    y: str,
    y_tilde: str,
    expand: int = 2,
) -> tuple[list[bool], float]:
    spans = char_edit_spans(y, y_tilde)
    enc = tokenizer(y, return_offsets_mapping=True, add_special_tokens=False)
    mask = spans_to_token_mask(spans, enc["offset_mapping"], expand=expand)
    return mask, edit_ratio(y, y_tilde)
