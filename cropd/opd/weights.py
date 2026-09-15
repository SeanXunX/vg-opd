from __future__ import annotations


def assemble_token_weights(
    n_tokens: int,
    contributions: list[tuple[list[bool], float]],
) -> list[float]:
    w = [0.0] * n_tokens
    for mask, scalar in contributions:
        if scalar <= 0:
            continue
        m = min(len(mask), n_tokens)
        for t in range(m):
            if mask[t] and scalar > w[t]:
                w[t] = scalar
    return w


def coverage(w: list[float]) -> float:
    return sum(1 for x in w if x > 0) / len(w) if w else 0.0
