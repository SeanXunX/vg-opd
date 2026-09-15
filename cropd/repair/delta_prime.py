from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cropd.repair.gate import RepairCandidate, criterion_score, gate_and_route

DEFAULT_AXIS_EXPERTS: dict[str, list[str]] = {
    "quantitative": ["quant"],
    "symbolic": ["symbolic"],
    "mechanistic": ["mech"],
    "evidence": ["evidence"],
}


def _expert_answer(question: str, expert: str, max_tokens: int) -> str | None:
    from cropd.repair.client import _client

    try:
        resp = _client().chat.completions.create(
            model=expert, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": question}],
        )
        text = resp.choices[0].message.content
        return text.strip() if text and text.strip() else None
    except Exception:
        return None


def probe_expert_answers(
    question: str,
    experts: list[str],
    max_tokens: int = 3072,
    concurrency: int = 4,
) -> dict[str, str | None]:
    with ThreadPoolExecutor(max(1, min(concurrency, len(experts)))) as ex:
        return dict(zip(experts, ex.map(lambda e: _expert_answer(question, e, max_tokens), experts)))


def delta_prime_candidates(
    y: str,
    expert_answers: dict[str, str | None],
    criterion: dict,
    axis_experts: dict[str, list[str]] | None = None,
    delta_min: float = 0.0,
    tau: float = 0.5,
) -> list[RepairCandidate]:
    w = criterion.get("weight", 1.0)
    if isinstance(w, (int, float)) and w < 0:
        return []
    experts = (axis_experts or DEFAULT_AXIS_EXPERTS).get(criterion.get("axis", ""), [])
    s0 = criterion_score(y, criterion)
    cands: list[RepairCandidate] = []
    for e in experts:
        y_k = expert_answers.get(e)
        c = RepairCandidate(expert=e, y_tilde=y_k or "", edit_ratio=0.0)
        if y_k is None or s0 is None:
            c.delta = None
        else:
            s1 = criterion_score(y_k, criterion)
            c.delta = None if s1 is None else s1 - s0
            c.detail = {"s0": s0, "s1": s1, "mode": "delta_prime"}
        cands.append(c)
    gate_and_route(cands, delta_min=delta_min, tau=tau, max_edit_ratio=float("inf"))
    return cands
