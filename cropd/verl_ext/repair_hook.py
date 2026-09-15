from __future__ import annotations

import asyncio
import os
import json
import random
from dataclasses import dataclass, field

from cropd.opd.weights import assemble_token_weights, coverage
from cropd.repair.edit_align import spans_to_token_mask, token_mask_for_repair
from cropd.repair.gate import RepairCandidate, criterion_score, gate_and_route, select_teacher
from cropd.repair.prompts import repair_messages
from cropd.repair.steps import split_steps

AXIS_EXPERTS = {
    "quantitative": ["quant"],
    "symbolic": ["symbolic"],
    "mechanistic": ["mech"],
    "evidence": ["evidence"],
}
OMEGA_GAMMA = float(os.environ.get("CROPD_OMEGA_GAMMA", "0.5"))


def omega_bar(w: float, gamma: float | None = None) -> float:
    g = OMEGA_GAMMA if gamma is None else gamma
    return (min(float(w), 5.0) / 5.0) ** g


@dataclass
class RepairOutcome:
    teacher_key: str
    weights: list[float]
    coverage: float
    n_criteria_gated: int
    debug: list[dict] = field(default_factory=list)
    per_teacher: dict[str, list[tuple[list[bool], float]]] = field(default_factory=dict)
    teacher_scalars: dict[str, float] = field(default_factory=dict)


RUBRIC_HEADER = "\n\nGrading rubric (a correct answer must satisfy every criterion below):\n"


def rubric_block(crits: list[dict]) -> str:
    lines = [f"- {c.get('text', '')}" for c in crits if c.get("text")]
    return (RUBRIC_HEADER + "\n".join(lines)) if lines else ""


def failed_criteria(criteria: list[dict], crit_detail: list[dict]) -> list[dict]:
    status = {d.get("cid"): d.get("pass") for d in crit_detail}
    return [c for c in criteria if status.get(c["cid"]) is False]


async def _expert_answer(client, model: str, question: str, max_tokens: int) -> str | None:
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": question}], timeout=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


async def repair_and_weight(
    question: str,
    y: str,
    criteria: list[dict],
    crit_detail: list[dict],
    client,
    tokenizer,
    experts: dict[str, str],
    delta_min: float = 0.0,
    tau: float = 0.5,
    expand: int = 2,
    max_tokens: int = 3072,
    attribution_repair: bool = True,
    attr_max_cov: float = 0.3,
    gate: bool = True,
    route_axis: str | None = None,
    uniform_w: bool = False,
    rubric_in_probe: bool = False,
) -> RepairOutcome | None:
    fails = [c for c in failed_criteria(criteria, crit_detail)
             if isinstance(c.get("weight", 1.0), (int, float)) and c.get("weight", 1.0) > 0]
    if not fails:
        return None

    def _crit_experts(c: dict) -> list[str]:
        if route_axis == "random":
            pool = [e for es in AXIS_EXPERTS.values() for e in es if e in experts]
            return [random.choice(pool)] if pool else []
        axis = route_axis if route_axis else c.get("axis", "")
        return [e for e in AXIS_EXPERTS.get(axis, []) if e in experts]

    if gate:
        needed = sorted({e for c in fails for e in _crit_experts(c)})
        if not needed:
            return None
        probe_q = question + rubric_block(fails) if rubric_in_probe else question
        probes = await asyncio.gather(*(_expert_answer(client, experts[e], probe_q, max_tokens) for e in needed))
        answers = dict(zip(needed, probes))
    else:
        answers = {}

    n_tok = len(tokenizer(y, add_special_tokens=False)["input_ids"])
    offsets = None
    per_teacher: dict[str, list[tuple[list[bool], float]]] = {}
    n_gated = 0
    debug: list[dict] = []

    for crit in fails:
        s0 = 0.0
        if gate:
            cand_experts = [e for e in _crit_experts(crit) if answers.get(e)]
            if not cand_experts:
                continue
            cands: list[RepairCandidate] = []
            for ek in cand_experts:
                s1 = await asyncio.to_thread(criterion_score, answers[ek], crit)
                c = RepairCandidate(expert=ek, y_tilde=answers[ek], edit_ratio=0.0)
                c.delta = None if s1 is None else s1 - s0
                c.detail = {"s0": s0, "s1": s1, "mode": "delta_prime"}
                cands.append(c)
            gate_and_route(cands, delta_min=delta_min, tau=tau, max_edit_ratio=float("inf"))
            best = select_teacher(cands)
            debug.append({"cid": crit.get("cid"),
                          "cands": [{"expert": c.expert, "delta": c.delta, "gated": c.gated,
                                     "q": round(c.q, 3)} for c in cands]})
            if best is None:
                continue
            best_expert, best_q = best.expert, best.q
        else:
            ce = _crit_experts(crit)
            if not ce:
                continue
            best_expert, best_q = ce[0], 1.0
            debug.append({"cid": crit.get("cid"), "mode": "no_gate",
                          "cands": [{"expert": best_expert, "gated": True, "q": 1.0}]})
        n_gated += 1

        mask: list[bool] | None = None
        if attribution_repair:
            try:
                resp = await client.chat.completions.create(
                    model=experts[best_expert], messages=repair_messages(question, y, crit),
                    max_tokens=max_tokens, temperature=0.0, timeout=600,
                )
                y_tilde = (resp.choices[0].message.content or "").strip()
            except Exception:
                y_tilde = ""
            if y_tilde:
                s_rep = await asyncio.to_thread(criterion_score, y_tilde, crit)
                if s_rep is not None and s_rep > s0:
                    m, _ratio = token_mask_for_repair(tokenizer, y, y_tilde, expand=expand)
                    if any(m) and sum(m) / max(len(m), 1) <= attr_max_cov:
                        mask = m
                        debug[-1]["attr"] = "edit_span"
        if mask is None:
            spans = split_steps(y)
            if offsets is None:
                offsets = tokenizer(y, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
            mask = spans_to_token_mask([spans[-1]], offsets, expand=0) if spans else [True] * n_tok
            debug[-1]["attr"] = "last_step"

        w = crit.get("weight", 0.5)
        w = float(w if isinstance(w, (int, float)) and w > 0 else 0.5)
        scalar = 1.0 if uniform_w else best_q * omega_bar(w)
        per_teacher.setdefault(best_expert, []).append((mask, scalar))

    if not per_teacher:
        return None
    teacher = max(per_teacher, key=lambda k: sum(s for _m, s in per_teacher[k]))
    w = assemble_token_weights(n_tok, per_teacher[teacher])
    return RepairOutcome(
        teacher_key=teacher, weights=w, coverage=coverage(w), n_criteria_gated=n_gated, debug=debug,
        per_teacher=per_teacher,
        teacher_scalars={k: max(sc for _m, sc in v) for k, v in per_teacher.items()},
    )


def parse_crit_fields(extra_info: dict, crit_detail_json: str) -> tuple[list[dict], list[dict]]:
    criteria = json.loads(extra_info.get("criteria") or "[]")
    detail = json.loads(crit_detail_json or "[]")
    return criteria, detail
