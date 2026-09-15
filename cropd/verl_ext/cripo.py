from __future__ import annotations

import asyncio
import json
import math
import os
import uuid

import torch

TEMPLATE = (
    "Given a response:\n{previous_response}\n"
    "Please revise the response with minimum necessary by modifying or deleting the parts that\n"
    "satisfy the following criteria:\n{satisfied_suppressed_criteria}\n"
    "Output only the revised response."
)


def grpo_advantages(scores: list[float], eps: float = 1e-6, norm_std: bool = True) -> list[float]:
    n = len(scores)
    mu = sum(scores) / n
    if not norm_std:
        return [s - mu for s in scores]
    var = sum((s - mu) ** 2 for s in scores) / n
    sd = math.sqrt(var)
    return [(s - mu) / (sd + eps) for s in scores]


def suppressed_criteria(r: dict[str, list[int]], advs: list[float]) -> list[str]:
    G = len(advs)
    out = []
    for cid, sat in r.items():
        S = [i for i, v in enumerate(sat) if v]
        if not S:
            continue
        tot = sum(advs[i] for i in S)
        if tot < -1e-9 or (abs(tot) <= 1e-9 and len(S) < G / 2):
            out.append(cid)
    return out


def rollout_suppressed(cs: list[str], r: dict[str, list[int]], i: int,
                       weights: dict[str, float], k: int = 3) -> list[str]:
    mine = [cid for cid in cs if r[cid][i]]
    mine.sort(key=lambda c: -weights.get(c, 0.0))
    return mine[:k]


def build_ctx_ids(tokenizer, response_text: str, criteria_texts: list[str]) -> list[int]:
    crit = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(criteria_texts))
    msg = TEMPLATE.replace("{previous_response}", response_text).replace(
        "{satisfied_suppressed_criteria}", crit)
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=True, add_generation_prompt=True))


def flip_mask(old_lp: torch.Tensor, t_lp: torch.Tensor, t_top1: torch.Tensor,
              alpha: float) -> torch.Tensor:
    has = t_top1 < 0
    return has & (old_lp - t_lp > 0) & (t_lp < math.log(alpha) + t_top1)


def _crit_from_extra_info(ei) -> tuple[dict[str, str], dict[str, float]]:
    crits = json.loads(str(dict(ei).get("criteria") or "[]"))
    texts, weights = {}, {}
    for c in crits:
        w = c.get("weight", 0)
        w = float(w) if isinstance(w, (int, float)) else 0.0
        if w > 0 and c.get("cid") and c.get("text"):
            texts[str(c["cid"])] = str(c["text"])
            weights[str(c["cid"])] = w
    return texts, weights


def _r_matrix(details: list[list[dict]], cids: set[str]) -> dict[str, list[int]]:
    r = {cid: [0] * len(details) for cid in cids}
    for i, det in enumerate(details):
        for d in det:
            cid = str(d.get("cid"))
            if cid in r and d.get("pass") is True:
                r[cid][i] = 1
    return r


async def _teacher_lp(server_manager, ctx_ids: list[int], resp_ids: list[int],
                      max_model_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    seq = ctx_ids + resp_ids
    if len(seq) + 1 > max_model_len:
        return None
    try:
        out = await server_manager.generate(
            request_id=f"cripo-{uuid.uuid4().hex}", prompt_ids=seq,
            sampling_params={"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1})
        top1 = out.extra_fields.get("prompt_logprobs")
        actual = out.extra_fields.get("prompt_actual_logprobs")
        if top1 is None or actual is None or len(top1) != len(seq):
            return None
        T, R = len(ctx_ids), len(resp_ids)
        a = torch.tensor([float(actual[T - 1 + r][0] if isinstance(actual[T - 1 + r], list)
                                else actual[T - 1 + r]) for r in range(R)], dtype=torch.float32)
        t = torch.tensor([float(top1[T - 1 + r][0]) for r in range(R)], dtype=torch.float32)
        return a, t
    except Exception:
        return None


async def attach_flip_inputs(outputs, index, extra_infos, server_manager, tokenizer,
                             max_model_len: int) -> None:
    alpha_k = int(os.environ.get("CROPD_CRIPO_K", "3"))
    groups: dict = {}
    for i, key in enumerate(index):
        groups.setdefault(key, []).append(i)
    n_qual = n_ok = 0
    tasks, meta = [], []
    for key, idxs in groups.items():
        scores = [outputs[i].reward_score for i in idxs]
        if any(s is None for s in scores) or len(idxs) < 2:
            continue
        advs = grpo_advantages([float(s) for s in scores])
        texts, weights = _crit_from_extra_info(extra_infos[idxs[0]])
        if not texts:
            continue
        details = []
        for i in idxs:
            rei = outputs[i].extra_fields.get("reward_extra_info") or {}
            cd = rei.get("crit_detail")
            cd = cd[0] if isinstance(cd, list) and cd else cd
            try:
                details.append(json.loads(cd) if isinstance(cd, str) else [])
            except Exception:
                details.append([])
        r = _r_matrix(details, set(texts))
        cs = suppressed_criteria(r, advs)
        if not cs:
            continue
        for gi, i in enumerate(idxs):
            if advs[gi] >= 0:
                continue
            mine = rollout_suppressed(cs, r, gi, weights, k=alpha_k)
            if not mine:
                continue
            n_qual += 1
            o = outputs[i]
            L = int(o.response_mask[0].sum().item())
            resp_ids = o.response_ids[0][:L].tolist()
            ctx = build_ctx_ids(
                tokenizer, tokenizer.decode(resp_ids, skip_special_tokens=True),
                [texts[c] for c in mine])
            tasks.append(_teacher_lp(server_manager, ctx, resp_ids, max_model_len))
            meta.append((i, L))
    results = await asyncio.gather(*tasks) if tasks else []
    for (i, L), res in zip(meta, results):
        if res is None:
            continue
        n_ok += 1
        o = outputs[i]
        W = o.response_ids.shape[1]
        a = torch.zeros(1, W, dtype=torch.float32)
        t = torch.zeros(1, W, dtype=torch.float32)
        a[0, :L], t[0, :L] = res
        o.extra_fields["cripo_t_lp"] = a
        o.extra_fields["cripo_t_top1"] = t
    if os.environ.get("CROPD_HOOK_DEBUG"):
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        print(f"[cripo] groups={len(groups)} sizes={sizes[:6]} qualified={n_qual} "
              f"teacher_ok={n_ok}", flush=True)


def apply_flip(batch, alpha: float | None = None, tau: float | None = None):
    if "cripo_t_lp" not in batch.batch.keys():
        return batch, {}
    alpha = float(os.environ.get("CROPD_CRIPO_ALPHA", "0.1")) if alpha is None else alpha
    tau = float(os.environ.get("CROPD_CRIPO_TAU", "0.1")) if tau is None else tau
    t_lp = batch.batch["cripo_t_lp"]
    t_top1 = batch.batch["cripo_t_top1"]
    old_lp = batch.batch["old_log_probs"]
    adv = batch.batch["advantages"]
    rmask = batch.batch["response_mask"].bool()
    m = flip_mask(old_lp, t_lp, t_top1, alpha) & rmask & (adv < 0)
    batch.batch["advantages"] = torch.where(m, torch.full_like(adv, tau), adv)
    metrics = {"cripo/n_flipped_tokens": int(m.sum().item()),
               "cripo/n_flipped_rows": int(m.any(-1).sum().item())}
    return batch, metrics
