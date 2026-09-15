from __future__ import annotations

import asyncio
import hashlib
import json
import os

import torch

from cropd.verl_ext.repair_hook import (AXIS_EXPERTS, failed_criteria, parse_crit_fields,
                                        repair_and_weight, rubric_block)

_CLIENT = None
_PROBE_CACHE: dict[str, dict] = {}
_PROBE_CACHE_MAX = 4096
EXPERTS = {e: e for es in AXIS_EXPERTS.values() for e in es}


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import AsyncOpenAI

        _CLIENT = AsyncOpenAI(
            base_url=os.environ.get("CROPD_REPAIR_BASE_URL", "http://localhost:8001/v1"),
            api_key="EMPTY", timeout=600, max_retries=0,
        )
    return _CLIENT


class _CachingClient:

    class _Chat:
        class _Completions:
            def __init__(self, base):
                self._base = base

            async def create(self, **kw):
                msgs = kw.get("messages") or []
                if len(msgs) == 1 and msgs[0].get("role") == "user":
                    key = hashlib.sha256((kw.get("model", "") + "|" + msgs[0]["content"]).encode()).hexdigest()
                    ent = _PROBE_CACHE.get(key)
                    if ent is None:
                        if len(_PROBE_CACHE) > _PROBE_CACHE_MAX:
                            _PROBE_CACHE.clear()
                        ent = {"lock": asyncio.Lock(), "resp": None}
                        _PROBE_CACHE[key] = ent
                    async with ent["lock"]:
                        if ent["resp"] is None:
                            ent["resp"] = await _client().chat.completions.create(**kw)
                    return ent["resp"]
                return await _client().chat.completions.create(**kw)

        def __init__(self, base):
            self.completions = self._Completions(base)

    def __init__(self):
        self.chat = self._Chat(self)


_CACHING = None


def _caching_client():
    global _CACHING
    if _CACHING is None:
        _CACHING = _CachingClient()
    return _CACHING


def _ent_lp_rank(ent) -> tuple[float, int]:
    if isinstance(ent, dict):
        return float(ent["logprob"]), int(ent.get("rank") or 10**9)
    return float(ent.logprob), int(getattr(ent, "rank", None) or 10**9)


async def _teacher_logprobs_via_sidecar(
    expert: str, sequence_ids: list[int], topk: int = 0
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        resp = await _client().completions.create(
            model=expert, prompt=sequence_ids, max_tokens=1, temperature=0.0,
            extra_body={"prompt_logprobs": topk},
        )
        plp = resp.choices[0].prompt_logprobs
        if len(plp) != len(sequence_ids):
            return None
        if topk == 0:
            vals = []
            for i, d in enumerate(plp):
                if not d:
                    vals.append(0.0)
                    continue
                ent = d.get(sequence_ids[i]) or d.get(str(sequence_ids[i])) or next(iter(d.values()))
                vals.append(_ent_lp_rank(ent)[0])
            ids = torch.tensor(sequence_ids, dtype=torch.int32).unsqueeze(-1)
            return ids, torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)
        ids_rows, lp_rows = [], []
        for i, d in enumerate(plp):
            if not d:
                ids_rows.append([sequence_ids[i]] * topk)
                lp_rows.append([0.0] * topk)
                continue
            ents = sorted(((int(k), *_ent_lp_rank(v)) for k, v in d.items()), key=lambda x: x[2])[:topk]
            while len(ents) < topk:
                ents.append((ents[-1][0], ents[-1][1], 10**9))
            ids_rows.append([e[0] for e in ents])
            lp_rows.append([e[1] for e in ents])
        return (torch.tensor(ids_rows, dtype=torch.int32), torch.tensor(lp_rows, dtype=torch.float32))
    except Exception:
        return None


async def _teacher_lps_with_ctx(expert: str, msgs: list[dict], prompt_ids: list[int],
                                response_ids: list[int], tokenizer, K: int):
    kk = max(1, K)
    S, R = len(prompt_ids) + len(response_ids), len(response_ids)
    t_prompt_ids = list(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    t = await _teacher_logprobs_via_sidecar(expert, t_prompt_ids + list(response_ids), topk=K)
    if t is None:
        return None
    t_ids_priv, t_lps_priv = t
    ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int32).unsqueeze(-1).repeat(1, kk)
    lps = torch.zeros(S, kk, dtype=torch.float32)
    ids[len(prompt_ids):] = t_ids_priv[len(t_prompt_ids):]
    lps[len(prompt_ids):] = t_lps_priv[len(t_prompt_ids):]
    return ids, lps


def _disagree_mask(t_ids: torch.Tensor, t_lps: torch.Tensor, response_ids: list[int], prompt_len: int,
                   K: int, q: float) -> torch.Tensor:
    R = len(response_ids)
    if K == 0:
        resp_lp = t_lps[prompt_len:, 0]
    else:
        ids_resp = t_ids[prompt_len:]
        target = torch.tensor(response_ids, dtype=ids_resp.dtype).unsqueeze(-1)
        match = ids_resp == target
        lp_resp = t_lps[prompt_len:]
        resp_lp = torch.where(match.any(-1), (lp_resp * match).sum(-1), torch.full((R,), -20.0))
    k = max(1, int(R * q))
    thr = torch.kthvalue(resp_lp, k).values
    m = (resp_lp <= thr).float()
    return torch.nn.functional.max_pool1d(m.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)


def _trace(output, prompt_ids, response_ids, outcome, w, ei, owners: dict | None = None) -> None:
    d = os.environ.get("CROPD_TRACE_DIR")
    if not d:
        return
    try:
        every = int(os.environ.get("CROPD_TRACE_EVERY", "10"))
        h = hashlib.sha256(bytes(str(prompt_ids[-32:]), "utf-8")).hexdigest()
        if int(h[:8], 16) % every:
            return
        R = len(response_ids)
        nz = (w > 0).nonzero().flatten().tolist()
        hist = [0] * 10
        for i in nz:
            hist[min(9, i * 10 // max(R, 1))] += 1
        rec = {
            "qh": h[:16], "capability": str(dict(ei or {}).get("capability") or ""),
            "teacher": outcome.teacher_key, "n_gated": outcome.n_criteria_gated,
            "coverage": round(float(outcome.coverage), 4), "R": R,
            "w_nonzero_frac": round(len(nz) / max(R, 1), 4),
            "w_max": round(float(w.max()), 4) if R else 0.0, "pos_hist": hist,
            "reward": None if output.reward_score is None else float(output.reward_score),
            "crit": outcome.debug,
            "teachers": owners or {},
        }
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"trace_{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def cropd_gate_and_teacher(output, prompt_ids: list[int], response_ids: list[int],
                                 sample_kwargs: dict | None, tokenizer) -> None:
    S = len(prompt_ids) + len(response_ids)
    R = len(response_ids)
    K = int(os.environ.get("CROPD_TEACHER_TOPK", "0"))
    kk = max(1, K)

    def zero_fill(reason: str = ""):
        if reason and os.environ.get("CROPD_HOOK_DEBUG"):
            print(f"[cropd-hook] zero: {reason}", flush=True)
        output.extra_fields["teacher_ids"] = (
            torch.tensor(prompt_ids + response_ids, dtype=torch.int32).unsqueeze(-1).repeat(1, kk)
        )
        output.extra_fields["teacher_logprobs"] = torch.zeros(S, kk, dtype=torch.float32)
        output.extra_fields["distill_weights"] = torch.zeros(R, dtype=torch.float32)

    try:
        sk = sample_kwargs or {}
        ei = sk.get("extra_info")
        ei = ei.item() if hasattr(ei, "item") else ei
        if os.environ.get("CROPD_HOOK_MODE", "vgopd") == "mopd":
            cap = str(dict(ei or {}).get("capability") or "")
            experts = AXIS_EXPERTS.get(cap)
            if not experts:
                return zero_fill(f"mopd unknown capability={cap!r}")
            t = None
            if os.environ.get("CROPD_TEACHER_RUBRIC", "0") == "1":
                criteria = json.loads(str(dict(ei or {}).get("criteria") or "[]"))
                prompt_msgs = sk.get("raw_prompt")
                prompt_msgs = prompt_msgs.tolist() if hasattr(prompt_msgs, "tolist") else prompt_msgs
                if criteria and prompt_msgs:
                    msgs = [dict(m) for m in prompt_msgs]
                    msgs[-1]["content"] = str(msgs[-1].get("content", "")) + rubric_block(criteria)
                    t = await _teacher_lps_with_ctx(experts[0], msgs, prompt_ids, response_ids, tokenizer, K)
            if t is None:
                t = await _teacher_logprobs_via_sidecar(experts[0], prompt_ids + response_ids, topk=K)
            if t is None:
                return zero_fill("mopd sidecar_none")
            output.extra_fields["teacher_ids"], output.extra_fields["teacher_logprobs"] = t
            w_mopd = float(os.environ.get("CROPD_MOPD_W", "1.0"))
            output.extra_fields["distill_weights"] = torch.full((R,), w_mopd, dtype=torch.float32)
            if os.environ.get("CROPD_HOOK_DEBUG"):
                print(f"[cropd-hook] mopd teacher={experts[0]} w={w_mopd}", flush=True)
            return
        if os.environ.get("CROPD_HOOK_MODE", "vgopd") == "opsd":
            criteria = json.loads(str(dict(ei or {}).get("criteria") or "[]"))
            prompt_msgs = sk.get("raw_prompt")
            prompt_msgs = prompt_msgs.tolist() if hasattr(prompt_msgs, "tolist") else prompt_msgs
            if not criteria or not prompt_msgs:
                return zero_fill(f"opsd no_criteria={not criteria} no_prompt={not prompt_msgs}")
            rubric = "\n".join(f"- {c.get('text', '')}" for c in criteria if c.get("text"))
            msgs = [dict(m) for m in prompt_msgs]
            msgs[-1]["content"] = (
                str(msgs[-1].get("content", ""))
                + "\n\nGrading rubric (privileged information; write an answer that satisfies "
                + "every criterion):\n" + rubric
            )
            t_prompt_ids = list(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            teacher_model = os.environ.get("CROPD_OPSD_TEACHER", "repair")
            t = await _teacher_logprobs_via_sidecar(teacher_model, t_prompt_ids + list(response_ids), topk=K)
            if t is None:
                return zero_fill("opsd sidecar_none")
            t_ids_priv, t_lps_priv = t
            ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int32).unsqueeze(-1).repeat(1, kk)
            lps = torch.zeros(S, kk, dtype=torch.float32)
            ids[len(prompt_ids):] = t_ids_priv[len(t_prompt_ids):]
            lps[len(prompt_ids):] = t_lps_priv[len(t_prompt_ids):]
            output.extra_fields["teacher_ids"] = ids
            output.extra_fields["teacher_logprobs"] = lps
            w_opsd = float(os.environ.get("CROPD_OPSD_W", "1.0"))
            output.extra_fields["distill_weights"] = torch.full((R,), w_opsd, dtype=torch.float32)
            if os.environ.get("CROPD_HOOK_DEBUG"):
                print(f"[cropd-hook] opsd rubric_n={len(criteria)} ctx={len(t_prompt_ids)}", flush=True)
            return
        rei = output.extra_fields.get("reward_extra_info") or {}
        crit_json = rei.get("crit_detail")
        crit_json = crit_json[0] if isinstance(crit_json, list) and crit_json else crit_json
        if not ei or not crit_json:
            return zero_fill(f"no_ei={not ei} no_crit={not crit_json} rei_keys={list(rei)[:6]}")
        criteria, detail = parse_crit_fields(dict(ei), crit_json)
        if not criteria or not detail:
            return zero_fill(f"empty parsed criteria={len(criteria)} detail={len(detail)}")

        question = str(dict(ei).get("question") or "")
        prompt_msgs = sk.get("raw_prompt")
        prompt_msgs = prompt_msgs.tolist() if hasattr(prompt_msgs, "tolist") else prompt_msgs
        if not question and prompt_msgs:
            question = prompt_msgs[-1].get("content", "")
        teacher_rubric = os.environ.get("CROPD_TEACHER_RUBRIC", "0") == "1"
        y = tokenizer.decode(response_ids, skip_special_tokens=True)

        if not question:
            return zero_fill("empty question")
        skip_thr = float(os.environ.get("CROPD_PROBE_SKIP_SCORE", "1.01"))
        if output.reward_score is not None and float(output.reward_score) >= skip_thr:
            return zero_fill()
        route_axis = None
        _rm = os.environ.get("CROPD_ROUTE_MODE", "criterion")
        if _rm == "domain":
            route_axis = str(dict(ei).get("capability") or "") or None
        elif _rm == "random":
            route_axis = "random"
        outcome = await repair_and_weight(
            question, y, criteria, detail, _caching_client(), tokenizer, EXPERTS,
            attribution_repair=os.environ.get("CROPD_ATTR_REPAIR", "1") != "0",
            gate=os.environ.get("CROPD_GATE", "1") != "0",
            route_axis=route_axis,
            uniform_w=os.environ.get("CROPD_UNIFORM_W", "0") == "1",
            rubric_in_probe=teacher_rubric,
            max_tokens=int(os.environ.get("CROPD_PROBE_MAX_TOKENS", "3072")),
        )
        if outcome is None:
            n_fail = sum(1 for d in detail if d.get("pass") is False)
            return zero_fill(f"no_gated fails={n_fail} q_len={len(question)}")

        multi = os.environ.get("CROPD_MULTI_TEACHER", "1") == "1"
        scalars = dict(outcome.teacher_scalars) or {
            outcome.teacher_key: (max(outcome.weights) if outcome.weights else 0.0)}
        if outcome.teacher_key not in scalars:
            scalars[outcome.teacher_key] = max(outcome.weights) if outcome.weights else 0.0
        if not multi:
            scalars = {outcome.teacher_key: scalars[outcome.teacher_key]}
        order = sorted(scalars, key=lambda k: (k != outcome.teacher_key, -scalars[k], k))
        attr_mode = os.environ.get("CROPD_ATTR_MODE", "disagree")

        async def _fetch(expert_key: str):
            if teacher_rubric and prompt_msgs:
                fails = [c for c in failed_criteria(criteria, detail)
                         if isinstance(c.get("weight", 1.0), (int, float)) and c.get("weight", 1.0) > 0]
                msgs = [dict(m) for m in prompt_msgs]
                msgs[-1]["content"] = str(msgs[-1].get("content", "")) + rubric_block(fails)
                return await _teacher_lps_with_ctx(EXPERTS[expert_key], msgs, prompt_ids, response_ids,
                                                   tokenizer, K)
            return await _teacher_logprobs_via_sidecar(EXPERTS[expert_key], prompt_ids + response_ids, topk=K)

        fetched: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for ek in (order if attr_mode == "disagree" else order[:1]):
            t = await _fetch(ek)
            if t is None:
                if ek == order[0]:
                    return zero_fill()
                continue
            fetched.append((ek, t[0], t[1]))
        t_ids, t_lps = fetched[0][1], fetched[0][2]
        P = len(prompt_ids)
        owners: dict[str, int] = {}
        if attr_mode == "full":
            scalar = float(scalars[order[0]])
            w = torch.full((R,), scalar, dtype=torch.float32)
            owners = {order[0]: R}
        elif attr_mode == "disagree":
            q = float(os.environ.get("CROPD_DISAGREE_Q", "0.10"))
            w = torch.zeros(R, dtype=torch.float32)
            t_ids, t_lps = t_ids.clone(), t_lps.clone()
            owner = torch.full((R,), -1, dtype=torch.int64)
            for idx, (ek, ids_k, lps_k) in sorted(enumerate(fetched), key=lambda x: (scalars[x[1][0]], -x[0])):
                take = _disagree_mask(ids_k, lps_k, response_ids, P, K, q) > 0
                w[take] = float(scalars[ek])
                owner[take] = idx
                t_ids[P:][take] = ids_k[P:][take]
                t_lps[P:][take] = lps_k[P:][take]
            owners = {fetched[i][0]: int((owner == i).sum()) for i in range(len(fetched))}
        elif attr_mode == "random":
            q = float(os.environ.get("CROPD_DISAGREE_Q", "0.10"))
            k = max(1, int(R * q))
            m = torch.zeros(R, dtype=torch.float32)
            m[torch.randperm(R)[:k]] = 1.0
            m = torch.nn.functional.max_pool1d(m.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)
            scalar = float(scalars[order[0]])
            w = (m * scalar).to(torch.float32)
            owners = {order[0]: int((w > 0).sum())}
        else:
            w = torch.zeros(R, dtype=torch.float32)
            n = min(R, len(outcome.weights))
            w[:n] = torch.tensor(outcome.weights[:n], dtype=torch.float32)
            owners = {order[0]: int((w > 0).sum())}
        output.extra_fields["teacher_ids"] = t_ids
        output.extra_fields["teacher_logprobs"] = t_lps
        output.extra_fields["distill_weights"] = w
        _trace(output, prompt_ids, response_ids, outcome, w, ei, owners)
        if os.environ.get("CROPD_HOOK_DEBUG"):
            print(f"[cropd-hook] teachers={owners} primary={outcome.teacher_key} gated={outcome.n_criteria_gated} "
                  f"cov={(w > 0).float().mean().item():.3f}", flush=True)
    except Exception as e:
        if os.environ.get("CROPD_HOOK_DEBUG"):
            print(f"[cropd-hook] error: {type(e).__name__}: {e}", flush=True)
        zero_fill()
