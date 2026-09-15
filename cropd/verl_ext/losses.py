from __future__ import annotations

import os
from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    DistillationLossSettings,
    register_distillation_loss,
)
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

INNER_KL = os.getenv("CROPD_VGOPD_INNER_KL", "k3")


def weighted_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weights: torch.Tensor,
    inner_kl: str = INNER_KL,
) -> torch.Tensor:
    assert student_log_probs.shape == teacher_log_probs.shape == weights.shape, (
        f"{student_log_probs.shape=} {teacher_log_probs.shape=} {weights.shape=}"
    )
    losses = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=inner_kl)
    return losses * weights


@register_distillation_loss(DistillationLossSettings(names=["vgopd_weighted"], use_estimator=True))
def compute_vgopd_weighted(
    config,
    distillation_config,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    student = no_padding_2_padding(model_output["log_probs"], data)
    teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.bool().to_padded_tensor(False)
    else:
        response_mask = response_mask.bool()

    w = data.get("distill_weights", None)
    if w is None:
        w = torch.ones_like(student)
    else:
        if w.is_nested:
            w = w.to_padded_tensor(0.0)
        w = w.to(dtype=student.dtype, device=student.device)

    losses = weighted_kl(student, teacher, w)

    w_valid = w[response_mask]
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, losses[response_mask].abs().mean()),
        "vgopd/w_mean": Metric(AggregationType.MEAN, w_valid.mean()),
        "vgopd/coverage": Metric(AggregationType.MEAN, (w_valid > 0).float().mean()),
    }
    return losses, metrics
