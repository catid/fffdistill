from __future__ import annotations

import torch
import torch.nn.functional as F

from cifar_mamba_fff.metrics import cosine_loss, normalized_mse


def distillation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    normalized_mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
    variance_weight: float = 0.0,
) -> torch.Tensor:
    loss = pred.float().new_tensor(0.0)
    if normalized_mse_weight:
        loss = loss + normalized_mse_weight * normalized_mse(pred, target)
    if cosine_weight:
        loss = loss + cosine_weight * cosine_loss(pred, target)
    if variance_weight:
        pred_var = pred.float().var(dim=0, unbiased=False)
        target_var = target.float().var(dim=0, unbiased=False)
        loss = loss + variance_weight * F.mse_loss(pred_var, target_var)
    return loss
