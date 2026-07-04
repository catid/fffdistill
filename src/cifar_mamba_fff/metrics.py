from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N, C], got {tuple(logits.shape)}")
    if target.ndim != 1:
        raise ValueError(f"target must be [N], got {tuple(target.shape)}")
    return float((logits.argmax(dim=1) == target).float().mean().item())


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (pred - target).square().sum() / (target.square().sum() + eps)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.flatten(0, -2)
    target_f = target.flatten(0, -2)
    return 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1, eps=eps).mean()


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
