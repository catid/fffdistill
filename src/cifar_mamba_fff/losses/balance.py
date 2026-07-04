"""Balance and routing regularizers for FFF routers."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Reduction = Literal["mean", "sum", "none"]


def _require_tensor(name: str, value: Tensor, *, validate_finite: bool = False) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if validate_finite and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def validate_finite_tensor(name: str, value: Tensor) -> None:
    """Debug validation helper for periodic fail-fast checks outside hot paths."""

    _require_tensor(name, value, validate_finite=True)


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")


def _mean_over_leading_dims(values: Tensor) -> Tensor:
    if values.ndim == 0:
        raise ValueError("expected at least one dimension")
    if values.ndim == 1:
        return values
    return values.mean(dim=tuple(range(values.ndim - 1)))


def _normalize_dim(ndim: int, dim: int) -> int:
    if ndim == 0:
        raise ValueError("expected at least one dimension")
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dim {dim} is out of range for tensor rank {ndim}")
    return normalized


def split_balance_loss(
    route_probs: Tensor,
    *,
    pair_probs: bool = True,
    target: float = 0.5,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize unbalanced binary splits.

    If ``pair_probs`` is true, ``route_probs`` must have shape
    ``(..., num_nodes, 2)`` and the second branch is treated as the right
    branch. Otherwise ``route_probs`` is interpreted as scalar right-branch
    probabilities with shape ``(..., num_nodes)``.
    """

    _require_tensor("route_probs", route_probs)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")

    if pair_probs:
        if route_probs.ndim < 2 or route_probs.shape[-1] != 2:
            raise ValueError("pair route_probs must have shape (..., num_nodes, 2)")
        right_probs = route_probs[..., 1]
    else:
        if route_probs.ndim < 1:
            raise ValueError("scalar route_probs must have shape (..., num_nodes)")
        right_probs = route_probs

    mean_right = _mean_over_leading_dims(right_probs)
    penalty = (mean_right - target) ** 2
    return _reduce(penalty, reduction)


def split_balance(
    route_probs: Tensor,
    *,
    pair_probs: bool = True,
    target: float = 0.5,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`split_balance_loss`."""

    return split_balance_loss(
        route_probs,
        pair_probs=pair_probs,
        target=target,
        reduction=reduction,
    )


def min_leaf_occupancy_loss(
    leaf_probs: Tensor,
    *,
    min_occupancy: float,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize leaves whose mean usage falls below ``min_occupancy``."""

    _require_tensor("leaf_probs", leaf_probs)
    if leaf_probs.ndim < 1:
        raise ValueError("leaf_probs must have shape (..., num_leaves)")
    if min_occupancy < 0.0:
        raise ValueError("min_occupancy must be non-negative")

    mean_usage = _mean_over_leading_dims(leaf_probs)
    penalty = F.relu(min_occupancy - mean_usage) ** 2
    return _reduce(penalty, reduction)


def min_leaf_occupancy(
    leaf_probs: Tensor,
    *,
    min_occupancy: float,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`min_leaf_occupancy_loss`."""

    return min_leaf_occupancy_loss(
        leaf_probs,
        min_occupancy=min_occupancy,
        reduction=reduction,
    )


def uniform_leaf_balance_loss(
    leaf_probs: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize deviation from uniform mean leaf usage."""

    _require_tensor("leaf_probs", leaf_probs)
    if leaf_probs.ndim < 1:
        raise ValueError("leaf_probs must have shape (..., num_leaves)")
    num_leaves = leaf_probs.shape[-1]
    if num_leaves < 1:
        raise ValueError("leaf_probs must contain at least one leaf")

    mean_usage = _mean_over_leading_dims(leaf_probs)
    target = mean_usage.new_full(mean_usage.shape, 1.0 / float(num_leaves))
    penalty = (mean_usage - target) ** 2
    return _reduce(penalty, reduction)


def uniform_leaf_balance(
    leaf_probs: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`uniform_leaf_balance_loss`."""

    return uniform_leaf_balance_loss(leaf_probs, reduction=reduction)


def route_margin_loss(
    route_logits: Tensor,
    *,
    target_margin: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Hinge penalty for binary route logits with too-small top-two margin."""

    _require_tensor("route_logits", route_logits)
    if route_logits.ndim < 1 or route_logits.shape[-1] != 2:
        raise ValueError("route_logits must have shape (..., 2)")
    if target_margin < 0.0:
        raise ValueError("target_margin must be non-negative")

    top2 = route_logits.topk(k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    penalty = F.relu(target_margin - margin) ** 2
    return _reduce(penalty, reduction)


def route_margin(
    route_logits: Tensor,
    *,
    target_margin: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`route_margin_loss`."""

    return route_margin_loss(
        route_logits,
        target_margin=target_margin,
        reduction=reduction,
    )


def route_entropy_loss(
    values: Tensor,
    *,
    from_logits: bool = True,
    dim: int = -1,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
    validate_probabilities: bool = False,
    validate_finite: bool = False,
) -> Tensor:
    """Return route entropy along ``dim``.

    This returns the raw non-negative entropy. Callers that want to encourage
    high entropy can subtract it from their objective.
    """

    _require_tensor("values", values, validate_finite=validate_finite)
    dim = _normalize_dim(values.ndim, dim)
    if values.shape[dim] < 2:
        raise ValueError("entropy dimension must contain at least two routes")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    if from_logits:
        log_probs = F.log_softmax(values, dim=dim)
        probs = log_probs.exp()
    else:
        if validate_probabilities and (values < 0).any().item():
            raise ValueError("probabilities must be non-negative")
        probs = values.clamp_min(eps)
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
        log_probs = probs.clamp_min(eps).log()

    entropy = -(probs * log_probs).sum(dim=dim)
    return _reduce(entropy, reduction)


def route_entropy(
    values: Tensor,
    *,
    from_logits: bool = True,
    dim: int = -1,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
    validate_probabilities: bool = False,
    validate_finite: bool = False,
) -> Tensor:
    """Alias for :func:`route_entropy_loss`."""

    return route_entropy_loss(
        values,
        from_logits=from_logits,
        dim=dim,
        eps=eps,
        reduction=reduction,
        validate_probabilities=validate_probabilities,
        validate_finite=validate_finite,
    )


def usage_cap_penalty(
    usage_rates: Tensor,
    *,
    cap: float,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize usage rates that exceed ``cap``."""

    _require_tensor("usage_rates", usage_rates)
    if cap < 0.0:
        raise ValueError("cap must be non-negative")

    penalty = F.relu(usage_rates - cap) ** 2
    return _reduce(penalty, reduction)


def master_fallback_usage_cap_penalty(
    master_usage: Tensor,
    fallback_usage: Tensor,
    *,
    master_cap: float,
    fallback_cap: float,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize master and fallback route usage above their caps."""

    _require_tensor("master_usage", master_usage)
    _require_tensor("fallback_usage", fallback_usage)
    if master_usage.shape != fallback_usage.shape:
        raise ValueError("master_usage and fallback_usage must have the same shape")

    master_penalty = usage_cap_penalty(master_usage, cap=master_cap, reduction="none")
    fallback_penalty = usage_cap_penalty(
        fallback_usage,
        cap=fallback_cap,
        reduction="none",
    )
    return _reduce(master_penalty + fallback_penalty, reduction)
