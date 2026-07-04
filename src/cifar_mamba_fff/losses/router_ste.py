"""Router straight-through estimators and assignment helpers."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Reduction = Literal["mean", "sum", "none"]


def _require_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _normalize_dim(ndim: int, dim: int) -> int:
    if ndim == 0:
        raise ValueError("expected at least one dimension")
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dim {dim} is out of range for tensor rank {ndim}")
    return normalized


def _require_router_logits(logits: Tensor, *, dim: int) -> int:
    _require_tensor("logits", logits)
    dim = _normalize_dim(logits.ndim, dim)
    if logits.shape[dim] < 2:
        raise ValueError("router dimension must contain at least two choices")
    return dim


def _require_temperature(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive finite value")


def _resolve_temperature(temperature: float, tau: float | None) -> float:
    if tau is not None:
        if temperature != 1.0:
            raise ValueError("specify only one of temperature or tau")
        temperature = tau
    _require_temperature("temperature", temperature)
    return temperature


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")


def _one_hot_argmax(logits: Tensor, *, dim: int) -> Tensor:
    indices = logits.argmax(dim=dim, keepdim=True)
    return torch.zeros_like(logits).scatter_(dim, indices, 1.0)


def no_ste_soft_router(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    tau: float | None = None,
    dim: int = -1,
) -> Tensor:
    """Return differentiable soft router probabilities."""

    dim = _require_router_logits(logits, dim=dim)
    temperature = _resolve_temperature(temperature, tau)
    return torch.softmax(logits / temperature, dim=dim)


def vanilla_ste(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    tau: float | None = None,
    dim: int = -1,
) -> Tensor:
    """Hard argmax forward pass with softmax straight-through gradients."""

    dim = _require_router_logits(logits, dim=dim)
    temperature = _resolve_temperature(temperature, tau)
    y_soft = no_ste_soft_router(logits, temperature=temperature, dim=dim)
    y_hard = _one_hot_argmax(logits, dim=dim)
    return y_hard - y_soft.detach() + y_soft


class _ClippedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, logits: Tensor, dim: int, clip: float) -> Tensor:
        ctx.dim = dim
        ctx.clip = clip
        return _one_hot_argmax(logits, dim=dim)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        del ctx.dim
        return grad_output.clamp(min=-ctx.clip, max=ctx.clip), None, None


def clipped_ste(
    logits: Tensor,
    *,
    clip: float = 1.0,
    dim: int = -1,
) -> Tensor:
    """Hard argmax forward pass with identity STE gradients clipped by value."""

    dim = _require_router_logits(logits, dim=dim)
    if clip <= 0.0 or not math.isfinite(clip):
        raise ValueError("clip must be a positive finite value")
    return _ClippedSTE.apply(logits, dim, float(clip))


def sigmoid_surrogate_ste(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    tau: float | None = None,
    dim: int = -1,
) -> Tensor:
    """Hard argmax forward pass with elementwise sigmoid surrogate gradients."""

    dim = _require_router_logits(logits, dim=dim)
    temperature = _resolve_temperature(temperature, tau)
    centered = logits - logits.mean(dim=dim, keepdim=True)
    surrogate = torch.sigmoid(centered / temperature)
    y_hard = _one_hot_argmax(logits, dim=dim)
    return y_hard - surrogate.detach() + surrogate


def _sample_gumbel_like(
    logits: Tensor,
    *,
    generator: torch.Generator | None,
    eps: float,
) -> Tensor:
    uniform = torch.rand(
        logits.shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    uniform = uniform.clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(uniform))


def st_gumbel(
    logits: Tensor,
    *,
    tau: float = 1.0,
    hard: bool = True,
    training: bool = True,
    dim: int = -1,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Straight-through Gumbel router.

    ``training=False`` uses deterministic argmax behavior and never samples
    Gumbel noise.
    """

    dim = _require_router_logits(logits, dim=dim)
    _require_temperature("tau", tau)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    if not training:
        if hard:
            return _one_hot_argmax(logits, dim=dim)
        return torch.softmax(logits / tau, dim=dim)

    noisy_logits = logits + _sample_gumbel_like(logits, generator=generator, eps=eps)
    y_soft = torch.softmax(noisy_logits / tau, dim=dim)
    if not hard:
        return y_soft
    y_hard = _one_hot_argmax(y_soft, dim=dim)
    return y_hard - y_soft.detach() + y_soft


def hard_em_targets(utility: Tensor, *, dim: int = -1) -> Tensor:
    """Return detached argmax expert targets for hard-EM style routing."""

    _require_tensor("utility", utility)
    dim = _normalize_dim(utility.ndim, dim)
    if utility.shape[dim] < 2:
        raise ValueError("utility dimension must contain at least two experts")
    return utility.argmax(dim=dim).detach()


def hard_em_target(utility: Tensor, *, dim: int = -1) -> Tensor:
    """Alias for :func:`hard_em_targets`."""

    return hard_em_targets(utility, dim=dim)


def utility_targeted_ce(
    router_logits: Tensor,
    utility: Tensor,
    *,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Cross-entropy from router logits to detached utility argmax targets."""

    _require_tensor("router_logits", router_logits)
    _require_tensor("utility", utility)
    _require_temperature("temperature", temperature)
    if router_logits.shape != utility.shape:
        raise ValueError("router_logits and utility must have the same shape")
    if router_logits.ndim < 2 or router_logits.shape[-1] < 2:
        raise ValueError("inputs must have shape (..., num_experts) with at least two experts")

    num_experts = router_logits.shape[-1]
    logits_flat = (router_logits / temperature).reshape(-1, num_experts)
    targets_flat = utility.argmax(dim=-1).reshape(-1).detach()
    return F.cross_entropy(logits_flat, targets_flat, reduction=reduction)


def utility_targeted_ce_loss(
    router_logits: Tensor,
    utility: Tensor,
    *,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`utility_targeted_ce`."""

    return utility_targeted_ce(
        router_logits,
        utility,
        temperature=temperature,
        reduction=reduction,
    )


def expert_choice_assignment(
    utility: Tensor,
    *,
    capacity: int,
) -> Tensor:
    """Assign each expert its top-``capacity`` tokens.

    Returns a boolean mask with shape ``(tokens, experts)``.
    """

    _require_tensor("utility", utility)
    if utility.ndim != 2:
        raise ValueError("utility must have shape (tokens, experts)")
    tokens, experts = utility.shape
    if tokens < 1 or experts < 1:
        raise ValueError("utility must contain at least one token and expert")
    if capacity < 1:
        raise ValueError("capacity must be at least 1")

    k = min(capacity, tokens)
    topk = utility.topk(k=k, dim=0).indices
    assignment = torch.zeros_like(utility, dtype=torch.bool)
    return assignment.scatter_(0, topk, True)


def hard_concrete_sample(
    log_alpha: Tensor,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
    training: bool = True,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample stretched hard-concrete gates in ``[0, 1]``."""

    _require_tensor("log_alpha", log_alpha)
    _require_temperature("beta", beta)
    if gamma >= 0.0:
        raise ValueError("gamma must be negative")
    if zeta <= 0.0 or zeta <= gamma:
        raise ValueError("zeta must be positive and greater than gamma")
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    if training:
        uniform = torch.rand(
            log_alpha.shape,
            dtype=log_alpha.dtype,
            device=log_alpha.device,
            generator=generator,
        )
        uniform = uniform.clamp(min=eps, max=1.0 - eps)
        noise = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((noise + log_alpha) / beta)
    else:
        relaxed = torch.sigmoid(log_alpha)

    stretched = relaxed * (zeta - gamma) + gamma
    return stretched.clamp(min=0.0, max=1.0)


def hard_concrete_expected_l0(
    log_alpha: Tensor,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> Tensor:
    """Return expected non-zero probability for hard-concrete gates."""

    _require_tensor("log_alpha", log_alpha)
    _require_temperature("beta", beta)
    if gamma >= 0.0:
        raise ValueError("gamma must be negative")
    if zeta <= 0.0 or zeta <= gamma:
        raise ValueError("zeta must be positive and greater than gamma")

    offset = beta * math.log(-gamma / zeta)
    return torch.sigmoid(log_alpha - offset)


def hard_concrete_gate(
    log_alpha: Tensor,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
    training: bool = True,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(sampled_gate, expected_l0)`` for hard-concrete gates."""

    sample = hard_concrete_sample(
        log_alpha,
        beta=beta,
        gamma=gamma,
        zeta=zeta,
        training=training,
        eps=eps,
        generator=generator,
    )
    expected_l0 = hard_concrete_expected_l0(
        log_alpha,
        beta=beta,
        gamma=gamma,
        zeta=zeta,
    )
    return sample, expected_l0


utility_targeted_cross_entropy = utility_targeted_ce
utility_targeted_ste = utility_targeted_ce
hard_em_utility_targets = hard_em_targets
hard_em_utility_ste = hard_em_targets
expert_choice_imitation = expert_choice_assignment
hard_concrete_row_gates = hard_concrete_gate
