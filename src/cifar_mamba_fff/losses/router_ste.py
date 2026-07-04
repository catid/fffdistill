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


def _one_hot_indices(indices: Tensor, *, like: Tensor, dim: int) -> Tensor:
    return torch.zeros_like(like).scatter_(dim, indices.unsqueeze(dim), 1.0)


def _require_matching_router_tensors(router_logits: Tensor, utility: Tensor, *, dim: int) -> int:
    _require_tensor("router_logits", router_logits)
    _require_tensor("utility", utility)
    dim = _normalize_dim(router_logits.ndim, dim)
    if router_logits.shape != utility.shape:
        raise ValueError("router_logits and utility must have the same shape")
    if router_logits.shape[dim] < 2:
        raise ValueError("router dimension must contain at least two experts")
    return dim


def _flatten_router_dim(values: Tensor, *, dim: int) -> Tensor:
    return values.movedim(dim, -1).reshape(-1, values.shape[dim])


def router_recipe_diagnostics(
    route_values: Tensor,
    *,
    logits: Tensor | None = None,
    utility: Tensor | None = None,
    dim: int = -1,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Return finite summary diagnostics for soft or hard router outputs."""

    _require_tensor("route_values", route_values)
    dim = _normalize_dim(route_values.ndim, dim)
    if route_values.shape[dim] < 2:
        raise ValueError("route dimension must contain at least two choices")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    flat_routes = _flatten_router_dim(route_values, dim=dim)
    nonnegative = flat_routes.clamp_min(0.0)
    mass = nonnegative.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs = nonnegative / mass
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    binary_entries = (flat_routes == 0.0) | (flat_routes == 1.0)
    one_hot_rows = binary_entries.all(dim=-1) & torch.isclose(
        flat_routes.sum(dim=-1),
        flat_routes.new_ones(flat_routes.shape[0]),
    )

    diagnostics = {
        "entropy_mean": entropy.mean(),
        "max_probability_mean": probs.max(dim=-1).values.mean(),
        "mass_error_mean": (flat_routes.sum(dim=-1) - 1.0).abs().mean(),
        "hard_fraction": one_hot_rows.to(dtype=flat_routes.dtype).mean(),
        "expert_usage": probs.mean(dim=0),
    }

    if logits is not None:
        _require_tensor("logits", logits)
        if logits.shape != route_values.shape:
            raise ValueError("logits and route_values must have the same shape")
        flat_logits = _flatten_router_dim(logits, dim=dim)
        top2 = flat_logits.topk(k=2, dim=-1).values
        diagnostics["logit_margin_mean"] = (top2[:, 0] - top2[:, 1]).mean()

    if utility is not None:
        _require_tensor("utility", utility)
        if utility.shape != route_values.shape:
            raise ValueError("utility and route_values must have the same shape")
        flat_utility = _flatten_router_dim(utility, dim=dim)
        diagnostics["selected_utility_mean"] = (probs * flat_utility).sum(dim=-1).mean()

    return diagnostics


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
    y_soft = torch.softmax(logits / temperature, dim=dim)
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


def utility_soft_targets(
    utility: Tensor,
    *,
    temperature: float = 1.0,
    dim: int = -1,
) -> Tensor:
    """Return detached soft targets induced by per-expert utility scores."""

    _require_tensor("utility", utility)
    dim = _normalize_dim(utility.ndim, dim)
    if utility.shape[dim] < 2:
        raise ValueError("utility dimension must contain at least two experts")
    temperature = _resolve_temperature(temperature, None)
    return torch.softmax(utility / temperature, dim=dim).detach()


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


def utility_targeted_ste(
    router_logits: Tensor,
    utility: Tensor,
    *,
    temperature: float = 1.0,
    utility_temperature: float = 1.0,
    hard: bool = True,
    dim: int = -1,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Route toward detached utility targets with gradients from router logits.

    ``hard=True`` uses a one-hot utility argmax forward pass. ``hard=False`` uses
    a detached utility-softmax target. In both cases, straight-through gradients
    flow through ``router_logits`` via a router softmax surrogate.
    """

    dim = _require_matching_router_tensors(router_logits, utility, dim=dim)
    temperature = _resolve_temperature(temperature, None)
    utility_temperature = _resolve_temperature(utility_temperature, None)

    router_probs = torch.softmax(router_logits / temperature, dim=dim)
    if hard:
        target_indices = utility.argmax(dim=dim).detach()
        forward_target = _one_hot_indices(target_indices, like=router_logits, dim=dim)
    else:
        forward_target = torch.softmax(utility / utility_temperature, dim=dim).detach()
    routed = forward_target.detach() - router_probs.detach() + router_probs
    if not return_diagnostics:
        return routed
    return routed, router_recipe_diagnostics(
        routed,
        logits=router_logits,
        utility=utility,
        dim=dim,
    )


def hard_em_utility_ste(
    router_logits: Tensor,
    utility: Tensor,
    *,
    temperature: float = 1.0,
    dim: int = -1,
    return_targets: bool = False,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Hard-EM router assignment with STE gradients through router logits."""

    dim = _require_matching_router_tensors(router_logits, utility, dim=dim)
    temperature = _resolve_temperature(temperature, None)
    targets = utility.argmax(dim=dim).detach()
    forward_target = _one_hot_indices(targets, like=router_logits, dim=dim)
    router_probs = torch.softmax(router_logits / temperature, dim=dim)
    routed = forward_target.detach() - router_probs.detach() + router_probs

    diagnostics = None
    if return_diagnostics:
        diagnostics = router_recipe_diagnostics(
            routed,
            logits=router_logits,
            utility=utility,
            dim=dim,
        )
    if return_targets and return_diagnostics:
        return routed, targets, diagnostics
    if return_targets:
        return routed, targets
    if return_diagnostics:
        return routed, diagnostics
    return routed


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


def expert_choice_imitation(
    router_logits: Tensor,
    utility: Tensor,
    *,
    capacity: int,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
    return_assignment: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Binary imitation loss for expert-choice token assignments."""

    _require_tensor("router_logits", router_logits)
    _require_tensor("utility", utility)
    _require_temperature("temperature", temperature)
    if router_logits.shape != utility.shape:
        raise ValueError("router_logits and utility must have the same shape")
    if router_logits.ndim != 2:
        raise ValueError("router_logits and utility must have shape (tokens, experts)")

    assignment = expert_choice_assignment(utility, capacity=capacity)
    targets = assignment.to(dtype=router_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        router_logits / temperature,
        targets,
        reduction=reduction,
    )
    if return_assignment:
        return loss, assignment
    return loss


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


def hard_concrete_row_gates(
    log_alpha: Tensor,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
    training: bool = True,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
    l0_weight: float = 1.0,
    reduction: Reduction = "mean",
    return_diagnostics: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return row gates, expected L0 probabilities, and optional diagnostics."""

    if l0_weight < 0.0 or not math.isfinite(l0_weight):
        raise ValueError("l0_weight must be a non-negative finite value")

    gates, expected_l0 = hard_concrete_gate(
        log_alpha,
        beta=beta,
        gamma=gamma,
        zeta=zeta,
        training=training,
        eps=eps,
        generator=generator,
    )
    if not return_diagnostics:
        return gates, expected_l0

    l0_penalty = _reduce(expected_l0, reduction) * l0_weight
    diagnostics = {
        "active_fraction": (gates > 0.0).to(dtype=gates.dtype).mean(),
        "expected_active_rows": expected_l0.sum(),
        "mean_expected_l0": expected_l0.mean(),
        "l0_penalty": l0_penalty,
    }
    return gates, expected_l0, diagnostics


utility_targeted_cross_entropy = utility_targeted_ce
hard_em_utility_targets = hard_em_targets
