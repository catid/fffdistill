from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .pace import OPTIMIZER_EXPERIMENTS_COMMIT, OPTIMIZER_EXPERIMENTS_SOURCE

POLAR_EXPRESS_UNSCALED: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = tuple(
    (
        a / POLAR_EXPRESS_SAFETY_FACTOR,
        b / POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for a, b, c in POLAR_EXPRESS_UNSCALED
)


def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    return x.reshape(effective_shape[0], math.prod(effective_shape[1:]))


class GramNewtonSchulz:
    """Batched Gram Newton-Schulz polar approximation used by Muon/NorMuon."""

    def __init__(
        self,
        *,
        coefficients: Iterable[Iterable[float]] | None = None,
        epsilon: float = 1e-7,
        reset_iterations: Iterable[int] = (2,),
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.coefficients = tuple(
            tuple(float(value) for value in row)
            for row in (coefficients or POLAR_EXPRESS_COEFFICIENTS)
        )
        self.epsilon = float(epsilon)
        self.reset_iterations = set(int(index) for index in reset_iterations)
        self.compute_dtype = compute_dtype

    def _compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        return torch.float16 if x.is_cuda else torch.float32

    @torch.no_grad()
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.ndim < 2:
            raise ValueError("GramNewtonSchulz expects a tensor with ndim >= 2")

        original_shape = matrix.shape
        x = matrix
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim > 3:
            x = x.reshape(-1, *x.shape[-2:])

        original_dtype = x.dtype
        x = x.to(torch.float32)
        transposed = x.size(-2) > x.size(-1)
        if transposed:
            x = x.mT

        x = x / (x.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        x = x.to(self._compute_dtype_for(x))

        if max(x.shape[-2:]) > min(x.shape[-2:]):
            x = self._gram_recurrence(x)
        else:
            x = self._standard_recurrence(x)

        if transposed:
            x = x.mT
        return x.to(original_dtype).reshape(original_shape)

    def _standard_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            gram = x @ x.mT
            poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            x = torch.baddbmm(x, poly, x, beta=a)
        return x

    def _gram_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        gram = x @ x.mT
        eye = torch.eye(gram.size(-1), device=x.device, dtype=x.dtype).expand(
            gram.size(0),
            -1,
            -1,
        ).contiguous()
        q: torch.Tensor | None = None

        for index, (a, b, c) in enumerate(self.coefficients):
            if index in self.reset_iterations and index != 0:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz reset reached without an inverse estimate")
                x = q @ x
                gram = x @ x.mT
                q = None

            z = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            if index == 0 or index in self.reset_iterations:
                q = z + a * eye
            else:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz inverse estimate was not initialized")
                q = torch.baddbmm(q, q, z, beta=a)

            if index < len(self.coefficients) - 1 and index + 1 not in self.reset_iterations:
                rz = torch.baddbmm(gram, gram, z, beta=a)
                gram = torch.baddbmm(rz, z, rz, beta=a)

        if q is None:
            raise RuntimeError("Gram Newton-Schulz finished without an inverse estimate")
        return q @ x


@torch.no_grad()
def _normuon_row_normalize(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    if tuple(second_momentum.shape[-2:]) != (update.shape[-2], 1):
        raise ValueError(
            f"NorMuon row second momentum shape {tuple(second_momentum.shape[-2:])} "
            f"!= {(update.shape[-2], 1)}"
        )
    dtype = update.dtype
    eps_t = torch.tensor(eps, dtype=dtype, device=update.device)
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    row_power = update.square().mean(dim=-1, keepdim=True).to(dtype)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    out = update * torch.rsqrt(second_momentum + eps_t)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps_t))


class MuonNorMuon(torch.optim.Optimizer):
    """Optimizer-experiments Muon+NorMuon ablation.

    This is not the official Muon package.  It exists only for T19 optimizer
    ablations where the requested NorMuon path cannot be expressed by wrapping
    official `SingleDeviceMuonWithAuxAdam` alone.  Parameter grouping remains
    this repo's audited Muon/AdamW split.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        ns_compute_dtype: torch.dtype | None = None,
    ) -> None:
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if normuon_eps <= 0.0:
            raise ValueError("normuon_eps must be positive")
        super().__init__(param_groups, defaults={})
        self.normuon_beta2 = float(normuon_beta2)
        self.normuon_eps = float(normuon_eps)
        self._orthogonalizer = GramNewtonSchulz(compute_dtype=ns_compute_dtype)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        beta = float(group.get("momentum", 0.95))
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("MuonNorMuon does not support sparse gradients")
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            grad = parameter.grad.detach().to(torch.float32)
            state = self.state[parameter]
            momentum = state.get("momentum_buffer")
            if momentum is None or momentum.shape != parameter.shape:
                momentum = state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
            momentum.lerp_(grad, 1.0 - beta)
            source = torch.lerp(grad, momentum, beta)
            matrix = _matrix_view(source)
            update = self._orthogonalizer(matrix)
            update = update * (0.2 * math.sqrt(max(update.shape[-2], update.shape[-1])))

            rows = update.shape[0]
            second = state.get("normuon_second_momentum")
            if second is None or second.shape != (rows, 1):
                second = state["normuon_second_momentum"] = torch.zeros(
                    rows,
                    1,
                    device=update.device,
                    dtype=torch.float32,
                )
            update = _normuon_row_normalize(
                update,
                second,
                beta2=self.normuon_beta2,
                eps=self.normuon_eps,
            )
            parameter.add_(update.reshape_as(parameter).to(parameter.dtype), alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("MuonNorMuon fallback does not support sparse gradients")
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            grad = parameter.grad.detach().to(torch.float32)
            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(float(beta1)).add_(grad, alpha=1.0 - float(beta1))
            exp_avg_sq.mul_(float(beta2)).addcmul_(grad, grad, value=1.0 - float(beta2))
            step = int(state["step"])
            update = exp_avg / max(1.0 - float(beta1) ** step, 1e-16)
            denom = (exp_avg_sq / max(1.0 - float(beta2) ** step, 1e-16)).sqrt().add_(eps)
            parameter.addcdiv_(update.to(parameter.dtype), denom.to(parameter.dtype), value=-lr)


def build_normuon_param_groups(
    muon_params: list[nn.Parameter],
    adamw_params: list[nn.Parameter],
    *,
    lr_muon: float,
    lr_adamw: float,
    weight_decay_muon: float,
    weight_decay_adamw: float,
    muon_momentum: float,
    adamw_betas: tuple[float, float],
    adamw_eps: float,
) -> list[dict[str, Any]]:
    return [
        {
            "params": muon_params,
            "lr": lr_muon,
            "momentum": muon_momentum,
            "weight_decay": weight_decay_muon,
            "use_muon": True,
        },
        {
            "params": adamw_params,
            "lr": lr_adamw,
            "betas": adamw_betas,
            "eps": adamw_eps,
            "weight_decay": weight_decay_adamw,
            "use_muon": False,
        },
    ]


__all__ = [
    "OPTIMIZER_EXPERIMENTS_COMMIT",
    "OPTIMIZER_EXPERIMENTS_SOURCE",
    "GramNewtonSchulz",
    "MuonNorMuon",
    "build_normuon_param_groups",
]
