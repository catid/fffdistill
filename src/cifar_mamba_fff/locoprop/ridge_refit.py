"""Local ridge refits for fixed LocoProp-S bases."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

SolveMethod = Literal["cholesky", "solve"]


@dataclass(frozen=True)
class RidgeRefitResult:
    """Weights and telemetry from a local ridge refit."""

    weights: torch.Tensor
    mse_before: float
    mse_after: float
    jitter_used: float
    used_fallback: bool
    attempts: int
    cholesky_info: int
    method: SolveMethod

    @property
    def jitter(self) -> float:
        """Backward-compatible short name for ``jitter_used``."""

        return self.jitter_used


def ridge_refit(
    A: torch.Tensor,
    Y: torch.Tensor,
    *,
    ridge_lambda: float = 1.0e-3,
    v0: torch.Tensor | None = None,
    alpha: float = 1.0,
    initial_jitter: float = 0.0,
    jitter_multiplier: float = 10.0,
    max_cholesky_attempts: int = 5,
    out_dtype: torch.dtype | None = None,
) -> RidgeRefitResult:
    """Refit output weights for a fixed activation/basis matrix.

    Solves the FP32 normal equation

        V* = (A.T @ A + ridge_lambda * I)^-1 (A.T @ Y + ridge_lambda * V0)

    with Cholesky first, jitter retries, and a dense solve fallback. If ``v0``
    is not supplied, it is treated as zeros for the solve and baseline MSE.
    ``alpha`` blends from the baseline weights to the solved weights.
    """

    _validate_inputs(
        A=A,
        Y=Y,
        v0=v0,
        ridge_lambda=ridge_lambda,
        alpha=alpha,
        initial_jitter=initial_jitter,
        jitter_multiplier=jitter_multiplier,
        max_cholesky_attempts=max_cholesky_attempts,
    )

    with torch.no_grad():
        a32 = A.detach().to(dtype=torch.float32)
        y32 = Y.detach().to(dtype=torch.float32)
        prior32 = v0.detach().to(dtype=torch.float32) if v0 is not None else None

        k = a32.shape[1]
        eye = torch.eye(k, device=a32.device, dtype=torch.float32)
        lhs = a32.transpose(0, 1).matmul(a32)
        if ridge_lambda != 0.0:
            lhs = lhs + float(ridge_lambda) * eye

        rhs = a32.transpose(0, 1).matmul(y32)
        if prior32 is not None and ridge_lambda != 0.0:
            rhs = rhs + float(ridge_lambda) * prior32

        solved, jitter_used, used_fallback, attempts, cholesky_info, method = _solve_with_retries(
            lhs=lhs,
            rhs=rhs,
            eye=eye,
            initial_jitter=float(initial_jitter),
            jitter_multiplier=float(jitter_multiplier),
            max_cholesky_attempts=int(max_cholesky_attempts),
        )

        baseline = torch.zeros_like(solved) if prior32 is None else prior32
        blended = baseline.lerp(solved, float(alpha))

        target_dtype = out_dtype
        if target_dtype is None:
            target_dtype = v0.dtype if v0 is not None else Y.dtype
        weights = blended.to(dtype=target_dtype)

        baseline_pred = a32.matmul(baseline)
        after_pred = a32.matmul(weights.float())
        mse_before = torch.mean((baseline_pred - y32).square()).item()
        mse_after = torch.mean((after_pred - y32).square()).item()

    return RidgeRefitResult(
        weights=weights,
        mse_before=float(mse_before),
        mse_after=float(mse_after),
        jitter_used=float(jitter_used),
        used_fallback=bool(used_fallback),
        attempts=int(attempts),
        cholesky_info=int(cholesky_info),
        method=method,
    )


def _solve_with_retries(
    *,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    eye: torch.Tensor,
    initial_jitter: float,
    jitter_multiplier: float,
    max_cholesky_attempts: int,
) -> tuple[torch.Tensor, float, bool, int, int, SolveMethod]:
    jitter = initial_jitter
    last_info = -1

    for attempt in range(1, max_cholesky_attempts + 1):
        matrix = lhs if jitter == 0.0 else lhs + jitter * eye
        chol, info = torch.linalg.cholesky_ex(matrix)
        last_info = int(info.item())
        if last_info == 0:
            return (
                torch.cholesky_solve(rhs, chol),
                jitter,
                False,
                attempt,
                last_info,
                "cholesky",
            )
        jitter = _next_jitter(lhs=lhs, current=jitter, multiplier=jitter_multiplier)

    matrix = lhs if jitter == 0.0 else lhs + jitter * eye
    return (
        torch.linalg.solve(matrix, rhs),
        jitter,
        True,
        max_cholesky_attempts,
        last_info,
        "solve",
    )


def _next_jitter(*, lhs: torch.Tensor, current: float, multiplier: float) -> float:
    if current > 0.0:
        return current * multiplier

    diag_scale = torch.mean(torch.abs(torch.diagonal(lhs))).item()
    if not isfinite(diag_scale):
        diag_scale = 1.0
    scale = max(float(diag_scale), 1.0)
    return torch.finfo(torch.float32).eps * scale


def _validate_inputs(
    *,
    A: torch.Tensor,
    Y: torch.Tensor,
    v0: torch.Tensor | None,
    ridge_lambda: float,
    alpha: float,
    initial_jitter: float,
    jitter_multiplier: float,
    max_cholesky_attempts: int,
) -> None:
    if A.ndim != 2:
        raise ValueError(f"A must have shape [N, K], got {tuple(A.shape)}")
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape [N, O], got {tuple(Y.shape)}")
    if A.shape[0] != Y.shape[0]:
        raise ValueError(
            f"A and Y must share N, got A.shape={tuple(A.shape)} and Y.shape={tuple(Y.shape)}"
        )
    if A.device != Y.device:
        raise ValueError(f"A and Y must be on the same device, got {A.device} and {Y.device}")
    if not torch.is_floating_point(A):
        raise TypeError(f"A must be floating point, got {A.dtype}")
    if not torch.is_floating_point(Y):
        raise TypeError(f"Y must be floating point, got {Y.dtype}")

    if v0 is not None:
        expected = (A.shape[1], Y.shape[1])
        if tuple(v0.shape) != expected:
            raise ValueError(f"v0 must have shape {expected}, got {tuple(v0.shape)}")
        if v0.device != A.device:
            raise ValueError(f"v0 must be on {A.device}, got {v0.device}")
        if not torch.is_floating_point(v0):
            raise TypeError(f"v0 must be floating point, got {v0.dtype}")

    _validate_nonnegative_finite("ridge_lambda", ridge_lambda)
    _validate_nonnegative_finite("initial_jitter", initial_jitter)
    _validate_nonnegative_finite("alpha", alpha)
    if alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not isinstance(max_cholesky_attempts, int) or max_cholesky_attempts < 0:
        raise ValueError(f"max_cholesky_attempts must be a non-negative int, got {max_cholesky_attempts}")
    jitter_multiplier_float = float(jitter_multiplier)
    if jitter_multiplier_float <= 1.0 or not isfinite(jitter_multiplier_float):
        raise ValueError(f"jitter_multiplier must be finite and > 1, got {jitter_multiplier}")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    value_float = float(value)
    if value_float < 0.0 or not isfinite(value_float):
        raise ValueError(f"{name} must be finite and non-negative, got {value}")


locoprop_ridge_refit = ridge_refit
