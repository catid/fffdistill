from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cifar_mamba_fff.locoprop.ridge_refit import RidgeRefitResult, ridge_refit


def _synthetic_problem(
    *,
    n: int = 96,
    k: int = 12,
    o: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(n, k, generator=generator)
    true_v = torch.randn(k, o, generator=generator)
    noise = 0.02 * torch.randn(n, o, generator=generator)
    Y = A.matmul(true_v) + noise
    return A, Y, true_v


def test_ridge_refit_decreases_mse_without_prior() -> None:
    A, Y, _ = _synthetic_problem(seed=1)

    result = ridge_refit(A, Y, ridge_lambda=1.0e-4)

    assert isinstance(result, RidgeRefitResult)
    assert result.weights.shape == (A.shape[1], Y.shape[1])
    assert result.mse_after <= result.mse_before + 1.0e-6
    assert math.isfinite(result.mse_before)
    assert math.isfinite(result.mse_after)
    assert result.jitter_used >= 0.0
    assert result.jitter == result.jitter_used
    assert result.used_fallback is False
    assert result.method == "cholesky"


def test_ridge_refit_decreases_mse_with_prior() -> None:
    A, Y, true_v = _synthetic_problem(seed=2)
    v0 = true_v + 0.5 * torch.randn_like(true_v)

    result = ridge_refit(A, Y, ridge_lambda=1.0e-2, v0=v0)

    assert result.weights.shape == v0.shape
    assert result.weights.dtype == v0.dtype
    assert result.mse_after <= result.mse_before + 1.0e-6


def test_ridge_refit_weights_do_not_require_grad_when_inputs_do() -> None:
    A, Y, true_v = _synthetic_problem(seed=9)
    A = A.detach().requires_grad_(True)
    Y = Y.detach().requires_grad_(True)
    v0 = (true_v + 0.5 * torch.randn_like(true_v)).detach().requires_grad_(True)

    with torch.enable_grad():
        result = ridge_refit(A, Y, ridge_lambda=1.0e-2, v0=v0)

    assert result.weights.shape == v0.shape
    assert result.weights.requires_grad is False
    assert result.weights.grad_fn is None
    assert result.mse_after <= result.mse_before + 1.0e-6


def test_ridge_refit_matches_closed_form_with_prior() -> None:
    A, Y, true_v = _synthetic_problem(seed=8)
    v0 = true_v + 0.25 * torch.randn_like(true_v)
    ridge_lambda = 3.0e-2

    result = ridge_refit(A, Y, ridge_lambda=ridge_lambda, v0=v0)

    eye = torch.eye(A.shape[1], dtype=torch.float32)
    lhs = A.float().T.matmul(A.float()) + ridge_lambda * eye
    rhs = A.float().T.matmul(Y.float()) + ridge_lambda * v0.float()
    expected = torch.linalg.solve(lhs, rhs)
    assert torch.allclose(result.weights, expected, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize("with_prior", [False, True])
def test_alpha_zero_preserves_baseline_mse(with_prior: bool) -> None:
    A, Y, true_v = _synthetic_problem(seed=3)
    v0 = true_v + torch.randn_like(true_v) if with_prior else None

    result = ridge_refit(A, Y, ridge_lambda=1.0e-3, v0=v0, alpha=0.0)

    assert result.mse_after == pytest.approx(result.mse_before, abs=1.0e-6)
    if v0 is None:
        assert torch.allclose(result.weights, torch.zeros_like(result.weights))
    else:
        assert torch.allclose(result.weights, v0)


def test_jitter_retry_handles_rank_deficient_basis() -> None:
    generator = torch.Generator().manual_seed(4)
    col = torch.randn(32, 1, generator=generator)
    A = torch.cat([col, col, torch.zeros(32, 1)], dim=1)
    Y = torch.randn(32, 2, generator=generator)

    result = ridge_refit(A, Y, ridge_lambda=0.0, max_cholesky_attempts=4)

    assert result.jitter_used > 0.0
    assert result.used_fallback is False
    assert result.method == "cholesky"
    assert result.mse_after <= result.mse_before + 1.0e-5


def test_solve_fallback_can_be_forced() -> None:
    A, Y, _ = _synthetic_problem(seed=5)

    result = ridge_refit(A, Y, ridge_lambda=1.0e-3, max_cholesky_attempts=0)

    assert result.used_fallback is True
    assert result.method == "solve"
    assert result.cholesky_info == -1
    assert result.mse_after <= result.mse_before + 1.0e-6


def test_bfloat16_inputs_accumulate_and_report_fp32_mse() -> None:
    A, Y, _ = _synthetic_problem(seed=6)

    result = ridge_refit(A.bfloat16(), Y.bfloat16(), ridge_lambda=1.0e-2)

    assert result.weights.dtype == torch.bfloat16
    assert math.isfinite(result.mse_before)
    assert math.isfinite(result.mse_after)
    assert result.mse_after <= result.mse_before + 1.0e-3


def test_shape_validation() -> None:
    A, Y, _ = _synthetic_problem(seed=7)

    with pytest.raises(ValueError, match="share N"):
        ridge_refit(A[:-1], Y)

    with pytest.raises(ValueError, match="v0"):
        ridge_refit(A, Y, v0=torch.randn(A.shape[1] + 1, Y.shape[1]))
