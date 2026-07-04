from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cifar_mamba_fff.losses.distill import distillation_loss
from cifar_mamba_fff.metrics import cosine_loss, normalized_mse


def test_normalized_mse_reduces_bfloat16_inputs_in_fp32() -> None:
    torch.manual_seed(101)
    pred = torch.randn(5, 7, dtype=torch.float32).bfloat16().requires_grad_()
    target = torch.randn(5, 7, dtype=torch.float32).bfloat16()

    loss = normalized_mse(pred, target)
    reference = (pred.float() - target.float()).square().sum() / (
        target.float().square().sum() + 1.0e-8
    )

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference, atol=0.0, rtol=0.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad.float()).all()


def test_cosine_loss_reduces_bfloat16_inputs_in_fp32() -> None:
    torch.manual_seed(102)
    pred = torch.randn(3, 4, 9, dtype=torch.float32).bfloat16().requires_grad_()
    target = torch.randn(3, 4, 9, dtype=torch.float32).bfloat16()

    loss = cosine_loss(pred, target)
    reference = 1.0 - F.cosine_similarity(
        pred.float().flatten(0, -2),
        target.float().flatten(0, -2),
        dim=-1,
        eps=1.0e-8,
    ).mean()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference, atol=0.0, rtol=0.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad.float()).all()


def test_distillation_loss_returns_fp32_for_bfloat16_inputs() -> None:
    torch.manual_seed(103)
    pred = torch.randn(6, 8, dtype=torch.float32).bfloat16().requires_grad_()
    target = torch.randn(6, 8, dtype=torch.float32).bfloat16()

    loss = distillation_loss(
        pred,
        target,
        normalized_mse_weight=0.7,
        cosine_weight=0.2,
        variance_weight=0.1,
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad.float()).all()


def test_distillation_loss_bfloat16_matches_fp32_reference() -> None:
    torch.manual_seed(104)
    pred = torch.randn(7, 6, dtype=torch.float32).bfloat16()
    target = torch.randn(7, 6, dtype=torch.float32).bfloat16()

    loss = distillation_loss(
        pred,
        target,
        normalized_mse_weight=0.6,
        cosine_weight=0.3,
        variance_weight=0.2,
    )
    pred_f = pred.float()
    target_f = target.float()
    reference = (
        0.6
        * (
            (pred_f - target_f).square().sum()
            / (target_f.square().sum() + 1.0e-8)
        )
        + 0.3
        * (
            1.0
            - F.cosine_similarity(
                pred_f.flatten(0, -2),
                target_f.flatten(0, -2),
                dim=-1,
                eps=1.0e-8,
            ).mean()
        )
        + 0.2
        * F.mse_loss(
            pred_f.var(dim=0, unbiased=False),
            target_f.var(dim=0, unbiased=False),
        )
    )

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference, atol=0.0, rtol=0.0)


def test_distillation_loss_zero_weights_returns_fp32_for_bfloat16_inputs() -> None:
    pred = torch.randn(3, 5, dtype=torch.float32).bfloat16()
    target = torch.randn(3, 5, dtype=torch.float32).bfloat16()

    loss = distillation_loss(
        pred,
        target,
        normalized_mse_weight=0.0,
        cosine_weight=0.0,
        variance_weight=0.0,
    )

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast coverage")
@pytest.mark.skipif(
    torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is not supported by this device",
)
def test_distillation_losses_return_fp32_under_cuda_bfloat16_autocast() -> None:
    pred = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mse = normalized_mse(pred, target)
        cosine = cosine_loss(pred, target)
        combined = distillation_loss(pred, target, cosine_weight=0.5, variance_weight=0.25)

    assert mse.dtype == torch.float32
    assert cosine.dtype == torch.float32
    assert combined.dtype == torch.float32
