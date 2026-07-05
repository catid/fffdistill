from __future__ import annotations

import pytest
import torch
from torch import nn

from cifar_mamba_fff.models.official_fastfeedforward_baseline import (
    _synchronize_if_cuda,
    detect_official_fff_capability,
    forward_smoke,
    make_matched_official_fff,
    make_official_fff,
    official_fff_active_rows_per_token_eval,
    official_fff_leaf_width_for_parameter_budget,
    official_fff_registered_parameter_count,
    official_fff_stored_rows,
    official_fff_trainable_parameter_count,
    run_official_fff_layer_regression_baseline,
)


def test_make_official_fff_matches_installed_signature_cpu() -> None:
    pytest.importorskip("fastfeedforward")

    input_width = 7
    leaf_width = 3
    output_width = 5
    depth = 2
    model = make_official_fff(
        input_width=input_width,
        leaf_width=leaf_width,
        output_width=output_width,
        depth=depth,
    )

    assert model.input_width == input_width
    assert model.leaf_width == leaf_width
    assert model.output_width == output_width
    assert int(model.depth.item()) == depth
    assert forward_smoke(model, input_width=input_width, device="cpu") == (2, output_width)

    y = model(torch.randn(4, input_width))
    assert y.shape == (4, output_width)


def test_official_fff_budget_formulas_match_installed_module() -> None:
    pytest.importorskip("fastfeedforward")

    model = make_official_fff(input_width=4, leaf_width=3, output_width=5, depth=1)

    assert official_fff_trainable_parameter_count(4, 3, 5, 1) == 75
    assert official_fff_registered_parameter_count(4, 3, 5, 1) == 76
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 75
    assert sum(parameter.numel() for parameter in model.parameters()) == 76
    assert official_fff_stored_rows(leaf_width=3, depth=1) == 7
    assert official_fff_active_rows_per_token_eval(leaf_width=3, depth=1) == 4


def test_official_fff_capability_selects_largest_leaf_width_under_budget() -> None:
    budget = official_fff_trainable_parameter_count(4, 3, 5, 1) + 11

    assert official_fff_leaf_width_for_parameter_budget(4, 5, budget, depth=1) == 3
    capability = detect_official_fff_capability(4, 5, budget, depth=1)

    assert capability.compatible
    assert capability.leaf_width == 3
    assert capability.trainable_parameters == 75
    assert capability.trainable_parameters <= capability.parameter_budget
    assert capability.registered_parameters == 76
    assert capability.stored_rows == 7
    assert capability.active_rows_per_token_eval == 4
    assert capability.log_record()["compatible"] is True


def test_official_fff_capability_reports_incompatible_budget_and_depth_zero_eval() -> None:
    too_small = official_fff_trainable_parameter_count(4, 1, 5, 1) - 1

    budget_capability = detect_official_fff_capability(4, 5, too_small, depth=1)
    depth_capability = detect_official_fff_capability(4, 5, 100, depth=0, require_eval=True)

    assert not budget_capability.compatible
    assert "cannot fit official fastfeedforward.FFF" in budget_capability.reason
    assert not depth_capability.compatible
    assert "depth=0" in depth_capability.reason


def test_make_matched_official_fff_preserves_shape_and_training_mode() -> None:
    pytest.importorskip("fastfeedforward")
    linear = nn.Linear(4, 5)
    linear.eval()
    budget = official_fff_trainable_parameter_count(4, 2, 5, 1)

    replacement, capability = make_matched_official_fff(
        linear,
        parameter_budget=budget,
        depth=1,
    )
    x = torch.randn(2, 3, 4)
    y = replacement(x)

    assert not replacement.training
    assert capability.leaf_width == 2
    assert y.shape == (2, 3, 5)
    assert sum(parameter.numel() for parameter in replacement.parameters() if parameter.requires_grad) == (
        capability.trainable_parameters
    )


def test_make_matched_official_fff_moves_parameters_to_linear_dtype() -> None:
    pytest.importorskip("fastfeedforward")
    linear = nn.Linear(4, 5, dtype=torch.float64)
    budget = official_fff_trainable_parameter_count(4, 2, 5, 1)

    replacement, _ = make_matched_official_fff(
        linear,
        parameter_budget=budget,
        depth=1,
    )

    floating_parameters = [
        parameter for parameter in replacement.parameters() if parameter.is_floating_point()
    ]
    assert floating_parameters
    assert {parameter.dtype for parameter in floating_parameters} == {torch.float64}
    assert {parameter.device for parameter in replacement.parameters()} == {linear.weight.device}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_make_matched_official_fff_returns_bf16_under_cuda_autocast() -> None:
    pytest.importorskip("fastfeedforward")
    linear = nn.Linear(8, 6, device="cuda")
    budget = official_fff_trainable_parameter_count(8, 2, 6, 1)
    replacement, _ = make_matched_official_fff(
        linear,
        parameter_budget=budget,
        depth=1,
    )
    x = torch.randn(4, 8, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = replacement(x)
        loss = y.float().square().mean()
    loss.backward()

    assert y.dtype == torch.bfloat16
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_official_fff_regression_harness_is_deterministic_for_layer_metrics() -> None:
    pytest.importorskip("fastfeedforward")
    torch.manual_seed(123)
    linear = nn.Linear(4, 3)
    inputs = torch.randn(6, 4)
    budget = official_fff_trainable_parameter_count(4, 2, 3, 1)

    first = run_official_fff_layer_regression_baseline(
        linear,
        inputs,
        parameter_budget=budget,
        depth=1,
        train_steps=2,
        lr=1e-2,
        seed=99,
        timing_iterations=1,
        timing_warmup=0,
    )
    second = run_official_fff_layer_regression_baseline(
        linear,
        inputs,
        parameter_budget=budget,
        depth=1,
        train_steps=2,
        lr=1e-2,
        seed=99,
        timing_iterations=1,
        timing_warmup=0,
    )

    assert first.capability == second.capability
    assert first.train_steps == 2
    assert first.token_count == 6
    assert first.output_shape == (6, 3)
    assert first.mse == pytest.approx(second.mse, abs=0.0)
    assert first.normalized_mse == pytest.approx(second.normalized_mse, abs=0.0)
    assert first.cosine_similarity == pytest.approx(second.cosine_similarity, abs=0.0)
    assert first.tokens_per_second > 0.0
    assert second.tokens_per_second > 0.0
    assert first.log_record()["capability"]["leaf_width"] == 2


def test_official_fff_regression_harness_accepts_non_contiguous_inputs() -> None:
    pytest.importorskip("fastfeedforward")
    linear = nn.Linear(4, 3)
    inputs = torch.randn(3, 2, 4).transpose(0, 1)
    budget = official_fff_trainable_parameter_count(4, 1, 3, 1)

    result = run_official_fff_layer_regression_baseline(
        linear,
        inputs,
        parameter_budget=budget,
        depth=1,
        train_steps=0,
        timing_iterations=1,
        timing_warmup=0,
    )

    assert not inputs.is_contiguous()
    assert result.token_count == 6
    assert result.output_shape == (2, 3, 3)


def test_official_fff_regression_harness_does_not_leak_seed() -> None:
    pytest.importorskip("fastfeedforward")
    linear = nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    budget = official_fff_trainable_parameter_count(4, 1, 3, 1)

    torch.manual_seed(2026)
    expected = torch.rand(3)
    torch.manual_seed(2026)
    run_official_fff_layer_regression_baseline(
        linear,
        inputs,
        parameter_budget=budget,
        depth=1,
        train_steps=0,
        seed=99,
        timing_iterations=1,
        timing_warmup=0,
    )

    torch.testing.assert_close(torch.rand(3), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_official_fff_regression_harness_does_not_leak_cuda_seed() -> None:
    pytest.importorskip("fastfeedforward")
    device = torch.device("cuda:0")
    linear = nn.Linear(4, 3, device=device)
    inputs = torch.randn(2, 4, device=device)
    budget = official_fff_trainable_parameter_count(4, 1, 3, 1)

    torch.cuda.manual_seed(2026)
    expected = torch.rand(3, device=device)
    torch.cuda.manual_seed(2026)
    run_official_fff_layer_regression_baseline(
        linear,
        inputs,
        parameter_budget=budget,
        depth=1,
        train_steps=0,
        seed=99,
        timing_iterations=1,
        timing_warmup=0,
    )

    torch.testing.assert_close(torch.rand(3, device=device), expected)


def test_cuda_synchronization_helper_only_syncs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []

    def fake_synchronize(device: torch.device) -> None:
        calls.append(device)

    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)

    _synchronize_if_cuda(torch.device("cpu"))
    assert calls == []

    cuda_device = torch.device("cuda:0")
    _synchronize_if_cuda(cuda_device)
    assert calls == [cuda_device]
