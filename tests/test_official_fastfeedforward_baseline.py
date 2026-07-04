from __future__ import annotations

import pytest
import torch
from torch import nn

from cifar_mamba_fff.models.official_fastfeedforward_baseline import (
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
