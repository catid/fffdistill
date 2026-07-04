from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cifar_mamba_fff.models.baseline_linears import (
    LowRankLinear,
    SmallerDenseLinear,
    dense_linear_parameter_count,
    linear_module_parameter_count,
    low_rank_linear_parameter_count,
    low_rank_rank_for_parameter_budget,
    make_matched_low_rank_linear,
    make_matched_smaller_dense_linear,
    smaller_dense_linear_parameter_count,
    smaller_dense_out_features_for_parameter_budget,
)


def test_parameter_count_helpers_match_module_shapes() -> None:
    assert dense_linear_parameter_count(6, 4, bias=True) == 28
    assert dense_linear_parameter_count(6, 4, bias=False) == 24
    assert low_rank_linear_parameter_count(6, 4, 2, bias=True) == 24
    assert low_rank_linear_parameter_count(6, 4, 2, bias=False) == 20
    assert smaller_dense_linear_parameter_count(6, 4, 3, bias=True) == 21
    assert smaller_dense_linear_parameter_count(6, 4, 3, bias=False) == 18


def test_parameter_budget_helpers_choose_largest_candidate_under_budget() -> None:
    low_rank_budget = low_rank_linear_parameter_count(6, 4, 2, bias=True) + 1
    rank = low_rank_rank_for_parameter_budget(6, 4, low_rank_budget, bias=True)
    assert rank == 2
    assert low_rank_linear_parameter_count(6, 4, rank, bias=True) <= low_rank_budget

    smaller_dense_budget = smaller_dense_linear_parameter_count(6, 4, 3, bias=True)
    active_out_features = smaller_dense_out_features_for_parameter_budget(
        6,
        4,
        smaller_dense_budget,
        bias=True,
    )
    assert active_out_features == 3
    assert (
        smaller_dense_linear_parameter_count(6, 4, active_out_features, bias=True)
        <= smaller_dense_budget
    )


def test_parameter_budget_helpers_reject_too_small_budget() -> None:
    with pytest.raises(ValueError, match="rank 1"):
        low_rank_rank_for_parameter_budget(6, 4, 13, bias=True)

    with pytest.raises(ValueError, match="1 dense output row"):
        smaller_dense_out_features_for_parameter_budget(6, 4, 6, bias=True)


def test_low_rank_linear_preserves_shape_dtype_bias_and_parameter_count() -> None:
    module = LowRankLinear(6, 4, 2, bias=False, dtype=torch.float64)
    x = torch.randn(2, 3, 6, dtype=torch.float64)

    y = module(x)

    assert y.shape == (2, 3, 4)
    assert y.dtype == torch.float64
    assert module.in_features == 6
    assert module.out_features == 4
    assert module.rank == 2
    assert module.bias is None
    assert module.parameter_count() == 20
    assert linear_module_parameter_count(module) == module.parameter_count()


def test_low_rank_linear_forward_matches_two_linear_factors() -> None:
    module = LowRankLinear(3, 2, 2, bias=True)
    with torch.no_grad():
        module.input_weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))
        module.output_weight.copy_(torch.tensor([[2.0, -1.0], [0.25, 0.5]]))
        module.bias.copy_(torch.tensor([0.5, -0.25]))
    x = torch.tensor([[1.0, -1.0, 0.5]])

    expected = F.linear(F.linear(x, module.input_weight), module.output_weight, module.bias)

    torch.testing.assert_close(module(x), expected)


def test_smaller_dense_linear_preserves_shape_and_pads_inactive_outputs() -> None:
    module = SmallerDenseLinear(3, 5, 2, bias=True)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.0, 2.0]]))
        module.bias.copy_(torch.tensor([0.25, -0.5]))
    x = torch.randn(4, 3)

    y = module(x)
    expected_active = F.linear(x, module.weight, module.bias)

    assert y.shape == (4, 5)
    torch.testing.assert_close(y[:, :2], expected_active)
    torch.testing.assert_close(y[:, 2:], torch.zeros(4, 3))
    assert module.parameter_count() == 8
    assert linear_module_parameter_count(module) == module.parameter_count()


def test_smaller_dense_linear_full_active_rows_matches_dense_shape() -> None:
    module = SmallerDenseLinear(3, 2, 2, bias=False, dtype=torch.float64)
    x = torch.randn(2, 0, 3, dtype=torch.float64)

    y = module(x)

    assert y.shape == (2, 0, 2)
    assert y.dtype == torch.float64
    assert module.bias is None


def test_matched_factories_preserve_linear_interface_dtype_and_mode() -> None:
    linear = nn.Linear(6, 4, bias=True, dtype=torch.float64)
    linear.eval()

    low_rank = make_matched_low_rank_linear(
        linear,
        parameter_budget=low_rank_linear_parameter_count(6, 4, 2, bias=True),
    )
    smaller_dense = make_matched_smaller_dense_linear(
        linear,
        parameter_budget=smaller_dense_linear_parameter_count(6, 4, 3, bias=True),
    )

    assert isinstance(low_rank, LowRankLinear)
    assert low_rank.rank == 2
    assert low_rank.in_features == linear.in_features
    assert low_rank.out_features == linear.out_features
    assert low_rank.input_weight.dtype == torch.float64
    assert not low_rank.training

    assert isinstance(smaller_dense, SmallerDenseLinear)
    assert smaller_dense.active_out_features == 3
    assert smaller_dense.in_features == linear.in_features
    assert smaller_dense.out_features == linear.out_features
    assert smaller_dense.weight.dtype == torch.float64
    assert not smaller_dense.training


def test_factories_reject_non_linear_modules() -> None:
    with pytest.raises(TypeError, match=r"expected nn\.Linear"):
        make_matched_low_rank_linear(nn.ReLU(), parameter_budget=10)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"expected nn\.Linear"):
        make_matched_smaller_dense_linear(nn.ReLU(), parameter_budget=10)  # type: ignore[arg-type]


def test_baseline_linears_module_does_not_import_fff() -> None:
    source = Path("src/cifar_mamba_fff/models/baseline_linears.py").read_text()
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert all("fff" not in imported_module.lower() for imported_module in imported_modules)
