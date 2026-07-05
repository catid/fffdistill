from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cifar_mamba_fff.models.sparse_row_column import (
    CheckerboardSparseMoELinear,
    CoupledRowColumnLinear,
    SparseColumnLinear,
    SparseRowLinear,
)


def test_sparse_row_linear_matches_selected_row_basis_forward() -> None:
    module = SparseRowLinear(3, 2, row_banks=3, rows_per_token=2, activation="relu", bias=True)
    with torch.no_grad():
        module.row_weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        module.row_bias.zero_()
        module.row_output.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.25], [-2.0, 1.0]]))
        module.bias.copy_(torch.tensor([0.1, -0.2]))
    x = torch.tensor([[3.0, 2.0, 1.0], [0.0, -1.0, 4.0]])

    route = module.route(x)
    selected_output = module.row_output[route.row_ids]
    expected = torch.einsum("nr,nro->no", route.row_scores, selected_output) + module.bias

    torch.testing.assert_close(module(x), expected)
    assert route.row_ids.shape == (2, 2)
    assert route.diagnostics["active_rows_per_token"] == 2
    assert route.diagnostics["active_columns_per_token"] == 2
    assert route.diagnostics["dead_rows"] == 0


def test_sparse_column_linear_activates_only_selected_output_blocks() -> None:
    module = SparseColumnLinear(3, 6, column_blocks=3, column_blocks_per_token=1, bias=True)
    with torch.no_grad():
        module.column_router_weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        module.column_router_bias.zero_()
        module.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10.0)
        module.bias.copy_(torch.arange(6, dtype=torch.float32) / 100.0)
    x = torch.tensor([[5.0, 2.0, 1.0], [0.0, 1.0, 4.0]])

    y = module(x)
    route = module.route(x)
    dense = F.linear(x, module.weight, module.bias)

    assert y.shape == (2, 6)
    for token in range(x.shape[0]):
        active = route.column_ids[token]
        inactive = torch.ones(6, dtype=torch.bool)
        inactive[active] = False
        torch.testing.assert_close(y[token, active], dense[token, active])
        torch.testing.assert_close(y[token, inactive], torch.zeros_like(y[token, inactive]))
    assert route.diagnostics["active_columns_per_token"] == 2
    assert route.diagnostics["estimated_active_flops_per_token"] == 12


def test_coupled_row_column_linear_scatters_only_selected_intersections() -> None:
    module = CoupledRowColumnLinear(
        3,
        4,
        row_banks=3,
        rows_per_token=2,
        column_blocks=2,
        column_blocks_per_token=1,
        activation="relu",
        bias=False,
    )
    with torch.no_grad():
        module.row_weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        module.row_bias.zero_()
        module.row_output.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0)
        module.column_router_weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        module.column_router_bias.zero_()
    x = torch.tensor([[3.0, 2.0, 1.0], [0.0, -1.0, 4.0]])

    y = module(x)
    route = module.route(x)
    expected = torch.zeros_like(y)
    for token in range(x.shape[0]):
        for column in route.column_ids[token]:
            expected[token, column] = (
                route.row_scores[token] * module.row_output[route.row_ids[token], column]
            ).sum()

    torch.testing.assert_close(y, expected)
    assert route.diagnostics["active_rows_per_token"] == 2
    assert route.diagnostics["active_columns_per_token"] == 2
    assert route.diagnostics["active_intersections_per_token"] == 4


def test_checkerboard_sparse_moe_matches_manual_expert_grid_forward() -> None:
    module = CheckerboardSparseMoELinear(
        2,
        4,
        row_experts=2,
        rows_per_token=1,
        column_blocks=2,
        column_blocks_per_token=1,
        bias=True,
    )
    with torch.no_grad():
        module.row_router_weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        module.row_router_bias.zero_()
        module.column_router_weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        module.column_router_bias.zero_()
        module.expert_weight.copy_(
            torch.tensor(
                [
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 2.0]],
                    ],
                    [
                        [[-1.0, 0.0], [0.0, -1.0]],
                        [[0.5, 0.0], [0.0, 0.5]],
                    ],
                ]
            )
        )
        module.expert_bias.zero_()
    x = torch.tensor([[3.0, 1.0], [0.0, 2.0]])

    y = module(x)
    route = module.route(x)
    expected = torch.zeros_like(y)
    for token in range(x.shape[0]):
        row = route.row_ids[token, 0]
        block = route.column_block_ids[token, 0]
        columns = route.column_ids[token]
        expected[token, columns] = F.linear(
            x[token],
            module.expert_weight[row, block],
            module.expert_bias[row, block],
        )

    torch.testing.assert_close(y, expected)
    assert route.diagnostics["active_intersections_per_token"] == 1
    assert route.diagnostics["expert_usage"].shape == (2, 2)
    assert route.diagnostics["dead_experts"] == 2


def test_sparse_row_column_modules_preserve_leading_shape_and_empty_batches() -> None:
    modules = [
        SparseRowLinear(5, 4, row_banks=3, rows_per_token=2),
        SparseColumnLinear(5, 4, column_blocks=2, column_blocks_per_token=1),
        CoupledRowColumnLinear(
            5,
            4,
            row_banks=3,
            rows_per_token=2,
            column_blocks=2,
            column_blocks_per_token=1,
        ),
        CheckerboardSparseMoELinear(
            5,
            4,
            row_experts=3,
            rows_per_token=2,
            column_blocks=2,
            column_blocks_per_token=1,
        ),
    ]

    for module in modules:
        y = module(torch.randn(2, 0, 5))

        assert y.shape == (2, 0, 4)
        assert module.diagnostics()["estimated_dense_flops_per_token"] == 40


def test_sparse_row_column_validation_rejects_incompatible_budgets() -> None:
    with pytest.raises(ValueError, match="rows_per_token"):
        SparseRowLinear(3, 4, row_banks=2, rows_per_token=3)
    with pytest.raises(ValueError, match="divisible"):
        SparseColumnLinear(3, 5, column_blocks=2, column_blocks_per_token=1)
    with pytest.raises(ValueError, match="column_blocks_per_token"):
        CoupledRowColumnLinear(
            3,
            4,
            row_banks=2,
            rows_per_token=1,
            column_blocks=2,
            column_blocks_per_token=3,
        )
    with pytest.raises(ValueError, match="expected input last dimension"):
        CheckerboardSparseMoELinear(
            3,
            4,
            row_experts=2,
            rows_per_token=1,
            column_blocks=2,
            column_blocks_per_token=1,
        )(torch.randn(2, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_sparse_row_column_optional_cuda_smoke() -> None:
    module = CheckerboardSparseMoELinear(
        8,
        8,
        row_experts=4,
        rows_per_token=2,
        column_blocks=4,
        column_blocks_per_token=2,
        device="cuda",
    )
    x = torch.randn(3, 8, device="cuda")

    y = module(x)
    diagnostics = module.diagnostics(x)

    assert y.shape == (3, 8)
    assert torch.isfinite(y).all()
    assert diagnostics["expert_usage"].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("module_cls", "kwargs"),
    [
        (
            SparseColumnLinear,
            {
                "column_blocks": 2,
                "column_blocks_per_token": 1,
            },
        ),
        (
            CoupledRowColumnLinear,
            {
                "row_banks": 4,
                "rows_per_token": 2,
                "column_blocks": 2,
                "column_blocks_per_token": 1,
            },
        ),
        (
            CheckerboardSparseMoELinear,
            {
                "row_experts": 2,
                "rows_per_token": 1,
                "column_blocks": 2,
                "column_blocks_per_token": 1,
            },
        ),
    ],
)
def test_sparse_column_paths_support_cuda_bf16_autocast_backward(
    module_cls: type[torch.nn.Module],
    kwargs: dict[str, object],
) -> None:
    module = module_cls(8, 8, device="cuda", **kwargs)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = module(x)
        loss = y.float().square().mean()
    loss.backward()

    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
