from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.models.fff_linear import FFFLinear, FFFLinearConfig


@pytest.mark.parametrize("activation", ["silu", "gelu", "relu"])
@pytest.mark.parametrize("hard_routing", [True, False])
def test_fff_linear_preserves_leading_shape(activation: str, hard_routing: bool) -> None:
    torch.manual_seed(1)
    layer = FFFLinear(
        8,
        4,
        depth=3,
        shared_rows=2,
        route_rows=2,
        leaf_rows=4,
        activation=activation,
        hard_routing=hard_routing,
        route_row_role="split_routing_output",
        route_result_rows=2,
        route_rows_output_count=3,
        master_leaf=True,
        fallback_leaf=True,
        region_leak=0.05,
    )

    x = torch.randn(2, 3, 5, 8)
    y = layer(x)

    assert y.shape == (2, 3, 5, 4)
    assert torch.isfinite(y).all()


def test_fff_linear_accepts_dataclass_config_and_legacy_contribution_alias() -> None:
    config = FFFLinearConfig(
        in_features=6,
        out_features=3,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        route_rows_contribute=True,
        route_rows_output_count=1,
        bias=False,
    )

    layer = FFFLinear(config)

    assert layer.route_row_role == "shared_routing_and_output"
    assert layer.config.route_rows_contribute is True
    assert layer.route_output_rows_per_token == 1
    assert layer(torch.randn(4, 6)).shape == (4, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "route_row_role": "routing_only",
        },
        {
            "route_row_role": "shared_routing_and_output",
            "route_rows_output_count": "all",
        },
        {
            "route_row_role": "split_routing_output",
            "route_result_rows": 2,
            "route_rows_output_count": "all",
        },
    ],
)
@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)])
def test_empty_leading_dimensions_preserve_route_shapes(
    kwargs: dict[str, object],
    shape: tuple[int, ...],
) -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=2,
        leaf_rows=2,
        **kwargs,
    )
    x = torch.randn(*shape)
    leading_shape = shape[:-1]

    y = layer(x)
    route_info = layer.route(x)

    assert y.shape == (*leading_shape, 4)
    assert route_info.leaf_ids.shape == leading_shape
    assert route_info.leaf_probs.shape == (*leading_shape, layer.leaves)
    assert route_info.node_ids.shape == (*leading_shape, layer.depth)
    assert route_info.route_values.shape == (*leading_shape, layer.depth, layer.route_rows)
    assert route_info.route_result_values.shape == (
        *leading_shape,
        layer.depth,
        layer.route_result_rows,
    )
    assert route_info.diagnostics["active_rows_per_token"].shape == leading_shape


def test_diagnostics_report_depth_leaves_stored_and_active_rows() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=1,
        route_rows=1,
        leaf_rows=2,
        bias=False,
    )

    diagnostics = layer.diagnostics()

    assert diagnostics["depth"] == 2
    assert diagnostics["leaves"] == 4
    assert diagnostics["stored_rows"] == 1 + 3 * 1 + 4 * 2
    assert diagnostics["route_row_role"] == "routing_only"
    assert diagnostics["max_visited_route_rows_per_token"] == 2
    assert diagnostics["max_route_output_rows_per_token"] == 0
    assert diagnostics["route_output_rows_per_token"] == 0
    assert diagnostics["active_rows_per_token"] == 3


@pytest.mark.parametrize(
    ("role", "route_result_rows", "count", "expected_route_outputs", "expected_stored"),
    [
        ("routing_only", 0, 0, 0, 1 + 3 * 2 + 4 * 1),
        ("shared_routing_and_output", 0, 1, 1, 1 + 3 * 2 + 4 * 1),
        ("shared_routing_and_output", 0, "all", 4, 1 + 3 * 2 + 4 * 1),
        ("split_routing_output", 2, 3, 3, 1 + 3 * 2 + 3 * 2 + 4 * 1),
    ],
)
def test_route_row_role_diagnostics(
    role: str,
    route_result_rows: int,
    count: int | str,
    expected_route_outputs: int,
    expected_stored: int,
) -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=1,
        route_rows=2,
        route_result_rows=route_result_rows,
        leaf_rows=1,
        route_row_role=role,
        route_rows_output_count=count,
        bias=False,
    )

    diagnostics = layer.diagnostics(torch.randn(3, 8))

    assert diagnostics["stored_rows"] == expected_stored
    assert diagnostics["route_output_rows_per_token"] == expected_route_outputs
    assert diagnostics["active_rows_per_token"].shape == (3,)
    assert torch.equal(
        diagnostics["active_rows_per_token"],
        torch.full((3,), 1 + 1 + expected_route_outputs),
    )


def test_route_output_fraction_bounds_active_rows() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=3,
        route_rows=2,
        leaf_rows=1,
        route_row_role="shared_routing_and_output",
        route_rows_output_count="all",
        route_rows_output_fraction=0.5,
        bias=False,
    )

    diagnostics = layer.diagnostics(torch.randn(2, 8))

    assert diagnostics["max_route_output_rows_per_token"] == 6
    assert diagnostics["route_output_rows_per_token"] == 3
    assert torch.equal(diagnostics["active_rows_per_token"], torch.full((2,), 4))


def test_enabling_shared_route_contribution_changes_outputs_and_active_rows() -> None:
    x = torch.full((2, 8), 0.5)
    base = FFFLinear(8, 4, depth=2, route_rows=1, leaf_rows=1, bias=False)
    shared = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=1,
        route_row_role="shared_routing_and_output",
        route_rows_output_count="all",
        bias=False,
    )
    with torch.no_grad():
        for layer in (base, shared):
            layer.route_weight.fill_(1.0)
            layer.route_bias.zero_()
            layer.leaf_weight.zero_()
            layer.leaf_bias.zero_()
            layer.leaf_output.zero_()
        shared.route_output.fill_(0.25)

    y_base = base(x)
    y_shared = shared(x)

    assert not torch.allclose(y_base, y_shared)
    assert torch.equal(base.route(x).diagnostics["active_rows_per_token"], torch.full((2,), 1))
    assert torch.equal(shared.route(x).diagnostics["active_rows_per_token"], torch.full((2,), 3))


def test_enabling_split_route_contribution_changes_outputs_and_active_rows() -> None:
    x = torch.full((2, 8), 0.5)
    split = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        route_result_rows=2,
        leaf_rows=1,
        route_row_role="split_routing_output",
        route_rows_output_count=1,
        bias=False,
    )
    with torch.no_grad():
        split.route_weight.fill_(1.0)
        split.route_bias.zero_()
        split.route_result_weight.fill_(1.0)
        split.route_result_bias.zero_()
        split.route_result_output.fill_(0.25)
        split.leaf_weight.zero_()
        split.leaf_bias.zero_()
        split.leaf_output.zero_()

    y = split(x)

    assert torch.count_nonzero(y) > 0
    assert torch.equal(split.route(x).diagnostics["active_rows_per_token"], torch.full((2,), 2))


def _assert_nonzero_finite_grad(name: str, parameter: torch.nn.Parameter) -> None:
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name
    assert parameter.grad.detach().abs().sum() > 0.0, name


def test_gradients_flow_for_soft_routing() -> None:
    torch.manual_seed(3)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=1,
        route_rows=2,
        leaf_rows=2,
        hard_routing=False,
        route_row_role="shared_routing_and_output",
        route_rows_output_count=2,
    )
    x = torch.randn(5, 8, requires_grad=True)

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in layer.named_parameters():
        _assert_nonzero_finite_grad(name, parameter)


@pytest.mark.parametrize(
    ("kwargs", "expected_parameter_names"),
    [
        (
            {
                "route_row_role": "routing_only",
            },
            {
                "shared_weight",
                "shared_bias",
                "shared_output",
                "route_weight",
                "route_bias",
                "leaf_weight",
                "leaf_bias",
                "leaf_output",
                "bias",
            },
        ),
        (
            {
                "route_row_role": "shared_routing_and_output",
                "route_rows_output_count": 2,
            },
            {
                "shared_weight",
                "shared_bias",
                "shared_output",
                "route_weight",
                "route_bias",
                "route_output",
                "leaf_weight",
                "leaf_bias",
                "leaf_output",
                "bias",
            },
        ),
        (
            {
                "route_row_role": "split_routing_output",
                "route_result_rows": 2,
                "route_rows_output_count": 2,
            },
            {
                "shared_weight",
                "shared_bias",
                "shared_output",
                "route_weight",
                "route_bias",
                "route_result_weight",
                "route_result_bias",
                "route_result_output",
                "leaf_weight",
                "leaf_bias",
                "leaf_output",
                "bias",
            },
        ),
    ],
)
def test_gradients_reach_applicable_route_leaf_and_shared_parameters(
    kwargs: dict[str, object],
    expected_parameter_names: set[str],
) -> None:
    torch.manual_seed(23)
    layer = FFFLinear(
        5,
        3,
        depth=2,
        shared_rows=2,
        route_rows=2,
        leaf_rows=2,
        hard_routing=False,
        **kwargs,
    )
    x = torch.randn(6, 5, requires_grad=True)
    target = torch.randn(6, 3)

    loss = torch.nn.functional.mse_loss(layer(x), target)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.detach().abs().sum() > 0.0
    assert set(dict(layer.named_parameters())) == expected_parameter_names
    for name, parameter in layer.named_parameters():
        _assert_nonzero_finite_grad(name, parameter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route_rows": 3},
        {"leaf_rows": 3},
        {"activation": "tanh"},
        {"train_temperature": 0.0},
        {"region_leak": -0.1},
        {"route_row_role": "split_routing_output", "route_result_rows": 0},
        {"route_row_role": "shared_routing_and_output", "route_result_rows": 1},
        {"route_row_role": "routing_only", "route_rows_output_count": 1},
        {"route_row_role": "unknown"},
        {"route_rows_output_fraction": 1.5},
    ],
)
def test_invalid_config_raises_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FFFLinear(8, 4, depth=2, **kwargs)


def test_bad_input_shape_raises_value_error() -> None:
    layer = FFFLinear(8, 4, depth=2)

    with pytest.raises(ValueError):
        layer(torch.randn(2, 7))
