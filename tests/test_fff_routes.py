from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.models.fff_linear import FFFLinear


def test_route_ids_follow_hard_binary_path_for_one_route_row() -> None:
    layer = FFFLinear(8, 4, depth=3, shared_rows=0, route_rows=1, leaf_rows=1)
    with torch.no_grad():
        layer.route_weight.zero_()
        layer.route_bias.fill_(-1.0)
        layer.route_bias[0, 0] = 1.0

    route_info = layer.route(torch.zeros(5, 8))

    assert route_info.leaf_ids.shape == (5,)
    assert torch.equal(route_info.leaf_ids, torch.full((5,), 4))
    assert torch.equal(route_info.node_ids, torch.tensor([[0, 2, 5]]).expand(5, 3))
    assert torch.equal(route_info.route_bits, torch.tensor([[1, 0, 0]]).expand(5, 3))
    assert int(route_info.leaf_ids.min()) >= 0
    assert int(route_info.leaf_ids.max()) < 8


def test_route_ids_follow_hard_binary_path_for_two_route_rows() -> None:
    layer = FFFLinear(8, 4, depth=2, shared_rows=0, route_rows=2, leaf_rows=1)
    with torch.no_grad():
        layer.route_weight.zero_()
        layer.route_bias[:, 0] = 1.0
        layer.route_bias[:, 1] = -1.0
        layer.route_bias[0, 0] = -1.0
        layer.route_bias[0, 1] = 1.0

    route_info = layer.route(torch.zeros(3, 8))

    assert torch.equal(route_info.leaf_ids, torch.full((3,), 2))
    assert torch.equal(route_info.node_ids, torch.tensor([[0, 2]]).expand(3, 2))
    assert torch.equal(route_info.route_bits, torch.tensor([[1, 0]]).expand(3, 2))
    assert route_info.route_logits.shape == (3, 2, 2)
    assert route_info.route_values.shape == (3, 2, 2)
    assert route_info.route_row_ids.shape == (3, 2, 2)


def test_soft_route_probs_sum_to_one_and_preserve_leaf_ids() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=0,
        route_rows=2,
        leaf_rows=1,
        hard_routing=False,
        train_temperature=0.7,
    )

    route_info = layer.route(torch.randn(6, 8))

    assert route_info.leaf_probs.shape == (6, 4)
    assert torch.allclose(route_info.leaf_probs.sum(dim=-1), torch.ones(6), atol=1e-6)
    assert torch.equal(route_info.leaf_ids, route_info.leaf_probs.argmax(dim=-1))
    assert route_info.leaf_weights.shape == (6, 4)


def test_region_leak_uses_fallback_leaf_when_enabled() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        hard_routing=True,
        region_leak=0.25,
        fallback_leaf=True,
        bias=False,
    )

    route_info = layer.route(torch.randn(4, 8))

    assert torch.allclose(route_info.leaf_weights.sum(dim=-1), torch.full((4,), 0.75))
    assert torch.equal(route_info.diagnostics["active_rows_per_token"], torch.full((4,), 4))


def test_region_leak_without_fallback_activates_all_regular_leaves() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=1,
        hard_routing=True,
        region_leak=0.1,
        bias=False,
    )

    route_info = layer.route(torch.randn(4, 8))

    assert torch.allclose(route_info.leaf_weights.sum(dim=-1), torch.ones(4))
    assert torch.equal(route_info.diagnostics["active_rows_per_token"], torch.full((4,), 4))


def test_master_leaf_adds_active_rows() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        master_leaf=True,
        bias=False,
    )

    route_info = layer.route(torch.randn(4, 8))

    assert torch.equal(route_info.diagnostics["active_rows_per_token"], torch.full((4,), 4))


def test_split_route_role_exposes_result_row_metadata() -> None:
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        route_result_rows=2,
        leaf_rows=1,
        route_row_role="split_routing_output",
        route_rows_output_count="all",
        bias=False,
    )

    route_info = layer.route(torch.randn(3, 8))

    assert route_info.route_values.shape == (3, 2, 1)
    assert route_info.route_result_values.shape == (3, 2, 2)
    assert route_info.route_result_row_ids.shape == (3, 2, 2)
    assert route_info.diagnostics["route_row_role"] == "split_routing_output"
    assert route_info.diagnostics["route_output_rows_per_token"] == 4
    assert torch.equal(route_info.diagnostics["active_rows_per_token"], torch.full((3,), 5))


def test_route_diagnostics_are_consistent_with_route_tensors() -> None:
    torch.manual_seed(19)
    layer = FFFLinear(
        6,
        3,
        depth=3,
        shared_rows=2,
        route_rows=2,
        route_result_rows=2,
        leaf_rows=2,
        hard_routing=False,
        route_row_role="split_routing_output",
        route_rows_output_count=3,
        route_rows_output_fraction=0.75,
        master_leaf=True,
        fallback_leaf=True,
        region_leak=0.2,
        bias=False,
    )

    route_info = layer.route(torch.randn(2, 4, 6))
    diagnostics = route_info.diagnostics
    active_rows = diagnostics["active_rows_per_token"]

    assert diagnostics["depth"] == layer.depth
    assert diagnostics["leaves"] == layer.leaves
    assert diagnostics["internal_nodes"] == layer.internal_nodes
    assert diagnostics["stored_rows"] == layer.stored_rows
    assert diagnostics["shared_rows"] == layer.shared_rows
    assert diagnostics["route_rows"] == layer.route_rows
    assert diagnostics["route_result_rows"] == layer.route_result_rows
    assert diagnostics["route_row_role"] == "split_routing_output"
    assert diagnostics["leaf_rows"] == layer.leaf_rows
    assert diagnostics["max_visited_route_rows_per_token"] == layer.depth * layer.route_rows
    assert diagnostics["max_route_output_rows_per_token"] == layer.depth * layer.route_result_rows
    assert diagnostics["route_output_rows_per_token"] == 3
    assert active_rows.shape == (2, 4)
    assert active_rows.dtype == torch.long
    assert torch.equal(active_rows, torch.full((2, 4), 25))
    assert diagnostics["mean_active_rows_per_token"] == pytest.approx(
        float(active_rows.float().mean())
    )

    assert torch.isfinite(route_info.leaf_probs).all()
    assert torch.allclose(route_info.leaf_probs.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert torch.isfinite(route_info.leaf_weights).all()
    assert torch.all(route_info.leaf_weights >= 0.0)
    assert torch.allclose(route_info.leaf_weights.sum(dim=-1), torch.full((2, 4), 0.8))
    assert torch.isfinite(route_info.route_logits).all()
    assert torch.isfinite(route_info.route_values).all()
    assert torch.isfinite(route_info.route_result_values).all()
    assert int(route_info.route_row_ids.min()) >= 0
    assert int(route_info.route_row_ids.max()) < layer.internal_nodes * layer.route_rows
    assert int(route_info.route_result_row_ids.min()) >= 0
    assert int(route_info.route_result_row_ids.max()) < (
        layer.internal_nodes * layer.route_result_rows
    )


def test_route_info_preserves_leading_shape() -> None:
    layer = FFFLinear(8, 4, depth=2, route_rows=1, leaf_rows=1)

    route_info = layer.route(torch.randn(2, 3, 8))

    assert route_info.leaf_ids.shape == (2, 3)
    assert route_info.leaf_probs.shape == (2, 3, 4)
    assert route_info.node_ids.shape == (2, 3, 2)
    assert route_info.route_logits.shape == (2, 3, 2, 2)
    assert route_info.diagnostics["active_rows_per_token"].shape == (2, 3)
