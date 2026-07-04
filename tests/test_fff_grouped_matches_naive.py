from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.models.fff_linear import FFFLinear

BF16_CLOSE_TOL = float(torch.finfo(torch.bfloat16).eps)

CUDA_BF16_ROUTE_ROLE_CASES = [
    pytest.param(
        {
            "depth": 2,
            "shared_rows": 1,
            "route_rows": 1,
            "leaf_rows": 2,
            "route_row_role": "routing_only",
            "bias": True,
        },
        id="routing-only",
    ),
    pytest.param(
        {
            "depth": 3,
            "shared_rows": 0,
            "route_rows": 2,
            "leaf_rows": 1,
            "route_row_role": "shared_routing_and_output",
            "route_rows_output_count": 2,
            "bias": False,
        },
        id="shared-routing-and-output",
    ),
    pytest.param(
        {
            "depth": 3,
            "shared_rows": 1,
            "route_rows": 1,
            "route_result_rows": 2,
            "leaf_rows": 2,
            "route_row_role": "split_routing_output",
            "route_rows_output_count": "all",
            "fallback_leaf": True,
            "region_leak": 0.05,
            "bias": False,
        },
        id="split-routing-output",
    ),
]


@pytest.mark.parametrize("hard_routing", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "depth": 2,
            "shared_rows": 1,
            "route_rows": 1,
            "leaf_rows": 2,
            "route_row_role": "routing_only",
            "bias": True,
        },
        {
            "depth": 3,
            "shared_rows": 0,
            "route_rows": 2,
            "leaf_rows": 1,
            "route_row_role": "shared_routing_and_output",
            "route_rows_output_count": 2,
            "bias": False,
        },
        {
            "depth": 2,
            "shared_rows": 2,
            "route_rows": 2,
            "leaf_rows": 4,
            "route_row_role": "shared_routing_and_output",
            "route_rows_output_count": "all",
            "route_rows_output_fraction": 0.5,
            "master_leaf": True,
            "bias": True,
        },
        {
            "depth": 3,
            "shared_rows": 1,
            "route_rows": 1,
            "route_result_rows": 2,
            "leaf_rows": 2,
            "route_row_role": "split_routing_output",
            "route_rows_output_count": "all",
            "fallback_leaf": True,
            "region_leak": 0.05,
            "bias": False,
        },
        {
            "depth": 2,
            "shared_rows": 0,
            "route_rows": 2,
            "route_result_rows": 1,
            "leaf_rows": 1,
            "route_row_role": "split_routing_output",
            "route_rows_output_count": 1,
            "route_rows_output_fraction": 1.0,
            "bias": True,
        },
    ],
)
def test_grouped_matches_naive_across_route_roles(
    hard_routing: bool,
    kwargs: dict[str, object],
) -> None:
    torch.manual_seed(10)
    layer = FFFLinear(8, 4, hard_routing=hard_routing, **kwargs)
    x = torch.randn(7, 8)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)

    assert torch.allclose(y_naive, y_grouped, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for BF16 FFF grouped-vs-naive coverage",
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is not supported by this device",
)
@pytest.mark.parametrize("hard_routing", [True, False])
@pytest.mark.parametrize("kwargs", CUDA_BF16_ROUTE_ROLE_CASES)
def test_cuda_bf16_grouped_matches_naive_across_route_roles(
    hard_routing: bool,
    kwargs: dict[str, object],
) -> None:
    torch.manual_seed(110)
    layer = FFFLinear(
        8,
        4,
        hard_routing=hard_routing,
        device="cuda",
        dtype=torch.bfloat16,
        **kwargs,
    )
    x = torch.randn(5, 8, device="cuda", dtype=torch.bfloat16)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)

    assert y_naive.dtype == torch.bfloat16
    assert y_grouped.dtype == torch.bfloat16
    if layer.route_row_role == "routing_only":
        assert layer.route_output_rows_per_token == 0
    else:
        assert layer.route_output_rows_per_token > 0
    torch.testing.assert_close(
        y_grouped,
        y_naive,
        atol=BF16_CLOSE_TOL,
        rtol=BF16_CLOSE_TOL,
    )


def test_grouped_matches_naive_for_high_rank_input() -> None:
    torch.manual_seed(11)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=1,
        route_rows=2,
        route_result_rows=2,
        leaf_rows=2,
        route_row_role="split_routing_output",
        route_rows_output_count=3,
        hard_routing=False,
    )
    x = torch.randn(2, 3, 5, 8)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)
    y_default = layer(x)

    assert y_naive.shape == (2, 3, 5, 4)
    assert torch.allclose(y_naive, y_grouped, atol=1e-5, rtol=1e-5)
    assert torch.allclose(y_default, y_grouped, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "depth": 2,
            "route_rows": 1,
            "leaf_rows": 1,
            "route_row_role": "routing_only",
        },
        {
            "depth": 3,
            "route_rows": 2,
            "leaf_rows": 2,
            "route_row_role": "shared_routing_and_output",
            "route_rows_output_count": "all",
        },
        {
            "depth": 2,
            "route_rows": 2,
            "route_result_rows": 2,
            "leaf_rows": 4,
            "route_row_role": "split_routing_output",
            "route_rows_output_count": "all",
        },
    ],
)
@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)])
def test_grouped_matches_naive_for_empty_leading_dimensions(
    kwargs: dict[str, object],
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(14)
    layer = FFFLinear(8, 4, hard_routing=False, **kwargs)
    x = torch.randn(*shape)
    expected_shape = (*shape[:-1], 4)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)
    y_default = layer(x)

    assert y_naive.shape == expected_shape
    assert y_grouped.shape == expected_shape
    assert y_default.shape == expected_shape
    assert torch.allclose(y_naive, y_grouped, atol=1e-5, rtol=1e-5)
    assert torch.allclose(y_default, y_grouped, atol=1e-5, rtol=1e-5)


def test_count_zero_disables_output_rows_even_for_output_role() -> None:
    torch.manual_seed(12)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=1,
        route_row_role="shared_routing_and_output",
        route_rows_output_count=0,
        bias=False,
    )
    x = torch.randn(5, 8)

    assert layer.route_output_rows_per_token == 0
    assert torch.allclose(layer.forward_naive(x), layer.forward_grouped(x), atol=1e-5, rtol=1e-5)


def test_legacy_route_rows_contribute_matches_shared_role_semantics() -> None:
    torch.manual_seed(13)
    legacy = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=1,
        route_rows_contribute=True,
        route_rows_output_count=1,
        bias=False,
    )
    explicit = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=1,
        route_row_role="shared_routing_and_output",
        route_rows_output_count=1,
        bias=False,
    )

    assert legacy.route_row_role == "shared_routing_and_output"
    assert explicit.route_row_role == "shared_routing_and_output"
    assert legacy.route_output_rows_per_token == explicit.route_output_rows_per_token == 1
