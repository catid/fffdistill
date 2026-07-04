from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.benchmark_fff import _build_parser, _run_benchmark
from cifar_mamba_fff.models.fff_linear import FFFLinear
from cifar_mamba_fff.profile import time_cuda_callable

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast coverage")
@pytest.mark.skipif(
    torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is not supported by this device",
)
@pytest.mark.parametrize("mode", ["grouped", "naive"])
def test_cuda_bf16_autocast_matches_linear_output_dtype_without_casting_parameters(
    mode: str,
) -> None:
    torch.manual_seed(111)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        shared_rows=1,
        route_rows=1,
        leaf_rows=2,
        hard_routing=True,
        route_row_role="shared_routing_and_output",
        route_rows_output_count=1,
        bias=True,
        device="cuda",
        dtype=torch.float32,
    )
    x = torch.randn(5, 8, device="cuda", dtype=torch.float32)

    y_no_autocast = layer(x, implementation=mode)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_autocast = layer(x, implementation=mode)

    assert y_no_autocast.dtype == torch.float32
    assert y_autocast.dtype == torch.bfloat16
    assert {param.dtype for param in layer.parameters()} == {torch.float32}


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


def test_hard_grouped_path_uses_selected_leaf_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(15)
    layer = FFFLinear(
        8,
        4,
        depth=3,
        shared_rows=1,
        route_rows=2,
        leaf_rows=2,
        hard_routing=True,
        bias=False,
    )
    x = torch.randn(11, 8)
    calls = 0
    original = layer._selected_leaf_output_grouped

    def wrapped_selected_leaf(
        flat: torch.Tensor,
        route_info: object,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(flat, route_info)  # type: ignore[arg-type]

    monkeypatch.setattr(layer, "_selected_leaf_output_grouped", wrapped_selected_leaf)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)

    assert calls == 1
    assert layer.diagnostics(x)["grouped_leaf_path"] == "selected_leaf"
    assert torch.allclose(y_naive, y_grouped, atol=1e-5, rtol=1e-5)


def test_selected_leaf_grouped_skips_zero_weight_regular_leaf() -> None:
    torch.manual_seed(17)
    layer = FFFLinear(
        4,
        3,
        depth=1,
        route_rows=1,
        leaf_rows=1,
        hard_routing=True,
        region_leak=1.0,
        fallback_leaf=True,
        bias=False,
    )
    with torch.no_grad():
        layer.leaf_weight[: layer.leaves].fill_(float("inf"))
    x = torch.randn(5, 4)

    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)

    assert torch.isfinite(y_naive).all()
    assert torch.isfinite(y_grouped).all()
    torch.testing.assert_close(y_grouped, y_naive)


def test_grouped_path_keeps_all_leaves_for_soft_or_leak_to_all() -> None:
    torch.manual_seed(16)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        hard_routing=True,
        region_leak=0.1,
        fallback_leaf=False,
        bias=False,
    )
    soft_layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        hard_routing=False,
        bias=False,
    )
    x = torch.randn(6, 8)

    assert layer.diagnostics(x)["grouped_leaf_path"] == "all_leaves"
    assert soft_layer.diagnostics(x)["grouped_leaf_path"] == "all_leaves"
    assert torch.allclose(layer.forward_naive(x), layer.forward_grouped(x), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        soft_layer.forward_naive(x),
        soft_layer.forward_grouped(x),
        atol=1e-5,
        rtol=1e-5,
    )


def test_eval_region_leak_uses_selected_leaf_grouped_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(161)
    layer = FFFLinear(
        8,
        4,
        depth=2,
        route_rows=1,
        leaf_rows=2,
        hard_routing=True,
        region_leak=0.1,
        fallback_leaf=False,
        bias=False,
    )
    x = torch.randn(7, 8)

    train_diagnostics = layer.diagnostics(x)
    assert train_diagnostics["region_leak"] == pytest.approx(0.1)
    assert train_diagnostics["effective_region_leak"] == pytest.approx(0.1)
    assert train_diagnostics["region_leak_policy"] == "train_only"
    assert train_diagnostics["grouped_leaf_path"] == "all_leaves"
    assert torch.equal(
        train_diagnostics["active_rows_per_token"],
        torch.full((7,), layer.leaves * layer.leaf_rows),
    )

    layer.eval()
    selected_calls = 0
    uniform_calls = 0
    original_selected = layer._selected_leaf_output_grouped
    original_uniform = layer._uniform_regular_leaf_output_grouped

    def wrapped_selected_leaf(
        flat: torch.Tensor,
        route_info: object,
    ) -> torch.Tensor:
        nonlocal selected_calls
        selected_calls += 1
        return original_selected(flat, route_info)  # type: ignore[arg-type]

    def wrapped_uniform(flat: torch.Tensor) -> torch.Tensor:
        nonlocal uniform_calls
        uniform_calls += 1
        return original_uniform(flat)

    monkeypatch.setattr(layer, "_selected_leaf_output_grouped", wrapped_selected_leaf)
    monkeypatch.setattr(layer, "_uniform_regular_leaf_output_grouped", wrapped_uniform)

    eval_diagnostics = layer.diagnostics(x)
    y_naive = layer.forward_naive(x)
    y_grouped = layer.forward_grouped(x)

    assert y_grouped.shape == (7, 4)
    assert selected_calls == 1
    assert uniform_calls == 0
    assert eval_diagnostics["region_leak"] == pytest.approx(0.1)
    assert eval_diagnostics["effective_region_leak"] == pytest.approx(0.0)
    assert eval_diagnostics["region_leak_policy"] == "train_only"
    assert eval_diagnostics["grouped_leaf_path"] == "selected_leaf"
    assert torch.equal(
        eval_diagnostics["active_rows_per_token"],
        torch.full((7,), layer.leaf_rows),
    )
    assert torch.allclose(y_naive, y_grouped, atol=1e-5, rtol=1e-5)


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
    assert legacy.route_output_rows_per_token == explicit.route_output_rows_per_token == 2


def test_time_cuda_callable_exposes_metadata_on_cpu() -> None:
    result = time_cuda_callable(lambda: None, iterations=2, items=4, allow_cpu=True)
    metadata = result.as_metadata(name="noop", tokens=4)

    assert metadata["name"] == "noop"
    assert metadata["tokens"] == 4
    assert metadata["iterations"] == 2
    assert metadata["seconds_per_iteration"] > 0.0
    assert metadata["items_per_second"] > 0.0


def test_time_cuda_callable_synchronizes_requested_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_devices: list[torch.device | None] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: seen_devices.append(device),
    )

    time_cuda_callable(lambda: None, iterations=1, items=1, device="cuda:1")

    assert seen_devices == [torch.device("cuda:1"), torch.device("cuda:1")]


def test_benchmark_quick_smoke_reports_grouped_throughput_metadata() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--quick-smoke", "true", "--device", "cpu"])

    rows = _run_benchmark(args)
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"dense", "fff_grouped", "fff_naive"}
    assert by_name["fff_grouped"]["tokens"] == 16
    assert by_name["fff_grouped"]["tokens_per_second"] > 0.0
    assert by_name["fff_grouped"]["grouped_leaf_path"] == "selected_leaf"
    assert by_name["fff_grouped"]["grouped_naive_max_abs_diff"] < 1e-5


def test_benchmark_reports_route_output_ablation_controls() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--quick-smoke",
            "true",
            "--device",
            "cpu",
            "--route-row-role",
            "shared_routing_and_output",
            "--route-rows-output-count",
            "all",
            "--route-rows-output-fraction",
            "0.5",
        ]
    )

    rows = _run_benchmark(args)
    grouped = {row["name"]: row for row in rows}["fff_grouped"]

    assert grouped["route_rows_contribute"] is True
    assert grouped["route_output_contributes"] is True
    assert grouped["route_row_role"] == "shared_routing_and_output"
    assert grouped["route_rows_output_count"] == "all"
    assert grouped["route_rows_output_fraction"] == pytest.approx(0.5)
    assert grouped["route_output_rows_per_token"] > 0


def test_benchmark_skips_naive_unless_requested() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--in-features",
            "8",
            "--out-features",
            "8",
            "--iterations",
            "1",
            "--warmup",
            "0",
        ]
    )

    rows = _run_benchmark(args)
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"dense", "fff_grouped"}
    assert "grouped_naive_max_abs_diff" not in by_name["fff_grouped"]
