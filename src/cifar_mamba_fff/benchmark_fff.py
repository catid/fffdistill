from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .models.fff_linear import FFFLinear, RouteRowRole
from .profile import TimingResult, time_cuda_callable
from .utils import bool_arg

BenchmarkKind = Literal[
    "dense",
    "fff_grouped",
    "fff_naive",
    "dense_backward",
    "fff_grouped_backward",
    "fff_naive_backward",
]


@dataclass(frozen=True)
class BenchmarkCase:
    name: BenchmarkKind
    forward: Callable[[], torch.Tensor]
    module: nn.Module


def _dtype_arg(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value!r}")


def _device_arg(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded FFFLinear microbenchmark")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--device", type=_device_arg, default=_device_arg("auto"))
    parser.add_argument("--dtype", type=_dtype_arg, default=torch.float32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=192)
    parser.add_argument("--out-features", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--shared-rows", type=int, default=2)
    parser.add_argument("--route-rows", type=int, default=2)
    parser.add_argument("--route-result-rows", type=int, default=2)
    parser.add_argument("--leaf-rows", type=int, default=2)
    parser.add_argument(
        "--route-row-role",
        choices=("routing_only", "shared_routing_and_output", "split_routing_output"),
        default="routing_only",
    )
    parser.add_argument("--route-rows-output-count", default=None)
    parser.add_argument("--route-rows-output-fraction", type=float, default=None)
    parser.add_argument("--region-leak", type=float, default=0.0)
    parser.add_argument("--hard-routing", type=bool_arg, default=True)
    parser.add_argument("--eval-mode", type=bool_arg, default=False)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--include-naive",
        type=bool_arg,
        default=False,
        help="Run the slow naive FFF path and emit grouped-vs-naive diff metadata.",
    )
    parser.add_argument(
        "--skip-naive",
        type=bool_arg,
        default=False,
        help="Skip the slow naive FFF path and omit grouped-vs-naive diff metadata.",
    )
    parser.add_argument(
        "--measure-backward",
        type=bool_arg,
        default=False,
        help="Also time one forward+backward optimizer-free step for each selected case.",
    )
    parser.add_argument("--json", type=bool_arg, default=True)
    return parser


def _route_rows_output_count_arg(value: str | None) -> int | Literal["all"] | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", ""}:
        return None
    if normalized == "all":
        return "all"
    parsed = int(normalized)
    if parsed < 0:
        raise argparse.ArgumentTypeError("route rows output count must be non-negative")
    return parsed


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _should_run_naive(args: argparse.Namespace) -> bool:
    if getattr(args, "skip_naive", False):
        return False
    return bool(getattr(args, "include_naive", False))


def _measure(
    *,
    fn: Callable[[], torch.Tensor],
    iterations: int,
    warmup: int,
    tokens: int,
    device: torch.device,
    grad_enabled: bool = False,
) -> tuple[torch.Tensor, TimingResult]:
    grad_context = torch.enable_grad if grad_enabled else torch.no_grad
    with grad_context():
        for _ in range(warmup):
            fn()
        _sync_if_cuda(device)
        result = time_cuda_callable(
            fn,
            iterations=iterations,
            items=tokens,
            allow_cpu=device.type == "cpu",
            device=device,
        )
        output = fn()
        _sync_if_cuda(device)
    return output, result


def _make_backward_step(
    *,
    forward: Callable[[], torch.Tensor],
    module: nn.Module,
    x: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def step() -> torch.Tensor:
        module.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        output = forward()
        loss = output.float().square().mean()
        loss.backward()
        return loss.detach()

    return step


def _run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    if args.quick_smoke:
        args.batch_size = min(args.batch_size, 16)
        args.iterations = 1
        args.warmup = 0
        args.in_features = min(args.in_features, 32)
        args.out_features = min(args.out_features, 32)
        args.depth = min(args.depth, 2)
        args.include_naive = True

    device = args.device
    dtype = args.dtype
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    route_row_role: RouteRowRole = args.route_row_role
    route_rows_output_count = _route_rows_output_count_arg(args.route_rows_output_count)
    if route_row_role == "split_routing_output" and args.route_result_rows <= 0:
        raise ValueError("route-result-rows must be positive for split_routing_output")

    torch.manual_seed(123)
    x = torch.randn(args.batch_size, args.in_features, device=device, dtype=dtype)
    if args.measure_backward:
        x.requires_grad_(True)
    dense = nn.Linear(args.in_features, args.out_features, device=device, dtype=dtype)
    fff = FFFLinear(
        args.in_features,
        args.out_features,
        depth=args.depth,
        shared_rows=args.shared_rows,
        route_rows=args.route_rows,
        route_result_rows=(
            args.route_result_rows if route_row_role == "split_routing_output" else 0
        ),
        leaf_rows=args.leaf_rows,
        route_row_role=route_row_role,
        route_rows_output_count=route_rows_output_count,
        route_rows_output_fraction=args.route_rows_output_fraction,
        region_leak=args.region_leak,
        hard_routing=args.hard_routing,
        device=device,
        dtype=dtype,
    )
    if args.eval_mode:
        dense.eval()
        fff.eval()

    diagnostics = fff.diagnostics(x)
    route_metadata = {
        "active_rows_per_token": diagnostics["mean_active_rows_per_token"],
        "stored_rows": diagnostics["stored_rows"],
        "effective_stored_rows": diagnostics["effective_stored_rows"],
        "effective_trainable_rows": diagnostics["effective_trainable_rows"],
        "route_rows_contribute": diagnostics["route_rows_contribute"],
        "route_output_contributes": diagnostics["route_output_contributes"],
        "route_row_role": diagnostics["route_row_role"],
        "route_result_rows": diagnostics["route_result_rows"],
        "route_rows_output_count": diagnostics["route_rows_output_count"],
        "route_rows_output_fraction": diagnostics["route_rows_output_fraction"],
        "region_leak": diagnostics["region_leak"],
        "effective_region_leak": diagnostics["effective_region_leak"],
        "region_leak_policy": diagnostics["region_leak_policy"],
        "max_visited_route_rows_per_token": diagnostics["max_visited_route_rows_per_token"],
        "max_route_output_rows_per_token": diagnostics["max_route_output_rows_per_token"],
        "stored_route_output_rows": diagnostics["stored_route_output_rows"],
        "effective_route_output_rows": diagnostics["effective_route_output_rows"],
        "unused_route_output_rows": diagnostics["unused_route_output_rows"],
        "unused_stored_route_output_rows": diagnostics["unused_stored_route_output_rows"],
        "route_output_rows_per_node": diagnostics["route_output_rows_per_node"],
        "route_output_rows_per_token": diagnostics["route_output_rows_per_token"],
        "grouped_leaf_path": diagnostics["grouped_leaf_path"],
    }
    cases: tuple[BenchmarkCase, ...] = (
        BenchmarkCase("dense", lambda: dense(x), dense),
        BenchmarkCase("fff_grouped", lambda: fff.forward_grouped(x), fff),
    )
    if _should_run_naive(args):
        cases = (*cases, BenchmarkCase("fff_naive", lambda: fff.forward_naive(x), fff))

    outputs: dict[BenchmarkKind, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        output, timing = _measure(
            fn=case.forward,
            iterations=args.iterations,
            warmup=args.warmup,
            tokens=args.batch_size,
            device=device,
        )
        outputs[case.name] = output.detach()
        rows.append(
            timing.as_metadata(
                name=case.name,
                phase="forward",
                device=str(device),
                dtype=str(dtype).removeprefix("torch."),
                tokens=args.batch_size,
                tokens_per_second=timing.items_per_second,
                grad_enabled=False,
                **route_metadata,
            )
        )

    if "fff_naive" in outputs:
        max_abs_diff = (outputs["fff_grouped"] - outputs["fff_naive"]).abs().max().item()
        for row in rows:
            row["grouped_naive_max_abs_diff"] = max_abs_diff
    if args.measure_backward:
        for case in cases:
            backward_name = f"{case.name}_backward"
            _, timing = _measure(
                fn=_make_backward_step(forward=case.forward, module=case.module, x=x),
                iterations=args.iterations,
                warmup=args.warmup,
                tokens=args.batch_size,
                device=device,
                grad_enabled=True,
            )
            rows.append(
                timing.as_metadata(
                    name=backward_name,
                    phase="forward_backward",
                    device=str(device),
                    dtype=str(dtype).removeprefix("torch."),
                    tokens=args.batch_size,
                    tokens_per_second=timing.items_per_second,
                    grad_enabled=True,
                    **route_metadata,
                )
            )
    return rows


def main() -> int:
    args = _build_parser().parse_args()
    rows = _run_benchmark(args)
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        for row in rows:
            print(
                "{name}: {tokens_per_second:.2f} tokens/s "
                "({seconds_per_iteration:.6f} s/iter, active_rows={active_rows_per_token})".format(
                    **row
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
