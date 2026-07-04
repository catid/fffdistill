from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn

from .models.fff_linear import FFFLinear, RouteRowRole
from .profile import TimingResult, time_cuda_callable
from .utils import bool_arg

BenchmarkKind = Literal["dense", "fff_grouped", "fff_naive"]


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
    parser.add_argument("--hard-routing", type=bool_arg, default=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--include-naive", type=bool_arg, default=False)
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


def _measure(
    *,
    fn: Callable[[], torch.Tensor],
    iterations: int,
    warmup: int,
    tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, TimingResult]:
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
        hard_routing=args.hard_routing,
        device=device,
        dtype=dtype,
    )

    diagnostics = fff.diagnostics(x)
    route_metadata = {
        "active_rows_per_token": diagnostics["mean_active_rows_per_token"],
        "stored_rows": diagnostics["stored_rows"],
        "route_row_role": diagnostics["route_row_role"],
        "route_output_rows_per_token": diagnostics["route_output_rows_per_token"],
        "grouped_leaf_path": diagnostics["grouped_leaf_path"],
    }
    cases: tuple[tuple[BenchmarkKind, Callable[[], torch.Tensor]], ...] = (
        ("dense", lambda: dense(x)),
        ("fff_grouped", lambda: fff.forward_grouped(x)),
    )
    if args.include_naive:
        cases = (*cases, ("fff_naive", lambda: fff.forward_naive(x)))

    outputs: dict[BenchmarkKind, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for name, fn in cases:
        output, timing = _measure(
            fn=fn,
            iterations=args.iterations,
            warmup=args.warmup,
            tokens=args.batch_size,
            device=device,
        )
        outputs[name] = output.detach()
        rows.append(
            timing.as_metadata(
                name=name,
                device=str(device),
                dtype=str(dtype).removeprefix("torch."),
                tokens=args.batch_size,
                tokens_per_second=timing.items_per_second,
                **route_metadata,
            )
        )

    if "fff_naive" in outputs:
        max_abs_diff = (outputs["fff_grouped"] - outputs["fff_naive"]).abs().max().item()
        for row in rows:
            row["grouped_naive_max_abs_diff"] = max_abs_diff
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
