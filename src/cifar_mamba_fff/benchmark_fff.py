from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .models.fff_linear import FFFLinear, RouteRowRole
from .profile import TimingResult, time_cuda_callable
from .utils import bool_arg

BenchmarkKind = Literal[
    "dense",
    "fff_grouped",
    "fff_grouped_compiled",
    "fff_naive",
    "dense_backward",
    "fff_grouped_backward",
    "fff_grouped_compiled_backward",
    "fff_naive_backward",
    "fff_route_setup",
    "fff_selected_leaf_kernel",
    "fff_all_leaf_kernel",
]


@dataclass(frozen=True)
class BenchmarkCase:
    name: BenchmarkKind
    forward: Callable[[], torch.Tensor]
    module: nn.Module


@dataclass(frozen=True)
class ProfileShape:
    name: str
    batch_size: int
    in_features: int
    out_features: int
    depth: int
    shared_rows: int
    route_rows: int
    route_result_rows: int
    leaf_rows: int
    route_row_role: RouteRowRole
    route_rows_output_count: int | Literal["all"] | None
    region_leak: float
    hard_routing: bool
    eval_mode: bool


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
    parser.add_argument(
        "--include-compiled",
        type=bool_arg,
        default=False,
        help="Also time torch.compile(FFFLinear.forward_grouped) when torch.compile is available.",
    )
    parser.add_argument(
        "--measure-components",
        type=bool_arg,
        default=False,
        help="Emit bounded component timings for route setup and regular leaf kernels.",
    )
    parser.add_argument(
        "--profile-report",
        choices=("none", "teacher-linear"),
        default="none",
        help="Run a reproducible multi-shape profiling matrix instead of one benchmark case.",
    )
    parser.add_argument(
        "--report-include-naive",
        type=bool_arg,
        default=True,
        help="Include the slow Python-loop naive path in profile-report mode.",
    )
    parser.add_argument("--report-markdown-out", type=Path, default=None)
    parser.add_argument("--json", type=bool_arg, default=True)
    return parser


def _route_rows_output_count_arg(value: str | int | None) -> int | Literal["all"] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise argparse.ArgumentTypeError("route rows output count must be non-negative")
        return value
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


def _measure_timing(
    *,
    fn: Callable[[], object],
    iterations: int,
    warmup: int,
    tokens: int,
    device: torch.device,
    grad_enabled: bool = False,
) -> TimingResult:
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
        _sync_if_cuda(device)
    return result


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


def _compile_grouped_forward(
    fff: FFFLinear,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return None
    return compile_fn(fff.forward_grouped, fullgraph=False)


def _hot_path_evidence(name: str) -> str:
    if name.startswith("fff_naive"):
        return "forward_naive loops over tokens and leaves for correctness comparison only"
    if name.startswith("fff_grouped"):
        return "forward_grouped routes the batch, then uses tensor gather/bmm/einsum kernels"
    if name.startswith("dense"):
        return "nn.Linear batched dense matmul"
    return "bounded component timing"


def _sort_bucket_strategy(row: dict[str, Any]) -> str:
    name = str(row["name"])
    if name.startswith("fff_grouped"):
        grouped_leaf_path = str(row.get("grouped_leaf_path", "unknown"))
        if grouped_leaf_path == "selected_leaf":
            return "not_used_selected_leaf_gather_bmm"
        if grouped_leaf_path == "all_leaves":
            return "not_used_all_leaf_einsum"
        return "not_used"
    if name.startswith("fff_naive"):
        return "not_applicable_python_loop"
    if name.startswith("dense"):
        return "not_applicable_dense"
    return "not_applicable_component"


def _annotate_relative_metrics(rows: list[dict[str, Any]]) -> None:
    dense_tokens_per_second_by_phase: dict[str, float] = {}
    tokens_per_second_by_name: dict[str, float] = {}
    for row in rows:
        name = str(row["name"])
        phase = str(row["phase"])
        tokens_per_second = float(row["tokens_per_second"])
        tokens_per_second_by_name[name] = tokens_per_second
        if name in {"dense", "dense_backward"}:
            dense_tokens_per_second_by_phase[phase] = tokens_per_second

    forward_naive_tokens = tokens_per_second_by_name.get("fff_naive")
    backward_naive_tokens = tokens_per_second_by_name.get("fff_naive_backward")

    for row in rows:
        name = str(row["name"])
        phase = str(row["phase"])
        tokens_per_second = float(row["tokens_per_second"])
        dense_tokens_per_second = dense_tokens_per_second_by_phase.get(phase)
        row["dense_tokens_per_second"] = dense_tokens_per_second
        row["tokens_per_second_fraction_of_dense"] = (
            tokens_per_second / dense_tokens_per_second
            if dense_tokens_per_second and dense_tokens_per_second > 0.0
            else None
        )
        row["dense_slowdown"] = (
            dense_tokens_per_second / tokens_per_second
            if dense_tokens_per_second and tokens_per_second > 0.0
            else None
        )
        row["python_per_token_hot_path"] = name.startswith("fff_naive")
        row["hot_path_evidence"] = _hot_path_evidence(name)
        row["sort_bucket_strategy"] = _sort_bucket_strategy(row)
        row["sort_bucket_seconds_per_iteration"] = 0.0 if name.startswith("fff_grouped") else None
        row["sort_bucket_overhead_fraction"] = 0.0 if name.startswith("fff_grouped") else None

        if (
            name.startswith("fff_grouped")
            and phase == "forward"
            and forward_naive_tokens
            and forward_naive_tokens > 0.0
        ):
            row["grouped_vs_naive_speedup"] = tokens_per_second / forward_naive_tokens
        elif (
            name.startswith("fff_grouped")
            and phase == "forward_backward"
            and backward_naive_tokens
            and backward_naive_tokens > 0.0
        ):
            row["grouped_vs_naive_speedup"] = tokens_per_second / backward_naive_tokens
        else:
            row["grouped_vs_naive_speedup"] = None


def _append_component_rows(
    *,
    rows: list[dict[str, Any]],
    fff: FFFLinear,
    x: torch.Tensor,
    route_metadata: dict[str, Any],
    iterations: int,
    warmup: int,
    tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    flat, _ = fff._flatten_input(x)
    with torch.no_grad():
        route_info = fff._route_flat(flat, hard=fff.config.hard_routing)
    route_timing = _measure_timing(
        fn=lambda: fff._route_flat(flat, hard=fff.config.hard_routing),
        iterations=iterations,
        warmup=warmup,
        tokens=tokens,
        device=device,
    )
    rows.append(
        route_timing.as_metadata(
            name="fff_route_setup",
            phase="component_forward",
            component_parent="fff_grouped",
            device=str(device),
            dtype=str(dtype).removeprefix("torch."),
            tokens=tokens,
            tokens_per_second=route_timing.items_per_second,
            grad_enabled=False,
            **route_metadata,
        )
    )

    if fff._can_use_selected_leaf_grouped_path():
        leaf_name: BenchmarkKind = "fff_selected_leaf_kernel"

        def leaf_fn() -> torch.Tensor:
            return fff._selected_leaf_output_grouped(flat, route_info)

    else:
        leaf_name = "fff_all_leaf_kernel"

        def leaf_fn() -> torch.Tensor:
            return fff._regular_leaf_output_grouped(flat, route_info)

    leaf_timing = _measure_timing(
        fn=leaf_fn,
        iterations=iterations,
        warmup=warmup,
        tokens=tokens,
        device=device,
    )
    rows.append(
        leaf_timing.as_metadata(
            name=leaf_name,
            phase="component_forward",
            component_parent="fff_grouped",
            device=str(device),
            dtype=str(dtype).removeprefix("torch."),
            tokens=tokens,
            tokens_per_second=leaf_timing.items_per_second,
            grad_enabled=False,
            **route_metadata,
        )
    )


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
    if args.include_compiled:
        compiled_forward = _compile_grouped_forward(fff)
        if compiled_forward is not None:
            cases = (
                *cases,
                BenchmarkCase("fff_grouped_compiled", lambda: compiled_forward(x), fff),
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

    if args.measure_components:
        _append_component_rows(
            rows=rows,
            fff=fff,
            x=x,
            route_metadata=route_metadata,
            iterations=args.iterations,
            warmup=args.warmup,
            tokens=args.batch_size,
            device=device,
            dtype=dtype,
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
    _annotate_relative_metrics(rows)
    return rows


def _teacher_linear_profile_shapes() -> tuple[ProfileShape, ...]:
    return (
        ProfileShape(
            name="mamba3_d_model_square_256",
            batch_size=1024,
            in_features=256,
            out_features=256,
            depth=5,
            shared_rows=38,
            route_rows=1,
            route_result_rows=2,
            leaf_rows=4,
            route_row_role="split_routing_output",
            route_rows_output_count="all",
            region_leak=0.01,
            hard_routing=True,
            eval_mode=True,
        ),
        ProfileShape(
            name="mamba3_expand_in_256x512",
            batch_size=1024,
            in_features=256,
            out_features=512,
            depth=5,
            shared_rows=38,
            route_rows=1,
            route_result_rows=2,
            leaf_rows=4,
            route_row_role="split_routing_output",
            route_rows_output_count="all",
            region_leak=0.01,
            hard_routing=True,
            eval_mode=True,
        ),
        ProfileShape(
            name="mamba3_expand_out_512x256",
            batch_size=1024,
            in_features=512,
            out_features=256,
            depth=5,
            shared_rows=38,
            route_rows=1,
            route_result_rows=2,
            leaf_rows=4,
            route_row_role="split_routing_output",
            route_rows_output_count="all",
            region_leak=0.01,
            hard_routing=True,
            eval_mode=True,
        ),
    )


def _profile_shapes(profile_report: str) -> tuple[ProfileShape, ...]:
    if profile_report == "teacher-linear":
        return _teacher_linear_profile_shapes()
    raise ValueError(f"unsupported profile report: {profile_report}")


def _args_for_profile_shape(
    args: argparse.Namespace,
    shape: ProfileShape,
) -> argparse.Namespace:
    shape_args = copy.copy(args)
    shape_args.batch_size = shape.batch_size
    shape_args.in_features = shape.in_features
    shape_args.out_features = shape.out_features
    shape_args.depth = shape.depth
    shape_args.shared_rows = shape.shared_rows
    shape_args.route_rows = shape.route_rows
    shape_args.route_result_rows = shape.route_result_rows
    shape_args.leaf_rows = shape.leaf_rows
    shape_args.route_row_role = shape.route_row_role
    shape_args.route_rows_output_count = shape.route_rows_output_count
    shape_args.route_rows_output_fraction = None
    shape_args.region_leak = shape.region_leak
    shape_args.hard_routing = shape.hard_routing
    shape_args.eval_mode = shape.eval_mode
    shape_args.include_naive = bool(args.report_include_naive)
    shape_args.skip_naive = not bool(args.report_include_naive)
    shape_args.measure_components = True
    shape_args.profile_report = "none"
    return shape_args


def _run_profile_report(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in _profile_shapes(args.profile_report):
        shape_args = _args_for_profile_shape(args, shape)
        shape_rows = _run_benchmark(shape_args)
        for row in shape_rows:
            row["profile_report"] = args.profile_report
            row["profile_case"] = shape.name
            row["requested_batch_size"] = shape.batch_size
            row["requested_in_features"] = shape.in_features
            row["requested_out_features"] = shape.out_features
        rows.extend(shape_rows)
    return rows


def _format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value >= 1_000_000.0:
            return f"{value / 1_000_000.0:.2f}M"
        if value >= 1_000.0:
            return f"{value / 1_000.0:.2f}K"
        return f"{value:.3f}"
    return str(value)


def _render_markdown_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Grouped FFF Throughput Profile",
        "",
        "Generated by `python -m cifar_mamba_fff.benchmark_fff "
        "--profile-report teacher-linear`.",
        "",
        "This report is a reproducible command/output structure. GPU utilization still "
        "requires wrapping the command with `nvidia-smi dmon`, DCGM, or an equivalent "
        "host sampler because this microbenchmark records PyTorch timing rows only.",
        "",
        "## Rows",
        "",
        "| case | path | phase | tokens/s | dense slowdown | Python per-token | sort/bucket | grouped path |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {name} | {phase} | {tps} | {slowdown} | {python_hot} | {sort} | {path} |".format(
                case=row.get("profile_case", ""),
                name=row["name"],
                phase=row["phase"],
                tps=_format_number(row["tokens_per_second"]),
                slowdown=_format_number(row.get("dense_slowdown")),
                python_hot=row["python_per_token_hot_path"],
                sort=row["sort_bucket_strategy"],
                path=row.get("grouped_leaf_path", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- `sort_bucket_seconds_per_iteration` is `0.0` for grouped rows because the "
            "current grouped implementation does not sort or bucket tokens by leaf; it "
            "uses selected-leaf tensor gather plus `bmm` for hard routing, and all-leaf "
            "`einsum` for soft/leaky training paths.",
            "- `fff_naive` is included only as a correctness and overhead reference. It "
            "has an intentional Python per-token loop and is not a deployable hot path.",
            "- `fff_grouped_compiled` appears only when `--include-compiled true` is used "
            "and `torch.compile` succeeds on the active PyTorch/device stack.",
            f"- Device requested: `{args.device}`. Dtype requested: `{args.dtype}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    rows = _run_profile_report(args) if args.profile_report != "none" else _run_benchmark(args)
    if args.report_markdown_out is not None:
        args.report_markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_markdown_out.write_text(_render_markdown_report(rows, args), encoding="utf-8")
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
