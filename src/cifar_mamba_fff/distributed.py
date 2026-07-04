from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .utils import bool_arg, git_commit

Strategy = Literal["auto", "single", "ddp"]


def distributed_available() -> bool:
    return dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def current_context() -> DistributedContext:
    return DistributedContext(
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=_env_int("WORLD_SIZE", 1),
        distributed=distributed_available(),
    )


def _select_backend(backend: str, device: torch.device) -> str:
    if backend != "auto":
        return backend
    return "nccl" if device.type == "cuda" else "gloo"


def init_distributed_if_needed(
    backend: str = "nccl",
    *,
    device: torch.device | str | None = None,
) -> bool:
    if not distributed_available():
        return False
    if dist.is_initialized():
        return True
    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    backend = _select_backend(backend, resolved_device)
    dist.init_process_group(backend=backend)
    return True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class TinyCifarBatchModel(nn.Module):
    def __init__(self, *, hidden_size: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device_arg(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _dtype_arg(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic CIFAR-shaped DDP smoke benchmark")
    parser.add_argument("--strategy", choices=("auto", "single", "ddp"), default="auto")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--device", type=_device_arg, default=_device_arg("auto"))
    parser.add_argument("--dtype", type=_dtype_arg, default=torch.float32)
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None, help="Optional JSONL path for rank-0 results.")
    parser.add_argument("--json", type=bool_arg, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "hidden_size", "classes", "iterations"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")


def _apply_quick_smoke(args: argparse.Namespace) -> None:
    if not args.quick_smoke:
        return
    args.batch_size = min(args.batch_size, 8)
    args.hidden_size = min(args.hidden_size, 32)
    args.iterations = 1
    args.warmup = 0


def _resolve_strategy(strategy: Strategy, context: DistributedContext) -> Literal["single", "ddp"]:
    if strategy == "auto":
        return "ddp" if context.distributed else "single"
    return strategy


def _resolve_device(
    requested_device: torch.device,
    *,
    strategy: Literal["single", "ddp"],
    context: DistributedContext,
) -> torch.device:
    device = requested_device
    if device.type == "cuda" and strategy == "ddp":
        if context.world_size > 1 and device.index is not None:
            raise ValueError(
                "Use --device auto or --device cuda for multi-process DDP; "
                "select physical GPUs with CUDA_VISIBLE_DEVICES."
            )
        if device.index is None:
            device = torch.device("cuda", context.local_rank)
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_benchmark_totals(
    *,
    elapsed_seconds: float,
    local_samples: int,
    device: torch.device,
    distributed: bool,
) -> tuple[float, int]:
    if not distributed:
        return elapsed_seconds, local_samples
    elapsed = torch.tensor(elapsed_seconds, device=device, dtype=torch.float64)
    samples = torch.tensor(local_samples, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(samples, op=dist.ReduceOp.SUM)
    return float(elapsed.item()), int(samples.item())


def _write_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any] | None:
    _validate_args(args)
    _apply_quick_smoke(args)
    context = current_context()
    strategy = _resolve_strategy(args.strategy, context)
    if strategy == "ddp" and not context.distributed:
        raise RuntimeError("DDP strategy requires torchrun-style RANK and WORLD_SIZE env vars")

    device = _resolve_device(args.device, strategy=strategy, context=context)
    dtype = args.dtype
    if device.type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32

    backend = _select_backend(args.backend, device)
    if strategy == "ddp":
        init_distributed_if_needed(args.backend, device=device)

    torch.manual_seed(args.seed + context.rank)
    model: nn.Module = TinyCifarBatchModel(hidden_size=args.hidden_size, classes=args.classes).to(
        device=device,
        dtype=dtype,
    )
    if strategy == "ddp":
        device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(args.batch_size, 3, 32, 32, device=device, dtype=dtype)
    y = torch.randint(0, args.classes, (args.batch_size,), device=device)

    start = 0.0
    total_steps = args.warmup + args.iterations
    for step in range(total_steps):
        if step == args.warmup:
            _sync_if_cuda(device)
            start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        optimizer.step()
    _sync_if_cuda(device)
    elapsed_seconds = time.perf_counter() - start

    world_size = context.world_size if strategy == "ddp" else 1
    local_samples = args.batch_size * args.iterations
    elapsed_seconds, global_samples = _reduce_benchmark_totals(
        elapsed_seconds=elapsed_seconds,
        local_samples=local_samples,
        device=device,
        distributed=strategy == "ddp",
    )
    record = {
        "event": "distributed_synthetic_cifar_benchmark",
        "strategy": strategy,
        "backend": backend if strategy == "ddp" else None,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "quick_smoke": args.quick_smoke,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "hidden_size": args.hidden_size,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "samples": global_samples,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": global_samples / elapsed_seconds if elapsed_seconds > 0 else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_commit(),
    }
    if context.rank == 0:
        if args.output is not None:
            _write_jsonl(args.output, record)
        return record
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record = _run_benchmark(args)
        if record is not None:
            if args.json:
                print(json.dumps(record, sort_keys=True))
            else:
                print(
                    "{strategy}: {samples_per_second:.2f} samples/s "
                    "({elapsed_seconds:.6f} s, world_size={world_size})".format(**record)
                )
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
