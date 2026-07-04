from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


def _cuda_device_arg(device: torch.device | str | int | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, int):
        return torch.device("cuda", device)
    requested = torch.device(device)
    if requested.type != "cuda":
        return None
    return requested


@dataclass(frozen=True)
class TimingResult:
    seconds: float
    iterations: int
    items_per_second: float

    @property
    def seconds_per_iteration(self) -> float:
        return self.seconds / self.iterations

    def as_metadata(self, **extra: object) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "iterations": self.iterations,
            "seconds_per_iteration": self.seconds_per_iteration,
            "items_per_second": self.items_per_second,
            **extra,
        }


def time_cuda_callable(
    fn: Callable[[], object],
    *,
    iterations: int,
    items: int,
    allow_cpu: bool = False,
    device: torch.device | str | int | None = None,
) -> TimingResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    cuda_device = _cuda_device_arg(device)
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is required for time_cuda_callable; pass allow_cpu=True for CPU timing")
    else:
        torch.cuda.synchronize(cuda_device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
    seconds = time.perf_counter() - start
    return TimingResult(seconds=seconds, iterations=iterations, items_per_second=items * iterations / seconds)
