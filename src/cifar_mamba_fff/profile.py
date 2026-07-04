from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimingResult:
    seconds: float
    iterations: int
    items_per_second: float


def time_cuda_callable(
    fn: Callable[[], object],
    *,
    iterations: int,
    items: int,
    allow_cpu: bool = False,
) -> TimingResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA is required for time_cuda_callable; pass allow_cpu=True for CPU timing")
    else:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return TimingResult(seconds=seconds, iterations=iterations, items_per_second=items * iterations / seconds)
