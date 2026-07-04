from __future__ import annotations

import os

import torch.distributed as dist


def distributed_available() -> bool:
    return dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed_if_needed(backend: str = "nccl") -> bool:
    if not distributed_available():
        return False
    dist.init_process_group(backend=backend)
    return True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
