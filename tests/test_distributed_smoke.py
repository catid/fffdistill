from __future__ import annotations

from cifar_mamba_fff.distributed import distributed_available


def test_distributed_available_is_boolean() -> None:
    assert isinstance(distributed_available(), bool)
