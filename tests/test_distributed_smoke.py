from __future__ import annotations

import json

import torch

from cifar_mamba_fff import distributed as distributed_module
from cifar_mamba_fff.distributed import (
    DistributedContext,
    _build_parser,
    _resolve_device,
    _run_benchmark,
    current_context,
    distributed_available,
    init_distributed_if_needed,
)


def test_distributed_available_is_boolean() -> None:
    assert isinstance(distributed_available(), bool)


def test_current_context_reads_torchrun_environment(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    context = current_context()

    assert context.rank == 3
    assert context.local_rank == 1
    assert context.world_size == 8
    assert context.distributed is True


def test_init_distributed_auto_backend_uses_gloo_for_cpu(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29599")
    monkeypatch.setattr(distributed_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        distributed_module.dist,
        "init_process_group",
        lambda backend: calls.append(backend),
    )

    assert init_distributed_if_needed("auto", device=torch.device("cpu")) is True
    assert calls == ["gloo"]


def test_implicit_cuda_device_is_mapped_to_index(monkeypatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(distributed_module.torch.cuda, "set_device", calls.append)
    context = DistributedContext(rank=2, local_rank=1, world_size=4, distributed=True)

    single = _resolve_device(torch.device("cuda"), strategy="single", context=context)
    ddp = _resolve_device(torch.device("cuda"), strategy="ddp", context=context)

    assert single == torch.device("cuda", 0)
    assert ddp == torch.device("cuda", 1)
    assert calls == [torch.device("cuda", 0), torch.device("cuda", 1)]


def test_explicit_cuda_index_is_rejected_for_multi_process_ddp(monkeypatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(distributed_module.torch.cuda, "set_device", calls.append)
    context = DistributedContext(rank=0, local_rank=0, world_size=2, distributed=True)

    try:
        _resolve_device(torch.device("cuda", 0), strategy="ddp", context=context)
    except ValueError as exc:
        assert "Use --device auto or --device cuda" in str(exc)
    else:
        raise AssertionError("explicit cuda index should be rejected for multi-process DDP")

    assert calls == []


def test_synthetic_single_process_benchmark_writes_jsonl(tmp_path, monkeypatch) -> None:
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    output_path = tmp_path / "single.jsonl"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--strategy",
            "single",
            "--device",
            "cpu",
            "--quick-smoke",
            "true",
            "--output",
            str(output_path),
        ]
    )

    record = _run_benchmark(args)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    written = json.loads(lines[0])

    assert record is not None
    assert len(lines) == 1
    assert written["strategy"] == "single"
    assert written["device"] == "cpu"
    assert written["quick_smoke"] is True
    assert written["samples"] == 8
    assert written["samples_per_second"] > 0.0


def test_ddp_strategy_requires_torchrun_environment(monkeypatch) -> None:
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    parser = _build_parser()
    args = parser.parse_args(["--strategy", "ddp", "--device", "cpu", "--quick-smoke", "true"])

    try:
        _run_benchmark(args)
    except RuntimeError as exc:
        assert "requires torchrun-style" in str(exc)
    else:
        raise AssertionError("DDP without torchrun env should fail")
