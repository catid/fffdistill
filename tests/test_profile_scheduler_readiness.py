from __future__ import annotations

import json
from pathlib import Path

import torch

from cifar_mamba_fff.benchmark_fff import _build_parser, _run_benchmark
from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.gpu_scheduler import (
    build_dry_run_jobs,
    build_train_teacher_command,
    parse_unavailable_slots,
    write_queue,
)
from cifar_mamba_fff.profile import time_cuda_callable


def test_time_cuda_callable_syncs_requested_cuda_device(monkeypatch) -> None:
    sync_calls: list[torch.device | None] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: sync_calls.append(device))

    result = time_cuda_callable(lambda: None, iterations=2, items=4, device="cuda:1")

    assert result.iterations == 2
    assert sync_calls == [torch.device("cuda:1"), torch.device("cuda:1")]


def test_benchmark_skip_naive_omits_naive_comparison_metadata() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["--quick-smoke", "true", "--device", "cpu", "--skip-naive", "true"]
    )

    rows = _run_benchmark(args)
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"dense", "fff_grouped"}
    assert "grouped_naive_max_abs_diff" not in by_name["dense"]
    assert "grouped_naive_max_abs_diff" not in by_name["fff_grouped"]


def test_benchmark_backward_rows_are_opt_in() -> None:
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
            "--measure-backward",
            "true",
        ]
    )

    rows = _run_benchmark(args)
    by_name = {row["name"]: row for row in rows}

    assert set(by_name) == {"dense", "fff_grouped", "dense_backward", "fff_grouped_backward"}
    assert by_name["dense"]["phase"] == "forward"
    assert by_name["fff_grouped_backward"]["phase"] == "forward_backward"
    assert by_name["fff_grouped_backward"]["tokens_per_second"] > 0.0


def test_scheduler_dry_run_jobs_bind_gpu_venv_python_and_output_dir(tmp_path) -> None:
    machines = [
        MachineSpec(
            name="work",
            host="localhost",
            gpus=2,
            role="local",
            workdir="/tmp/repo",
        )
    ]

    jobs = build_dry_run_jobs(
        machines,
        quick_smoke=True,
        dry_run=True,
        python_bin="venv/bin/python",
    )

    assert len(jobs) == 2
    for gpu_id, job in enumerate(jobs):
        expected_output_dir = Path("outputs/scheduler_smoke/work") / str(gpu_id)
        assert job.output_dir == expected_output_dir
        assert job.command.startswith("PYTHONPATH=src ")
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in job.command
        assert "venv/bin/python -m cifar_mamba_fff.train_teacher" in job.command
        assert f"--output-dir {expected_output_dir}" in job.command
        assert job.metadata["cuda_visible_devices"] == str(gpu_id)
        assert job.metadata["output_dir"] == str(expected_output_dir)

    queue_path = tmp_path / "job_queue.jsonl"
    write_queue(queue_path, jobs)
    records = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]

    assert records[0]["output_dir"] == "outputs/scheduler_smoke/work/0"
    assert records[0]["metadata"]["output_dir"] == "outputs/scheduler_smoke/work/0"


def test_scheduler_filters_unavailable_slots() -> None:
    machines = [
        MachineSpec(
            name="foureyes",
            host="foureyes",
            gpus=4,
            role="remote",
            workdir="/home/catid/fffdistill",
        )
    ]

    jobs = build_dry_run_jobs(
        machines,
        quick_smoke=True,
        dry_run=True,
        unavailable_slots=parse_unavailable_slots(["foureyes:2", "foureyes:3"]),
    )

    assert [(job.machine, job.gpu_id) for job in jobs] == [("foureyes", 0), ("foureyes", 1)]


def test_scheduler_default_command_uses_dot_venv_python() -> None:
    command = build_train_teacher_command(
        gpu_id=3,
        output_dir=Path("outputs/scheduler_smoke/work/3"),
    )

    assert command.startswith("PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m ")
    assert "--quick-smoke true" in command
    assert "--output-dir outputs/scheduler_smoke/work/3" in command


def test_scheduler_command_quotes_paths_with_spaces() -> None:
    command = build_train_teacher_command(
        gpu_id=0,
        output_dir=Path("outputs/scheduler smoke/work gpu0"),
        python_bin=".venv with spaces/bin/python",
    )

    assert "'.venv with spaces/bin/python'" in command
    assert "--output-dir 'outputs/scheduler smoke/work gpu0'" in command
