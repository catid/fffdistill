from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from shlex import quote

import pytest
import torch

from cifar_mamba_fff import gpu_scheduler
from cifar_mamba_fff.benchmark_fff import _build_parser, _run_benchmark
from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.gpu_scheduler import (
    GpuJob,
    JobStatus,
    build_dry_run_jobs,
    build_teacher_hpo_command,
    build_train_teacher_command,
    launch_detached_job,
    parse_unavailable_slots,
    read_detached_job_status,
    resolve_collect_root,
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
        assert job.command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID ")
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in job.command
        assert "venv/bin/python -m cifar_mamba_fff.train_teacher" in job.command
        assert f"--output-dir {expected_output_dir}" in job.command
        assert f"--seed {1337 + gpu_id}" in job.command
        assert job.seed == 1337 + gpu_id
        assert job.metadata["cuda_device_order"] == "PCI_BUS_ID"
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


def test_scheduler_hpo_jobs_bind_gpu_seed_and_unique_output_dir() -> None:
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
        job_kind="teacher_hpo",
        run_id="smoke-001",
        hpo_trials_per_job=1,
        hpo_max_attempts_per_job=16,
        max_train_steps=1,
        max_val_steps=1,
    )

    assert len(jobs) == 2
    for gpu_id, job in enumerate(jobs):
        expected_output_dir = Path("outputs/scheduler_hpo/smoke-001/work") / str(gpu_id)
        assert job.output_dir == expected_output_dir
        assert job.command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID ")
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in job.command
        assert ".venv/bin/python -m cifar_mamba_fff.hpo.teacher_hpo" in job.command
        assert "--execute-trials true" in job.command
        assert "--max-trials 1" in job.command
        assert "--max-attempts 16" in job.command
        assert "--max-train-steps 1" in job.command
        assert "--max-val-steps 1" in job.command
        assert f"--seed {1337 + gpu_id}" in job.command
        assert f"--output-dir {expected_output_dir}" in job.command
        assert job.metadata["job_kind"] == "teacher_hpo"
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert len({job.seed for job in jobs}) == len(jobs)


def test_scheduler_rejects_duplicate_job_seed() -> None:
    jobs = [
        GpuJob(command="cmd", output_dir=Path("a"), seed=1),
        GpuJob(command="cmd", output_dir=Path("b"), seed=1),
    ]

    with pytest.raises(ValueError, match="duplicate job seed"):
        gpu_scheduler._validate_jobs_unique(jobs)


def test_scheduler_default_command_uses_dot_venv_python() -> None:
    command = build_train_teacher_command(
        gpu_id=3,
        output_dir=Path("outputs/scheduler_smoke/work/3"),
    )

    assert command.startswith(
        "PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m "
    )
    assert "--quick-smoke true" in command
    assert "--output-dir outputs/scheduler_smoke/work/3" in command


def test_scheduler_command_includes_seed_when_provided() -> None:
    command = build_train_teacher_command(
        gpu_id=0,
        output_dir=Path("outputs/scheduler_smoke/work/0"),
        seed=2026,
    )

    assert "--seed 2026" in command


def test_scheduler_hpo_command_quotes_and_uses_cuda_visible_device() -> None:
    command = build_teacher_hpo_command(
        gpu_id=2,
        output_dir=Path("outputs/scheduler hpo/work gpu2"),
        python_bin=".venv with spaces/bin/python",
        seed=2026,
        max_train_steps=1,
        max_val_steps=1,
        prune_min_value=0.2,
    )

    assert command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ")
    assert "'.venv with spaces/bin/python'" in command
    assert "--output-dir 'outputs/scheduler hpo/work gpu2'" in command
    assert "--seed 2026" in command
    assert "--max-train-steps 1" in command
    assert "--max-val-steps 1" in command
    assert "--prune-min-value 0.2" in command


def test_scheduler_main_threads_hpo_args_and_requires_cifar_preflight(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    machines_path = tmp_path / "machines.yaml"
    machines_path.write_text(
        "\n".join(
            [
                "machines:",
                "  work:",
                "    host: localhost",
                "    gpus: 1",
                "    role: local",
                f"    workdir: {tmp_path}",
            ]
        ),
        encoding="utf-8",
    )
    preflight_requirements: list[bool] = []

    def fake_preflight_machine(*args, **kwargs):
        preflight_requirements.append(bool(kwargs["require_cifar10_train"]))
        return gpu_scheduler.PreflightResult(
            machine="work",
            ok=False,
            returncode=1,
            commit="commit",
            stdout="",
            stderr="synthetic preflight stop",
        )

    monkeypatch.setattr(gpu_scheduler, "preflight_machine", fake_preflight_machine)
    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "commit")

    rc = gpu_scheduler.main(
        [
            "--machines",
            str(machines_path),
            "--queue-out",
            str(tmp_path / "queue.jsonl"),
            "--launch-results-out",
            str(tmp_path / "launch.jsonl"),
            "--dry-run",
            "false",
            "--quick-smoke",
            "true",
            "--smoke-mode",
            "metadata",
            "--job-kind",
            "teacher_hpo",
            "--run-id",
            "hpo-smoke",
            "--hpo-trials-per-job",
            "1",
            "--hpo-max-attempts-per-job",
            "5",
            "--max-train-steps",
            "1",
            "--max-val-steps",
            "1",
            "--wait",
            "false",
        ]
    )

    assert rc == 0
    assert preflight_requirements == [True]
    queue_records = [
        json.loads(line)
        for line in (tmp_path / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert queue_records[0]["output_dir"] == "outputs/scheduler_hpo/hpo-smoke/work/0"
    assert "--max-attempts 5" in queue_records[0]["command"]
    assert "--max-train-steps 1" in queue_records[0]["command"]
    assert "--max-val-steps 1" in queue_records[0]["command"]
    assert "recorded 1 GPU slots" in capsys.readouterr().out


def test_scheduler_default_collect_root_is_run_scoped() -> None:
    assert resolve_collect_root(None, run_id="teacher-hpo-001") == Path(
        "outputs/scheduler_collected/teacher-hpo-001"
    )
    assert resolve_collect_root(None, run_id=None) == Path("outputs/scheduler_collected")
    assert resolve_collect_root("outputs/custom_collect", run_id="teacher-hpo-001") == Path(
        "outputs/custom_collect"
    )


def test_scheduler_command_quotes_paths_with_spaces() -> None:
    command = build_train_teacher_command(
        gpu_id=0,
        output_dir=Path("outputs/scheduler smoke/work gpu0"),
        python_bin=".venv with spaces/bin/python",
    )

    assert "'.venv with spaces/bin/python'" in command
    assert "--output-dir 'outputs/scheduler smoke/work gpu0'" in command


def _wait_for_terminal_status(
    spec: MachineSpec,
    job: GpuJob,
    *,
    timeout_s: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    terminal = {
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED_INFRA.value,
        JobStatus.FAILED_LOGIC.value,
        JobStatus.CANCELLED.value,
    }
    while time.monotonic() < deadline:
        status = read_detached_job_status(spec, job)
        if str(status["status"]) in terminal:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not reach terminal status: {job.record()}")


def test_detached_local_job_records_success_sidecars(tmp_path) -> None:
    spec = MachineSpec(
        name="local",
        host="localhost",
        gpus=1,
        role="local",
        workdir=str(tmp_path),
    )
    command = f"{quote(sys.executable)} -c {quote('print(\"detached-ok\")')}"
    job = GpuJob(
        command=command,
        output_dir=Path("job_success"),
        machine="local",
        gpu_id=0,
        seed=123,
        metadata={"purpose": "unit-test"},
    )

    launch = launch_detached_job(spec, job)
    assert launch.ok is True
    assert launch.status == JobStatus.RUNNING

    status = _wait_for_terminal_status(spec, job)

    assert status["status"] == JobStatus.SUCCEEDED.value
    assert status["returncode"] == 0
    assert status["machine"] == "local"
    assert status["gpu_id"] == 0
    assert (tmp_path / "job_success" / "stdout.log").read_text(encoding="utf-8").strip() == "detached-ok"
    assert (tmp_path / "job_success" / "stderr.log").exists()
    assert (tmp_path / "job_success" / "pid.txt").exists()
    assert (tmp_path / "job_success" / "heartbeat.txt").exists()
    assert (tmp_path / "job_success" / "exit_code.txt").read_text(encoding="utf-8").strip() == "0"


def test_detached_local_job_classifies_nonzero_as_logic_failure(tmp_path) -> None:
    spec = MachineSpec(
        name="local",
        host="localhost",
        gpus=1,
        role="local",
        workdir=str(tmp_path),
    )
    command = f"{quote(sys.executable)} -c {quote('raise SystemExit(3)')}"
    job = GpuJob(
        command=command,
        output_dir=Path("job_logic_failure"),
        machine="local",
        gpu_id=0,
        seed=123,
    )

    launch = launch_detached_job(spec, job)
    assert launch.ok is True

    status = _wait_for_terminal_status(spec, job)

    assert status["status"] == JobStatus.FAILED_LOGIC.value
    assert status["returncode"] == 3
    assert (tmp_path / "job_logic_failure" / "exit_code.txt").read_text(encoding="utf-8").strip() == "3"
