from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from shlex import quote

import pytest
import torch

from cifar_mamba_fff import gpu_scheduler
from cifar_mamba_fff.benchmark_fff import _build_parser, _run_benchmark, _run_profile_report
from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.gpu_scheduler import (
    GpuJob,
    JobStatus,
    build_config_manifest,
    build_distill_hpo_command,
    build_dry_run_jobs,
    build_finetune_hpo_command,
    build_student_final_command,
    build_student_final_jobs,
    build_teacher_hpo_command,
    build_train_teacher_command,
    config_manifest_paths_from_jobs,
    launch_detached_job,
    parse_unavailable_slots,
    preflight_machine,
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
    assert by_name["dense"]["grad_enabled"] is False
    assert by_name["fff_grouped"]["grad_enabled"] is False
    assert by_name["fff_grouped_backward"]["phase"] == "forward_backward"
    assert by_name["dense_backward"]["grad_enabled"] is True
    assert by_name["fff_grouped_backward"]["grad_enabled"] is True
    assert by_name["fff_grouped_backward"]["tokens_per_second"] > 0.0


def test_benchmark_reports_dense_overhead_and_hot_path_metadata() -> None:
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
            "--include-naive",
            "true",
            "--measure-components",
            "true",
        ]
    )

    rows = _run_benchmark(args)
    by_name = {row["name"]: row for row in rows}

    assert {
        "dense",
        "fff_grouped",
        "fff_naive",
        "fff_route_setup",
        "fff_selected_leaf_kernel",
    } <= set(by_name)
    assert by_name["dense"]["dense_slowdown"] == pytest.approx(1.0)
    assert by_name["fff_grouped"]["dense_slowdown"] > 0.0
    assert by_name["fff_grouped"]["tokens_per_second_fraction_of_dense"] > 0.0
    assert by_name["fff_grouped"]["grouped_vs_naive_speedup"] > 0.0
    assert by_name["fff_grouped"]["python_per_token_hot_path"] is False
    assert by_name["fff_naive"]["python_per_token_hot_path"] is True
    assert by_name["fff_grouped"]["sort_bucket_strategy"] == "not_used_selected_leaf_gather_bmm"
    assert by_name["fff_grouped"]["sort_bucket_seconds_per_iteration"] == 0.0
    assert by_name["fff_grouped"]["sort_bucket_overhead_fraction"] == 0.0


def test_teacher_linear_profile_report_emits_reproducible_shape_matrix() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--profile-report",
            "teacher-linear",
            "--quick-smoke",
            "true",
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--report-include-naive",
            "false",
        ]
    )

    rows = _run_profile_report(args)
    cases = {row["profile_case"] for row in rows}
    names_by_case: dict[str, set[str]] = {}
    for row in rows:
        names_by_case.setdefault(row["profile_case"], set()).add(row["name"])

    assert cases == {
        "mamba3_d_model_square_256",
        "mamba3_expand_in_256x512",
        "mamba3_expand_out_512x256",
    }
    for case in cases:
        assert {"dense", "fff_grouped", "fff_route_setup", "fff_selected_leaf_kernel"} <= (
            names_by_case[case]
        )
    grouped = next(row for row in rows if row["name"] == "fff_grouped")
    assert grouped["profile_report"] == "teacher-linear"
    assert grouped["requested_batch_size"] == 1024
    assert grouped["tokens"] == 16
    assert grouped["requested_in_features"] in {256, 512}
    assert grouped["python_per_token_hot_path"] is False
    assert grouped["sort_bucket_strategy"] == "not_used_selected_leaf_gather_bmm"


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
        assert job.metadata["seed_base"] == 1337

    queue_path = tmp_path / "job_queue.jsonl"
    write_queue(queue_path, jobs)
    records = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]

    assert records[0]["output_dir"] == "outputs/scheduler_smoke/work/0"
    assert records[0]["metadata"]["output_dir"] == "outputs/scheduler_smoke/work/0"


def test_config_manifest_paths_from_jobs_collects_config_arguments() -> None:
    jobs = [
        GpuJob(
            command=(
                "PYTHONPATH=src .venv/bin/python -m runner "
                "--base-config configs/base.yaml --hpo-config=configs/hpo.yaml "
                "--selection-record docs/final_eval_selection/case.json"
            ),
            output_dir=Path("outputs/job"),
            machine="work",
            gpu_id=0,
        )
    ]

    assert config_manifest_paths_from_jobs(jobs) == [
        "configs/base.yaml",
        "configs/hpo.yaml",
        "docs/final_eval_selection/case.json",
    ]


def test_preflight_records_and_refuses_config_manifest_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("seed: 1\n", encoding="utf-8")
    local_manifest = build_config_manifest(["configs/base.yaml"])
    assert local_manifest is not None
    remote_manifest = {
        **local_manifest,
        "sha256": "0" * 64,
    }
    spec = MachineSpec(name="work", host="localhost", gpus=1, role="local", workdir=str(tmp_path))

    monkeypatch.setattr(gpu_scheduler, "_git_worktree_clean", lambda: True)
    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "abc123")

    def fake_run_remote(*args, **kwargs):
        return {
            "ok": True,
            "returncode": 0,
            "stdout": (
                "abc123\n"
                "Python 3.12.0\n"
                f"{gpu_scheduler.CONFIG_MANIFEST_PREFIX}"
                f"{json.dumps(remote_manifest, sort_keys=True)}\n"
            ),
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)

    result = preflight_machine(
        spec,
        python_bin="python3",
        expected_commit="abc123",
        expected_config_manifest=local_manifest,
    )

    assert result.ok is False
    assert result.local_config_manifest_sha256 == local_manifest["sha256"]
    assert result.remote_config_manifest_sha256 == "0" * 64
    assert "remote config manifest mismatch" in result.stderr


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


def test_scheduler_distill_grid_offsets_use_queued_job_index_after_unavailable_slots() -> None:
    machines = [
        MachineSpec(
            name="ripper",
            host="ripper",
            gpus=4,
            role="remote",
            workdir="/home/catid/fffdistill",
        )
    ]

    jobs = build_dry_run_jobs(
        machines,
        quick_smoke=False,
        dry_run=True,
        job_kind="distill_hpo",
        distill_teacher_checkpoint="checkpoints/teacher/best.pt",
        unavailable_slots=parse_unavailable_slots(["ripper:1"]),
        distill_grid_offset_base=3,
    )

    assert [(job.machine, job.gpu_id) for job in jobs] == [
        ("ripper", 0),
        ("ripper", 2),
        ("ripper", 3),
    ]
    assert [job.metadata["distill_grid_offset"] for job in jobs] == [3, 4, 5]
    assert "--grid-offset 3" in jobs[0].command
    assert "--grid-offset 4" in jobs[1].command
    assert "--grid-offset 5" in jobs[2].command


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
        assert job.metadata["seed_base"] == 1337
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert len({job.seed for job in jobs}) == len(jobs)


def test_scheduler_distill_hpo_jobs_bind_gpu_seed_checkpoint_config_and_samples() -> None:
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
        job_kind="distill_hpo",
        run_id="distill-smoke-001",
        hpo_trials_per_job=2,
        hpo_max_attempts_per_job=8,
        distill_teacher_checkpoint="checkpoints/teacher best.pt",
        distill_base_config="configs/distill base.yaml",
        distill_hpo_config="configs/distill hpo.yaml",
        distill_sample_split="val",
        distill_max_sample_batches=3,
        seed_base=7000,
    )

    assert len(jobs) == 2
    for gpu_id, job in enumerate(jobs):
        expected_output_dir = Path("outputs/scheduler_distill_hpo/distill-smoke-001/work") / str(
            gpu_id
        )
        assert job.output_dir == expected_output_dir
        assert job.command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID ")
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in job.command
        assert ".venv/bin/python -m cifar_mamba_fff.hpo.distill_hpo" in job.command
        assert "--execute-trials true" in job.command
        assert "--max-trials 2" in job.command
        assert "--max-attempts 8" in job.command
        assert "--teacher-checkpoint 'checkpoints/teacher best.pt'" in job.command
        assert "--base-config 'configs/distill base.yaml'" in job.command
        assert "--hpo-config 'configs/distill hpo.yaml'" in job.command
        assert "--sample-split val" in job.command
        assert "--max-sample-batches 3" in job.command
        assert f"--seed {7000 + gpu_id}" in job.command
        assert f"--grid-offset {gpu_id * 2}" in job.command
        assert f"--output-dir {expected_output_dir}" in job.command
        assert job.seed == 7000 + gpu_id
        assert job.metadata["job_kind"] == "distill_hpo"
        assert job.metadata["seed_base"] == 7000
        assert job.metadata["distill_teacher_checkpoint"] == "checkpoints/teacher best.pt"
        assert job.metadata["distill_base_config"] == "configs/distill base.yaml"
        assert job.metadata["distill_hpo_config"] == "configs/distill hpo.yaml"
        assert job.metadata["distill_sample_split"] == "val"
        assert job.metadata["distill_max_sample_batches"] == 3
        assert job.metadata["distill_grid_offset"] == gpu_id * 2
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert len({job.seed for job in jobs}) == len(jobs)


def test_scheduler_finetune_hpo_jobs_bind_gpu_seed_config_and_offsets() -> None:
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
        job_kind="finetune_hpo",
        run_id="finetune-smoke-001",
        hpo_trials_per_job=2,
        finetune_base_config="configs/finetune base.yaml",
        finetune_hpo_config="configs/finetune hpo.yaml",
        finetune_grid_offset_base=5,
        max_train_steps=3,
        max_val_steps=1,
        seed_base=8000,
    )

    assert len(jobs) == 2
    for gpu_id, job in enumerate(jobs):
        expected_output_dir = Path("outputs/scheduler_finetune_hpo/finetune-smoke-001/work") / str(
            gpu_id
        )
        assert job.output_dir == expected_output_dir
        assert job.command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID ")
        assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in job.command
        assert ".venv/bin/python -m cifar_mamba_fff.hpo.finetune_hpo" in job.command
        assert "--max-trials 2" in job.command
        assert "--base-config 'configs/finetune base.yaml'" in job.command
        assert "--hpo-config 'configs/finetune hpo.yaml'" in job.command
        assert f"--seed {8000 + gpu_id}" in job.command
        assert f"--grid-offset {5 + gpu_id * 2}" in job.command
        assert "--max-train-steps 3" in job.command
        assert "--max-val-steps 1" in job.command
        assert f"--output-dir {expected_output_dir}" in job.command
        assert job.seed == 8000 + gpu_id
        assert job.metadata["job_kind"] == "finetune_hpo"
        assert job.metadata["seed_base"] == 8000
        assert job.metadata["finetune_base_config"] == "configs/finetune base.yaml"
        assert job.metadata["finetune_hpo_config"] == "configs/finetune hpo.yaml"
        assert job.metadata["finetune_grid_offset"] == 5 + gpu_id * 2
    assert len({job.output_dir for job in jobs}) == len(jobs)
    assert len({job.seed for job in jobs}) == len(jobs)


def test_scheduler_student_final_jobs_bind_manifest_rows_and_allow_reused_seeds(
    tmp_path: Path,
) -> None:
    machines = [
        MachineSpec(
            name="work",
            host="localhost",
            gpus=2,
            role="local",
            workdir="/tmp/repo",
        )
    ]
    manifest = tmp_path / "student_final_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "machine": "work",
                        "gpu": 0,
                        "case": "baseline_seed21001",
                        "family": "baseline",
                        "seed": 21001,
                        "best_val_accuracy": 0.91,
                        "checkpoint_path": "outputs/a/student_best.pt",
                        "checkpoint_sha256": "a" * 64,
                        "selection_record": "docs/selection/baseline_seed21001.json",
                    }
                ),
                json.dumps(
                    {
                        "machine": "work",
                        "gpu": 1,
                        "case": "wsd_seed21001",
                        "family": "wsd",
                        "seed": 21001,
                        "best_val_accuracy": 0.90,
                        "checkpoint_path": "outputs/b/student_best.pt",
                        "checkpoint_sha256": "b" * 64,
                        "selection_record": "docs/selection/wsd_seed21001.json",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    jobs = build_student_final_jobs(
        machines,
        manifest=manifest,
        quick_smoke=False,
        dry_run=True,
        run_id="student-final",
        batch_size=512,
        num_workers=4,
    )

    assert [(job.machine, job.gpu_id, job.seed) for job in jobs] == [
        ("work", 0, 21001),
        ("work", 1, 21001),
    ]
    assert jobs[0].output_dir == Path("outputs/scheduler_student_final/student-final/work/0")
    assert ".venv/bin/python -m cifar_mamba_fff.evaluate_student" in jobs[0].command
    assert "--checkpoint outputs/a/student_best.pt" in jobs[0].command
    assert "--selection-record docs/selection/baseline_seed21001.json" in jobs[0].command
    assert "--batch-size 512" in jobs[0].command
    assert "--num-workers 4" in jobs[0].command
    assert jobs[0].metadata["job_kind"] == "student_final"
    assert jobs[0].metadata["case"] == "baseline_seed21001"
    assert jobs[1].metadata["family"] == "wsd"


def test_scheduler_custom_seed_base_changes_hpo_job_seeds() -> None:
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
        run_id="smoke-002",
        seed_base=9000,
    )

    assert [job.seed for job in jobs] == [9000, 9001]
    assert "--seed 9000" in jobs[0].command
    assert "--seed 9001" in jobs[1].command
    assert all(job.metadata["seed_base"] == 9000 for job in jobs)
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


def test_scheduler_distill_hpo_command_quotes_and_uses_cuda_visible_device() -> None:
    command = build_distill_hpo_command(
        gpu_id=2,
        output_dir=Path("outputs/scheduler distill hpo/work gpu2"),
        python_bin=".venv with spaces/bin/python",
        seed=2026,
        quick_smoke=False,
        teacher_checkpoint="checkpoints/teacher best.pt",
        base_config="configs/distill base.yaml",
        hpo_config="configs/distill hpo.yaml",
        sample_split="val",
        max_sample_batches=3,
        max_trials=4,
        max_attempts=9,
    )

    assert command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ")
    assert "'.venv with spaces/bin/python'" in command
    assert "-m cifar_mamba_fff.hpo.distill_hpo" in command
    assert "--output-dir 'outputs/scheduler distill hpo/work gpu2'" in command
    assert "--teacher-checkpoint 'checkpoints/teacher best.pt'" in command
    assert "--base-config 'configs/distill base.yaml'" in command
    assert "--hpo-config 'configs/distill hpo.yaml'" in command
    assert "--quick-smoke false" in command
    assert "--sample-split val" in command
    assert "--max-sample-batches 3" in command
    assert "--max-trials 4" in command
    assert "--max-attempts 9" in command
    assert "--grid-offset 0" in command
    assert "--seed 2026" in command


def test_scheduler_finetune_hpo_command_quotes_and_uses_cuda_visible_device() -> None:
    command = build_finetune_hpo_command(
        gpu_id=2,
        output_dir=Path("outputs/scheduler finetune hpo/work gpu2"),
        python_bin=".venv with spaces/bin/python",
        seed=2026,
        quick_smoke=False,
        base_config="configs/finetune base.yaml",
        hpo_config="configs/finetune hpo.yaml",
        max_trials=4,
        grid_offset=9,
        max_train_steps=5,
        max_val_steps=2,
    )

    assert command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ")
    assert "'.venv with spaces/bin/python'" in command
    assert "-m cifar_mamba_fff.hpo.finetune_hpo" in command
    assert "--output-dir 'outputs/scheduler finetune hpo/work gpu2'" in command
    assert "--base-config 'configs/finetune base.yaml'" in command
    assert "--hpo-config 'configs/finetune hpo.yaml'" in command
    assert "--quick-smoke false" in command
    assert "--max-trials 4" in command
    assert "--grid-offset 9" in command
    assert "--max-train-steps 5" in command
    assert "--max-val-steps 2" in command
    assert "--seed 2026" in command


def test_scheduler_student_final_command_quotes_and_uses_cuda_visible_device() -> None:
    command = build_student_final_command(
        gpu_id=2,
        output_dir=Path("outputs/student final/work gpu2"),
        python_bin=".venv with spaces/bin/python",
        checkpoint="outputs/check point/student_best.pt",
        selection_record="docs/selection record.json",
        quick_smoke=False,
        batch_size=512,
        num_workers=4,
        min_selected_val_accuracy=0.85,
    )

    assert command.startswith("PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ")
    assert "'.venv with spaces/bin/python'" in command
    assert "-m cifar_mamba_fff.evaluate_student" in command
    assert "--checkpoint 'outputs/check point/student_best.pt'" in command
    assert "--selection-record 'docs/selection record.json'" in command
    assert "--output-dir 'outputs/student final/work gpu2'" in command
    assert "--quick-smoke false" in command
    assert "--batch-size 512" in command
    assert "--num-workers 4" in command
    assert "--min-selected-val-accuracy 0.85" in command


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
    monkeypatch.setattr(gpu_scheduler, "_git_worktree_clean", lambda: True)

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
            "--seed-base",
            "8123",
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
    assert "--seed 8123" in queue_records[0]["command"]
    assert queue_records[0]["seed"] == 8123
    assert queue_records[0]["metadata"]["seed_base"] == 8123
    assert "--max-attempts 5" in queue_records[0]["command"]
    assert "--max-train-steps 1" in queue_records[0]["command"]
    assert "--max-val-steps 1" in queue_records[0]["command"]
    assert "recorded 1 GPU slots" in capsys.readouterr().out


def test_scheduler_main_threads_distill_hpo_args_and_requires_cifar_preflight(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "distill base.yaml").write_text("model: distill\n", encoding="utf-8")
    (configs_dir / "distill hpo.yaml").write_text("trials: 2\n", encoding="utf-8")
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
            stderr="synthetic distill preflight stop",
        )

    def fake_launch_detached_jobs(*args, **kwargs):
        assert args[1] == []
        return []

    monkeypatch.setattr(gpu_scheduler, "preflight_machine", fake_preflight_machine)
    monkeypatch.setattr(gpu_scheduler, "launch_detached_jobs", fake_launch_detached_jobs)
    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "commit")
    monkeypatch.setattr(gpu_scheduler, "_git_worktree_clean", lambda: True)

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
            "distill_hpo",
            "--run-id",
            "distill-smoke",
            "--distill-teacher-checkpoint",
            "checkpoints/teacher best.pt",
            "--distill-base-config",
            "configs/distill base.yaml",
            "--distill-hpo-config",
            "configs/distill hpo.yaml",
            "--distill-sample-split",
            "val",
            "--distill-max-sample-batches",
            "4",
            "--hpo-trials-per-job",
            "2",
            "--hpo-max-attempts-per-job",
            "7",
            "--seed-base",
            "8123",
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
    assert queue_records[0]["output_dir"] == "outputs/scheduler_distill_hpo/distill-smoke/work/0"
    assert "--seed 8123" in queue_records[0]["command"]
    assert queue_records[0]["seed"] == 8123
    assert queue_records[0]["metadata"]["job_kind"] == "distill_hpo"
    assert queue_records[0]["metadata"]["seed_base"] == 8123
    assert queue_records[0]["metadata"]["distill_teacher_checkpoint"] == "checkpoints/teacher best.pt"
    assert queue_records[0]["metadata"]["distill_base_config"] == "configs/distill base.yaml"
    assert queue_records[0]["metadata"]["distill_hpo_config"] == "configs/distill hpo.yaml"
    assert queue_records[0]["metadata"]["distill_sample_split"] == "val"
    assert queue_records[0]["metadata"]["distill_max_sample_batches"] == 4
    assert "--teacher-checkpoint 'checkpoints/teacher best.pt'" in queue_records[0]["command"]
    assert "--base-config 'configs/distill base.yaml'" in queue_records[0]["command"]
    assert "--hpo-config 'configs/distill hpo.yaml'" in queue_records[0]["command"]
    assert "--sample-split val" in queue_records[0]["command"]
    assert "--max-sample-batches 4" in queue_records[0]["command"]
    assert "--max-trials 2" in queue_records[0]["command"]
    assert "--max-attempts 7" in queue_records[0]["command"]
    assert "--grid-offset 0" in queue_records[0]["command"]
    assert queue_records[0]["metadata"]["distill_grid_offset"] == 0
    assert "recorded 1 GPU slots" in capsys.readouterr().out


def test_scheduler_main_threads_finetune_hpo_args_and_requires_cifar_preflight(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "finetune base.yaml").write_text("model: finetune\n", encoding="utf-8")
    (configs_dir / "finetune hpo.yaml").write_text("trials: 2\n", encoding="utf-8")
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
            stderr="synthetic finetune preflight stop",
        )

    def fake_launch_detached_jobs(*args, **kwargs):
        assert args[1] == []
        return []

    monkeypatch.setattr(gpu_scheduler, "preflight_machine", fake_preflight_machine)
    monkeypatch.setattr(gpu_scheduler, "launch_detached_jobs", fake_launch_detached_jobs)
    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "commit")
    monkeypatch.setattr(gpu_scheduler, "_git_worktree_clean", lambda: True)

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
            "finetune_hpo",
            "--run-id",
            "finetune-smoke",
            "--finetune-base-config",
            "configs/finetune base.yaml",
            "--finetune-hpo-config",
            "configs/finetune hpo.yaml",
            "--finetune-grid-offset-base",
            "4",
            "--hpo-trials-per-job",
            "2",
            "--seed-base",
            "8123",
            "--max-train-steps",
            "3",
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
    assert queue_records[0]["output_dir"] == "outputs/scheduler_finetune_hpo/finetune-smoke/work/0"
    assert "--seed 8123" in queue_records[0]["command"]
    assert queue_records[0]["seed"] == 8123
    assert queue_records[0]["metadata"]["job_kind"] == "finetune_hpo"
    assert queue_records[0]["metadata"]["seed_base"] == 8123
    assert queue_records[0]["metadata"]["finetune_base_config"] == "configs/finetune base.yaml"
    assert queue_records[0]["metadata"]["finetune_hpo_config"] == "configs/finetune hpo.yaml"
    assert queue_records[0]["metadata"]["finetune_grid_offset"] == 4
    assert "--base-config 'configs/finetune base.yaml'" in queue_records[0]["command"]
    assert "--hpo-config 'configs/finetune hpo.yaml'" in queue_records[0]["command"]
    assert "--max-trials 2" in queue_records[0]["command"]
    assert "--grid-offset 4" in queue_records[0]["command"]
    assert "--max-train-steps 3" in queue_records[0]["command"]
    assert "--max-val-steps 1" in queue_records[0]["command"]
    assert "recorded 1 GPU slots" in capsys.readouterr().out


def test_scheduler_main_threads_student_final_manifest_and_requires_cifar_preflight(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    selection_dir = tmp_path / "docs" / "selection"
    selection_dir.mkdir(parents=True)
    (selection_dir / "baseline_seed21001.json").write_text(
        json.dumps({"case": "baseline_seed21001"}) + "\n",
        encoding="utf-8",
    )
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
    manifest = tmp_path / "student_final_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "machine": "work",
                "gpu": 0,
                "case": "baseline_seed21001",
                "family": "baseline",
                "seed": 21001,
                "best_val_accuracy": 0.91,
                "checkpoint_path": "outputs/a/student_best.pt",
                "checkpoint_sha256": "a" * 64,
                "selection_record": "docs/selection/baseline_seed21001.json",
            }
        )
        + "\n",
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
            stderr="synthetic student final preflight stop",
        )

    def fake_launch_detached_jobs(*args, **kwargs):
        assert args[1] == []
        return []

    monkeypatch.setattr(gpu_scheduler, "preflight_machine", fake_preflight_machine)
    monkeypatch.setattr(gpu_scheduler, "launch_detached_jobs", fake_launch_detached_jobs)
    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "commit")
    monkeypatch.setattr(gpu_scheduler, "_git_worktree_clean", lambda: True)

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
            "false",
            "--allow-long-jobs",
            "true",
            "--job-kind",
            "student_final",
            "--run-id",
            "student-final",
            "--student-final-manifest",
            str(manifest),
            "--student-final-batch-size",
            "512",
            "--student-final-num-workers",
            "4",
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
    assert queue_records[0]["output_dir"] == "outputs/scheduler_student_final/student-final/work/0"
    assert queue_records[0]["seed"] == 21001
    assert queue_records[0]["metadata"]["job_kind"] == "student_final"
    assert queue_records[0]["metadata"]["case"] == "baseline_seed21001"
    assert "--checkpoint outputs/a/student_best.pt" in queue_records[0]["command"]
    assert "--selection-record docs/selection/baseline_seed21001.json" in queue_records[0]["command"]
    assert "--batch-size 512" in queue_records[0]["command"]
    assert "--num-workers 4" in queue_records[0]["command"]
    assert "recorded 1 GPU slots" in capsys.readouterr().out


def test_scheduler_main_rejects_non_smoke_distill_hpo_without_checkpoint(tmp_path) -> None:
    with pytest.raises(
        RuntimeError,
        match="--distill-teacher-checkpoint is required for non-smoke distill_hpo",
    ):
        gpu_scheduler.main(
            [
                "--machines",
                str(tmp_path / "missing-machines.yaml"),
                "--queue-out",
                str(tmp_path / "queue.jsonl"),
                "--dry-run",
                "true",
                "--quick-smoke",
                "false",
                "--job-kind",
                "distill_hpo",
            ]
        )


def test_scheduler_main_rejects_zero_distill_max_sample_batches(tmp_path) -> None:
    with pytest.raises(ValueError, match="--distill-max-sample-batches must be positive"):
        gpu_scheduler.main(
            [
                "--machines",
                str(tmp_path / "missing-machines.yaml"),
                "--queue-out",
                str(tmp_path / "queue.jsonl"),
                "--dry-run",
                "true",
                "--job-kind",
                "distill_hpo",
                "--distill-max-sample-batches",
                "0",
            ]
        )


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


def _init_clean_git_repo(path: Path, ignored: str) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    (path / ".gitignore").write_text(ignored, encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "base",
        ],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_detached_local_job_records_success_sidecars(tmp_path) -> None:
    _init_clean_git_repo(tmp_path, "job_success/\n")
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
    assert launch.status in {JobStatus.RUNNING, JobStatus.SUCCEEDED}

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
    _init_clean_git_repo(tmp_path, "job_logic_failure/\n")
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
