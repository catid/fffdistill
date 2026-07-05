from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shlex import quote
from shlex import split as shlex_split
from typing import Any

from .cluster import MachineSpec, load_machines, read_remote_text, run_remote, write_remote_text
from .utils import bool_arg, git_commit

DEFAULT_PYTHON_BIN = ".venv/bin/python"
DEFAULT_SMOKE_MODE = "metadata"
DEFAULT_JOB_KIND = "teacher_train"
SCHEDULER_JOB_KINDS = ("teacher_train", "teacher_hpo", "distill_hpo", "finetune_hpo", "student_final")
ROOT_COLLECTED_ARTIFACT_NAMES = (
    "status.json",
    "stdout.log",
    "stderr.log",
    "exit_code.txt",
    "heartbeat.txt",
    "run_context.json",
    "metrics_summary.json",
    "metrics.jsonl",
    "teacher_hpo_summary.json",
    "teacher_hpo_events.jsonl",
    "distill_hpo_summary.json",
    "finetune_hpo_summary.json",
    "finetune_hpo_summary.csv",
    "teacher_final_test_metrics.json",
    "teacher_final_test_metrics.jsonl",
    "student_final_test_metrics.json",
    "student_final_test_metrics.jsonl",
)
TRIAL_COLLECTED_ARTIFACT_NAMES = (
    "trial_config.json",
    "trial_summary.json",
    "finetune_config.yaml",
    "run_context.json",
    "metrics_summary.json",
    "metrics.jsonl",
    "student_assembly_manifest.json",
    "student_assembly_manifest.csv",
    "distill_config.yaml",
    "trial_result.json",
    "distill_summary.json",
    "layer_summary.json",
    "layer_metrics.jsonl",
)
COLLECTED_ARTIFACT_NAMES = (
    *ROOT_COLLECTED_ARTIFACT_NAMES,
    *(f"trials/trial_000000/{name}" for name in TRIAL_COLLECTED_ARTIFACT_NAMES),
)
CONFIG_PATH_ARGUMENTS = frozenset(
    {
        "--base-config",
        "--hpo-config",
        "--config",
        "--selection-record",
    }
)
CONFIG_MANIFEST_PREFIX = "CONFIG_MANIFEST "


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_INFRA = "failed_infra"
    FAILED_LOGIC = "failed_logic"
    CANCELLED = "cancelled"


@dataclass
class GpuJob:
    command: str
    output_dir: Path
    machine: str | None = None
    gpu_id: int | None = None
    seed: int | None = None
    status: JobStatus = JobStatus.QUEUED
    metadata: dict[str, object] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        local_commit = git_commit()
        return {
            "command": self.command,
            "output_dir": str(self.output_dir),
            "machine": self.machine,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "status": self.status,
            "metadata": self.metadata,
            "git_commit": local_commit,
            "local_git_commit": local_commit,
        }


@dataclass(frozen=True)
class DetachedJobFiles:
    output_dir: Path
    script: Path
    status: Path
    stdout: Path
    stderr: Path
    pid: Path
    exit_code: Path
    heartbeat: Path

    def record(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "script": str(self.script),
            "status": str(self.status),
            "stdout": str(self.stdout),
            "stderr": str(self.stderr),
            "pid": str(self.pid),
            "exit_code": str(self.exit_code),
            "heartbeat": str(self.heartbeat),
        }


@dataclass(frozen=True)
class LaunchResult:
    ok: bool
    status: JobStatus
    job: GpuJob
    files: DetachedJobFiles
    launch_stdout: str
    launch_stderr: str


@dataclass(frozen=True)
class PreflightResult:
    machine: str
    ok: bool
    returncode: int | None
    commit: str | None
    expected_commit: str | None = None
    local_commit: str = ""
    remote_commit: str | None = None
    local_clean: bool | None = None
    remote_clean: bool | None = None
    local_config_manifest_sha256: str | None = None
    remote_config_manifest_sha256: str | None = None
    local_config_manifest: dict[str, Any] | None = None
    remote_config_manifest: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""

    def record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "machine": self.machine,
            "ok": self.ok,
            "returncode": self.returncode,
            "commit": self.commit,
            "expected_commit": self.expected_commit,
            "local_commit": self.local_commit,
            "remote_commit": self.remote_commit,
            "local_clean": self.local_clean,
            "remote_clean": self.remote_clean,
            "stdout": self.stdout.strip(),
            "stderr": self.stderr.strip(),
        }
        if self.local_config_manifest is not None or self.remote_config_manifest is not None:
            record.update(
                {
                    "local_config_manifest_sha256": self.local_config_manifest_sha256,
                    "remote_config_manifest_sha256": self.remote_config_manifest_sha256,
                    "local_config_manifest": self.local_config_manifest,
                    "remote_config_manifest": self.remote_config_manifest,
                }
            )
        return record


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_worktree_clean() -> bool:
    checks = (
        ("git", "diff", "--quiet"),
        ("git", "diff", "--cached", "--quiet"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    )
    for command in checks[:2]:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            return False
    completed = subprocess.run(checks[2], check=False, capture_output=True, text=True)
    return completed.returncode == 0 and not completed.stdout.strip()


def _append_stderr(stderr: str, message: str) -> str:
    return (stderr.rstrip() + "\n" + message).strip()


def _relative_path(path: Path, *, field_name: str) -> Path:
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative to the repository workdir")
    return path


def _normalize_manifest_path(path: str | Path) -> str:
    relative = _relative_path(Path(path), field_name="config manifest path")
    if any(part == ".." for part in relative.parts):
        raise ValueError(f"config manifest path must stay inside repository: {path}")
    return relative.as_posix()


def build_config_manifest(paths: list[str | Path]) -> dict[str, Any] | None:
    normalized = sorted({_normalize_manifest_path(path) for path in paths if str(path)})
    if not normalized:
        return None
    files: list[dict[str, object]] = []
    for path_text in normalized:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"config manifest path not found: {path_text}")
        data = path.read_bytes()
        files.append(
            {
                "path": path_text,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _config_manifest_paths(manifest: dict[str, Any] | None) -> list[str]:
    if manifest is None:
        return []
    paths: list[str] = []
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if path:
            paths.append(str(path))
    return paths


def config_manifest_paths_from_jobs(jobs: list[GpuJob]) -> list[str]:
    paths: list[str | Path] = []
    for job in jobs:
        try:
            parts = shlex_split(job.command)
        except ValueError as exc:
            raise ValueError(f"cannot parse job command for config manifest: {job.command}") from exc
        for index, token in enumerate(parts):
            if token in CONFIG_PATH_ARGUMENTS:
                if index + 1 >= len(parts):
                    raise ValueError(f"{token} in job command is missing a path")
                paths.append(parts[index + 1])
                continue
            for flag in CONFIG_PATH_ARGUMENTS:
                prefix = f"{flag}="
                if token.startswith(prefix):
                    paths.append(token[len(prefix) :])
    return sorted({_normalize_manifest_path(path) for path in paths if str(path)})


def _config_manifest_python_source() -> str:
    return "\n".join(
        [
            "import hashlib, json, sys",
            "from pathlib import Path",
            "paths = json.loads(sys.argv[1])",
            "files = []",
            "for path_text in sorted(set(paths)):",
            "    path = Path(path_text)",
            "    if path.is_absolute() or '..' in path.parts:",
            "        raise SystemExit(f'invalid config manifest path: {path_text}')",
            "    if not path.is_file():",
            "        raise SystemExit(f'missing config manifest path: {path_text}')",
            "    data = path.read_bytes()",
            "    files.append({",
            "        'path': path_text,",
            "        'size_bytes': len(data),",
            "        'sha256': hashlib.sha256(data).hexdigest(),",
            "    })",
            "canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')",
            "manifest = {",
            "    'algorithm': 'sha256',",
            "    'files': files,",
            "    'sha256': hashlib.sha256(canonical).hexdigest(),",
            "}",
            f"print({CONFIG_MANIFEST_PREFIX!r} + json.dumps(manifest, sort_keys=True))",
        ]
    )


def _remote_config_manifest_command(paths: list[str | Path], *, python_bin: str) -> str | None:
    normalized = sorted({_normalize_manifest_path(path) for path in paths if str(path)})
    if not normalized:
        return None
    return (
        f"{quote(python_bin)} -c {quote(_config_manifest_python_source())} "
        f"{quote(json.dumps(normalized))}"
    )


def _extract_remote_config_manifest(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith(CONFIG_MANIFEST_PREFIX):
            return json.loads(line[len(CONFIG_MANIFEST_PREFIX) :])
    return None


def detached_job_files(job: GpuJob) -> DetachedJobFiles:
    output_dir = _relative_path(job.output_dir, field_name="job.output_dir")
    return DetachedJobFiles(
        output_dir=output_dir,
        script=output_dir / "launch.sh",
        status=output_dir / "status.json",
        stdout=output_dir / "stdout.log",
        stderr=output_dir / "stderr.log",
        pid=output_dir / "pid.txt",
        exit_code=output_dir / "exit_code.txt",
        heartbeat=output_dir / "heartbeat.txt",
    )


def enumerate_slots(machines: list[MachineSpec]) -> list[tuple[str, int]]:
    return [(machine.name, gpu_id) for machine in machines for gpu_id in range(machine.gpus)]


def parse_unavailable_slots(values: list[str]) -> set[tuple[str, int]]:
    unavailable: set[tuple[str, int]] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if ":" not in normalized:
            raise ValueError("unavailable slots must use machine:gpu_id syntax")
        machine, gpu_text = normalized.split(":", maxsplit=1)
        if not machine:
            raise ValueError("unavailable slot machine must be non-empty")
        try:
            gpu_id = int(gpu_text)
        except ValueError as exc:
            raise ValueError(f"invalid GPU id in unavailable slot {value!r}") from exc
        if gpu_id < 0:
            raise ValueError("unavailable slot GPU id must be non-negative")
        unavailable.add((machine, gpu_id))
    return unavailable


def build_train_teacher_command(
    *,
    gpu_id: int,
    output_dir: Path,
    python_bin: str = DEFAULT_PYTHON_BIN,
    quick_smoke: bool = True,
    smoke_mode: str = DEFAULT_SMOKE_MODE,
    seed: int | None = None,
) -> str:
    seed_arg = "" if seed is None else f" --seed {int(seed)}"
    return (
        f"PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} {quote(python_bin)} "
        "-m cifar_mamba_fff.train_teacher "
        f"--quick-smoke {str(quick_smoke).lower()} "
        f"--smoke-mode {quote(smoke_mode)} "
        f"--output-dir {quote(str(output_dir))}"
        f"{seed_arg}"
    )


def build_teacher_hpo_command(
    *,
    gpu_id: int,
    output_dir: Path,
    seed: int,
    python_bin: str = DEFAULT_PYTHON_BIN,
    quick_smoke: bool = True,
    base_config: str = "configs/teacher_default.yaml",
    hpo_config: str = "configs/teacher_hpo.yaml",
    max_trials: int = 1,
    max_attempts: int = 32,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    prune_min_value: float | None = None,
) -> str:
    command = (
        f"PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{quote(python_bin)} -m cifar_mamba_fff.hpo.teacher_hpo "
        f"--base-config {quote(base_config)} "
        f"--hpo-config {quote(hpo_config)} "
        f"--output-dir {quote(str(output_dir))} "
        f"--quick-smoke {str(quick_smoke).lower()} "
        "--execute-trials true "
        f"--max-trials {max_trials} "
        f"--max-attempts {max_attempts} "
        f"--seed {seed}"
    )
    if max_train_steps is not None:
        command += f" --max-train-steps {int(max_train_steps)}"
    if max_val_steps is not None:
        command += f" --max-val-steps {int(max_val_steps)}"
    if prune_min_value is not None:
        command += f" --prune-min-value {float(prune_min_value)}"
    return command


def build_distill_hpo_command(
    *,
    gpu_id: int,
    output_dir: Path,
    seed: int,
    python_bin: str = DEFAULT_PYTHON_BIN,
    quick_smoke: bool = True,
    teacher_checkpoint: str | None = None,
    base_config: str = "configs/fff_distill_default.yaml",
    hpo_config: str = "configs/fff_distill_hpo.yaml",
    sample_split: str = "train_eval",
    max_sample_batches: int = 1,
    max_trials: int = 1,
    max_attempts: int = 32,
    grid_offset: int = 0,
) -> str:
    if sample_split not in {"train", "train_eval", "val"}:
        raise ValueError("sample_split must be train, train_eval, or val")
    if max_sample_batches <= 0:
        raise ValueError("max_sample_batches must be positive")
    if grid_offset < 0:
        raise ValueError("grid_offset must be non-negative")
    command = (
        f"PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{quote(python_bin)} -m cifar_mamba_fff.hpo.distill_hpo "
        f"--base-config {quote(base_config)} "
        f"--hpo-config {quote(hpo_config)} "
        f"--output-dir {quote(str(output_dir))} "
        f"--quick-smoke {str(quick_smoke).lower()} "
        "--execute-trials true "
        f"--sample-split {quote(sample_split)} "
        f"--max-sample-batches {int(max_sample_batches)} "
        f"--max-trials {int(max_trials)} "
        f"--max-attempts {int(max_attempts)} "
        f"--grid-offset {int(grid_offset)} "
        f"--seed {int(seed)}"
    )
    if teacher_checkpoint is not None:
        command += f" --teacher-checkpoint {quote(teacher_checkpoint)}"
    return command


def build_finetune_hpo_command(
    *,
    gpu_id: int,
    output_dir: Path,
    seed: int,
    python_bin: str = DEFAULT_PYTHON_BIN,
    quick_smoke: bool = True,
    base_config: str = "configs/finetune_default.yaml",
    hpo_config: str = "configs/finetune_hpo.yaml",
    max_trials: int = 1,
    grid_offset: int = 0,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> str:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if grid_offset < 0:
        raise ValueError("grid_offset must be non-negative")
    command = (
        f"PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{quote(python_bin)} -m cifar_mamba_fff.hpo.finetune_hpo "
        f"--base-config {quote(base_config)} "
        f"--hpo-config {quote(hpo_config)} "
        f"--output-dir {quote(str(output_dir))} "
        f"--quick-smoke {str(quick_smoke).lower()} "
        f"--max-trials {int(max_trials)} "
        f"--grid-offset {int(grid_offset)} "
        f"--seed {int(seed)}"
    )
    if max_train_steps is not None:
        command += f" --max-train-steps {int(max_train_steps)}"
    if max_val_steps is not None:
        command += f" --max-val-steps {int(max_val_steps)}"
    return command


def build_student_final_command(
    *,
    gpu_id: int,
    output_dir: Path,
    checkpoint: str,
    selection_record: str,
    python_bin: str = DEFAULT_PYTHON_BIN,
    quick_smoke: bool = False,
    max_test_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    min_selected_val_accuracy: float = 0.90,
    allow_below_target: bool = False,
) -> str:
    command = (
        f"PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{quote(python_bin)} -m cifar_mamba_fff.evaluate_student "
        f"--checkpoint {quote(checkpoint)} "
        f"--selection-record {quote(selection_record)} "
        f"--output-dir {quote(str(output_dir))} "
        f"--quick-smoke {str(quick_smoke).lower()} "
        f"--min-selected-val-accuracy {float(min_selected_val_accuracy)} "
        f"--allow-below-target {str(allow_below_target).lower()}"
    )
    if max_test_steps is not None:
        command += f" --max-test-steps {int(max_test_steps)}"
    if batch_size is not None:
        command += f" --batch-size {int(batch_size)}"
    if num_workers is not None:
        command += f" --num-workers {int(num_workers)}"
    return command


def _load_student_final_manifest(path: str | Path | None) -> list[dict[str, object]]:
    if path is None:
        raise ValueError("--student-final-manifest is required for student_final jobs")
    manifest_path = Path(path)
    records: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain a JSON object")
            records.append(payload)
    if not records:
        raise ValueError(f"{manifest_path} contains no student final jobs")
    return records


def _manifest_str(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if value in (None, ""):
        raise ValueError(f"student final manifest row missing required field {key}")
    return str(value)


def _manifest_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if value in (None, ""):
        raise ValueError(f"student final manifest row missing required integer field {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"student final manifest field {key} must be an integer") from exc


def build_student_final_jobs(
    machines: list[MachineSpec],
    *,
    manifest: str | Path | None,
    quick_smoke: bool,
    dry_run: bool,
    python_bin: str = DEFAULT_PYTHON_BIN,
    output_root: Path | None = None,
    run_id: str | None = None,
    unavailable_slots: set[tuple[str, int]] | None = None,
    max_jobs: int | None = None,
    max_test_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    min_selected_val_accuracy: float = 0.90,
    allow_below_target: bool = False,
) -> list[GpuJob]:
    specs_by_name = {machine.name: machine for machine in machines}
    unavailable_slots = unavailable_slots or set()
    resolved_output_root = output_root or Path("outputs/scheduler_student_final")
    if run_id is not None:
        resolved_output_root = resolved_output_root / run_id
    jobs: list[GpuJob] = []
    for row_index, record in enumerate(_load_student_final_manifest(manifest)):
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
        machine = _manifest_str(record, "machine")
        if machine not in specs_by_name:
            raise ValueError(f"student final manifest uses unknown machine {machine!r}")
        gpu_id = _manifest_int(record, "gpu")
        if gpu_id < 0 or gpu_id >= specs_by_name[machine].gpus:
            raise ValueError(f"student final manifest has invalid GPU {machine}:{gpu_id}")
        if (machine, gpu_id) in unavailable_slots:
            continue
        case = _manifest_str(record, "case")
        checkpoint = _manifest_str(record, "checkpoint_path")
        selection_record = _manifest_str(record, "selection_record")
        seed = _manifest_int(record, "seed")
        output_dir = resolved_output_root / machine / str(gpu_id)
        command = build_student_final_command(
            gpu_id=gpu_id,
            output_dir=output_dir,
            checkpoint=checkpoint,
            selection_record=selection_record,
            python_bin=python_bin,
            quick_smoke=quick_smoke,
            max_test_steps=max_test_steps,
            batch_size=batch_size,
            num_workers=num_workers,
            min_selected_val_accuracy=min_selected_val_accuracy,
            allow_below_target=allow_below_target,
        )
        jobs.append(
            GpuJob(
                command=command,
                output_dir=output_dir,
                machine=machine,
                gpu_id=gpu_id,
                seed=seed,
                metadata={
                    "quick_smoke": quick_smoke,
                    "dry_run": dry_run,
                    "job_kind": "student_final",
                    "case": case,
                    "family": _manifest_str(record, "family"),
                    "selection_record": selection_record,
                    "checkpoint_path": checkpoint,
                    "selected_val_accuracy": record.get("best_val_accuracy"),
                    "checkpoint_sha256": record.get("checkpoint_sha256"),
                    "manifest_row_index": row_index,
                    "cuda_device_order": "PCI_BUS_ID",
                    "cuda_visible_devices": str(gpu_id),
                    "python_bin": python_bin,
                    "output_dir": str(output_dir),
                    "max_test_steps": max_test_steps,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                },
            )
        )
    _validate_jobs_unique(jobs, require_unique_seeds=False)
    return jobs


def build_dry_run_jobs(
    machines: list[MachineSpec],
    *,
    quick_smoke: bool,
    dry_run: bool,
    python_bin: str = DEFAULT_PYTHON_BIN,
    smoke_mode: str = DEFAULT_SMOKE_MODE,
    job_kind: str = DEFAULT_JOB_KIND,
    teacher_base_config: str = "configs/teacher_default.yaml",
    teacher_hpo_config: str = "configs/teacher_hpo.yaml",
    distill_base_config: str = "configs/fff_distill_default.yaml",
    distill_hpo_config: str = "configs/fff_distill_hpo.yaml",
    finetune_base_config: str = "configs/finetune_default.yaml",
    finetune_hpo_config: str = "configs/finetune_hpo.yaml",
    distill_teacher_checkpoint: str | None = None,
    distill_sample_split: str = "train_eval",
    distill_max_sample_batches: int = 1,
    distill_grid_offset_base: int = 0,
    finetune_grid_offset_base: int = 0,
    hpo_trials_per_job: int = 1,
    hpo_max_attempts_per_job: int = 32,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    prune_min_value: float | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    unavailable_slots: set[tuple[str, int]] | None = None,
    max_jobs: int | None = None,
    seed_base: int = 1337,
) -> list[GpuJob]:
    if job_kind not in SCHEDULER_JOB_KINDS:
        raise ValueError(f"job_kind must be one of: {', '.join(SCHEDULER_JOB_KINDS)}")
    if job_kind == "student_final":
        raise ValueError("student_final jobs require build_student_final_jobs")
    if seed_base < 0:
        raise ValueError("seed_base must be non-negative")
    if distill_max_sample_batches <= 0:
        raise ValueError("distill_max_sample_batches must be positive")
    if distill_grid_offset_base < 0:
        raise ValueError("distill_grid_offset_base must be non-negative")
    if finetune_grid_offset_base < 0:
        raise ValueError("finetune_grid_offset_base must be non-negative")
    unavailable_slots = unavailable_slots or set()
    jobs: list[GpuJob] = []
    for idx, (machine, gpu_id) in enumerate(enumerate_slots(machines)):
        if (machine, gpu_id) in unavailable_slots:
            continue
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
        seed = seed_base + idx
        queued_job_index = len(jobs)
        default_root = Path(
            "outputs/scheduler_hpo"
            if job_kind == "teacher_hpo"
            else "outputs/scheduler_distill_hpo"
            if job_kind == "distill_hpo"
            else "outputs/scheduler_finetune_hpo"
            if job_kind == "finetune_hpo"
            else "outputs/scheduler_smoke"
        )
        resolved_output_root = output_root or default_root
        if run_id is not None:
            resolved_output_root = resolved_output_root / run_id
        output_dir = resolved_output_root / machine / str(gpu_id)
        hpo_grid_offset = queued_job_index * hpo_trials_per_job
        distill_grid_offset = distill_grid_offset_base + hpo_grid_offset
        finetune_grid_offset = finetune_grid_offset_base + hpo_grid_offset
        if job_kind == "teacher_hpo":
            command = build_teacher_hpo_command(
                gpu_id=gpu_id,
                output_dir=output_dir,
                seed=seed,
                python_bin=python_bin,
                quick_smoke=quick_smoke,
                base_config=teacher_base_config,
                hpo_config=teacher_hpo_config,
                max_trials=hpo_trials_per_job,
                max_attempts=hpo_max_attempts_per_job,
                max_train_steps=max_train_steps,
                max_val_steps=max_val_steps,
                prune_min_value=prune_min_value,
            )
        elif job_kind == "distill_hpo":
            command = build_distill_hpo_command(
                gpu_id=gpu_id,
                output_dir=output_dir,
                seed=seed,
                python_bin=python_bin,
                quick_smoke=quick_smoke,
                teacher_checkpoint=distill_teacher_checkpoint,
                base_config=distill_base_config,
                hpo_config=distill_hpo_config,
                sample_split=distill_sample_split,
                max_sample_batches=distill_max_sample_batches,
                max_trials=hpo_trials_per_job,
                max_attempts=hpo_max_attempts_per_job,
                grid_offset=distill_grid_offset,
            )
        elif job_kind == "finetune_hpo":
            command = build_finetune_hpo_command(
                gpu_id=gpu_id,
                output_dir=output_dir,
                seed=seed,
                python_bin=python_bin,
                quick_smoke=quick_smoke,
                base_config=finetune_base_config,
                hpo_config=finetune_hpo_config,
                max_trials=hpo_trials_per_job,
                grid_offset=finetune_grid_offset,
                max_train_steps=max_train_steps,
                max_val_steps=max_val_steps,
            )
        else:
            command = build_train_teacher_command(
                gpu_id=gpu_id,
                output_dir=output_dir,
                python_bin=python_bin,
                quick_smoke=quick_smoke,
                smoke_mode=smoke_mode,
                seed=seed,
            )
        jobs.append(
            GpuJob(
                command=command,
                output_dir=output_dir,
                machine=machine,
                gpu_id=gpu_id,
                seed=seed,
                metadata={
                    "quick_smoke": quick_smoke,
                    "dry_run": dry_run,
                    "job_kind": job_kind,
                    "smoke_mode": smoke_mode,
                    "cuda_device_order": "PCI_BUS_ID",
                    "cuda_visible_devices": str(gpu_id),
                    "python_bin": python_bin,
                    "output_dir": str(output_dir),
                    "seed_base": seed_base,
                    "distill_base_config": distill_base_config
                    if job_kind == "distill_hpo"
                    else None,
                    "distill_hpo_config": distill_hpo_config
                    if job_kind == "distill_hpo"
                    else None,
                    "distill_teacher_checkpoint": distill_teacher_checkpoint
                    if job_kind == "distill_hpo"
                    else None,
                    "distill_sample_split": distill_sample_split
                    if job_kind == "distill_hpo"
                    else None,
                    "distill_max_sample_batches": distill_max_sample_batches
                    if job_kind == "distill_hpo"
                    else None,
                    "distill_grid_offset": distill_grid_offset
                    if job_kind == "distill_hpo"
                    else None,
                    "finetune_base_config": finetune_base_config
                    if job_kind == "finetune_hpo"
                    else None,
                    "finetune_hpo_config": finetune_hpo_config
                    if job_kind == "finetune_hpo"
                    else None,
                    "finetune_grid_offset": finetune_grid_offset
                    if job_kind == "finetune_hpo"
                    else None,
                },
            )
        )
    _validate_jobs_unique(jobs)
    return jobs


def _validate_jobs_unique(jobs: list[GpuJob], *, require_unique_seeds: bool = True) -> None:
    output_dirs: set[Path] = set()
    seeds: set[int] = set()
    for job in jobs:
        if job.output_dir in output_dirs:
            raise ValueError(f"duplicate job output_dir: {job.output_dir}")
        output_dirs.add(job.output_dir)
        if job.seed is None or not require_unique_seeds:
            continue
        if job.seed in seeds:
            raise ValueError(f"duplicate job seed: {job.seed}")
        seeds.add(job.seed)


def write_queue(path: Path, jobs: list[GpuJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.record(), sort_keys=True) + "\n")


def preflight_machine(
    spec: MachineSpec,
    *,
    python_bin: str = DEFAULT_PYTHON_BIN,
    expected_commit: str | None = None,
    config_paths: list[str | Path] | None = None,
    expected_config_manifest: dict[str, Any] | None = None,
    require_cifar10_train: bool = False,
    data_dir: str | Path = "data/cifar10",
    timeout_s: int = 20,
) -> PreflightResult:
    local_clean = _git_worktree_clean()
    local_config_manifest = expected_config_manifest
    if local_config_manifest is None:
        local_config_manifest = build_config_manifest(list(config_paths or []))
    manifest_command = _remote_config_manifest_command(
        _config_manifest_paths(local_config_manifest),
        python_bin=python_bin,
    )
    commands = [
        "test -d .git",
        "git rev-parse HEAD",
        "git diff --quiet || { echo 'remote worktree has unstaged changes' >&2; exit 121; }",
        "git diff --cached --quiet || { echo 'remote worktree has staged changes' >&2; exit 121; }",
        (
            "test -z \"$(git ls-files --others --exclude-standard)\" || "
            "{ echo 'remote worktree has untracked non-ignored files' >&2; "
            "git ls-files --others --exclude-standard >&2; exit 121; }"
        ),
        f"test -x {quote(python_bin)}",
        f"{quote(python_bin)} --version",
    ]
    if manifest_command is not None:
        commands.append(manifest_command)
    if require_cifar10_train:
        commands.append(
            f"{quote(python_bin)} scripts/prepare_cifar10.py "
            f"--data-dir {quote(str(data_dir))} "
            "--output-json outputs/cifar10_preflight.json "
            "--download false --extract false"
        )
    command = " && ".join(commands)
    result = run_remote(spec, command, timeout_s=timeout_s)
    stdout = str(result["stdout"])
    commit = stdout.splitlines()[0].strip() if stdout.splitlines() else None
    local_commit = git_commit()
    ok = bool(result["ok"])
    stderr = str(result["stderr"])
    remote_clean = result["returncode"] != 121
    try:
        remote_config_manifest = _extract_remote_config_manifest(stdout)
    except json.JSONDecodeError as exc:
        ok = False
        stderr = _append_stderr(stderr, f"remote config manifest is invalid JSON: {exc}")
        remote_config_manifest = None
    if expected_commit is not None and commit != expected_commit:
        ok = False
        stderr = _append_stderr(
            stderr,
            f"remote commit/config mismatch: expected {expected_commit}, got {commit}",
        )
    if local_config_manifest is not None:
        local_sha = str(local_config_manifest["sha256"])
        remote_sha = (
            str(remote_config_manifest["sha256"])
            if remote_config_manifest is not None and remote_config_manifest.get("sha256")
            else None
        )
        if remote_sha != local_sha:
            ok = False
            stderr = _append_stderr(
                stderr,
                f"remote config manifest mismatch: expected {local_sha}, got {remote_sha or 'missing'}",
            )
    if not local_clean:
        ok = False
        stderr = _append_stderr(stderr, "local worktree is dirty")
    return PreflightResult(
        machine=spec.name,
        ok=ok,
        returncode=result["returncode"],
        commit=commit,
        expected_commit=expected_commit,
        local_commit=local_commit,
        remote_commit=commit,
        local_clean=local_clean,
        remote_clean=remote_clean,
        local_config_manifest_sha256=(
            str(local_config_manifest["sha256"]) if local_config_manifest is not None else None
        ),
        remote_config_manifest_sha256=(
            str(remote_config_manifest["sha256"])
            if remote_config_manifest is not None and remote_config_manifest.get("sha256")
            else None
        ),
        local_config_manifest=local_config_manifest,
        remote_config_manifest=remote_config_manifest,
        stdout=stdout,
        stderr=stderr,
    )


def _job_static_metadata(
    job: GpuJob,
    files: DetachedJobFiles,
    *,
    expected_commit: str | None = None,
) -> dict[str, object]:
    local_commit = git_commit()
    return {
        "command": job.command,
        "output_dir": str(job.output_dir),
        "machine": job.machine,
        "gpu_id": job.gpu_id,
        "seed": job.seed,
        "metadata": job.metadata,
        "git_commit": local_commit,
        "local_git_commit": local_commit,
        "expected_git_commit": expected_commit,
        "files": files.record(),
    }


def render_detached_launch_script(
    job: GpuJob,
    *,
    expected_commit: str | None = None,
) -> str:
    files = detached_job_files(job)
    static_metadata = json.dumps(
        _job_static_metadata(job, files, expected_commit=expected_commit),
        sort_keys=True,
    )
    expected_commit_value = "" if expected_commit is None else expected_commit
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            "set +u",
            "set +o pipefail",
            f"OUT_DIR={quote(str(files.output_dir))}",
            f"COMMAND={quote(job.command)}",
            f"EXPECTED_GIT_COMMIT={quote(expected_commit_value)}",
            f"STATIC_METADATA={quote(static_metadata)}",
            "export OUT_DIR COMMAND EXPECTED_GIT_COMMIT STATIC_METADATA",
            'mkdir -p "$OUT_DIR"',
            'printf "%s\\n" "$$" > "$OUT_DIR/pid.txt"',
            "write_status() {",
            '  JOB_STATUS="$1" JOB_RETURN_CODE="$2" UPDATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
            "python3 - <<'PY'",
            "import json",
            "import os",
            "from pathlib import Path",
            "out_dir = Path(os.environ['OUT_DIR'])",
            "payload = json.loads(os.environ['STATIC_METADATA'])",
            "return_code = os.environ['JOB_RETURN_CODE']",
            "payload.update({",
            "    'status': os.environ['JOB_STATUS'],",
            "    'returncode': None if return_code == '' else int(return_code),",
            "    'updated_at': os.environ['UPDATED_AT'],",
            "})",
            "payload['remote_git_commit'] = os.environ.get('REMOTE_GIT_COMMIT') or None",
            "remote_git_clean = os.environ.get('REMOTE_GIT_CLEAN')",
            "if remote_git_clean is not None:",
            "    payload['remote_git_clean'] = remote_git_clean == 'true'",
            "prelaunch_error = os.environ.get('PRELAUNCH_ERROR')",
            "if prelaunch_error:",
            "    payload['prelaunch_error'] = prelaunch_error",
            "pid_path = out_dir / 'pid.txt'",
            "if pid_path.exists():",
            "    payload['pid'] = int(pid_path.read_text(encoding='utf-8').strip())",
            "status_path = out_dir / 'status.json'",
            "tmp_path = out_dir / f'status.json.tmp.{os.getpid()}'",
            "tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
            "os.replace(tmp_path, status_path)",
            "(out_dir / 'heartbeat.txt').write_text(os.environ['UPDATED_AT'] + '\\n', encoding='utf-8')",
            "PY",
            "}",
            "heartbeat_loop() {",
            "  while true; do",
            "    date -u +%Y-%m-%dT%H:%M:%SZ > \"$OUT_DIR/heartbeat.txt\"",
            "    sleep 30",
            "  done",
            "}",
            'REMOTE_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"',
            "export REMOTE_GIT_COMMIT",
            'if [ -n "$EXPECTED_GIT_COMMIT" ] && [ "$REMOTE_GIT_COMMIT" != "$EXPECTED_GIT_COMMIT" ]; then',
            '  PRELAUNCH_ERROR="remote commit/config mismatch: expected ${EXPECTED_GIT_COMMIT}, got ${REMOTE_GIT_COMMIT:-unknown}"',
            "  export PRELAUNCH_ERROR",
            '  printf "%s\\n" "$PRELAUNCH_ERROR" > "$OUT_DIR/stderr.log"',
            '  printf "%s\\n" "125" > "$OUT_DIR/exit_code.txt"',
            '  write_status "failed_infra" "125"',
            "  exit 125",
            "fi",
            "REMOTE_GIT_CLEAN=true",
            "if ! git diff --quiet 2>/dev/null; then",
            "  REMOTE_GIT_CLEAN=false",
            "  PRELAUNCH_ERROR='remote worktree has unstaged changes'",
            "elif ! git diff --cached --quiet 2>/dev/null; then",
            "  REMOTE_GIT_CLEAN=false",
            "  PRELAUNCH_ERROR='remote worktree has staged changes'",
            "elif [ -n \"$(git ls-files --others --exclude-standard 2>/dev/null)\" ]; then",
            "  REMOTE_GIT_CLEAN=false",
            "  PRELAUNCH_ERROR='remote worktree has untracked non-ignored files'",
            "fi",
            "export REMOTE_GIT_CLEAN PRELAUNCH_ERROR",
            'if [ "$REMOTE_GIT_CLEAN" != "true" ]; then',
            '  printf "%s\\n" "$PRELAUNCH_ERROR" > "$OUT_DIR/stderr.log"',
            '  printf "%s\\n" "125" > "$OUT_DIR/exit_code.txt"',
            '  write_status "failed_infra" "125"',
            "  exit 125",
            "fi",
            'write_status "running" ""',
            "heartbeat_loop &",
            "heartbeat_pid=$!",
            'bash -lc "$COMMAND" > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"',
            "rc=$?",
            'kill "$heartbeat_pid" >/dev/null 2>&1 || true',
            'wait "$heartbeat_pid" 2>/dev/null || true',
            'printf "%s\\n" "$rc" > "$OUT_DIR/exit_code.txt"',
            'if [ "$rc" -eq 0 ]; then',
            '  final_status="succeeded"',
            'elif [ "$rc" -eq 126 ] || [ "$rc" -eq 127 ] || [ "$rc" -ge 128 ]; then',
            '  final_status="failed_infra"',
            "else",
            '  final_status="failed_logic"',
            "fi",
            'write_status "$final_status" "$rc"',
            'exit "$rc"',
            "",
        ]
    )


def probe_detached_job_liveness(
    spec: MachineSpec,
    files: DetachedJobFiles,
    *,
    timeout_s: int = 20,
) -> dict[str, Any]:
    command = "\n".join(
        [
            "python3 - <<'PY'",
            "import json",
            "import os",
            "import sys",
            "from pathlib import Path",
            f"pid_path = Path({str(files.pid)!r})",
            f"heartbeat_path = Path({str(files.heartbeat)!r})",
            "if not pid_path.exists():",
            "    raise SystemExit(44)",
            "try:",
            "    pid = int(pid_path.read_text(encoding='utf-8').strip())",
            "except Exception:",
            "    raise SystemExit(44)",
            "try:",
            "    os.kill(pid, 0)",
            "except ProcessLookupError:",
            "    raise SystemExit(45)",
            "except PermissionError:",
            "    pass",
            "payload = {'pid': pid, 'heartbeat_exists': heartbeat_path.exists()}",
            "if heartbeat_path.exists():",
            "    payload['heartbeat_mtime'] = heartbeat_path.stat().st_mtime",
            "print(json.dumps(payload, sort_keys=True))",
            "PY",
        ]
    )
    return run_remote(spec, command, timeout_s=timeout_s)


def build_detached_launch_command(files: DetachedJobFiles) -> str:
    script = quote(str(files.script))
    status = quote(str(files.status))
    return (
        f"if command -v setsid >/dev/null 2>&1; then "
        f"setsid -f bash {script} >/dev/null 2>&1 </dev/null; "
        f"else nohup bash {script} >/dev/null 2>&1 </dev/null & fi; "
        f"for i in 1 2 3 4 5; do test -s {status} && break; sleep 0.2; done"
    )


def launch_detached_job(
    spec: MachineSpec,
    job: GpuJob,
    *,
    timeout_s: int = 20,
    expected_commit: str | None = None,
) -> LaunchResult:
    files = detached_job_files(job)
    script = render_detached_launch_script(job, expected_commit=expected_commit)
    write_result = write_remote_text(spec, files.script, script, executable=True, timeout_s=timeout_s)
    if not write_result["ok"]:
        job.status = JobStatus.FAILED_INFRA
        return LaunchResult(
            ok=False,
            status=job.status,
            job=job,
            files=files,
            launch_stdout=str(write_result["stdout"]),
            launch_stderr=str(write_result["stderr"]),
        )

    launch_command = build_detached_launch_command(files)
    launch_result = run_remote(spec, launch_command, timeout_s=timeout_s)
    status_probe = read_remote_text(spec, files.status, timeout_s=timeout_s)
    if status_probe["ok"]:
        try:
            payload = json.loads(str(status_probe["stdout"]))
            job.status = JobStatus(str(payload["status"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            job.status = JobStatus.RUNNING if launch_result["ok"] else JobStatus.FAILED_INFRA
    elif not launch_result["ok"]:
        job.status = JobStatus.FAILED_INFRA
    else:
        job.status = JobStatus.RUNNING
    return LaunchResult(
        ok=job.status != JobStatus.FAILED_INFRA,
        status=job.status,
        job=job,
        files=files,
        launch_stdout=str(launch_result["stdout"]),
        launch_stderr=str(launch_result["stderr"]),
    )


def read_detached_job_status(
    spec: MachineSpec,
    job: GpuJob,
    *,
    timeout_s: int = 20,
) -> dict[str, Any]:
    files = detached_job_files(job)
    result = read_remote_text(spec, files.status, timeout_s=timeout_s)
    if not result["ok"]:
        if job.status == JobStatus.RUNNING and result["returncode"] in {None, 44, 255}:
            return {
                "status": JobStatus.RUNNING,
                "returncode": None,
                "stderr": str(result["stderr"]) or "status file not written yet",
                "updated_at": _utc_now(),
                "files": files.record(),
            }
        job.status = JobStatus.FAILED_INFRA
        return {
            "status": JobStatus.FAILED_INFRA,
            "returncode": result["returncode"],
            "stderr": result["stderr"],
            "updated_at": _utc_now(),
            "files": files.record(),
        }
    try:
        payload = json.loads(str(result["stdout"]))
        status = JobStatus(str(payload["status"]))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        if job.status == JobStatus.RUNNING:
            return {
                "status": JobStatus.RUNNING,
                "returncode": None,
                "stderr": f"transient unreadable status: {type(exc).__name__}: {exc}",
                "updated_at": _utc_now(),
                "files": files.record(),
            }
        job.status = JobStatus.FAILED_INFRA
        return {
            "status": JobStatus.FAILED_INFRA,
            "returncode": None,
            "stderr": f"invalid terminal status payload: {type(exc).__name__}: {exc}",
            "updated_at": _utc_now(),
            "files": files.record(),
        }
    if status == JobStatus.RUNNING:
        liveness = probe_detached_job_liveness(spec, files, timeout_s=timeout_s)
        if not liveness["ok"]:
            returncode = liveness["returncode"]
            if returncode in {None, 255}:
                payload["stderr"] = str(liveness["stderr"]) or "transient liveness probe failure"
                job.status = JobStatus.RUNNING
                return payload
            job.status = JobStatus.FAILED_INFRA
            payload = dict(payload)
            payload.update(
                {
                    "status": JobStatus.FAILED_INFRA.value,
                    "returncode": returncode,
                    "stderr": str(liveness["stderr"]) or "detached job pid is not live",
                    "updated_at": _utc_now(),
                    "files": files.record(),
                }
            )
            return payload
        try:
            payload["liveness"] = json.loads(str(liveness["stdout"]) or "{}")
        except json.JSONDecodeError:
            payload["liveness"] = {"raw": str(liveness["stdout"])}
    job.status = status
    return payload


def launch_detached_jobs(
    machines: list[MachineSpec],
    jobs: list[GpuJob],
    *,
    timeout_s: int = 20,
    expected_commit: str | None = None,
) -> list[LaunchResult]:
    by_name = {machine.name: machine for machine in machines}
    results: list[LaunchResult] = []
    for job in jobs:
        if job.machine is None:
            raise ValueError("job.machine is required for detached launch")
        try:
            spec = by_name[job.machine]
        except KeyError as exc:
            raise ValueError(f"unknown job machine: {job.machine}") from exc
        results.append(
            launch_detached_job(
                spec,
                job,
                timeout_s=timeout_s,
                expected_commit=expected_commit,
            )
        )
    return results


def _list_remote_trial_dirs(
    spec: MachineSpec,
    output_dir: Path,
    *,
    timeout_s: int,
) -> dict[str, Any]:
    trials_dir = output_dir / "trials"
    command = "\n".join(
        [
            "python3 - <<'PY'",
            "import json",
            "from pathlib import Path",
            f"path = Path({str(trials_dir)!r})",
            "if not path.exists():",
            "    print('[]')",
            "elif not path.is_dir():",
            "    raise SystemExit(45)",
            "else:",
            "    print(json.dumps(sorted(",
            "        child.name for child in path.iterdir()",
            "        if child.is_dir() and child.name.startswith('trial_')",
            "    )))",
            "PY",
        ]
    )
    result = run_remote(spec, command, timeout_s=timeout_s)
    if not result["ok"]:
        return {
            "ok": False,
            "returncode": result["returncode"],
            "stderr": str(result["stderr"]),
            "trial_dirs": [],
        }
    try:
        trial_dirs = json.loads(str(result["stdout"]) or "[]")
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "returncode": result["returncode"],
            "stderr": f"invalid trial directory listing: {type(exc).__name__}: {exc}",
            "trial_dirs": [],
        }
    return {
        "ok": True,
        "returncode": result["returncode"],
        "stderr": str(result["stderr"]),
        "trial_dirs": [str(name) for name in trial_dirs],
    }


def _trial_dir_name_from_record(record: dict[str, Any]) -> str | None:
    output_dir = record.get("output_dir")
    if output_dir is not None:
        name = Path(str(output_dir)).name
        if name.startswith("trial_"):
            return name
    trial_index = record.get("trial_index")
    if trial_index is None:
        return None
    try:
        return f"trial_{int(trial_index):06d}"
    except (TypeError, ValueError):
        return None


def _expected_trial_dirs_from_summary(payload: dict[str, Any]) -> list[str]:
    trials = payload.get("trials")
    expected: set[str] = set()
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            name = _trial_dir_name_from_record(trial)
            if name is not None:
                expected.add(name)
    if expected:
        return sorted(expected)

    accepted_trials = payload.get("accepted_trials")
    if accepted_trials is None:
        return []
    try:
        count = int(accepted_trials)
        grid_offset = int(payload.get("grid_offset", 0))
    except (TypeError, ValueError):
        return []
    if count <= 0 or count > 100_000 or grid_offset < 0:
        return []
    return [f"trial_{grid_offset + index:06d}" for index in range(count)]


def _expected_trial_dirs_from_collected_summaries(files: dict[str, Any]) -> list[str]:
    expected: set[str] = set()
    for artifact_name in (
        "teacher_hpo_summary.json",
        "distill_hpo_summary.json",
        "finetune_hpo_summary.json",
    ):
        record = files.get(artifact_name)
        if not isinstance(record, dict) or not record.get("ok") or record.get("truncated"):
            continue
        local_path = record.get("local_path")
        if local_path is None:
            continue
        try:
            payload = json.loads(Path(str(local_path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            expected.update(_expected_trial_dirs_from_summary(payload))
    return sorted(expected)


def _is_metrics_artifact(artifact_name: str) -> bool:
    path = Path(artifact_name)
    return "metrics" in path.name or path.name in {
        "teacher_hpo_events.jsonl",
        "layer_metrics.jsonl",
    }


def _collect_one_artifact(
    spec: MachineSpec,
    source_path: Path,
    artifact_name: str,
    *,
    destination: Path,
    timeout_s: int,
    max_bytes: int,
) -> dict[str, Any]:
    result = read_remote_text(
        spec,
        source_path,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
    )
    record: dict[str, Any] = {
        "source": str(source_path),
        "ok": bool(result["ok"]),
        "returncode": result["returncode"],
        "stderr": str(result["stderr"]),
        "remote_size_bytes": result.get("remote_size_bytes"),
        "copied_size_bytes": result.get("copied_size_bytes"),
        "local_size_bytes": None,
        "truncated": bool(result.get("truncated")) if result.get("truncated") is not None else None,
    }
    if result["ok"]:
        target = destination / artifact_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(result["stdout"]), encoding="utf-8")
        record["local_path"] = str(target)
        record["local_size_bytes"] = target.stat().st_size
    return record


def collect_detached_job_artifacts(
    spec: MachineSpec,
    job: GpuJob,
    *,
    local_root: Path,
    timeout_s: int = 20,
    max_bytes: int = 1_048_576,
) -> dict[str, Any]:
    files = detached_job_files(job)
    destination = local_root / str(job.machine) / str(job.gpu_id)
    destination.mkdir(parents=True, exist_ok=True)
    collected: dict[str, Any] = {
        "status": "artifacts_collected",
        "machine": job.machine,
        "gpu_id": job.gpu_id,
        "source_output_dir": str(files.output_dir),
        "local_output_dir": str(destination),
        "files": {},
        "trial_dirs": {
            "remote": [],
            "expected": [],
            "missing": [],
            "listing_ok": None,
            "listing_stderr": "",
        },
        "artifact_integrity": {
            "truncated": False,
            "truncated_files": [],
            "metrics_truncated_files": [],
            "missing_files": [],
        },
    }
    for artifact_name in ROOT_COLLECTED_ARTIFACT_NAMES:
        relative_path = files.output_dir / artifact_name
        collected["files"][artifact_name] = _collect_one_artifact(
            spec,
            relative_path,
            artifact_name,
            destination=destination,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
        )

    listing = _list_remote_trial_dirs(spec, files.output_dir, timeout_s=timeout_s)
    remote_trial_dirs = sorted(set(str(name) for name in listing["trial_dirs"]))
    expected_trial_dirs = _expected_trial_dirs_from_collected_summaries(collected["files"])
    trial_dirs = sorted(set(remote_trial_dirs) | set(expected_trial_dirs))
    missing_trial_dirs = sorted(set(expected_trial_dirs) - set(remote_trial_dirs))
    collected["trial_dirs"] = {
        "remote": remote_trial_dirs,
        "expected": expected_trial_dirs,
        "missing": missing_trial_dirs,
        "listing_ok": bool(listing["ok"]),
        "listing_stderr": str(listing["stderr"]),
    }

    for trial_dir in trial_dirs:
        for artifact_name in TRIAL_COLLECTED_ARTIFACT_NAMES:
            relative_name = f"trials/{trial_dir}/{artifact_name}"
            relative_path = files.output_dir / relative_name
            collected["files"][relative_name] = _collect_one_artifact(
                spec,
                relative_path,
                relative_name,
                destination=destination,
                timeout_s=timeout_s,
                max_bytes=max_bytes,
            )

    truncated_files = sorted(
        name
        for name, record in collected["files"].items()
        if isinstance(record, dict) and bool(record.get("truncated"))
    )
    metrics_truncated_files = [name for name in truncated_files if _is_metrics_artifact(name)]
    missing_files = sorted(
        name
        for name, record in collected["files"].items()
        if isinstance(record, dict) and not bool(record.get("ok")) and record.get("returncode") == 44
    )
    if metrics_truncated_files:
        collected["status"] = "artifacts_collected_truncated_metrics"
    elif truncated_files:
        collected["status"] = "artifacts_collected_truncated"
    collected["artifact_integrity"] = {
        "truncated": bool(truncated_files),
        "truncated_files": truncated_files,
        "metrics_truncated_files": metrics_truncated_files,
        "missing_files": missing_files,
    }
    return collected


def resolve_collect_root(collect_root: str | Path | None, *, run_id: str | None) -> Path:
    if collect_root is not None:
        return Path(collect_root)
    base = Path("outputs/scheduler_collected")
    if run_id is None:
        return base
    return base / run_id


def wait_for_jobs(
    machines: list[MachineSpec],
    jobs: list[GpuJob],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> list[dict[str, Any]]:
    by_name = {machine.name: machine for machine in machines}
    deadline = time.monotonic() + timeout_s
    latest: dict[tuple[str | None, int | None], dict[str, Any]] = {}
    terminal = {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED_INFRA,
        JobStatus.FAILED_LOGIC,
        JobStatus.CANCELLED,
    }
    while True:
        all_terminal = True
        for job in jobs:
            if job.machine is None:
                raise ValueError("job.machine is required for status polling")
            key = (job.machine, job.gpu_id)
            if job.status in terminal:
                continue
            status = read_detached_job_status(by_name[job.machine], job)
            latest[key] = status
            if JobStatus(str(status["status"])) not in terminal:
                all_terminal = False
        if all_terminal or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)
    return [latest.get((job.machine, job.gpu_id), job.record()) for job in jobs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="configs/machines.yaml")
    parser.add_argument("--queue-out", default="outputs/job_queue.jsonl")
    parser.add_argument("--quick-smoke", type=bool_arg, default=True)
    parser.add_argument(
        "--smoke-mode",
        choices=("metadata", "train"),
        default=DEFAULT_SMOKE_MODE,
        help="Use metadata for scheduler readiness; train requires CIFAR/CUDA smoke gate.",
    )
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--job-kind",
        choices=SCHEDULER_JOB_KINDS,
        default=DEFAULT_JOB_KIND,
        help=(
            "Launch teacher train/smoke, one-GPU teacher HPO, one-GPU distill HPO, "
            "one-GPU fine-tune HPO, or manifest-driven student final-test jobs."
        ),
    )
    parser.add_argument("--teacher-base-config", default="configs/teacher_default.yaml")
    parser.add_argument("--teacher-hpo-config", default="configs/teacher_hpo.yaml")
    parser.add_argument("--distill-teacher-checkpoint", default=None)
    parser.add_argument("--distill-base-config", default="configs/fff_distill_default.yaml")
    parser.add_argument("--distill-hpo-config", default="configs/fff_distill_hpo.yaml")
    parser.add_argument("--finetune-base-config", default="configs/finetune_default.yaml")
    parser.add_argument("--finetune-hpo-config", default="configs/finetune_hpo.yaml")
    parser.add_argument(
        "--distill-sample-split",
        choices=("train", "train_eval", "val"),
        default="train_eval",
        help="CIFAR split used to sample teacher Linear activations for distillation HPO.",
    )
    parser.add_argument("--distill-max-sample-batches", type=int, default=1)
    parser.add_argument(
        "--distill-grid-offset-base",
        type=int,
        default=0,
        help=(
            "Base grid offset for distill_hpo jobs. Each queued job adds its zero-based "
            "queued-job index so parallel grid sweeps cover distinct candidates."
        ),
    )
    parser.add_argument(
        "--finetune-grid-offset-base",
        type=int,
        default=0,
        help=(
            "Base grid offset for finetune_hpo jobs. Each queued job adds its zero-based "
            "queued-job index so parallel fine-tune grids cover distinct candidates."
        ),
    )
    parser.add_argument("--hpo-trials-per-job", type=int, default=1)
    parser.add_argument("--hpo-max-attempts-per-job", type=int, default=32)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1337,
        help="Base seed added to enumerated slot index. Change this for each HPO wave.",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--student-final-manifest", default=None)
    parser.add_argument("--student-final-max-test-steps", type=int, default=None)
    parser.add_argument("--student-final-batch-size", type=int, default=None)
    parser.add_argument("--student-final-num-workers", type=int, default=None)
    parser.add_argument("--student-final-min-selected-val-accuracy", type=float, default=0.90)
    parser.add_argument("--student-final-allow-below-target", type=bool_arg, default=False)
    parser.add_argument("--prune-min-value", type=float, default=None)
    parser.add_argument(
        "--job-output-root",
        default=None,
        help="Override scheduler output root. Defaults depend on --job-kind.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id inserted under the scheduler output root to avoid output collisions.",
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--unavailable-slot",
        action="append",
        default=[],
        help="Exclude a currently occupied GPU slot from queued jobs, as machine:gpu_id. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        type=bool_arg,
        default=True,
        help="Only materialize queue metadata.",
    )
    parser.add_argument(
        "--allow-long-jobs",
        type=bool_arg,
        default=False,
        help="Permit non-smoke launches. Keep false until environment, profiler, and GPU gates pass.",
    )
    parser.add_argument("--wait", type=bool_arg, default=True)
    parser.add_argument("--launch-timeout-s", type=int, default=20)
    parser.add_argument("--wait-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--preflight", type=bool_arg, default=True)
    parser.add_argument(
        "--expected-commit",
        default=git_commit(),
        help="Require remote workdirs to match this git commit before launch. Use 'any' to disable.",
    )
    parser.add_argument(
        "--collect-root",
        default=None,
        help=(
            "Local directory where small logs/status/metrics are copied from each launched job. "
            "Defaults to outputs/scheduler_collected/<run-id> when --run-id is set, otherwise "
            "outputs/scheduler_collected."
        ),
    )
    parser.add_argument(
        "--launch-results-out",
        default="outputs/scheduler_launch_results.jsonl",
        help="JSONL launch/status records for detached launches.",
    )
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive when set")
    if args.hpo_trials_per_job <= 0:
        raise ValueError("--hpo-trials-per-job must be positive")
    if args.hpo_max_attempts_per_job < args.hpo_trials_per_job:
        raise ValueError("--hpo-max-attempts-per-job must be >= --hpo-trials-per-job")
    if args.distill_max_sample_batches <= 0:
        raise ValueError("--distill-max-sample-batches must be positive")
    if args.distill_grid_offset_base < 0:
        raise ValueError("--distill-grid-offset-base must be non-negative")
    if args.job_kind == "distill_hpo" and not args.quick_smoke and args.distill_teacher_checkpoint is None:
        raise RuntimeError("--distill-teacher-checkpoint is required for non-smoke distill_hpo")
    if not args.dry_run and not args.quick_smoke and not args.allow_long_jobs:
        raise RuntimeError("refusing non-smoke scheduler launch without --allow-long-jobs true")
    if not args.dry_run and not _git_worktree_clean():
        raise RuntimeError("refusing scheduler launch from a dirty local git worktree")
    collect_root = resolve_collect_root(args.collect_root, run_id=args.run_id)

    machines = load_machines(args.machines)
    unavailable_slots = parse_unavailable_slots(args.unavailable_slot)
    if args.job_kind == "student_final":
        jobs = build_student_final_jobs(
            machines,
            manifest=args.student_final_manifest,
            quick_smoke=args.quick_smoke,
            dry_run=args.dry_run,
            python_bin=args.python_bin,
            output_root=Path(args.job_output_root) if args.job_output_root is not None else None,
            run_id=args.run_id,
            unavailable_slots=unavailable_slots,
            max_jobs=args.max_jobs,
            max_test_steps=args.student_final_max_test_steps,
            batch_size=args.student_final_batch_size,
            num_workers=args.student_final_num_workers,
            min_selected_val_accuracy=args.student_final_min_selected_val_accuracy,
            allow_below_target=args.student_final_allow_below_target,
        )
    else:
        jobs = build_dry_run_jobs(
            machines,
            quick_smoke=args.quick_smoke,
            dry_run=args.dry_run,
            python_bin=args.python_bin,
            smoke_mode=args.smoke_mode,
            job_kind=args.job_kind,
            teacher_base_config=args.teacher_base_config,
            teacher_hpo_config=args.teacher_hpo_config,
            distill_base_config=args.distill_base_config,
            distill_hpo_config=args.distill_hpo_config,
            finetune_base_config=args.finetune_base_config,
            finetune_hpo_config=args.finetune_hpo_config,
            distill_teacher_checkpoint=args.distill_teacher_checkpoint,
            distill_sample_split=args.distill_sample_split,
            distill_max_sample_batches=args.distill_max_sample_batches,
            distill_grid_offset_base=args.distill_grid_offset_base,
            finetune_grid_offset_base=args.finetune_grid_offset_base,
            hpo_trials_per_job=args.hpo_trials_per_job,
            hpo_max_attempts_per_job=args.hpo_max_attempts_per_job,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            prune_min_value=args.prune_min_value,
            output_root=Path(args.job_output_root) if args.job_output_root is not None else None,
            run_id=args.run_id,
            unavailable_slots=unavailable_slots,
            max_jobs=args.max_jobs,
            seed_base=args.seed_base,
        )
    write_queue(Path(args.queue_out), jobs)
    print(f"recorded {len(jobs)} GPU slots in {args.queue_out}")
    if not args.dry_run:
        launchable_jobs = list(jobs)
        skipped_results: list[dict[str, Any]] = []
        preflight_records_by_machine: dict[str, dict[str, object]] = {}
        expected_launch_commit = None if args.expected_commit == "any" else str(args.expected_commit)
        config_manifest = build_config_manifest(config_manifest_paths_from_jobs(jobs))
        if args.preflight:
            specs_by_name = {machine.name: machine for machine in machines}
            machines_with_jobs = sorted({str(job.machine) for job in jobs if job.machine is not None})
            preflight_by_machine = {
                machine_name: preflight_machine(
                    specs_by_name[machine_name],
                    python_bin=args.python_bin,
                    expected_commit=expected_launch_commit,
                    expected_config_manifest=config_manifest,
                    require_cifar10_train=args.smoke_mode == "train"
                    or args.job_kind
                    in {"teacher_hpo", "distill_hpo", "finetune_hpo", "student_final"},
                    timeout_s=args.launch_timeout_s,
                )
                for machine_name in machines_with_jobs
            }
            preflight_records_by_machine = {
                machine_name: preflight.record()
                for machine_name, preflight in preflight_by_machine.items()
            }
            launchable_jobs = []
            for job in jobs:
                if job.machine is None:
                    raise ValueError("job.machine is required for detached launch")
                preflight = preflight_by_machine[str(job.machine)]
                if preflight.ok:
                    launchable_jobs.append(job)
                    continue
                job.status = JobStatus.FAILED_INFRA
                skipped_results.append(
                    {
                        "ok": False,
                        "status": JobStatus.FAILED_INFRA,
                        "machine": job.machine,
                        "gpu_id": job.gpu_id,
                        "output_dir": str(job.output_dir),
                        "preflight": preflight.record(),
                    }
                )
        launch_results = launch_detached_jobs(
            machines,
            launchable_jobs,
            timeout_s=args.launch_timeout_s,
            expected_commit=expected_launch_commit,
        )
        Path(args.launch_results_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.launch_results_out).open("w", encoding="utf-8") as handle:
            for skipped in skipped_results:
                handle.write(json.dumps(skipped, sort_keys=True) + "\n")
            for result in launch_results:
                handle.write(
                    json.dumps(
                        {
                            "ok": result.ok,
                            "status": result.status,
                            "machine": result.job.machine,
                            "gpu_id": result.job.gpu_id,
                            "output_dir": str(result.job.output_dir),
                            "launch_stdout": result.launch_stdout.strip(),
                            "launch_stderr": result.launch_stderr.strip(),
                            "preflight": preflight_records_by_machine.get(
                                str(result.job.machine)
                            ),
                            "files": result.files.record(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if args.wait:
                for status in wait_for_jobs(
                    machines,
                    launchable_jobs,
                    timeout_s=args.wait_timeout_s,
                    poll_interval_s=args.poll_interval_s,
                ):
                    handle.write(json.dumps(status, sort_keys=True) + "\n")
                specs_by_name = {machine.name: machine for machine in machines}
                for job in launchable_jobs:
                    artifacts = collect_detached_job_artifacts(
                        specs_by_name[str(job.machine)],
                        job,
                        local_root=collect_root,
                        timeout_s=args.launch_timeout_s,
                    )
                    handle.write(
                        json.dumps(
                            {"status": "artifacts_collected", **artifacts},
                            sort_keys=True,
                        )
                        + "\n"
                    )
        write_queue(Path(args.queue_out), jobs)
        print(f"launch/status records written to {args.launch_results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
