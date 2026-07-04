from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from shlex import quote
from typing import Any

from .cluster import MachineSpec, load_machines, read_remote_text, run_remote, write_remote_text
from .utils import bool_arg, git_commit

DEFAULT_PYTHON_BIN = ".venv/bin/python"
DEFAULT_SMOKE_MODE = "metadata"
DEFAULT_JOB_KIND = "teacher_train"
COLLECTED_ARTIFACT_NAMES = (
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
    "trials/trial_000000/trial_config.json",
    "trials/trial_000000/trial_summary.json",
    "trials/trial_000000/run_context.json",
    "trials/trial_000000/metrics_summary.json",
    "trials/trial_000000/metrics.jsonl",
)


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
        return {
            "command": self.command,
            "output_dir": str(self.output_dir),
            "machine": self.machine,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "status": self.status,
            "metadata": self.metadata,
            "git_commit": git_commit(),
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
    stdout: str
    stderr: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _relative_path(path: Path, *, field_name: str) -> Path:
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative to the repository workdir")
    return path


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
    hpo_trials_per_job: int = 1,
    hpo_max_attempts_per_job: int = 32,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    prune_min_value: float | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    unavailable_slots: set[tuple[str, int]] | None = None,
    max_jobs: int | None = None,
) -> list[GpuJob]:
    if job_kind not in {"teacher_train", "teacher_hpo"}:
        raise ValueError("job_kind must be teacher_train or teacher_hpo")
    unavailable_slots = unavailable_slots or set()
    jobs: list[GpuJob] = []
    for idx, (machine, gpu_id) in enumerate(enumerate_slots(machines)):
        if (machine, gpu_id) in unavailable_slots:
            continue
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
        seed = 1337 + idx
        default_root = Path("outputs/scheduler_hpo" if job_kind == "teacher_hpo" else "outputs/scheduler_smoke")
        resolved_output_root = output_root or default_root
        if run_id is not None:
            resolved_output_root = resolved_output_root / run_id
        output_dir = resolved_output_root / machine / str(gpu_id)
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
                },
            )
        )
    _validate_jobs_unique(jobs)
    return jobs


def _validate_jobs_unique(jobs: list[GpuJob]) -> None:
    output_dirs: set[Path] = set()
    seeds: set[int] = set()
    for job in jobs:
        if job.output_dir in output_dirs:
            raise ValueError(f"duplicate job output_dir: {job.output_dir}")
        output_dirs.add(job.output_dir)
        if job.seed is None:
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
    require_cifar10_train: bool = False,
    data_dir: str | Path = "data/cifar10",
    timeout_s: int = 20,
) -> PreflightResult:
    commands = [
        "test -d .git",
        "git rev-parse HEAD",
        f"test -x {quote(python_bin)}",
        f"{quote(python_bin)} --version",
    ]
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
    ok = bool(result["ok"])
    stderr = str(result["stderr"])
    if expected_commit is not None and commit != expected_commit:
        ok = False
        stderr = (
            stderr.rstrip()
            + f"\nremote commit mismatch: expected {expected_commit}, got {commit}"
        ).strip()
    return PreflightResult(
        machine=spec.name,
        ok=ok,
        returncode=result["returncode"],
        commit=commit,
        stdout=stdout,
        stderr=stderr,
    )


def _job_static_metadata(job: GpuJob, files: DetachedJobFiles) -> dict[str, object]:
    return {
        "command": job.command,
        "output_dir": str(job.output_dir),
        "machine": job.machine,
        "gpu_id": job.gpu_id,
        "seed": job.seed,
        "metadata": job.metadata,
        "git_commit": git_commit(),
        "files": files.record(),
    }


def render_detached_launch_script(job: GpuJob) -> str:
    files = detached_job_files(job)
    static_metadata = json.dumps(_job_static_metadata(job, files), sort_keys=True)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            "set +u",
            "set +o pipefail",
            f"OUT_DIR={quote(str(files.output_dir))}",
            f"COMMAND={quote(job.command)}",
            f"STATIC_METADATA={quote(static_metadata)}",
            "export OUT_DIR STATIC_METADATA",
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


def launch_detached_job(spec: MachineSpec, job: GpuJob, *, timeout_s: int = 20) -> LaunchResult:
    files = detached_job_files(job)
    script = render_detached_launch_script(job)
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
    if not launch_result["ok"]:
        status_probe = read_remote_text(spec, files.status, timeout_s=timeout_s)
        if status_probe["ok"]:
            try:
                payload = json.loads(str(status_probe["stdout"]))
                job.status = JobStatus(str(payload["status"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                job.status = JobStatus.RUNNING
        else:
            job.status = JobStatus.FAILED_INFRA
    else:
        job.status = JobStatus.RUNNING
    return LaunchResult(
        ok=bool(launch_result["ok"]) or job.status != JobStatus.FAILED_INFRA,
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
        results.append(launch_detached_job(spec, job, timeout_s=timeout_s))
    return results


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
        "machine": job.machine,
        "gpu_id": job.gpu_id,
        "source_output_dir": str(files.output_dir),
        "local_output_dir": str(destination),
        "files": {},
    }
    for artifact_name in COLLECTED_ARTIFACT_NAMES:
        relative_path = files.output_dir / artifact_name
        result = read_remote_text(
            spec,
            relative_path,
            max_bytes=max_bytes,
            timeout_s=timeout_s,
        )
        record = {
            "source": str(relative_path),
            "ok": bool(result["ok"]),
            "returncode": result["returncode"],
            "stderr": str(result["stderr"]),
        }
        if result["ok"]:
            target = destination / artifact_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(result["stdout"]), encoding="utf-8")
            record["local_path"] = str(target)
        collected["files"][artifact_name] = record
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
        choices=("teacher_train", "teacher_hpo"),
        default=DEFAULT_JOB_KIND,
        help="Launch teacher train/smoke commands or one-GPU teacher HPO jobs.",
    )
    parser.add_argument("--teacher-base-config", default="configs/teacher_default.yaml")
    parser.add_argument("--teacher-hpo-config", default="configs/teacher_hpo.yaml")
    parser.add_argument("--hpo-trials-per-job", type=int, default=1)
    parser.add_argument("--hpo-max-attempts-per-job", type=int, default=32)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
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
    if not args.dry_run and not args.quick_smoke and not args.allow_long_jobs:
        raise RuntimeError("refusing non-smoke scheduler launch without --allow-long-jobs true")
    collect_root = resolve_collect_root(args.collect_root, run_id=args.run_id)

    machines = load_machines(args.machines)
    unavailable_slots = parse_unavailable_slots(args.unavailable_slot)
    jobs = build_dry_run_jobs(
        machines,
        quick_smoke=args.quick_smoke,
        dry_run=args.dry_run,
        python_bin=args.python_bin,
        smoke_mode=args.smoke_mode,
        job_kind=args.job_kind,
        teacher_base_config=args.teacher_base_config,
        teacher_hpo_config=args.teacher_hpo_config,
        hpo_trials_per_job=args.hpo_trials_per_job,
        hpo_max_attempts_per_job=args.hpo_max_attempts_per_job,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        prune_min_value=args.prune_min_value,
        output_root=Path(args.job_output_root) if args.job_output_root is not None else None,
        run_id=args.run_id,
        unavailable_slots=unavailable_slots,
        max_jobs=args.max_jobs,
    )
    write_queue(Path(args.queue_out), jobs)
    print(f"recorded {len(jobs)} GPU slots in {args.queue_out}")
    if not args.dry_run:
        launchable_jobs = list(jobs)
        skipped_results: list[dict[str, Any]] = []
        if args.preflight:
            specs_by_name = {machine.name: machine for machine in machines}
            machines_with_jobs = sorted({str(job.machine) for job in jobs if job.machine is not None})
            preflight_by_machine = {
                machine_name: preflight_machine(
                    specs_by_name[machine_name],
                    python_bin=args.python_bin,
                    expected_commit=None
                    if args.expected_commit == "any"
                    else str(args.expected_commit),
                    require_cifar10_train=args.smoke_mode == "train"
                    or args.job_kind == "teacher_hpo",
                    timeout_s=args.launch_timeout_s,
                )
                for machine_name in machines_with_jobs
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
                        "preflight": {
                            "returncode": preflight.returncode,
                            "commit": preflight.commit,
                            "stdout": preflight.stdout.strip(),
                            "stderr": preflight.stderr.strip(),
                        },
                    }
                )
        launch_results = launch_detached_jobs(
            machines,
            launchable_jobs,
            timeout_s=args.launch_timeout_s,
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
