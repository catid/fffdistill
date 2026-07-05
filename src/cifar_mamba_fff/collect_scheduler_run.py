from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from shlex import quote
from typing import Any

from .cluster import MachineSpec, load_machines, read_remote_text, run_remote
from .gpu_scheduler import (
    GpuJob,
    JobStatus,
    collect_detached_job_artifacts,
    resolve_collect_root,
)
from .utils import bool_arg, write_json

TERMINAL_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED_INFRA,
    JobStatus.FAILED_LOGIC,
    JobStatus.CANCELLED,
}


def default_scheduler_output_root(job_kind: str) -> Path:
    if job_kind == "teacher_hpo":
        return Path("outputs/scheduler_hpo")
    if job_kind == "distill_hpo":
        return Path("outputs/scheduler_distill_hpo")
    if job_kind == "finetune_hpo":
        return Path("outputs/scheduler_finetune_hpo")
    if job_kind == "student_final":
        return Path("outputs/scheduler_student_final")
    if job_kind == "smoke":
        return Path("outputs/scheduler_smoke")
    raise ValueError(
        "job_kind must be one of: teacher_hpo, distill_hpo, finetune_hpo, "
        "student_final, smoke"
    )


def scheduler_run_root(*, job_kind: str, run_id: str, output_root: Path | None) -> Path:
    root = output_root or default_scheduler_output_root(job_kind)
    return root / run_id


def _status_paths(spec: MachineSpec, run_root: Path, *, timeout_s: int) -> list[Path]:
    command = (
        f"if test -d {quote(str(run_root))}; then "
        f"find {quote(str(run_root))} -mindepth 3 -maxdepth 3 -name status.json -print; "
        "fi"
    )
    result = run_remote(spec, command, timeout_s=timeout_s)
    if not result["ok"]:
        stderr = str(result.get("stderr", "")).strip()
        stdout = str(result.get("stdout", "")).strip()
        detail = stderr or stdout or f"returncode={result.get('returncode')}"
        raise RuntimeError(
            f"could not discover scheduler statuses on {spec.name}:{run_root}: {detail}"
        )
    return [Path(line.strip()) for line in str(result["stdout"]).splitlines() if line.strip()]


def _status_payload(spec: MachineSpec, status_path: Path, *, timeout_s: int) -> dict[str, Any]:
    result = read_remote_text(spec, status_path, timeout_s=timeout_s)
    if not result["ok"]:
        raise RuntimeError(
            f"could not read {spec.name}:{status_path}: {result['stderr']}"
        )
    loaded = json.loads(str(result["stdout"]))
    if not isinstance(loaded, dict):
        raise ValueError(f"{spec.name}:{status_path} did not contain a JSON object")
    return loaded


def _job_from_status(status_path: Path, payload: dict[str, Any], *, job_kind: str) -> GpuJob:
    status_text = str(payload.get("status", JobStatus.RUNNING.value))
    try:
        status = JobStatus(status_text)
    except ValueError:
        status = JobStatus.FAILED_INFRA
    output_dir = status_path.parent
    payload_output_dir = payload.get("output_dir")
    if payload_output_dir not in (None, "") and Path(str(payload_output_dir)) != output_dir:
        raise ValueError(
            f"{status_path} has output_dir={payload_output_dir!r}, expected {str(output_dir)!r}"
        )
    expected_machine = status_path.parent.parent.name
    payload_machine = payload.get("machine")
    if payload_machine not in (None, "") and str(payload_machine) != expected_machine:
        raise ValueError(
            f"{status_path} has machine={payload_machine!r}, expected {expected_machine!r}"
        )
    expected_gpu_raw = status_path.parent.name
    expected_gpu_id = int(expected_gpu_raw) if expected_gpu_raw.isdecimal() else None
    gpu_raw = payload.get("gpu_id", payload.get("gpu"))
    gpu_id = int(gpu_raw) if gpu_raw not in (None, "") else None
    if gpu_id != expected_gpu_id:
        raise ValueError(f"{status_path} has gpu_id={gpu_id!r}, expected {expected_gpu_id!r}")
    seed_raw = payload.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "") else None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload_job_kind = metadata.get("job_kind")
    if payload_job_kind not in (None, "") and str(payload_job_kind) != job_kind:
        raise ValueError(
            f"{status_path} has metadata.job_kind={payload_job_kind!r}, expected {job_kind!r}"
        )
    return GpuJob(
        command=str(payload.get("command", "")),
        output_dir=output_dir,
        machine=expected_machine,
        gpu_id=gpu_id,
        seed=seed,
        status=status,
        metadata=dict(metadata),
    )


def discover_scheduler_jobs(
    machines: Sequence[MachineSpec],
    *,
    job_kind: str,
    run_id: str,
    output_root: Path | None = None,
    timeout_s: int = 20,
) -> list[tuple[MachineSpec, GpuJob]]:
    run_root = scheduler_run_root(job_kind=job_kind, run_id=run_id, output_root=output_root)
    discovered: list[tuple[MachineSpec, GpuJob]] = []
    seen: set[tuple[str, int | None, str]] = set()
    for spec in machines:
        for status_path in _status_paths(spec, run_root, timeout_s=timeout_s):
            payload = _status_payload(spec, status_path, timeout_s=timeout_s)
            job = _job_from_status(status_path, payload, job_kind=job_kind)
            key = (str(job.machine), job.gpu_id, str(job.output_dir))
            if key in seen:
                continue
            seen.add(key)
            discovered.append((spec, job))
    return discovered


def collect_scheduler_run(
    machines: Sequence[MachineSpec],
    *,
    job_kind: str,
    run_id: str,
    output_root: Path | None = None,
    collect_root: Path | None = None,
    include_running: bool = False,
    timeout_s: int = 20,
    max_bytes: int = 1_048_576,
) -> dict[str, Any]:
    destination = resolve_collect_root(collect_root, run_id=run_id)
    jobs = discover_scheduler_jobs(
        machines,
        job_kind=job_kind,
        run_id=run_id,
        output_root=output_root,
        timeout_s=timeout_s,
    )
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for spec, job in jobs:
        if not include_running and job.status not in TERMINAL_STATUSES:
            skipped.append(
                {
                    "machine": job.machine,
                    "gpu_id": job.gpu_id,
                    "output_dir": str(job.output_dir),
                    "status": job.status.value,
                    "reason": "non-terminal",
                }
            )
            continue
        records.append(
            collect_detached_job_artifacts(
                spec,
                job,
                local_root=destination,
                timeout_s=timeout_s,
                max_bytes=max_bytes,
            )
        )
    summary: dict[str, Any] = {
        "run_id": run_id,
        "job_kind": job_kind,
        "collect_root": str(destination),
        "output_root": str(output_root) if output_root is not None else "",
        "discovered_jobs": len(jobs),
        "collected_jobs": len(records),
        "skipped_jobs": len(skipped),
        "include_running": include_running,
        "artifacts": records,
        "skipped": skipped,
    }
    write_json(destination / "collection_manifest.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="configs/machines.yaml")
    parser.add_argument(
        "--job-kind",
        choices=("teacher_hpo", "distill_hpo", "finetune_hpo", "student_final", "smoke"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-output-root", default=None)
    parser.add_argument("--collect-root", default=None)
    parser.add_argument("--include-running", type=bool_arg, default=False)
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    parser.add_argument("--expect-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")
    machines = load_machines(args.machines)
    summary = collect_scheduler_run(
        machines,
        job_kind=args.job_kind,
        run_id=args.run_id,
        output_root=Path(args.job_output_root) if args.job_output_root else None,
        collect_root=Path(args.collect_root) if args.collect_root else None,
        include_running=args.include_running,
        timeout_s=args.timeout_s,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    if args.expect_jobs is not None and int(summary["collected_jobs"]) != args.expect_jobs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
