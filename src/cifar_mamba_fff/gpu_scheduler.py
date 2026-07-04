from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from shlex import quote

from .cluster import MachineSpec, load_machines
from .utils import bool_arg, git_commit

DEFAULT_PYTHON_BIN = ".venv/bin/python"


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
) -> str:
    return (
        f"PYTHONPATH=src CUDA_VISIBLE_DEVICES={gpu_id} {quote(python_bin)} "
        "-m cifar_mamba_fff.train_teacher "
        f"--quick-smoke {str(quick_smoke).lower()} "
        f"--output-dir {quote(str(output_dir))}"
    )


def build_dry_run_jobs(
    machines: list[MachineSpec],
    *,
    quick_smoke: bool,
    dry_run: bool,
    python_bin: str = DEFAULT_PYTHON_BIN,
    unavailable_slots: set[tuple[str, int]] | None = None,
) -> list[GpuJob]:
    unavailable_slots = unavailable_slots or set()
    jobs: list[GpuJob] = []
    for idx, (machine, gpu_id) in enumerate(enumerate_slots(machines)):
        if (machine, gpu_id) in unavailable_slots:
            continue
        output_dir = Path("outputs/scheduler_smoke") / machine / str(gpu_id)
        jobs.append(
            GpuJob(
                command=build_train_teacher_command(
                    gpu_id=gpu_id,
                    output_dir=output_dir,
                    python_bin=python_bin,
                    quick_smoke=quick_smoke,
                ),
                output_dir=output_dir,
                machine=machine,
                gpu_id=gpu_id,
                seed=1337 + idx,
                metadata={
                    "quick_smoke": quick_smoke,
                    "dry_run": dry_run,
                    "cuda_visible_devices": str(gpu_id),
                    "python_bin": python_bin,
                    "output_dir": str(output_dir),
                },
            )
        )
    return jobs


def write_queue(path: Path, jobs: list[GpuJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.record(), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="configs/machines.yaml")
    parser.add_argument("--queue-out", default="outputs/job_queue.jsonl")
    parser.add_argument("--quick-smoke", type=bool_arg, default=True)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
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
        help="Only materialize queue metadata. Real remote launch is implemented after env gates pass.",
    )
    args = parser.parse_args()

    machines = load_machines(args.machines)
    unavailable_slots = parse_unavailable_slots(args.unavailable_slot)
    jobs = build_dry_run_jobs(
        machines,
        quick_smoke=args.quick_smoke,
        dry_run=args.dry_run,
        python_bin=args.python_bin,
        unavailable_slots=unavailable_slots,
    )
    write_queue(Path(args.queue_out), jobs)
    print(f"recorded {len(jobs)} GPU slots in {args.queue_out}")
    if not args.dry_run:
        raise RuntimeError("real scheduler launch is blocked until verify_env and verify_cluster pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
