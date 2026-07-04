from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .cluster import MachineSpec, load_machines
from .utils import bool_arg, git_commit


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


def write_queue(path: Path, jobs: list[GpuJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.record(), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", default="configs/machines.yaml")
    parser.add_argument("--queue-out", default="outputs/job_queue.jsonl")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument(
        "--dry-run",
        type=bool_arg,
        default=True,
        help="Only materialize queue metadata. Real remote launch is implemented after env gates pass.",
    )
    args = parser.parse_args()

    machines = load_machines(args.machines)
    slots = enumerate_slots(machines)
    jobs = [
        GpuJob(
            command="python -m cifar_mamba_fff.train_teacher --quick-smoke true",
            output_dir=Path("outputs/scheduler_smoke") / machine / str(gpu_id),
            machine=machine,
            gpu_id=gpu_id,
            seed=1337 + idx,
            metadata={"quick_smoke": args.quick_smoke, "dry_run": args.dry_run},
        )
        for idx, (machine, gpu_id) in enumerate(slots)
    ]
    write_queue(Path(args.queue_out), jobs)
    print(f"recorded {len(jobs)} GPU slots in {args.queue_out}")
    if not args.dry_run:
        raise RuntimeError("real scheduler launch is blocked until verify_env and verify_cluster pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
