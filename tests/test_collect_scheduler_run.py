from __future__ import annotations

import json
from pathlib import Path

import pytest

from cifar_mamba_fff import collect_scheduler_run as collector
from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.gpu_scheduler import GpuJob, JobStatus


def test_scheduler_run_root_uses_job_kind_default_root() -> None:
    assert collector.scheduler_run_root(
        job_kind="finetune_hpo",
        run_id="run-a",
        output_root=None,
    ) == Path("outputs/scheduler_finetune_hpo/run-a")
    assert collector.scheduler_run_root(
        job_kind="distill_hpo",
        run_id="run-b",
        output_root=Path("custom"),
    ) == Path("custom/run-b")


def test_discover_scheduler_jobs_reads_remote_status(monkeypatch) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir="/repo",
    )
    status_path = Path("outputs/scheduler_finetune_hpo/run-a/work/1/status.json")

    def fake_run_remote(machine: MachineSpec, command: str, timeout_s: int = 60) -> dict[str, object]:
        assert machine == spec
        assert "find outputs/scheduler_finetune_hpo/run-a" in command
        return {"ok": True, "returncode": 0, "stdout": f"{status_path}\n", "stderr": ""}

    def fake_read_remote_text(
        machine: MachineSpec,
        path: str | Path,
        *,
        max_bytes: int = 262_144,
        timeout_s: int = 60,
    ) -> dict[str, object]:
        assert machine == spec
        assert Path(path) == status_path
        payload = {
            "command": "CUDA_VISIBLE_DEVICES=1 train",
            "output_dir": str(status_path.parent),
            "machine": "work",
            "gpu_id": 1,
            "seed": 123,
            "status": "succeeded",
            "metadata": {"job_kind": "finetune_hpo"},
        }
        return {"ok": True, "returncode": 0, "stdout": json.dumps(payload), "stderr": ""}

    monkeypatch.setattr(collector, "run_remote", fake_run_remote)
    monkeypatch.setattr(collector, "read_remote_text", fake_read_remote_text)

    jobs = collector.discover_scheduler_jobs(
        [spec],
        job_kind="finetune_hpo",
        run_id="run-a",
    )

    assert len(jobs) == 1
    discovered_spec, job = jobs[0]
    assert discovered_spec == spec
    assert job.machine == "work"
    assert job.gpu_id == 1
    assert job.seed == 123
    assert job.status == JobStatus.SUCCEEDED
    assert job.output_dir == status_path.parent


def test_discover_scheduler_jobs_treats_missing_run_dir_as_empty(tmp_path) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir=str(tmp_path),
    )

    jobs = collector.discover_scheduler_jobs(
        [spec],
        job_kind="finetune_hpo",
        run_id="missing-run",
        timeout_s=3,
    )

    assert jobs == []


def test_discover_scheduler_jobs_fails_on_status_discovery_error(monkeypatch) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir="/missing/repo",
    )

    monkeypatch.setattr(
        collector,
        "run_remote",
        lambda *_args, **_kwargs: {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "bash: line 1: cd: /missing/repo: No such file or directory",
        },
    )
    monkeypatch.setattr(
        collector,
        "read_remote_text",
        lambda *_args, **_kwargs: pytest.fail("status payloads should not be read"),
    )

    with pytest.raises(RuntimeError, match="could not discover scheduler statuses on work"):
        collector.discover_scheduler_jobs([spec], job_kind="finetune_hpo", run_id="run-a")


def test_discover_scheduler_jobs_rejects_mismatched_status_output_dir(monkeypatch) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir="/repo",
    )
    status_path = Path("outputs/scheduler_finetune_hpo/run-a/work/1/status.json")

    monkeypatch.setattr(
        collector,
        "run_remote",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": f"{status_path}\n",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        collector,
        "read_remote_text",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps(
                {
                    "output_dir": "outputs/scheduler_finetune_hpo/run-a/work/0",
                    "machine": "work",
                    "gpu_id": 1,
                    "status": "succeeded",
                    "metadata": {"job_kind": "finetune_hpo"},
                }
            ),
            "stderr": "",
        },
    )

    try:
        collector.discover_scheduler_jobs([spec], job_kind="finetune_hpo", run_id="run-a")
    except ValueError as exc:
        assert "output_dir" in str(exc)
    else:
        raise AssertionError("mismatched status output_dir was accepted")


def test_discover_scheduler_jobs_rejects_mismatched_status_job_kind(monkeypatch) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir="/repo",
    )
    status_path = Path("outputs/scheduler_finetune_hpo/run-a/work/1/status.json")

    monkeypatch.setattr(
        collector,
        "run_remote",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": f"{status_path}\n",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        collector,
        "read_remote_text",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps(
                {
                    "output_dir": str(status_path.parent),
                    "machine": "work",
                    "gpu_id": 1,
                    "status": "succeeded",
                    "metadata": {"job_kind": "distill_hpo"},
                }
            ),
            "stderr": "",
        },
    )

    try:
        collector.discover_scheduler_jobs([spec], job_kind="finetune_hpo", run_id="run-a")
    except ValueError as exc:
        assert "job_kind" in str(exc)
    else:
        raise AssertionError("mismatched status job kind was accepted")


def test_collect_scheduler_run_skips_running_jobs_by_default(monkeypatch, tmp_path) -> None:
    spec = MachineSpec(
        name="work",
        host="localhost",
        gpus=2,
        role="local",
        workdir="/repo",
    )
    succeeded = GpuJob(
        command="ok",
        output_dir=Path("outputs/scheduler_finetune_hpo/run-a/work/0"),
        machine="work",
        gpu_id=0,
        status=JobStatus.SUCCEEDED,
    )
    running = GpuJob(
        command="run",
        output_dir=Path("outputs/scheduler_finetune_hpo/run-a/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )

    monkeypatch.setattr(
        collector,
        "discover_scheduler_jobs",
        lambda *args, **kwargs: [(spec, succeeded), (spec, running)],
    )

    collected: list[GpuJob] = []

    def fake_collect(
        machine: MachineSpec,
        job: GpuJob,
        *,
        local_root: Path,
        timeout_s: int = 20,
        max_bytes: int = 1_048_576,
    ) -> dict[str, object]:
        collected.append(job)
        return {
            "status": "artifacts_collected",
            "machine": job.machine,
            "gpu_id": job.gpu_id,
            "local_output_dir": str(local_root / str(job.machine) / str(job.gpu_id)),
        }

    monkeypatch.setattr(collector, "collect_detached_job_artifacts", fake_collect)

    summary = collector.collect_scheduler_run(
        [spec],
        job_kind="finetune_hpo",
        run_id="run-a",
        collect_root=tmp_path,
    )

    assert [job.gpu_id for job in collected] == [0]
    assert summary["discovered_jobs"] == 2
    assert summary["collected_jobs"] == 1
    assert summary["skipped_jobs"] == 1
    assert summary["skipped"][0]["reason"] == "non-terminal"
    assert (tmp_path / "collection_manifest.json").exists()
