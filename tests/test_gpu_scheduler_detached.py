from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from cifar_mamba_fff import gpu_scheduler
from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.gpu_scheduler import GpuJob, JobStatus

DETACHED_API_MESSAGE = (
    "Expected cifar_mamba_fff.gpu_scheduler detached helpers: "
    "detached_job_files(job: GpuJob), render_detached_launch_script(job: GpuJob), "
    "and read_detached_job_status(spec: MachineSpec, job: GpuJob, *, timeout_s: int)."
)


def _require_public_helper(name: str):
    helper = getattr(gpu_scheduler, name, None)
    if helper is None:
        pytest.fail(f"{DETACHED_API_MESSAGE} Missing helper: {name}.")
    return helper


def test_local_detached_job_files_and_script_construct_status_and_log_paths() -> None:
    detached_job_files = _require_public_helper("detached_job_files")
    render_detached_launch_script = _require_public_helper("render_detached_launch_script")
    job = GpuJob(
        command=(
            "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
            "-m cifar_mamba_fff.train_teacher --quick-smoke true "
            "--output-dir outputs/scheduler_smoke/work/0"
        ),
        output_dir=Path("outputs/scheduler_smoke/work/0"),
        machine="work",
        gpu_id=0,
        seed=1337,
        metadata={"quick_smoke": True},
    )

    files = detached_job_files(job)
    assert files.output_dir == Path("outputs/scheduler_smoke/work/0")
    assert files.script == Path("outputs/scheduler_smoke/work/0/launch.sh")
    assert files.status == Path("outputs/scheduler_smoke/work/0/status.json")
    assert files.stdout == Path("outputs/scheduler_smoke/work/0/stdout.log")
    assert files.stderr == Path("outputs/scheduler_smoke/work/0/stderr.log")
    assert files.pid == Path("outputs/scheduler_smoke/work/0/pid.txt")
    assert files.exit_code == Path("outputs/scheduler_smoke/work/0/exit_code.txt")
    assert files.heartbeat == Path("outputs/scheduler_smoke/work/0/heartbeat.txt")

    script_text = render_detached_launch_script(job)
    assert "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0" in script_text
    assert ".venv/bin/python -m cifar_mamba_fff.train_teacher" in script_text
    assert 'bash -lc "$COMMAND" > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"' in script_text
    assert "(out_dir / 'status.json').write_text" in script_text
    assert "(out_dir / 'heartbeat.txt').write_text" in script_text
    assert 'final_status="succeeded"' in script_text
    assert 'final_status="failed_logic"' in script_text


@pytest.mark.parametrize(
    ("payload_status", "returncode"),
    [(JobStatus.SUCCEEDED, 0), (JobStatus.FAILED_LOGIC, 2)],
)
def test_read_detached_job_status_classifies_success_and_logic_failure(
    monkeypatch, payload_status: JobStatus, returncode: int
) -> None:
    detached_job_files = _require_public_helper("detached_job_files")
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")
    expected_status_path = detached_job_files(job).status
    read_calls: list[tuple[MachineSpec, Path, int]] = []

    def fake_read_remote_text(
        spec_arg: MachineSpec,
        path: Path,
        *,
        timeout_s: int,
    ) -> Mapping[str, object]:
        read_calls.append((spec_arg, path, timeout_s))
        return {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"status": payload_status.value, "returncode": returncode}),
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "read_remote_text", fake_read_remote_text)

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert read_calls == [(spec, expected_status_path, 7)]
    assert record["status"] == payload_status.value
    assert record["returncode"] == returncode
    assert job.status == payload_status


def test_preflight_machine_rejects_commit_mismatch(monkeypatch) -> None:
    preflight_machine = _require_public_helper("preflight_machine")
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    def fake_run_remote(
        spec_arg: MachineSpec,
        command: str,
        *,
        timeout_s: int,
    ) -> Mapping[str, object]:
        assert spec_arg == spec
        assert "git rev-parse HEAD" in command
        assert timeout_s == 3
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "oldcommit\nPython 3.12.11\n",
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)

    result = preflight_machine(
        spec,
        python_bin=".venv/bin/python",
        expected_commit="newcommit",
        timeout_s=3,
    )

    assert result.ok is False
    assert result.commit == "oldcommit"
    assert "remote commit mismatch: expected newcommit, got oldcommit" in result.stderr
