from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from cifar_mamba_fff import gpu_scheduler
from cifar_mamba_fff.cluster import MachineSpec, read_remote_text
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
            "PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python "
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
    assert "PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0" in script_text
    assert ".venv/bin/python -m cifar_mamba_fff.train_teacher" in script_text
    assert 'bash -lc "$COMMAND" > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"' in script_text
    assert "status.json.tmp" in script_text
    assert "os.replace(tmp_path, status_path)" in script_text
    assert "heartbeat_loop &" in script_text
    assert "(out_dir / 'heartbeat.txt').write_text" in script_text
    assert 'final_status="succeeded"' in script_text
    assert '[ "$rc" -ge 128 ]' in script_text
    assert 'final_status="failed_infra"' in script_text
    assert 'final_status="failed_logic"' in script_text


def test_collected_artifact_names_include_hpo_trial_config() -> None:
    assert "teacher_hpo_summary.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "teacher_hpo_events.jsonl" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/trial_config.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/trial_summary.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES


def test_collected_artifact_names_include_distill_hpo_summaries() -> None:
    assert "distill_hpo_summary.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/distill_config.yaml" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/trial_result.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/distill_summary.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/layer_summary.json" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES
    assert "trials/trial_000000/layer_metrics.jsonl" in gpu_scheduler.COLLECTED_ARTIFACT_NAMES


def test_read_remote_text_reports_copy_integrity_for_truncated_file(tmp_path: Path) -> None:
    spec = MachineSpec(name="local", host="localhost", gpus=0, role="local", workdir=str(tmp_path))
    (tmp_path / "metrics.jsonl").write_text("0123456789", encoding="utf-8")

    result = read_remote_text(spec, "metrics.jsonl", max_bytes=4, timeout_s=3)

    assert result["ok"] is True
    assert result["stdout"] == "6789"
    assert result["remote_size_bytes"] == 10
    assert result["copied_size_bytes"] == 4
    assert result["truncated"] is True


def test_collect_detached_job_artifacts_marks_truncated_metrics_and_missing_trials(
    monkeypatch, tmp_path: Path
) -> None:
    collect_detached_job_artifacts = _require_public_helper("collect_detached_job_artifacts")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer",
        output_dir=Path("outputs/hpo/work/0"),
        machine="work",
        gpu_id=0,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")
    output_dir = gpu_scheduler.detached_job_files(job).output_dir

    def payload_result(text: str, *, truncated: bool = False, remote_size: int | None = None):
        copied_size = len(text.encode("utf-8"))
        return {
            "ok": True,
            "returncode": 0,
            "stdout": text,
            "stderr": "",
            "remote_size_bytes": remote_size if remote_size is not None else copied_size,
            "copied_size_bytes": copied_size,
            "truncated": truncated,
        }

    def fake_read_remote_text(
        spec_arg: MachineSpec,
        path: Path,
        *,
        max_bytes: int,
        timeout_s: int,
    ) -> Mapping[str, object]:
        assert spec_arg == spec
        assert max_bytes == 4
        assert timeout_s == 9
        if path == output_dir / "teacher_hpo_summary.json":
            return payload_result(
                json.dumps(
                    {
                        "mode": "teacher_hpo",
                        "accepted_trials": 3,
                        "status": "completed",
                    }
                )
            )
        if path == output_dir / "trials/trial_000000/metrics.jsonl":
            return payload_result("tail", truncated=True, remote_size=100)
        if path == output_dir / "trials/trial_000001/trial_config.json":
            return payload_result(json.dumps({"trial_index": 1}))
        return {
            "ok": False,
            "returncode": 44,
            "stdout": "",
            "stderr": "",
            "remote_size_bytes": None,
            "copied_size_bytes": None,
            "truncated": None,
        }

    def fake_run_remote(
        spec_arg: MachineSpec,
        command: str,
        *,
        timeout_s: int,
    ) -> Mapping[str, object]:
        assert spec_arg == spec
        assert timeout_s == 9
        assert "outputs/hpo/work/0/trials" in command
        return {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps(["trial_000000", "trial_000001"]),
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "read_remote_text", fake_read_remote_text)
    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)

    record = collect_detached_job_artifacts(
        spec,
        job,
        local_root=tmp_path,
        timeout_s=9,
        max_bytes=4,
    )

    assert record["status"] == "artifacts_collected_truncated_metrics"
    assert record["trial_dirs"] == {
        "remote": ["trial_000000", "trial_000001"],
        "expected": ["trial_000000", "trial_000001", "trial_000002"],
        "missing": ["trial_000002"],
        "listing_ok": True,
        "listing_stderr": "",
    }
    assert "trials/trial_000001/trial_config.json" in record["files"]
    assert "trials/trial_000002/trial_config.json" in record["files"]
    truncated_record = record["files"]["trials/trial_000000/metrics.jsonl"]
    assert truncated_record["remote_size_bytes"] == 100
    assert truncated_record["copied_size_bytes"] == 4
    assert truncated_record["local_size_bytes"] == 4
    assert truncated_record["truncated"] is True
    assert record["artifact_integrity"]["metrics_truncated_files"] == [
        "trials/trial_000000/metrics.jsonl"
    ]


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


def test_read_detached_job_status_treats_truncated_json_as_running(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": "{",
            "stderr": "",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.RUNNING
    assert "transient unreadable status" in record["stderr"]
    assert job.status == JobStatus.RUNNING


def test_read_detached_job_status_treats_rc255_as_running(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": False,
            "returncode": 255,
            "stdout": "",
            "stderr": "ssh transient failure",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.RUNNING
    assert "ssh transient failure" in record["stderr"]
    assert job.status == JobStatus.RUNNING


def test_read_detached_job_status_treats_timeout_as_running(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "timed out after 7 seconds",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.RUNNING
    assert "timed out" in record["stderr"]
    assert job.status == JobStatus.RUNNING


def test_read_detached_job_status_marks_dead_running_pid_failed_infra(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"status": JobStatus.RUNNING.value, "returncode": None}),
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        gpu_scheduler,
        "probe_detached_job_liveness",
        lambda spec_arg, files, *, timeout_s: {
            "ok": False,
            "returncode": 45,
            "stdout": "",
            "stderr": "",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.FAILED_INFRA
    assert record["returncode"] == 45
    assert "pid is not live" in record["stderr"]
    assert job.status == JobStatus.FAILED_INFRA


def test_read_detached_job_status_keeps_running_on_liveness_timeout(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"status": JobStatus.RUNNING.value, "returncode": None}),
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        gpu_scheduler,
        "probe_detached_job_liveness",
        lambda spec_arg, files, *, timeout_s: {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "timed out",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.RUNNING
    assert "timed out" in record["stderr"]
    assert job.status == JobStatus.RUNNING


def test_read_detached_job_status_records_liveness_for_running_job(monkeypatch) -> None:
    read_detached_job_status = _require_public_helper("read_detached_job_status")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/1"),
        machine="work",
        gpu_id=1,
        status=JobStatus.RUNNING,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"status": JobStatus.RUNNING.value, "returncode": None}),
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        gpu_scheduler,
        "probe_detached_job_liveness",
        lambda spec_arg, files, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"pid": 123, "heartbeat_exists": True}),
            "stderr": "",
        },
    )

    record = read_detached_job_status(spec, job, timeout_s=7)

    assert record["status"] == JobStatus.RUNNING
    assert record["liveness"] == {"pid": 123, "heartbeat_exists": True}
    assert job.status == JobStatus.RUNNING


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
    assert result.expected_commit == "newcommit"
    assert result.remote_commit == "oldcommit"
    assert result.local_commit
    assert "remote commit/config mismatch: expected newcommit, got oldcommit" in result.stderr


def test_preflight_machine_records_matching_local_and_remote_commits(monkeypatch) -> None:
    preflight_machine = _require_public_helper("preflight_machine")
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")

    monkeypatch.setattr(gpu_scheduler, "git_commit", lambda: "localcommit")

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
            "stdout": "localcommit\nPython 3.12.11\n",
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)

    result = preflight_machine(
        spec,
        python_bin=".venv/bin/python",
        expected_commit="localcommit",
        timeout_s=3,
    )

    assert result.ok is True
    assert result.record() == {
        "machine": "work",
        "ok": True,
        "returncode": 0,
        "commit": "localcommit",
        "expected_commit": "localcommit",
        "local_commit": "localcommit",
        "remote_commit": "localcommit",
        "stdout": "localcommit\nPython 3.12.11",
        "stderr": "",
    }


def test_detached_launch_script_refuses_commit_mismatch_before_command(tmp_path: Path) -> None:
    render_detached_launch_script = _require_public_helper("render_detached_launch_script")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
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
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    remote_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
    ).strip()
    expected_commit = "0" * 40
    job = GpuJob(
        command="touch outputs/sync_guard/work/0/should_not_run",
        output_dir=Path("outputs/sync_guard/work/0"),
        machine="work",
        gpu_id=0,
    )
    script_path = tmp_path / "launch.sh"
    script_path.write_text(
        render_detached_launch_script(job, expected_commit=expected_commit),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    output_dir = tmp_path / "outputs/sync_guard/work/0"
    payload = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert completed.returncode == 125
    assert not (output_dir / "should_not_run").exists()
    assert payload["status"] == JobStatus.FAILED_INFRA
    assert payload["returncode"] == 125
    assert payload["expected_git_commit"] == expected_commit
    assert payload["remote_git_commit"] == remote_commit
    assert "remote commit/config mismatch" in payload["prelaunch_error"]
    assert "remote commit/config mismatch" in (output_dir / "stderr.log").read_text(
        encoding="utf-8"
    )


def test_preflight_machine_can_require_cifar10_train_readiness(monkeypatch) -> None:
    preflight_machine = _require_public_helper("preflight_machine")
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")
    commands: list[str] = []

    def fake_run_remote(
        spec_arg: MachineSpec,
        command: str,
        *,
        timeout_s: int,
    ) -> Mapping[str, object]:
        assert spec_arg == spec
        assert timeout_s == 3
        commands.append(command)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "newcommit\nPython 3.12.11\n",
            "stderr": "",
        }

    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)

    result = preflight_machine(
        spec,
        python_bin=".venv/bin/python",
        expected_commit="newcommit",
        require_cifar10_train=True,
        data_dir="data/cifar10",
        timeout_s=3,
    )

    assert result.ok is True
    assert "scripts/prepare_cifar10.py" in commands[0]
    assert "--download false --extract false" in commands[0]


def test_launch_detached_job_probes_status_after_launch_timeout(monkeypatch) -> None:
    launch_detached_job = _require_public_helper("launch_detached_job")
    build_detached_launch_command = _require_public_helper("build_detached_launch_command")
    job = GpuJob(
        command="PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trainer",
        output_dir=Path("outputs/scheduler_smoke/work/0"),
        machine="work",
        gpu_id=0,
    )
    spec = MachineSpec(name="work", host="localhost", gpus=2, role="local", workdir="/repo")
    launch_commands: list[str] = []
    launch_command = build_detached_launch_command(gpu_scheduler.detached_job_files(job))
    assert "setsid -f bash" in launch_command
    assert "nohup bash" in launch_command
    assert "</dev/null" in launch_command
    assert "status.json" in launch_command

    monkeypatch.setattr(
        gpu_scheduler,
        "write_remote_text",
        lambda spec_arg, path, text, *, executable, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    def fake_run_remote(
        spec_arg: MachineSpec,
        command: str,
        *,
        timeout_s: int,
    ) -> Mapping[str, object]:
        assert spec_arg == spec
        launch_commands.append(command)
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "timed out after 20 seconds",
        }

    monkeypatch.setattr(gpu_scheduler, "run_remote", fake_run_remote)
    monkeypatch.setattr(
        gpu_scheduler,
        "read_remote_text",
        lambda spec_arg, path, *, timeout_s: {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps({"status": JobStatus.RUNNING.value, "returncode": None}),
            "stderr": "",
        },
    )

    result = launch_detached_job(spec, job, timeout_s=20)

    assert result.ok is True
    assert result.status == JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING
    assert "</dev/null" in launch_commands[0]
