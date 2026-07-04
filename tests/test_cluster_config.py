from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from cifar_mamba_fff.cluster import load_machines


def _load_verify_cluster():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_cluster.py"
    spec = importlib.util.spec_from_file_location("verify_cluster", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_machine_config_loads_expected_slots() -> None:
    machines = load_machines("configs/machines.yaml")
    by_name = {machine.name: machine for machine in machines}
    assert by_name["work"].gpus == 2
    assert by_name["ripper"].gpus == 4
    assert by_name["foureyes"].gpus == 4
    assert by_name["ai"].gpus == 2
    assert by_name["work"].workdir == "/home/catid/fff"
    assert sum(machine.gpus for machine in machines) == 12


def test_cluster_inventory_marks_missing_python_unavailable(monkeypatch) -> None:
    verify_cluster = _load_verify_cluster()

    def fake_run(command: list[str], *, timeout_s: int) -> dict[str, object]:
        joined = " ".join(command)
        if "python3 --version" in joined:
            return {"ok": False, "returncode": 127, "stdout": "", "stderr": "python3: not found"}
        if "nvidia-smi" in joined:
            return {"ok": True, "returncode": 0, "stdout": "GPU", "stderr": ""}
        if "cd " in joined and "pwd" in joined:
            return {"ok": True, "returncode": 0, "stdout": "/tmp/repo", "stderr": ""}
        if "test -x .venv/bin/python" in joined:
            return {"ok": False, "returncode": 1, "stdout": "", "stderr": ""}
        return {"ok": True, "returncode": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(verify_cluster, "_run", fake_run)
    machine = verify_cluster._inventory_machine(
        "testbox",
        {"host": "localhost", "role": "local", "gpus": 1, "workdir": "/tmp/repo"},
        timeout_s=1,
    )

    assert machine["gpu_available"] is True
    assert machine["python_available"] is False
    assert machine["available"] is False


def test_cluster_inventory_requires_expected_gpu_count(monkeypatch) -> None:
    verify_cluster = _load_verify_cluster()

    def fake_run(command: list[str], *, timeout_s: int) -> dict[str, object]:
        joined = " ".join(command)
        if "nvidia-smi" in joined:
            return {"ok": True, "returncode": 0, "stdout": "GPU0\nGPU1", "stderr": ""}
        return {"ok": True, "returncode": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(verify_cluster, "_run", fake_run)
    machine = verify_cluster._inventory_machine(
        "testbox",
        {"host": "localhost", "role": "local", "gpus": 4, "workdir": "/tmp/repo"},
        timeout_s=1,
    )

    assert machine["detected_gpus"] == 2
    assert machine["gpu_count_matches"] is False
    assert machine["available"] is False
