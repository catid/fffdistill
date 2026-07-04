from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

from .utils import load_yaml


@dataclass(frozen=True)
class MachineSpec:
    name: str
    host: str
    gpus: int
    role: str
    workdir: str

    def validate(self) -> None:
        if self.gpus < 0:
            raise ValueError(f"{self.name}: gpus must be non-negative")
        if self.role not in {"local", "remote"}:
            raise ValueError(f"{self.name}: role must be local or remote")
        if not self.workdir:
            raise ValueError(f"{self.name}: workdir must be non-empty")


def load_machines(path: str | Path = "configs/machines.yaml") -> list[MachineSpec]:
    raw = load_yaml(path)
    machines = raw.get("machines")
    if not isinstance(machines, dict):
        raise ValueError("machines.yaml must contain a machines mapping")
    specs = [
        MachineSpec(
            name=name,
            host=str(cfg["host"]),
            gpus=int(cfg["gpus"]),
            role=str(cfg["role"]),
            workdir=str(cfg["workdir"]),
        )
        for name, cfg in machines.items()
    ]
    for spec in specs:
        spec.validate()
    return specs


def run_remote(spec: MachineSpec, command: str, timeout_s: int = 60) -> dict[str, Any]:
    repo_command = f"cd {quote(spec.workdir)} && {command}"
    argv = ["bash", "-lc", repo_command]
    if spec.role == "remote":
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", spec.host, repo_command]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
