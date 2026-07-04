from __future__ import annotations

import base64
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


def _machine_argv(spec: MachineSpec, command: str) -> list[str]:
    if spec.role == "local" or spec.host in {"localhost", "127.0.0.1"}:
        return ["bash", "-lc", command]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", spec.host, command]


def run_machine(spec: MachineSpec, command: str, timeout_s: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            _machine_argv(spec, command),
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


def run_remote(spec: MachineSpec, command: str, timeout_s: int = 60) -> dict[str, Any]:
    repo_command = f"cd {quote(spec.workdir)} && {command}"
    return run_machine(spec, repo_command, timeout_s=timeout_s)


def write_remote_text(
    spec: MachineSpec,
    path: str | Path,
    text: str,
    *,
    executable: bool = False,
    timeout_s: int = 60,
) -> dict[str, Any]:
    remote_path = Path(path)
    if remote_path.is_absolute():
        raise ValueError("remote repository paths must be relative")
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    chmod = "path.chmod(path.stat().st_mode | 0o111)" if executable else ""
    command = "\n".join(
        [
            "python3 - <<'PY'",
            "import base64",
            "from pathlib import Path",
            f"path = Path({str(remote_path)!r})",
            "path.parent.mkdir(parents=True, exist_ok=True)",
            f"path.write_bytes(base64.b64decode({encoded!r}))",
            chmod,
            "PY",
        ]
    )
    return run_remote(spec, command, timeout_s=timeout_s)


def read_remote_text(
    spec: MachineSpec,
    path: str | Path,
    *,
    max_bytes: int = 262_144,
    timeout_s: int = 60,
) -> dict[str, Any]:
    remote_path = Path(path)
    if remote_path.is_absolute():
        raise ValueError("remote repository paths must be relative")
    command = (
        "python3 - <<'PY'\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"path = Path({str(remote_path)!r})\n"
        "if not path.exists():\n"
        "    raise SystemExit(44)\n"
        f"sys.stdout.buffer.write(path.read_bytes()[-{int(max_bytes)}:])\n"
        "PY"
    )
    return run_remote(spec, command, timeout_s=timeout_s)
