#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from shlex import quote

import yaml


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def _run(command: list[str], *, timeout_s: int) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}


def _machine_command(host: str, role: str, command: str) -> list[str]:
    if role == "local" or host in {"localhost", "127.0.0.1"}:
        return ["bash", "-lc", command]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command]


def _repo_command(workdir: str, command: str) -> str:
    return f"cd {quote(workdir)} && {command}"


def _gpu_count(nvidia_smi_stdout: str) -> int:
    return sum(1 for line in nvidia_smi_stdout.splitlines() if line.strip())


def _inventory_machine(name: str, spec: dict[str, object], timeout_s: int) -> dict[str, object]:
    host = str(spec["host"])
    role = str(spec["role"])
    workdir = str(spec["workdir"])
    commands = {
        "hostname": "hostname",
        "nvidia_smi": "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
        "python": "python3 --version 2>&1",
        "repo_workdir": _repo_command(workdir, "pwd"),
        "venv_python": _repo_command(workdir, "test -x .venv/bin/python && .venv/bin/python --version 2>&1"),
        "workdir": "pwd",
    }
    results = {
        key: _run(_machine_command(host, role, command), timeout_s=timeout_s)
        for key, command in commands.items()
    }
    expected_gpus = int(spec["gpus"])
    detected_gpus = _gpu_count(str(results["nvidia_smi"]["stdout"])) if results["nvidia_smi"]["ok"] else 0
    gpu_count_matches = detected_gpus == expected_gpus
    gpu_available = bool(results["hostname"]["ok"]) and bool(results["nvidia_smi"]["ok"]) and gpu_count_matches
    python_available = bool(results["python"]["ok"])
    workdir_available = bool(results["repo_workdir"]["ok"])
    venv_available = bool(results["venv_python"]["ok"])
    return {
        "name": name,
        "host": host,
        "role": role,
        "workdir": workdir,
        "expected_gpus": expected_gpus,
        "detected_gpus": detected_gpus,
        "available": gpu_available and python_available and workdir_available and venv_available,
        "gpu_available": gpu_available,
        "gpu_count_matches": gpu_count_matches,
        "python_available": python_available,
        "workdir_available": workdir_available,
        "venv_available": venv_available,
        "commands": results,
    }


def _write_markdown(inventory: dict[str, object], path: Path) -> None:
    lines = [
        "# Cluster Inventory",
        "",
        f"Generated: {inventory['generated_at']}",
        "",
        "| machine | host | role | workdir | expected GPUs | detected GPUs | available | GPU summary |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for machine in inventory["machines"]:
        smi = machine["commands"]["nvidia_smi"]
        gpu_summary = str(smi["stdout"]).replace("\n", "<br>") if smi["ok"] else str(smi["stderr"])
        python = machine["commands"]["python"]
        if not machine["python_available"]:
            python_summary = f"python unavailable: {python['stderr'] or python['stdout']}"
            gpu_summary = f"{gpu_summary}<br>{python_summary}"
        elif not machine["workdir_available"]:
            workdir = machine["commands"]["repo_workdir"]
            gpu_summary = f"{gpu_summary}<br>workdir unavailable: {workdir['stderr'] or workdir['stdout']}"
        elif not machine["venv_available"]:
            gpu_summary = f"{gpu_summary}<br>project venv not verified"
        if not machine["gpu_count_matches"]:
            gpu_summary = f"{gpu_summary}<br>detected GPU count does not match config"
        lines.append(
            "| {name} | {host} | {role} | {workdir} | {expected_gpus} | {detected_gpus} | {available} | {gpu_summary} |".format(
                **machine,
                gpu_summary=gpu_summary.replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/machines.yaml")
    parser.add_argument("--quick-smoke", type=_bool_arg, default=False)
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument(
        "--allow-incomplete",
        type=_bool_arg,
        default=False,
        help="Write inventory and exit 0 even when one or more machine environments are unavailable.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    machines = config.get("machines")
    if not isinstance(machines, dict):
        raise ValueError("machines config must contain a mapping at key 'machines'")
    for name, spec in machines.items():
        if not isinstance(spec, dict) or not spec.get("workdir"):
            raise ValueError(f"machine {name} must define workdir")

    selected = machines
    if args.quick_smoke:
        selected = {name: spec for name, spec in machines.items() if spec.get("role") == "local"}

    inventory = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": args.config,
        "quick_smoke": args.quick_smoke,
        "machines": [
            _inventory_machine(name, spec, timeout_s=args.timeout_s) for name, spec in selected.items()
        ],
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/cluster_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(inventory, Path("docs/cluster_inventory.md"))
    print(json.dumps(inventory, indent=2))
    unavailable = [machine["name"] for machine in inventory["machines"] if not machine["available"]]
    if unavailable and not args.allow_incomplete:
        joined = ", ".join(unavailable)
        raise SystemExit(f"cluster verification incomplete for: {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
