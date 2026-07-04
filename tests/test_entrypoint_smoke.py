from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_metadata_entrypoints_quick_smoke(tmp_path) -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "cifar_mamba_fff.train_teacher",
            "--quick-smoke",
            "true",
            "--smoke-mode",
            "metadata",
            "--output-dir",
            str(tmp_path / "teacher"),
        ],
        [
            sys.executable,
            "-m",
            "cifar_mamba_fff.distill_linears",
            "--quick-smoke",
            "true",
            "--output-dir",
            str(tmp_path / "distill"),
        ],
        [
            sys.executable,
            "-m",
            "cifar_mamba_fff.finetune_student",
            "--quick-smoke",
            "true",
            "--output-dir",
            str(tmp_path / "finetune"),
        ],
        [
            sys.executable,
            "-m",
            "cifar_mamba_fff.hpo.finetune_hpo",
            "--quick-smoke",
            "true",
            "--plan-only",
            "true",
            "--output-dir",
            str(tmp_path / "finetune_hpo"),
        ],
    ]
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
        assert completed.returncode == 0, completed.stderr
