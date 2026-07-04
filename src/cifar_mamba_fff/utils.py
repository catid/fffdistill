from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def require_file(path: str | Path, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


@dataclass(frozen=True)
class RunContext:
    output_dir: Path
    seed: int
    quick_smoke: bool

    def prepare(self, *, configure_cuda: bool = True) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(self.seed)
        if configure_cuda:
            configure_torch()

    def metadata(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "quick_smoke": self.quick_smoke,
            "git_commit": git_commit(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
