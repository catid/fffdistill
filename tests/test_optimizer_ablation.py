from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

if importlib.util.find_spec("torchvision") is None:
    torchvision_stub = types.ModuleType("torchvision")
    datasets_stub = types.ModuleType("torchvision.datasets")
    transforms_stub = types.ModuleType("torchvision.transforms")
    utils_stub = types.ModuleType("torchvision.datasets.utils")

    class _ImportOnlyCIFAR10:
        train_list: ClassVar[list[tuple[str, str]]] = []
        meta: ClassVar[dict[str, str]] = {}

    datasets_stub.CIFAR10 = _ImportOnlyCIFAR10
    utils_stub.check_integrity = lambda *args, **kwargs: True
    torchvision_stub.datasets = datasets_stub
    torchvision_stub.transforms = transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.datasets"] = datasets_stub
    sys.modules["torchvision.datasets.utils"] = utils_stub
    sys.modules["torchvision.transforms"] = transforms_stub

import cifar_mamba_fff.train_teacher as train_teacher
from cifar_mamba_fff.hpo.optimizer_ablation import run_optimizer_ablation
from cifar_mamba_fff.optim.normuon import MuonNorMuon
from cifar_mamba_fff.optim.pace import PaceOptimizer
from cifar_mamba_fff.train_teacher import TeacherTrainConfig


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(torch.relu(self.hidden(x))))


class FakeOfficialMuon(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)


def _one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def test_pace_muon_wraps_official_muon_baseline(monkeypatch) -> None:
    monkeypatch.setattr(train_teacher, "_import_official_muon_class", lambda: FakeOfficialMuon)
    model = TinyNet()
    config = TeacherTrainConfig(optimizer="pace_muon", pace_pullback_c=1e-3)

    optimizer, summary = train_teacher.build_training_optimizer(model, config)

    assert isinstance(optimizer, PaceOptimizer)
    assert isinstance(optimizer.base_optimizer, FakeOfficialMuon)
    assert summary["optimizer_family"] == "pace_muon"
    assert summary["base_optimizer_family"] == "muon_adamw"
    assert summary["uses_ema_eval"] is True
    assert summary["optimizer_experiments_commit"] == "689568d71ebe92093e5f5bf433127a5184ef0c35"
    assert train_teacher._scheduler_optimizer(optimizer) is optimizer.base_optimizer

    live_before = [parameter.detach().clone() for parameter in model.parameters()]
    _one_step(model, optimizer)
    with optimizer.use_ema_weights():
        ema_weights = [parameter.detach().clone() for parameter in model.parameters()]
        assert any(
            not torch.allclose(live, ema)
            for live, ema in zip(live_before, ema_weights, strict=True)
        )
    assert optimizer._swapped is False


def test_normuon_and_pace_normuon_are_explicit_non_official_ablations() -> None:
    model = TinyNet()
    config = TeacherTrainConfig(optimizer="normuon_adamw")

    optimizer, summary = train_teacher.build_training_optimizer(model, config)

    assert isinstance(optimizer, MuonNorMuon)
    assert summary["optimizer_family"] == "normuon_adamw"
    assert "optimizer_experiments" in str(summary["optimizer_source"])
    _one_step(model, optimizer)
    hidden_state = optimizer.state[model.hidden.weight]
    assert hidden_state["normuon_second_momentum"].shape == (4, 1)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    pace_optimizer, pace_summary = train_teacher.build_training_optimizer(
        TinyNet(),
        TeacherTrainConfig(optimizer="pace_normuon", pace_pullback_c=0.0),
    )
    assert isinstance(pace_optimizer, PaceOptimizer)
    assert isinstance(pace_optimizer.base_optimizer, MuonNorMuon)
    assert pace_summary["optimizer_family"] == "pace_normuon"
    assert pace_summary["base_optimizer_family"] == "normuon_adamw"
    assert pace_summary["uses_ema_eval"] is True


def _write_ablation_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_optimizer_ablation_reuses_shared_seed_set_for_every_case(tmp_path: Path) -> None:
    ablation_config = _write_ablation_config(
        tmp_path / "ablation.yaml",
        """
seed: 999
seeds: [101, 103]
cases:
  - name: muon_wsd
    train:
      optimizer: muon_adamw
      schedule: wsd
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
      warmup_epochs: 0
      mixup: 0.0
      cutmix: 0.0
  - name: pace_cosine
    train:
      optimizer: pace_muon
      schedule: cosine
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
      warmup_epochs: 0
      mixup: 0.0
      cutmix: 0.0
""",
    )
    calls: list[tuple[str, int, str, str]] = []

    def fake_training(run_config, output_dir: Path, **kwargs):
        case = output_dir.parent.name if output_dir.name.startswith("seed_") else output_dir.name
        calls.append((case, run_config.seed, run_config.train.optimizer, run_config.train.schedule))
        return {
            "best_val_accuracy": 0.25 + (run_config.seed / 10000.0),
            "train_steps_total": 2,
            "train_images_seen": 32,
            "elapsed_seconds": 1.0,
            "train_images_per_second": 32.0,
            "train_images_per_second_train_only": 64.0,
            "parameter_count": 123,
            "optimizer_summary": {
                "uses_ema_eval": run_config.train.optimizer.startswith("pace"),
                "optimizer_source": "fake",
            },
        }

    summary = run_optimizer_ablation(
        base_config_path=Path("configs/teacher_default.yaml"),
        ablation_config_path=ablation_config,
        output_dir=tmp_path / "out",
        quick_smoke=True,
        max_train_steps=2,
        max_val_steps=1,
        seed=5000,
        training_fn=fake_training,
    )

    assert calls == [
        ("muon_wsd", 101, "muon_adamw", "wsd"),
        ("muon_wsd", 103, "muon_adamw", "wsd"),
        ("pace_cosine", 101, "pace_muon", "cosine"),
        ("pace_cosine", 103, "pace_muon", "cosine"),
    ]
    assert summary["seed_list"] == [101, 103]
    assert [row["seed_list"] for row in summary["results"]] == [[101, 103], [101, 103]]
    assert [row["seed_count"] for row in summary["results"]] == [2, 2]
    assert {row["optimizer"] for row in summary["results"]} == {"muon_adamw", "pace_muon"}
    assert {row["schedule"] for row in summary["results"]} == {"wsd", "cosine"}

    with (tmp_path / "out" / "optimizer_ablation_summary.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [json.loads(row["seed_list"]) for row in csv_rows] == [[101, 103], [101, 103]]
    assert [row["seed_count"] for row in csv_rows] == ["2", "2"]


def test_optimizer_ablation_case_offset_does_not_change_case_seed_set(tmp_path: Path) -> None:
    ablation_config = _write_ablation_config(
        tmp_path / "ablation.yaml",
        """
seeds: [11, 12]
cases:
  - name: case_a
    train:
      optimizer: muon_adamw
      schedule: cosine
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
  - name: case_b
    train:
      optimizer: pace_muon
      schedule: wsd
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
  - name: case_c
    train:
      optimizer: muon_adamw
      schedule: wsd
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
""",
    )
    seen: list[tuple[str, int]] = []

    def fake_training(run_config, output_dir: Path, **kwargs):
        seen.append((output_dir.parent.name, run_config.seed))
        return {
            "best_val_accuracy": 0.1,
            "train_steps_total": 1,
            "train_images_seen": 16,
            "optimizer_summary": {"uses_ema_eval": False, "optimizer_source": "fake"},
        }

    summary = run_optimizer_ablation(
        base_config_path=Path("configs/teacher_default.yaml"),
        ablation_config_path=ablation_config,
        output_dir=tmp_path / "out",
        quick_smoke=True,
        max_train_steps=1,
        max_val_steps=1,
        case_offset=1,
        case_limit=1,
        seed=9000,
        training_fn=fake_training,
    )

    assert seen == [("case_b", 11), ("case_b", 12)]
    assert summary["results"][0]["case"] == "case_b"
    assert summary["results"][0]["seed_list"] == [11, 12]


def test_normuon_ablation_requires_and_records_update_rms_calibration_note(tmp_path: Path) -> None:
    missing_note = _write_ablation_config(
        tmp_path / "missing_note.yaml",
        """
seeds: [101]
cases:
  - name: normuon_wsd
    train:
      optimizer: normuon_adamw
      schedule: wsd
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
""",
    )

    with pytest.raises(ValueError, match="update_rms_calibration_note"):
        run_optimizer_ablation(
            base_config_path=Path("configs/teacher_default.yaml"),
            ablation_config_path=missing_note,
            output_dir=tmp_path / "missing",
            quick_smoke=True,
            max_train_steps=1,
            max_val_steps=1,
            training_fn=lambda *args, **kwargs: {"best_val_accuracy": 0.1},
        )

    note = "NorMuon update RMS is not matched in this smoke; run an optimizer-specific LR sweep before quality claims."
    with_note = _write_ablation_config(
        tmp_path / "with_note.yaml",
        f"""
seeds: [101]
update_rms_calibration_note: "{note}"
cases:
  - name: normuon_wsd
    train:
      optimizer: normuon_adamw
      schedule: wsd
      epochs: 1
      batch_size_per_gpu: 16
      num_workers: 0
""",
    )

    summary = run_optimizer_ablation(
        base_config_path=Path("configs/teacher_default.yaml"),
        ablation_config_path=with_note,
        output_dir=tmp_path / "with_note",
        quick_smoke=True,
        max_train_steps=1,
        max_val_steps=1,
        training_fn=lambda run_config, **kwargs: {
            "best_val_accuracy": 0.1,
            "train_steps_total": 1,
            "train_images_seen": 16,
            "optimizer_summary": {"uses_ema_eval": False, "optimizer_source": "fake"},
        },
    )

    assert summary["update_rms_calibration_note"] == note
    assert summary["results"][0]["update_rms_calibration_note"] == note
