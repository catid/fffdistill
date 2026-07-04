from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cifar_mamba_fff.finetune_student import (
    assemble_fff_student_from_artifacts,
    kd_loss,
    load_finetune_run_config,
    parse_finetune_run_config,
)
from cifar_mamba_fff.hpo.finetune_hpo import (
    run_finetune_hpo_trials,
    write_finetune_hpo_trial_plan,
)
from cifar_mamba_fff.models.replacement import make_fff_replacement


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _minimal_config(tmp_path: Path) -> dict[str, object]:
    return {
        "seed": 7,
        "teacher_checkpoint": str(tmp_path / "teacher.pt"),
        "dataset": {
            "name": "cifar10",
            "data_dir": str(tmp_path / "cifar10"),
            "train_size": 128,
            "val_size": 32,
        },
        "student": {
            "source": "distill_artifacts",
            "distill_artifact_root": str(tmp_path / "artifacts"),
            "distill_config": str(tmp_path / "distill.yaml"),
            "min_in_features": 1,
            "min_out_features": 1,
        },
        "train": {
            "epochs": 1,
            "batch_size_per_gpu": 8,
            "num_workers": 0,
            "optimizer": "muon_adamw",
            "schedule": "wsd",
        },
        "losses": {
            "kd_temperature": 2.0,
            "lambda_kd": 0.7,
            "lambda_ce": 0.3,
            "lambda_hidden": 0.0,
            "lambda_balance": 0.0,
        },
    }


def test_finetune_default_config_parses_train_val_only() -> None:
    config = load_finetune_run_config("configs/finetune_default.yaml", quick_smoke=True)

    assert config.teacher_checkpoint is not None
    assert config.data.use_test is False
    assert config.student.source == "distill_artifacts"
    assert config.train.optimizer == "muon_adamw"
    assert config.losses.kd_temperature == pytest.approx(4.0)


def test_finetune_config_rejects_unknown_keys_and_test_access(tmp_path: Path) -> None:
    raw = _minimal_config(tmp_path)
    with pytest.raises(ValueError, match="unknown keys"):
        parse_finetune_run_config(raw | {"surprise": 1}, quick_smoke=True)

    bad_dataset = dict(raw)
    bad_dataset["dataset"] = dict(raw["dataset"], use_test=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train-owned keys"):
        parse_finetune_run_config(bad_dataset, quick_smoke=True)

    missing_teacher = dict(raw)
    missing_teacher["teacher_checkpoint"] = None
    with pytest.raises(ValueError, match="teacher_checkpoint is required"):
        parse_finetune_run_config(missing_teacher, quick_smoke=False)


def test_kd_loss_is_finite_and_nonnegative() -> None:
    student = torch.tensor([[1.0, 0.0], [0.5, -0.25]])
    teacher = torch.tensor([[0.8, 0.2], [0.0, 1.0]])

    loss = kd_loss(student, teacher, temperature=2.0)

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_assemble_fff_student_from_artifacts_loads_strict_state(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    layer_dir = artifact_root / "work" / "0" / "trials" / "trial_000000" / "layers" / "proj"
    layer_dir.mkdir(parents=True)
    distill_config = tmp_path / "distill.yaml"
    distill_config.write_text(
        "\n".join(
            [
                "fff:",
                "  depth: 2",
                "  shared_rows: 0",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  activation: silu",
                "  hard_routing: true",
                "  route_rows_contribute: false",
                "  bias: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base = TinyLinearModel()
    replacement = make_fff_replacement(base.proj, config={"depth": 2, "leaf_rows": 1, "route_rows": 1})
    state_path = layer_dir / "fff_state.pt"
    torch.save(replacement.state_dict(), state_path)
    summary_path = artifact_root / "work" / "0" / "trials" / "trial_000000" / "layer_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "name": "proj",
                    "replacement_path": str(state_path),
                    "final_normalized_mse": 0.1,
                    "final_cosine_similarity": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = assemble_fff_student_from_artifacts(
        TinyLinearModel(),
        artifact_root=artifact_root,
        distill_config_path=distill_config,
        min_in_features=1,
        min_out_features=1,
        require_full_replacement=True,
        device=torch.device("cpu"),
    )

    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert result.manifest[0].name == "proj"


def test_assemble_fff_student_rejects_incomplete_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "layer_summary.json").write_text("[]", encoding="utf-8")
    distill_config = tmp_path / "distill.yaml"
    distill_config.write_text("fff:\n  depth: 2\n  route_rows: 1\n  leaf_rows: 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match eligible"):
        assemble_fff_student_from_artifacts(
            TinyLinearModel(),
            artifact_root=artifact_root,
            distill_config_path=distill_config,
            min_in_features=1,
            min_out_features=1,
            require_full_replacement=True,
            device=torch.device("cpu"),
        )


def test_finetune_hpo_plan_and_execute_with_fake_runner(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {
        "study_name": "test_finetune",
        "cases": [
            {
                "name": "official_wsd",
                "fine_tune_epochs": 2,
                "lr_muon": 0.001,
                "schedule": "wsd",
                "kd_temperature": 4,
            }
        ],
    }
    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "plan",
        max_trials=1,
    )
    assert plan["test_accessed"] is False
    trial_config = Path(str(plan["trials"][0]["config_path"]))
    assert trial_config.exists()

    def fake_runner(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "succeeded",
            "summary": {
                "best_val_accuracy": 0.25,
                "train_steps_total": 1,
                "student_replacement_count": 64,
                "optimizer": "muon_adamw",
                "schedule": "wsd",
            },
            "test_accessed": False,
        }

    summary = run_finetune_hpo_trials(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "execute",
        max_trials=1,
        trial_runner=fake_runner,
    )
    assert summary["succeeded"] == 1
    assert summary["test_accessed"] is False


def test_finetune_hpo_rejects_test_accessed_trial(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {"cases": [{"name": "bad", "fine_tune_epochs": 1}]}

    def fake_runner(**_kwargs: object) -> dict[str, object]:
        return {"status": "succeeded", "test_accessed": True}

    with pytest.raises(RuntimeError, match="zero successful"):
        run_finetune_hpo_trials(
            base_config=base,
            hpo_config=hpo,
            output_dir=tmp_path / "execute",
            max_trials=1,
            trial_runner=fake_runner,
        )
