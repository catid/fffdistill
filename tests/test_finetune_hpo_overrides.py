from __future__ import annotations

from pathlib import Path

import yaml

from cifar_mamba_fff.finetune_student import parse_finetune_run_config
from cifar_mamba_fff.hpo.finetune_hpo import write_finetune_hpo_trial_plan


def _base_finetune_config(tmp_path: Path) -> dict[str, object]:
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
            "distill_artifact_root": str(tmp_path / "old_artifacts"),
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


def test_finetune_hpo_overrides_student_distill_artifact_root(tmp_path: Path) -> None:
    new_root = tmp_path / "new_assembled_artifacts"
    hpo_config = {
        "cases": [
            {
                "name": "new_artifacts",
                "distill_artifact_root": str(new_root),
            }
        ]
    }

    plan = write_finetune_hpo_trial_plan(
        base_config=_base_finetune_config(tmp_path),
        hpo_config=hpo_config,
        output_dir=tmp_path / "plan",
        max_trials=1,
    )

    trial_config = yaml.safe_load(
        Path(str(plan["trials"][0]["config_path"])).read_text(encoding="utf-8")
    )
    assert plan["test_accessed"] is False
    assert "distill_artifact_root" not in trial_config
    assert trial_config["student"]["distill_artifact_root"] == str(new_root)

    parsed = parse_finetune_run_config(trial_config, quick_smoke=False)
    assert parsed.student.distill_artifact_root == new_root
    assert parsed.data.use_test is False
