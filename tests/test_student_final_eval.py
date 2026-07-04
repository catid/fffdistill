from __future__ import annotations

from pathlib import Path

import pytest

from cifar_mamba_fff.evaluate_student import (
    _checkpoint_config_to_finetune_input,
    format_student_final_result,
    selected_student_val_accuracy,
    validate_student_final_eval_threshold,
)


def test_checkpoint_config_normalizer_accepts_resolved_data_section() -> None:
    normalized = _checkpoint_config_to_finetune_input(
        {
            "seed": 1,
            "data": {
                "data_dir": "data/cifar10",
                "train_size": 45000,
                "val_size": 5000,
                "batch_size": 32,
                "num_workers": 8,
                "seed": 1,
                "quick_smoke": False,
                "use_test": False,
            },
            "train": {},
            "student": {},
            "losses": {},
        }
    )

    dataset = normalized["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["name"] == "cifar10"
    assert dataset["data_dir"] == "data/cifar10"
    assert "use_test" not in dataset
    assert "batch_size" not in dataset


def test_selected_student_val_accuracy_requires_validation_metrics() -> None:
    assert selected_student_val_accuracy({"metrics": {"val_accuracy": 0.42}}) == pytest.approx(0.42)

    with pytest.raises(ValueError, match="validation metrics"):
        selected_student_val_accuracy({"metrics": {}})


def test_student_final_eval_threshold_requires_explicit_failure_override() -> None:
    with pytest.raises(ValueError, match="below required"):
        validate_student_final_eval_threshold(
            selected_val_accuracy=0.4,
            min_selected_val_accuracy=0.5,
            allow_below_target=False,
        )

    validate_student_final_eval_threshold(
        selected_val_accuracy=0.4,
        min_selected_val_accuracy=0.5,
        allow_below_target=True,
    )


def test_format_student_final_result_separates_partial_and_full_metrics() -> None:
    partial = format_student_final_result(
        checkpoint_path=Path("student_best.pt"),
        selected_val_accuracy_value=0.4,
        metrics={"val_accuracy": 0.3, "val_loss": 1.2, "val_steps": 1.0},
        elapsed_seconds=2.0,
        replacement_count=64,
        eligible_count=64,
        quick_smoke=True,
        max_test_steps=None,
    )
    full = format_student_final_result(
        checkpoint_path=Path("student_best.pt"),
        selected_val_accuracy_value=0.4,
        metrics={"val_accuracy": 0.31, "val_loss": 1.1, "val_steps": 10.0},
        elapsed_seconds=20.0,
        replacement_count=64,
        eligible_count=64,
        quick_smoke=False,
        max_test_steps=None,
    )

    assert partial["test_accessed"] is True
    assert partial["partial_test_evaluation"] is True
    assert partial["test_accuracy_partial"] == pytest.approx(0.3)
    assert "test_accuracy" not in partial
    assert full["partial_test_evaluation"] is False
    assert full["test_accuracy"] == pytest.approx(0.31)
    assert "test_accuracy_partial" not in full
