from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cifar_mamba_fff.summarize_generated_sublinear import collect_rows, main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trial_dir(root: Path, machine: str, gpu: int, trial: str) -> Path:
    return root / machine / str(gpu) / "trials" / trial


def test_generated_sublinear_summary_extracts_budget_accounting(tmp_path: Path) -> None:
    root = tmp_path / "collected" / "generated_run"
    trial = _trial_dir(root, "work", 0, "trial_000006_sparse_column_seed31001")
    _write_json(
        root / "work" / "0" / "status.json",
        {"machine": "work", "gpu_id": 0, "status": "succeeded"},
    )
    _write_json(
        trial / "trial_result.json",
        {
            "case": "sparse_column_seed31001",
            "overrides": {"seed": 31001, "student_source": "sparse_column"},
            "status": "succeeded",
            "test_accessed": False,
            "result": {
                "status": "succeeded",
                "test_accessed": False,
                "summary": {
                    "best_val_accuracy": 0.5,
                    "train_steps_total": 12,
                    "test_accessed": False,
                    "student_source": "sparse_column",
                    "student_replacement_count": 2,
                    "eligible_linear_count": 2,
                    "train_images_per_second": 123.0,
                    "train_images_per_second_train_only": 130.0,
                    "checkpoint_path": "outputs/run/student_best.pt",
                    "metrics_path": "outputs/run/metrics.jsonl",
                },
            },
        },
    )
    descriptor = (
        "generated:sparse_column:budget=100:row_banks=64:budget_tolerance_frac=0:"
        "rows_per_token=4:column_blocks=2:column_blocks_per_token=1:"
        "diagnostics={'stored_rows': 0, 'stored_columns': 8, "
        "'active_rows_per_token': 0, 'active_columns_per_token': 4, "
        "'estimated_active_flops_per_token': 64, "
        "'estimated_routing_flops_per_token': 32, "
        "'estimated_dense_flops_per_token': 128}"
    )
    _write_json(
        trial / "student_assembly_manifest.json",
        {
            "source": "sparse_column",
            "replacement_count": 2,
            "eligible_count": 2,
            "layers": [
                {
                    "name": "a",
                    "replacement_path": descriptor,
                    "in_features": 8,
                    "out_features": 8,
                    "parameters": 100,
                    "final_normalized_mse": None,
                    "final_cosine_similarity": None,
                },
                {
                    "name": "b",
                    "replacement_path": descriptor,
                    "in_features": 8,
                    "out_features": 8,
                    "parameters": 120,
                    "final_normalized_mse": None,
                    "final_cosine_similarity": None,
                },
            ],
        },
    )

    rows, skipped = collect_rows([root])

    assert skipped == 0
    assert len(rows) == 1
    row = rows[0]
    assert row["family"] == "sparse_column"
    assert row["replacement_parameters_total"] == pytest.approx(220)
    assert row["mean_active_rows_per_token"] == pytest.approx(0)
    assert row["mean_active_columns_per_token"] == pytest.approx(4)
    assert row["mean_active_flops_per_token"] == pytest.approx(64)
    assert row["mse_available"] is False
    assert row["cosine_available"] is False


def test_generated_sublinear_cli_skips_failed_and_writes_docs(tmp_path: Path) -> None:
    root = tmp_path / "collected" / "generated_run"
    trial = _trial_dir(root, "ai", 1, "trial_000006_sparse_column_seed31001")
    _write_json(
        trial / "trial_result.json",
        {
            "case": "sparse_column_seed31001",
            "status": "failed_logic",
            "test_accessed": False,
            "result": {"status": "failed_logic", "test_accessed": False},
        },
    )
    good = _trial_dir(root, "work", 0, "trial_000003_sparse_row_seed31001")
    _write_json(
        root / "work" / "0" / "status.json",
        {"machine": "work", "gpu_id": 0, "status": "succeeded"},
    )
    _write_json(
        good / "trial_result.json",
        {
            "case": "sparse_row_seed31001",
            "overrides": {"seed": 31001, "student_source": "sparse_row"},
            "status": "succeeded",
            "test_accessed": False,
            "result": {
                "status": "succeeded",
                "test_accessed": False,
                "summary": {
                    "best_val_accuracy": 0.25,
                    "train_steps_total": 3,
                    "test_accessed": False,
                    "student_source": "sparse_row",
                    "student_replacement_count": 1,
                    "eligible_linear_count": 1,
                    "checkpoint_path": "outputs/run/student_best.pt",
                },
            },
        },
    )
    _write_json(
        good / "student_assembly_manifest.json",
        {
            "source": "sparse_row",
            "replacement_count": 1,
            "eligible_count": 1,
            "layers": [
                {
                    "name": "a",
                    "replacement_path": "generated:sparse_row:diagnostics={'active_rows_per_token': 4}",
                    "in_features": 4,
                    "out_features": 2,
                    "parameters": 10,
                    "final_normalized_mse": 0.1,
                    "final_cosine_similarity": 0.9,
                }
            ],
        },
    )
    trials_csv = tmp_path / "trials.csv"
    families_csv = tmp_path / "families.csv"
    summary_md = tmp_path / "summary.md"

    assert (
        main(
            [
                "--collected-root",
                str(root),
                "--csv-out",
                str(trials_csv),
                "--family-csv-out",
                str(families_csv),
                "--markdown-out",
                str(summary_md),
                "--expect-rows",
                "1",
            ]
        )
        == 0
    )

    with trials_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case"] == "sparse_row_seed31001"
    assert rows[0]["mse_available"] == "true"
    assert rows[0]["mean_final_normalized_mse"] == "0.100000"
    assert "Skipped failed trial rows: `1`" in summary_md.read_text(encoding="utf-8")
