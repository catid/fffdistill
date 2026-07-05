from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cifar_mamba_fff.full_student_pareto import (
    PARETO_COLUMNS,
    build_full_student_pareto_rows,
    main,
    write_csv,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_full_student_pareto_merges_validation_final_and_distill(tmp_path: Path) -> None:
    validation_csv = tmp_path / "validation_families.csv"
    final_csv = tmp_path / "final_families.csv"
    distill_csv = tmp_path / "distill_layerwise.csv"
    _write_csv(
        validation_csv,
        [
            {
                "family": "alpha",
                "seed_count": 2,
                "seeds": "1,2",
                "mean_best_val_accuracy": 0.80,
                "test_accessed": "false",
            },
            {
                "family": "beta",
                "seed_count": 1,
                "seeds": "3",
                "mean_best_val_accuracy": 0.70,
                "test_accessed": "false",
            },
        ],
    )
    _write_csv(
        final_csv,
        [
            {
                "family": "alpha",
                "seed_count": 2,
                "seeds": "1,2",
                "mean_selected_val_accuracy": 0.81,
                "mean_test_accuracy": 0.79,
                "test_accessed": "true",
            }
        ],
    )
    _write_csv(
        distill_csv,
        [
            {
                "case_name": "alpha_seed1",
                "layer": "layer0",
                "active_rows_per_token": 2,
                "stored_rows": 10,
                "effective_stored_rows": 8,
                "estimated_active_flops_per_token": 100,
                "estimated_routing_flops_per_token": 5,
                "estimated_dense_flops_per_token": 200,
                "tokens_per_second": 1000,
            },
            {
                "case_name": "alpha_seed1",
                "layer": "layer1",
                "active_rows_per_token": 3,
                "stored_rows": 20,
                "effective_stored_rows": 16,
                "estimated_active_flops_per_token": 150,
                "estimated_routing_flops_per_token": 6,
                "estimated_dense_flops_per_token": 300,
                "tokens_per_second": 500,
            },
            {
                "case_name": "alpha_seed2",
                "layer": "layer0",
                "active_rows_per_token": 4,
                "stored_rows": 12,
                "effective_stored_rows": 9,
                "estimated_active_flops_per_token": 120,
                "estimated_routing_flops_per_token": 7,
                "estimated_dense_flops_per_token": 220,
                "tokens_per_second": 750,
            },
            {
                "case_name": "alpha_seed2",
                "layer": "layer1",
                "active_rows_per_token": 6,
                "stored_rows": 24,
                "effective_stored_rows": 18,
                "estimated_active_flops_per_token": 180,
                "estimated_routing_flops_per_token": 8,
                "estimated_dense_flops_per_token": 330,
                "tokens_per_second": 250,
            },
        ],
    )

    rows = build_full_student_pareto_rows([validation_csv, final_csv], [distill_csv])
    by_method = {row["method"]: row for row in rows}

    assert list(by_method) == ["alpha", "beta"]
    assert by_method["alpha"]["validation_accuracy"] == pytest.approx(0.80)
    assert by_method["alpha"]["final_test_accuracy"] == pytest.approx(0.79)
    assert by_method["alpha"]["active_rows_per_token"] == pytest.approx(7.5)
    assert by_method["alpha"]["stored_rows"] == pytest.approx(33.0)
    assert by_method["alpha"]["effective_stored_rows"] == pytest.approx(25.5)
    assert by_method["alpha"]["estimated_active_flops_per_token"] == pytest.approx(275.0)
    assert by_method["alpha"]["estimated_routing_flops_per_token"] == pytest.approx(13.0)
    assert by_method["alpha"]["estimated_dense_flops_per_token"] == pytest.approx(525.0)
    assert by_method["alpha"]["tokens_per_second"] == pytest.approx(625.0)
    assert by_method["alpha"]["seed_count"] == 2
    assert by_method["alpha"]["test_accessed"] is True
    assert str(validation_csv) in by_method["alpha"]["validation_source_files"]
    assert str(final_csv) in by_method["alpha"]["final_test_source_files"]
    assert str(distill_csv) in by_method["alpha"]["distill_source_files"]

    assert by_method["beta"]["validation_accuracy"] == pytest.approx(0.70)
    assert by_method["beta"]["final_test_accuracy"] is None
    assert by_method["beta"]["active_rows_per_token"] is None
    assert by_method["beta"]["seed_count"] == 1
    assert by_method["beta"]["test_accessed"] is False


def test_write_csv_preserves_validation_only_blanks(tmp_path: Path) -> None:
    validation_csv = tmp_path / "validation_families.csv"
    out_csv = tmp_path / "pareto.csv"
    _write_csv(
        validation_csv,
        [
            {
                "family": "validation_only",
                "seed_count": 3,
                "seeds": "7,8,9",
                "mean_best_val_accuracy": 0.91,
                "test_accessed": "false",
            }
        ],
    )

    rows = build_full_student_pareto_rows([validation_csv])
    write_csv(out_csv, rows)

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == PARETO_COLUMNS
    assert written[0]["method"] == "validation_only"
    assert written[0]["validation_accuracy"] == "0.910000"
    assert written[0]["final_test_accuracy"] == ""
    assert written[0]["active_rows_per_token"] == ""
    assert written[0]["tokens_per_second"] == ""
    assert written[0]["seed_count"] == "3"
    assert written[0]["test_accessed"] == "false"


def test_cli_writes_pareto_csv_and_checks_expected_rows(tmp_path: Path) -> None:
    final_csv = tmp_path / "final_families.csv"
    out_csv = tmp_path / "pareto.csv"
    _write_csv(
        final_csv,
        [
            {
                "family": "gamma",
                "seed_count": 2,
                "seeds": "1,2",
                "mean_selected_val_accuracy": 0.88,
                "mean_test_accuracy": 0.87,
                "test_accessed": "true",
            }
        ],
    )

    assert (
        main(
            [
                "--finetune-csv",
                str(final_csv),
                "--csv-out",
                str(out_csv),
                "--expect-rows",
                "1",
            ]
        )
        == 0
    )

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["method"] == "gamma"
    assert written[0]["validation_accuracy"] == "0.880000"
    assert written[0]["final_test_accuracy"] == "0.870000"
    assert written[0]["test_accessed"] == "true"

    with pytest.raises(RuntimeError, match="expected 2 full-student Pareto rows, found 1"):
        main(
            [
                "--finetune-csv",
                str(final_csv),
                "--csv-out",
                str(tmp_path / "bad.csv"),
                "--expect-rows",
                "2",
            ]
        )
