from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from cifar_mamba_fff.fairness import (
    EXPECTED_FAIRNESS_ROWS,
    EXPECTED_GC5_FAMILY_ROWS,
    build_fairness_rows,
    expected_fairness_rows,
    validate_fairness_rows,
    write_fairness_reports,
)


def test_validate_fairness_rows_allows_test_only_on_final_splits() -> None:
    validate_fairness_rows(
        [
            {"method": "teacher", "split": "final_test", "test_accessed": "true"},
            {"method": "student", "split": "partial_final_test", "test_accessed": "true"},
            {"method": "ablation", "split": "validation_smoke", "test_accessed": "false"},
        ],
        expected_rows=3,
        expected_test_access_rows=2,
    )

    with pytest.raises(ValueError, match="non-final split"):
        validate_fairness_rows(
            [{"method": "bad", "split": "validation_smoke", "test_accessed": "true"}],
            expected_rows=1,
            expected_test_access_rows=1,
        )

    with pytest.raises(ValueError, match="test-access rows"):
        validate_fairness_rows(
            [{"method": "teacher", "split": "final_test", "test_accessed": "true"}],
            expected_rows=1,
            expected_test_access_rows=2,
        )

    validate_fairness_rows(
        [{"method": "teacher", "split": "final_test", "test_accessed": " 1 "}],
        expected_rows=1,
        expected_test_access_rows=1,
    )

    with pytest.raises(ValueError, match="invalid boolean"):
        validate_fairness_rows(
            [{"method": "bad", "split": "validation_smoke", "test_accessed": "maybe"}],
            expected_rows=1,
            expected_test_access_rows=0,
        )


def test_build_fairness_rows_from_committed_docs_contains_required_baseline_evidence() -> None:
    rows = build_fairness_rows(Path("docs"))
    by_method = {str(row["method"]): row for row in rows}

    assert len(rows) == expected_fairness_rows(Path("docs")) >= EXPECTED_FAIRNESS_ROWS
    assert by_method["dense_mamba3_teacher"]["split"] == "final_test"
    assert (
        by_method["official_fastfeedforward_fff"]["status"]
        == "shape_budget_regression_harness_available"
    )
    assert by_method["fff_student_stage_h_no_balance_cosine"]["split"] == "final_test"
    assert by_method["fff_student_stage_h_no_balance_cosine"]["status"] == "validation_leader_full_test"
    assert by_method["fff_student_stage_h_no_balance_cosine"]["test_accessed"] == "true"
    assert by_method["fff_student_stage_h_no_balance_cosine"]["final_test_accuracy"] == "0.915133"
    assert by_method["fff_student_stage_h_baseline_cosine"]["final_test_accuracy"] == "0.909133"
    assert by_method["fff_student_stage_h_wsd"]["final_test_accuracy"] == "0.905700"
    assert by_method["fff_student_stage_h_low_lr_cosine"]["final_test_accuracy"] == "0.889133"
    assert by_method["dense_teacher_copied_student"]["status"] == "validation_selected_full_test"
    assert by_method["dense_teacher_copied_student"]["split"] == "final_test"
    assert by_method["dense_teacher_copied_student"]["test_accessed"] == "true"
    assert by_method["dense_teacher_copied_student"]["validation_accuracy"] == "0.931667"
    assert by_method["dense_teacher_copied_student"]["final_test_accuracy"] == "0.927867"
    assert by_method["matched_low_rank_linear"]["status"] == "validation_selected_full_test"
    assert by_method["matched_low_rank_linear"]["validation_accuracy"] == "0.926400"
    assert by_method["matched_low_rank_linear"]["final_test_accuracy"] == "0.921200"
    assert by_method["matched_smaller_dense_linear"]["status"] == "validation_selected_full_test"
    assert by_method["matched_smaller_dense_linear"]["validation_accuracy"] == "0.931933"
    assert by_method["matched_smaller_dense_linear"]["final_test_accuracy"] == "0.927500"
    assert by_method["shared_only_rows_baseline"]["status"] == "validation_selected_full_test"
    assert by_method["shared_only_rows_baseline"]["validation_accuracy"] == "0.910000"
    assert by_method["shared_only_rows_baseline"]["final_test_accuracy"] == "0.906267"
    assert (
        by_method["assembled_fff_stage_f_validation_capture_legacy"]["split"]
        == "legacy_validation_capture_layerwise"
    )
    assert (
        by_method["assembled_fff_stage_f_validation_capture_legacy"]["status"]
        == "superseded_leakage_limited"
    )
    assert (
        by_method["assembled_fff_stage_f_train_eval_layerwise"]["split"]
        == "train_eval_layerwise_holdout"
    )
    assert (
        by_method["assembled_fff_stage_f_train_eval_layerwise"]["evidence"]
        == "docs/t13_stage_f_train_eval_layerwise_summary.csv"
    )
    assert by_method["assembled_fff_stage_f_train_eval_layerwise"]["test_accessed"] == "false"
    assert by_method["optimizer_official_muon_cosine"]["seeds"] == "7331"
    assert any(str(row["method"]).startswith("route_output_") for row in rows)
    assert any(str(row["method"]).startswith("optimizer_") for row in rows)
    assert any(str(row["split"]) == "partial_final_test" for row in rows)


def test_write_fairness_reports_writes_markdown_and_csv(tmp_path: Path) -> None:
    summary = write_fairness_reports(
        docs_dir=Path("docs"),
        markdown_path=tmp_path / "fairness.md",
        csv_path=tmp_path / "fairness.csv",
    )

    assert int(summary["rows"]) > 0
    assert (tmp_path / "fairness.md").read_text(encoding="utf-8").startswith("# T15")
    csv_text = (tmp_path / "fairness.csv").read_text(encoding="utf-8")
    assert "dense_mamba3_teacher" in csv_text
    assert "matched_low_rank_linear" in csv_text


def test_build_fairness_rows_rejects_missing_shared_only_family(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    family_path = docs_dir / "t15_baselines_final_test_families.csv"
    with family_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows = [row for row in rows if row["family"] != "shared_only"]
    with family_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="expected 4 rows"):
        build_fairness_rows(docs_dir)


def test_build_fairness_rows_adds_optional_gc5_validation_rows(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)

    rows = build_fairness_rows(docs_dir)
    by_method = {str(row["method"]): row for row in rows}

    assert expected_fairness_rows(docs_dir) == EXPECTED_FAIRNESS_ROWS + EXPECTED_GC5_FAMILY_ROWS
    assert len(rows) == EXPECTED_FAIRNESS_ROWS + EXPECTED_GC5_FAMILY_ROWS
    assert by_method["gc5_official_muon_cosine_lr_base"]["status"] == "completed"
    assert by_method["gc5_official_muon_cosine_lr_base"]["split"] == "validation"
    assert by_method["gc5_official_muon_cosine_lr_base"]["validation_accuracy"] == "0.914800"
    assert by_method["gc5_official_muon_wsd_lr_base"]["test_accessed"] == "false"


def test_build_fairness_rows_rejects_partial_gc5_validation_rows(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    gc5_path = docs_dir / "fff_gc5_optimizer_wsd_validation_families.csv"
    with gc5_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    with gc5_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:2])

    with pytest.raises(ValueError, match="expected 15 rows"):
        build_fairness_rows(docs_dir)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"family": "unexpected_gc5_family"}, "unexpected GC5 families"),
        ({"test_accessed": "true"}, "accessed CIFAR-10 test"),
        ({"seeds": "9999"}, "seed 1337"),
        ({"best_seed": "9999"}, "seed 1337"),
        ({"seed_count": "2"}, "exactly one trial and one seed"),
        ({"trials": "2"}, "exactly one trial and one seed"),
        ({"mean_train_steps": "123"}, "4218-step budget"),
    ],
)
def test_build_fairness_rows_rejects_malformed_gc5_validation_rows(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    gc5_path = docs_dir / "fff_gc5_optimizer_wsd_validation_families.csv"
    with gc5_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0].update(updates)
    with gc5_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        build_fairness_rows(docs_dir)


def test_build_fairness_rows_rejects_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required teacher summary"):
        build_fairness_rows(tmp_path)
