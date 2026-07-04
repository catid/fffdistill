from __future__ import annotations

from pathlib import Path

import pytest

from cifar_mamba_fff.fairness import (
    build_fairness_rows,
    validate_fairness_rows,
    write_fairness_reports,
)


def test_validate_fairness_rows_allows_test_only_on_final_splits() -> None:
    validate_fairness_rows(
        [
            {"method": "teacher", "split": "final_test", "test_accessed": "true"},
            {"method": "student", "split": "partial_final_test", "test_accessed": "true"},
            {"method": "ablation", "split": "validation_smoke", "test_accessed": "false"},
        ]
    )

    with pytest.raises(ValueError, match="non-final split"):
        validate_fairness_rows(
            [{"method": "bad", "split": "validation_smoke", "test_accessed": "true"}]
        )


def test_build_fairness_rows_from_committed_docs_contains_required_placeholders() -> None:
    rows = build_fairness_rows(Path("docs"))
    by_method = {str(row["method"]): row for row in rows}

    assert by_method["dense_mamba3_teacher"]["split"] == "final_test"
    assert by_method["official_fastfeedforward_fff"]["status"] == "shape_compatible_forward_tested"
    assert by_method["dense_teacher_copied_student"]["status"] == "not_run"
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
