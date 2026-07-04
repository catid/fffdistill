from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from cifar_mamba_fff.fairness import write_fairness_reports
from cifar_mamba_fff.make_report import validate_final_report


def test_validate_final_report_accepts_committed_report() -> None:
    validate_final_report(Path("docs/final_report.md"))


def test_validate_final_report_rejects_missing_sections(tmp_path: Path) -> None:
    report = tmp_path / "final_report.md"
    report.write_text("# Final Report\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required snippets"):
        validate_final_report(report)


def test_validate_final_report_rejects_source_metric_mismatch(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    write_fairness_reports(docs_dir=docs_dir)

    report = docs_dir / "final_report.md"
    text = report.read_text(encoding="utf-8")
    report.write_text(
        text.replace(
            "Selected validation accuracy: `0.9418`",
            "Selected validation accuracy: `0.0000`",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher selected validation accuracy"):
        validate_final_report(report)


def test_validate_final_report_rejects_stale_fairness_csv(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    write_fairness_reports(docs_dir=docs_dir)

    fairness_csv = docs_dir / "t15_fairness_summary.csv"
    with fairness_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["final_test_accuracy"] = "0.000000"
    with fairness_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="stale"):
        validate_final_report(docs_dir / "final_report.md")


def _rewrite_csv_row(path: Path, *, row_index: int, updates: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[row_index].update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_final_report_rejects_stale_corrected_stage_f_metric(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    source = docs_dir / "t13_stage_f_train_eval_layerwise_summary.csv"
    _rewrite_csv_row(source, row_index=0, updates={"final_normalized_mse": "9.0"})
    write_fairness_reports(docs_dir=docs_dir)

    with pytest.raises(ValueError, match="corrected Stage F mean final normalized MSE"):
        validate_final_report(docs_dir / "final_report.md")


def test_validate_final_report_rejects_corrected_stage_f_test_access(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    source = docs_dir / "t13_stage_f_train_eval_layerwise_summary.csv"
    _rewrite_csv_row(source, row_index=0, updates={"test_accessed": "true"})

    with pytest.raises(ValueError, match="non-final split"):
        validate_final_report(docs_dir / "final_report.md")


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("sample_split", "val", "sample_split"),
        ("metric_split", "train", "metric_split"),
    ],
)
def test_validate_final_report_rejects_corrected_stage_f_split_drift(
    tmp_path: Path,
    field: str,
    bad_value: str,
    message: str,
) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    source = docs_dir / "t13_stage_f_train_eval_layerwise_summary.csv"
    _rewrite_csv_row(source, row_index=0, updates={field: bad_value})
    write_fairness_reports(docs_dir=docs_dir)

    with pytest.raises(ValueError, match=message):
        validate_final_report(docs_dir / "final_report.md")
