from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from cifar_mamba_fff.fairness import write_fairness_reports
from cifar_mamba_fff.make_report import GC5_EXPECTED_FAMILIES, validate_final_report


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


def _rewrite_jsonl_record(path: Path, *, record_index: int, updates: dict[str, object]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[record_index].update(updates)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_gc5_validation_csvs(docs_dir: Path) -> None:
    trial_path = docs_dir / "fff_gc5_optimizer_wsd_validation_trials.csv"
    family_path = docs_dir / "fff_gc5_optimizer_wsd_validation_families.csv"
    with trial_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "machine",
                "gpu",
                "case",
                "family",
                "seed",
                "best_val_accuracy",
                "train_steps",
                "test_accessed",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for index, family in enumerate(sorted(GC5_EXPECTED_FAMILIES)):
            writer.writerow(
                {
                    "run": "gc5_test",
                    "machine": "work",
                    "gpu": str(index % 2),
                    "case": family,
                    "family": family,
                    "seed": "1337",
                    "best_val_accuracy": "0.914800"
                    if family == "official_muon_cosine_lr_base"
                    else "0.800000",
                    "train_steps": "4218",
                    "test_accessed": "false",
                    "checkpoint_path": f"outputs/gc5/{family}/student_best.pt",
                }
            )
    with family_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "trials",
                "seed_count",
                "seeds",
                "mean_best_val_accuracy",
                "std_best_val_accuracy",
                "mean_train_steps",
                "test_accessed",
                "best_case",
                "best_seed",
                "best_val_accuracy",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for family in sorted(GC5_EXPECTED_FAMILIES):
            accuracy = "0.914800" if family == "official_muon_cosine_lr_base" else "0.800000"
            writer.writerow(
                {
                    "family": family,
                    "trials": "1",
                    "seed_count": "1",
                    "seeds": "1337",
                    "mean_best_val_accuracy": accuracy,
                    "std_best_val_accuracy": "0.000000",
                    "mean_train_steps": "4218.000000",
                    "test_accessed": "false",
                    "best_case": family,
                    "best_seed": "1337",
                    "best_val_accuracy": accuracy,
                    "checkpoint_path": f"outputs/gc5/{family}/student_best.pt",
                }
            )


def test_validate_final_report_rejects_stage_h_selection_rank_drift(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    source = docs_dir / "stage_h_validation_full3ep_families.csv"
    _rewrite_csv_row(source, row_index=0, updates={"mean_best_val_accuracy": "0.100000"})
    write_fairness_reports(docs_dir=docs_dir)

    with pytest.raises(ValueError, match="does not select no_balance_cosine"):
        validate_final_report(docs_dir / "final_report.md")


def test_validate_final_report_rejects_test_accessed_selection_manifest(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    source = docs_dir / "stage_h_final_selection_manifest.jsonl"
    _rewrite_jsonl_record(source, record_index=0, updates={"test_accessed": True})
    write_fairness_reports(docs_dir=docs_dir)

    with pytest.raises(ValueError, match="selection manifest accessed CIFAR-10 test"):
        validate_final_report(docs_dir / "final_report.md")


def test_validate_final_report_checks_optional_gc5_validation_metrics(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    shutil.copytree(Path("docs"), docs_dir)
    _write_gc5_validation_csvs(docs_dir)
    write_fairness_reports(docs_dir=docs_dir)
    report = docs_dir / "final_report.md"
    report.write_text(
        report.read_text(encoding="utf-8").replace(
            "The table contains 30 rows",
            "The table contains 45 rows",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="GC5 validation trial row count"):
        validate_final_report(report)

    report.write_text(
        report.read_text(encoding="utf-8")
        + "\nGC5 optimizer/WSD validation rows: `15`\n"
        + "GC5 best validation family: `official_muon_cosine_lr_base`\n"
        + "GC5 best validation accuracy: `0.914800`\n",
        encoding="utf-8",
    )
    validate_final_report(report)


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
