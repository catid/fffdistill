from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .utils import bool_arg

FAIRNESS_COLUMNS = [
    "method",
    "evidence",
    "split",
    "status",
    "test_accessed",
    "validation_accuracy",
    "final_test_accuracy",
    "partial_test_accuracy",
    "distillation_nmse",
    "cosine_similarity",
    "tokens_per_second",
    "active_rows_per_token",
    "stored_rows",
    "train_steps",
    "seeds",
    "budget",
    "fairness_note",
]
EXPECTED_ROW_COUNTS = {
    "t13_stage_f_layerwise_summary.csv": 64,
    "t13_stage_f_train_eval_layerwise_summary.csv": 64,
    "t20_route_row_output_ablation_results.csv": 7,
    "t19_optimizer_ablation_summary.csv": 6,
    "t14_finetune_summary.csv": 8,
    "t15_missing_baseline_validation_families.csv": 4,
    "stage_h_final_full_test_families.csv": 1,
}
EXPECTED_FAIRNESS_ROWS = 30
EXPECTED_TEST_ACCESS_ROWS = 3
GC5_FAMILY_SUMMARY = "fff_gc5_optimizer_wsd_validation_families.csv"
FFF_BANK_OPTIMIZER_CAVEAT = (
    "Current Muon grouping sends only hidden 2D matrix parameters to Muon; "
    "assembled FFF replacement banks such as route_weight, route_output, "
    "route_result_weight, route_result_output, leaf_weight, and leaf_output "
    "are 3D tensors and use AdamW fallback unless a future tested bank-specific "
    "Muon grouping is implemented."
)


def _read_csv(path: Path, *, expected_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required fairness source CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path} expected {expected_rows} rows, found {len(rows)}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAIRNESS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FAIRNESS_COLUMNS})


def _float_text(value: object, *, digits: int = 6) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _int_text(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def _mean(rows: Iterable[Mapping[str, str]], key: str) -> str:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return ""
    return _float_text(sum(values) / len(values))


def _parse_bool(value: object, *, field: str = "test_accessed") -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        raise ValueError(f"{field} is missing a boolean value")
    try:
        return bool_arg(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"{field} has invalid boolean value {value!r}") from exc


def _bool_text(value: object, *, field: str = "test_accessed") -> str:
    return str(_parse_bool(value, field=field)).lower()


def _all_false(rows: Iterable[Mapping[str, str]], key: str = "test_accessed") -> bool:
    return all(not _parse_bool(row.get(key), field=key) for row in rows)


def _test_accessed(row: Mapping[str, object]) -> bool:
    return _parse_bool(row.get("test_accessed"), field="test_accessed")


def _extract_backtick_float(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        return ""
    return _float_text(match.group(1))


def _row(**kwargs: object) -> dict[str, object]:
    return {column: kwargs.get(column, "") for column in FAIRNESS_COLUMNS}


def expected_fairness_rows(docs_dir: Path = Path("docs")) -> int:
    expected = EXPECTED_FAIRNESS_ROWS
    gc5_path = docs_dir / GC5_FAMILY_SUMMARY
    if gc5_path.exists():
        expected += len(_read_csv(gc5_path))
    return expected


def _teacher_row(docs_dir: Path) -> dict[str, object]:
    path = docs_dir / "t06_teacher_hpo_final_summary.md"
    if not path.exists():
        raise FileNotFoundError(f"required teacher summary is missing: {path}")
    text = path.read_text(encoding="utf-8")
    validation_accuracy = _extract_backtick_float(text, r"Best validation accuracy: `([0-9.]+)`")
    final_test_accuracy = _extract_backtick_float(text, r"Final test accuracy: `([0-9.]+)`")
    if not validation_accuracy or not final_test_accuracy:
        raise ValueError(f"teacher summary is missing validation/final test accuracy: {path}")
    return _row(
        method="dense_mamba3_teacher",
        evidence=str(path),
        split="final_test",
        status="selected_full_test",
        test_accessed="true",
        validation_accuracy=validation_accuracy,
        final_test_accuracy=final_test_accuracy,
        train_steps="175 epochs",
        seeds="1 selected checkpoint",
        budget="Full teacher HPO selection followed by one full CIFAR-10 test evaluation.",
        fairness_note="Teacher is the dense reference; test was accessed only after validation selection.",
    )


def _stage_f_row(docs_dir: Path) -> dict[str, object]:
    path = docs_dir / "t13_stage_f_layerwise_summary.csv"
    rows = _read_csv(path, expected_rows=EXPECTED_ROW_COUNTS[path.name])
    return _row(
        method="assembled_fff_stage_f_validation_capture_legacy",
        evidence=str(path),
        split="legacy_validation_capture_layerwise",
        status="superseded_leakage_limited",
        test_accessed=_bool_text(not _all_false(rows)),
        distillation_nmse=_mean(rows, "final_nmse"),
        cosine_similarity=_mean(rows, "final_cosine_similarity"),
        tokens_per_second=_mean(rows, "tokens_per_second"),
        active_rows_per_token=_mean(rows, "active_rows_per_token"),
        stored_rows=_mean(rows, "stored_rows"),
        train_steps="64 layers x short layerwise budgets",
        seeds="1",
        budget=(
            "Legacy validation-split layerwise distillation, 2 sample batches per layer shard; "
            "leakage-limited for layerwise validation metrics."
        ),
        fairness_note=(
            "Layerwise MSE evidence only; not an end-to-end student accuracy comparison. "
            "Superseded for corrected layerwise evidence by the train_eval held-out row."
        ),
    )


def _stage_f_train_eval_row(docs_dir: Path) -> dict[str, object]:
    path = docs_dir / "t13_stage_f_train_eval_layerwise_summary.csv"
    rows = _read_csv(path, expected_rows=EXPECTED_ROW_COUNTS[path.name])
    return _row(
        method="assembled_fff_stage_f_train_eval_layerwise",
        evidence=str(path),
        split="train_eval_layerwise_holdout",
        status="completed",
        test_accessed=_bool_text(not _all_false(rows)),
        distillation_nmse=_mean(rows, "final_normalized_mse"),
        cosine_similarity=_mean(rows, "final_cosine_similarity"),
        tokens_per_second=_mean(rows, "tokens_per_second"),
        active_rows_per_token=_mean(rows, "active_rows_per_token"),
        stored_rows=_mean(rows, "stored_rows"),
        train_steps="64 layers x short layerwise budgets",
        seeds="12 launch seeds, one shard per GPU slot",
        budget=(
            "Corrected train_eval activation capture, 2 sample batches per layer shard, "
            "10 percent held-out token metric split."
        ),
        fairness_note=(
            "Layerwise held-out token MSE evidence from CIFAR-10 train images with eval/no-augmentation "
            "transform; not an end-to-end student accuracy comparison and not CIFAR-10 final test."
        ),
    )


def _t20_rows(docs_dir: Path) -> list[dict[str, object]]:
    path = docs_dir / "t20_route_row_output_ablation_results.csv"
    rows = _read_csv(path, expected_rows=EXPECTED_ROW_COUNTS[path.name])
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            _row(
                method=f"route_output_{row.get('case_name', '')}",
                evidence=str(path),
                split="validation_single_layer",
                status="completed",
                test_accessed=_bool_text(row.get("test_accessed")),
                validation_accuracy=_float_text(row.get("validation_accuracy_after_replacement")),
                distillation_nmse=_float_text(row.get("final_nmse")),
                cosine_similarity=_float_text(row.get("final_cosine_similarity")),
                tokens_per_second=_float_text(row.get("tokens_per_second")),
                active_rows_per_token=_float_text(row.get("active_rows_per_token")),
                stored_rows=_int_text(row.get("effective_stored_rows") or row.get("stored_rows")),
                train_steps="single representative layer short budget",
                seeds="1",
                budget=(
                    "Matched representative layer and token budget; active/stored rows are "
                    "reported per route-output setting."
                ),
                fairness_note="Single-layer validation ablation; not a full-student comparison.",
            )
        )
    return out


def _t19_rows(docs_dir: Path) -> list[dict[str, object]]:
    path = docs_dir / "t19_optimizer_ablation_summary.csv"
    rows = _read_csv(path, expected_rows=EXPECTED_ROW_COUNTS[path.name])
    out: list[dict[str, object]] = []
    for row in rows:
        seeds = row.get("seed_list") or row.get("seed") or ""
        out.append(
            _row(
                method=f"optimizer_{row.get('case', '')}",
                evidence=str(path),
                split="validation_smoke",
                status="completed",
                test_accessed=_bool_text(row.get("test_accessed")),
                validation_accuracy=_float_text(row.get("best_val_accuracy")),
                tokens_per_second=_float_text(row.get("train_images_per_second_train_only")),
                train_steps=_int_text(row.get("train_steps_total")),
                seeds=seeds,
                budget=(
                    "Equal two-step teacher smoke budget; no FFF replacement banks; "
                    "not a quality ranking."
                ),
                fairness_note=(
                    f"optimizer={row.get('optimizer', '')}, schedule={row.get('schedule', '')}; "
                    "WSD is reported separately from optimizer family. "
                    "These teacher rows are not FFF-bank optimizer evidence."
                ),
            )
        )
    return out


def _t14_rows(docs_dir: Path) -> list[dict[str, object]]:
    path = docs_dir / "t14_finetune_summary.csv"
    rows = _read_csv(path, expected_rows=EXPECTED_ROW_COUNTS[path.name])
    out: list[dict[str, object]] = []
    for row in rows:
        purpose = row.get("purpose", "")
        status = row.get("status", "")
        partial_match = re.search(r"partial_test_accuracy=([0-9.]+)", status)
        test_accessed = _test_accessed(row)
        out.append(
            _row(
                method=f"t14_{Path(row.get('run', '')).name}",
                evidence=str(path),
                split="partial_final_test" if test_accessed else "validation_smoke",
                status=status,
                test_accessed=_bool_text(test_accessed),
                partial_test_accuracy=_float_text(partial_match.group(1)) if partial_match else "",
                train_steps="1 bounded step" if "one train" in purpose else "",
                seeds="1",
                budget=purpose,
                fairness_note=(
                    "T14 smoke/final-eval plumbing evidence; not a full final accuracy claim. "
                    f"{FFF_BANK_OPTIMIZER_CAVEAT}"
                ),
            )
        )
    return out


def _stage_h_final_student_row(docs_dir: Path) -> dict[str, object]:
    family_path = docs_dir / "stage_h_final_full_test_families.csv"
    rows = _read_csv(family_path, expected_rows=EXPECTED_ROW_COUNTS[family_path.name])
    row = rows[0]
    if row.get("family") != "no_balance_cosine":
        raise ValueError(f"{family_path} expected no_balance_cosine final family")
    if _parse_bool(row.get("partial_test_evaluation"), field="partial_test_evaluation"):
        raise ValueError(f"{family_path} contains partial final-test evidence")
    if not _parse_bool(row.get("test_accessed"), field="test_accessed"):
        raise ValueError(f"{family_path} final student row did not access CIFAR-10 test")
    if _int_text(row.get("seed_count")) != "3":
        raise ValueError(f"{family_path} expected three selected final-test seeds")
    return _row(
        method="fff_student_stage_h_no_balance_cosine",
        evidence=str(family_path),
        split="final_test",
        status="selected_full_test",
        test_accessed="true",
        validation_accuracy=_float_text(row.get("mean_selected_val_accuracy")),
        final_test_accuracy=_float_text(row.get("mean_test_accuracy")),
        train_steps="4218 per selected seed",
        seeds=row.get("seeds", ""),
        budget=(
            "Validation-selected no_balance_cosine family, three seeds, full CIFAR-10 "
            "test evaluation after selection."
        ),
        fairness_note=(
            f"Mean/std final test accuracy {_float_text(row.get('mean_test_accuracy'))}/"
            f"{_float_text(row.get('std_test_accuracy'))}; best case {row.get('best_case', '')} "
            f"seed {row.get('best_seed', '')} reached {_float_text(row.get('best_test_accuracy'))}. "
            "Selection manifests are committed in docs/stage_h_final_selection_manifest.jsonl."
        ),
    )


def _baseline_rows(docs_dir: Path) -> list[dict[str, object]]:
    evidence = "tests/test_official_fastfeedforward_baseline.py"
    rows = [
        _row(
            method="official_fastfeedforward_fff",
            evidence=evidence,
            split="shape_smoke",
            status="shape_compatible_forward_tested",
            test_accessed="false",
            budget="CPU API/shape smoke only.",
            fairness_note=(
                "Installed fastfeedforward.FFF wrapper is tested where shape-compatible; "
                "no matched-budget layerwise metric has been run yet."
            ),
        ),
    ]
    family_path = docs_dir / "t15_missing_baseline_validation_families.csv"
    baseline_methods = {
        "dense_copy": (
            "dense_teacher_copied_student",
            "Dense teacher-copied student sanity baseline; validation-only, no CIFAR-10 test access.",
        ),
        "low_rank": (
            "matched_low_rank_linear",
            "Matched low-rank Linear baseline using the same three-epoch validation budget.",
        ),
        "smaller_dense": (
            "matched_smaller_dense_linear",
            "Matched smaller-dense Linear baseline using the same three-epoch validation budget.",
        ),
        "shared_only": (
            "shared_only_rows_baseline",
            "Shared-only rows baseline using the same three-epoch validation budget.",
        ),
    }
    required_families = {"dense_copy", "low_rank", "smaller_dense", "shared_only"}
    seen_families: set[str] = set()
    family_rows = _read_csv(family_path, expected_rows=EXPECTED_ROW_COUNTS[family_path.name])
    for family in family_rows:
        family_name = family.get("family", "")
        if family_name not in baseline_methods:
            raise ValueError(f"unexpected baseline family in {family_path}: {family_name!r}")
        seen_families.add(family_name)
        method, note_prefix = baseline_methods[family_name]
        rows.append(
            _row(
                method=method,
                evidence=str(family_path),
                split="validation",
                status="completed",
                test_accessed=_bool_text(family.get("test_accessed")),
                validation_accuracy=_float_text(family.get("mean_best_val_accuracy")),
                train_steps=_int_text(family.get("mean_train_steps")),
                seeds=family.get("seeds", ""),
                budget=(
                    "Three seeds, three fine-tune epochs, 4218 train steps each, "
                    "same teacher checkpoint and validation-only selection protocol."
                ),
                fairness_note=(
                    f"{note_prefix} Best case {family.get('best_case', '')} "
                    f"seed {family.get('best_seed', '')} reached validation accuracy "
                    f"{_float_text(family.get('best_val_accuracy'))}."
                ),
            )
        )
    missing_required = sorted(required_families - seen_families)
    if missing_required:
        raise ValueError(f"{family_path} missing required baseline families: {missing_required}")
    return rows


def _gc5_rows(docs_dir: Path) -> list[dict[str, object]]:
    family_path = docs_dir / GC5_FAMILY_SUMMARY
    if not family_path.exists():
        return []
    rows = _read_csv(family_path)
    if not rows:
        raise ValueError(f"{family_path} exists but has no optimizer validation rows")
    out: list[dict[str, object]] = []
    for family in rows:
        family_name = family.get("family", "")
        out.append(
            _row(
                method=f"gc5_{family_name}",
                evidence=str(family_path),
                split="validation",
                status="completed",
                test_accessed=_bool_text(family.get("test_accessed")),
                validation_accuracy=_float_text(family.get("mean_best_val_accuracy")),
                tokens_per_second="",
                train_steps=_int_text(family.get("mean_train_steps")),
                seeds=family.get("seeds", ""),
                budget=(
                    "GC5 matched-budget full-student fine-tune validation cell; "
                    "same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access."
                ),
                fairness_note=(
                    f"Best case {family.get('best_case', '')} seed {family.get('best_seed', '')} "
                    f"reached validation accuracy {_float_text(family.get('best_val_accuracy'))}. "
                    "Optimizer family and LR schedule are encoded in the case/family name and reported "
                    "separately from the two-step T19 smoke rows."
                ),
            )
        )
    return out


def build_fairness_rows(docs_dir: Path = Path("docs")) -> list[dict[str, object]]:
    rows = [_teacher_row(docs_dir), _stage_f_train_eval_row(docs_dir), _stage_f_row(docs_dir)]
    rows.extend(_t20_rows(docs_dir))
    rows.extend(_t19_rows(docs_dir))
    rows.extend(_t14_rows(docs_dir))
    rows.append(_stage_h_final_student_row(docs_dir))
    rows.extend(_baseline_rows(docs_dir))
    rows.extend(_gc5_rows(docs_dir))
    validate_fairness_rows(rows, expected_rows=expected_fairness_rows(docs_dir))
    return rows


def validate_fairness_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_rows: int | None = EXPECTED_FAIRNESS_ROWS,
    expected_test_access_rows: int | None = EXPECTED_TEST_ACCESS_ROWS,
) -> None:
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} fairness rows, found {len(rows)}")
    test_access_rows = 0
    for row in rows:
        test_accessed = _test_accessed(row)
        split = str(row.get("split", ""))
        if test_accessed:
            test_access_rows += 1
        if test_accessed and split not in {"final_test", "partial_final_test"}:
            raise ValueError(
                f"row {row.get('method')} reports test access on non-final split {split!r}"
            )
    if expected_test_access_rows is not None and test_access_rows != expected_test_access_rows:
        raise ValueError(
            f"expected {expected_test_access_rows} test-access rows, found {test_access_rows}"
        )


def write_fairness_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# T15 Fair Baseline And Comparison Checks",
        "",
        "This table aggregates committed evidence only. Missing baselines are listed as",
        "`not_run` instead of being filled with invented metrics. Smoke, validation, partial",
        "final-test, and full final-test rows are deliberately separated.",
        "",
        "| Method | Split | Status | Test accessed | Val acc | Final/partial test acc | NMSE | Tokens/s | Budget note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        test_metric = row.get("final_test_accuracy") or row.get("partial_test_accuracy") or ""
        lines.append(
            "| {method} | {split} | {status} | {test_accessed} | {val} | {test} | {nmse} | {tps} | {budget} |".format(
                method=row.get("method", ""),
                split=row.get("split", ""),
                status=row.get("status", ""),
                test_accessed=row.get("test_accessed", ""),
                val=row.get("validation_accuracy", ""),
                test=test_metric,
                nmse=row.get("distillation_nmse", ""),
                tps=row.get("tokens_per_second", ""),
                budget=str(row.get("budget", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Fairness Checks",
            "",
            "- CIFAR-10 test access appears only in `final_test` or `partial_final_test` rows.",
            "- Optimizer ablations use equal two-step smoke budgets and are not ranked as final quality results.",
            f"- {FFF_BANK_OPTIMIZER_CAVEAT} Any assembled-student optimizer conclusion must "
            "state whether replacement banks used AdamW fallback or a tested Muon bank grouping.",
            "- Route-output ablations use one representative layer with reported active/stored row budgets.",
            "- Stage F layerwise rows are legacy validation-capture MSE/cosine/throughput evidence, "
            "not clean held-out validation metrics and not final accuracy.",
            "- Stage F train-eval rows use CIFAR-10 train images with eval/no-augmentation transforms "
            "and held-out token metrics; they are clean layerwise distillation evidence, not final accuracy.",
            "- Required baselines without committed metrics are explicitly marked `not_run`; "
            "completed baseline rows are validation-only unless separately marked final-test.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fairness_reports(
    *,
    docs_dir: Path = Path("docs"),
    markdown_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, object]:
    rows = build_fairness_rows(docs_dir)
    markdown_path = markdown_path or docs_dir / "t15_fairness_summary.md"
    csv_path = csv_path or docs_dir / "t15_fairness_summary.csv"
    write_fairness_markdown(markdown_path, rows)
    _write_csv(csv_path, rows)
    return {
        "rows": len(rows),
        "markdown_path": str(markdown_path),
        "csv_path": str(csv_path),
        "test_access_rows": sum(_test_accessed(row) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()
    summary = write_fairness_reports(
        docs_dir=Path(args.docs_dir),
        markdown_path=Path(args.markdown_path) if args.markdown_path else None,
        csv_path=Path(args.csv_path) if args.csv_path else None,
    )
    print(f"fairness summary complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
