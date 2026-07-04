from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAIRNESS_COLUMNS)
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


def _all_false(rows: Iterable[Mapping[str, str]], key: str = "test_accessed") -> bool:
    return all(str(row.get(key, "")).strip().lower() in {"", "false", "0"} for row in rows)


def _extract_backtick_float(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        return ""
    return _float_text(match.group(1))


def _row(**kwargs: object) -> dict[str, object]:
    return {column: kwargs.get(column, "") for column in FAIRNESS_COLUMNS}


def _teacher_row(docs_dir: Path) -> dict[str, object]:
    path = docs_dir / "t06_teacher_hpo_final_summary.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _row(
        method="dense_mamba3_teacher",
        evidence=str(path),
        split="final_test",
        status="selected_full_test",
        test_accessed="true",
        validation_accuracy=_extract_backtick_float(text, r"Best validation accuracy: `([0-9.]+)`"),
        final_test_accuracy=_extract_backtick_float(text, r"Final test accuracy: `([0-9.]+)`"),
        train_steps="175 epochs",
        seeds="1 selected checkpoint",
        budget="Full teacher HPO selection followed by one full CIFAR-10 test evaluation.",
        fairness_note="Teacher is the dense reference; test was accessed only after validation selection.",
    )


def _stage_f_row(docs_dir: Path) -> dict[str, object]:
    path = docs_dir / "t13_stage_f_layerwise_summary.csv"
    rows = _read_csv(path)
    return _row(
        method="assembled_fff_stage_f_layerwise",
        evidence=str(path),
        split="validation_layerwise",
        status="completed",
        test_accessed=str(not _all_false(rows)).lower(),
        distillation_nmse=_mean(rows, "final_nmse"),
        cosine_similarity=_mean(rows, "final_cosine_similarity"),
        tokens_per_second=_mean(rows, "tokens_per_second"),
        active_rows_per_token=_mean(rows, "active_rows_per_token"),
        stored_rows=_mean(rows, "stored_rows"),
        train_steps="64 layers x short layerwise budgets",
        seeds="1",
        budget="Validation-split layerwise distillation, 2 sample batches per layer shard.",
        fairness_note="Layerwise MSE evidence only; not an end-to-end student accuracy comparison.",
    )


def _t20_rows(docs_dir: Path) -> list[dict[str, object]]:
    path = docs_dir / "t20_route_row_output_ablation_results.csv"
    rows = _read_csv(path)
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            _row(
                method=f"route_output_{row.get('case_name', '')}",
                evidence=str(path),
                split="validation_single_layer",
                status="completed",
                test_accessed=str(row.get("test_accessed", "")).lower(),
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
    rows = _read_csv(path)
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            _row(
                method=f"optimizer_{row.get('case', '')}",
                evidence=str(path),
                split="validation_smoke",
                status="completed",
                test_accessed=str(row.get("test_accessed", "")).lower(),
                validation_accuracy=_float_text(row.get("best_val_accuracy")),
                tokens_per_second=_float_text(row.get("train_images_per_second_train_only")),
                train_steps=_int_text(row.get("train_steps_total")),
                seeds="1",
                budget="Equal two-step teacher smoke budget; not a quality ranking.",
                fairness_note=(
                    f"optimizer={row.get('optimizer', '')}, schedule={row.get('schedule', '')}; "
                    "WSD is reported separately from optimizer family."
                ),
            )
        )
    return out


def _t14_rows(docs_dir: Path) -> list[dict[str, object]]:
    path = docs_dir / "t14_finetune_summary.csv"
    rows = _read_csv(path)
    out: list[dict[str, object]] = []
    for row in rows:
        purpose = row.get("purpose", "")
        status = row.get("status", "")
        partial_match = re.search(r"partial_test_accuracy=([0-9.]+)", status)
        out.append(
            _row(
                method=f"t14_{Path(row.get('run', '')).name}",
                evidence=str(path),
                split="partial_final_test" if row.get("test_accessed") == "true" else "validation_smoke",
                status=status,
                test_accessed=str(row.get("test_accessed", "")).lower(),
                partial_test_accuracy=_float_text(partial_match.group(1)) if partial_match else "",
                train_steps="1 bounded step" if "one train" in purpose else "",
                seeds="1",
                budget=purpose,
                fairness_note="T14 smoke/final-eval plumbing evidence; not a full final accuracy claim.",
            )
        )
    return out


def _baseline_placeholder_rows(docs_dir: Path) -> list[dict[str, object]]:
    evidence = "tests/test_official_fastfeedforward_baseline.py"
    return [
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
        _row(
            method="dense_teacher_copied_student",
            evidence=str(docs_dir / "t14_finetune_summary.md"),
            split="not_run",
            status="not_run",
            test_accessed="false",
            budget="Not executed as a separate T15 baseline.",
            fairness_note="Required sanity baseline remains an explicit limitation for final claims.",
        ),
        _row(
            method="shared_only_rows_baseline",
            evidence=str(docs_dir / "t13_stage_f_layerwise_summary.csv"),
            split="not_run",
            status="not_run",
            test_accessed="false",
            budget="Not executed as a matched full-layer baseline.",
            fairness_note="FFF HPO includes shared rows, but a shared-only full baseline is not available.",
        ),
        _row(
            method="matched_low_rank_linear",
            evidence=str(docs_dir / "t15_fairness_summary.md"),
            split="not_run",
            status="not_run",
            test_accessed="false",
            budget="Not executed.",
            fairness_note="Marked as optional/easy baseline in the project plan; no metric claimed.",
        ),
        _row(
            method="matched_smaller_dense_linear",
            evidence=str(docs_dir / "t15_fairness_summary.md"),
            split="not_run",
            status="not_run",
            test_accessed="false",
            budget="Not executed.",
            fairness_note="Marked as optional/easy baseline in the project plan; no metric claimed.",
        ),
    ]


def build_fairness_rows(docs_dir: Path = Path("docs")) -> list[dict[str, object]]:
    rows = [_teacher_row(docs_dir), _stage_f_row(docs_dir)]
    rows.extend(_t20_rows(docs_dir))
    rows.extend(_t19_rows(docs_dir))
    rows.extend(_t14_rows(docs_dir))
    rows.extend(_baseline_placeholder_rows(docs_dir))
    validate_fairness_rows(rows)
    return rows


def validate_fairness_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        test_accessed = str(row.get("test_accessed", "")).lower() == "true"
        split = str(row.get("split", ""))
        if test_accessed and split not in {"final_test", "partial_final_test"}:
            raise ValueError(
                f"row {row.get('method')} reports test access on non-final split {split!r}"
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
            "- Route-output ablations use one representative layer with reported active/stored row budgets.",
            "- Stage F layerwise rows report validation-split MSE/cosine/throughput, not final accuracy.",
            "- Required baselines without committed metrics are explicitly marked `not_run`.",
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
        "test_access_rows": sum(str(row.get("test_accessed", "")).lower() == "true" for row in rows),
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
