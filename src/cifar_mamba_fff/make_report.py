from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from .fairness import (
    EXPECTED_TEST_ACCESS_ROWS,
    FAIRNESS_COLUMNS,
    build_fairness_rows,
    expected_fairness_rows,
    validate_fairness_rows,
)
from .utils import bool_arg

REQUIRED_SNIPPETS = (
    "## References",
    "## Environment",
    "## Beads And Git",
    "## Teacher",
    "## FFF Validation",
    "## STE And Routing",
    "## Architecture Sweep",
    "## Layerwise Distillation",
    "## End-To-End KD",
    "## Fairness",
    "## Pareto Inputs",
    "## Best Recipes",
    "## Limitations",
    "https://arxiv.org/pdf/2603.15569",
    "https://github.com/state-spaces/mamba",
    "https://github.com/KellerJordan/Muon",
    "https://github.com/pbelcak/fastfeedforward",
    "https://arxiv.org/pdf/2106.06199",
    "Stage C compared seven router recipes",
    "Stage D swept 10 one-layer architecture/recipe trials",
    "Legacy validation-capture Stage F",
    "docs/t13_stage_f_train_eval_layerwise_summary.csv",
    "Corrected Stage F train-eval rerun",
    "Full Stage H all-family FFF student final CIFAR-10 test",
    "test_accessed=true",
)

GC5_EXPECTED_FAMILIES = {
    "official_muon_cosine_lr_low",
    "official_muon_cosine_lr_base",
    "official_muon_cosine_lr_high",
    "official_muon_wsd_lr_low",
    "official_muon_wsd_lr_base",
    "official_muon_wsd_lr_high",
    "pace_muon_cosine_lr_low",
    "pace_muon_cosine_lr_base",
    "pace_muon_cosine_lr_high",
    "normuon_cosine_lr_low",
    "normuon_cosine_lr_base",
    "normuon_cosine_lr_high",
    "pace_normuon_cosine_lr_low",
    "pace_normuon_cosine_lr_base",
    "pace_normuon_cosine_lr_high",
}


def _csv_text(value: object) -> str:
    return "" if value is None else str(value)


def _canonical_evidence(value: object) -> str:
    text = _csv_text(value)
    try:
        return str(Path(text).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return text


def _canonical_fairness_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for row in rows:
        canonical.append(
            {
                column: (
                    _canonical_evidence(row.get(column, ""))
                    if column == "evidence"
                    else _csv_text(row.get(column, ""))
                )
                for column in FAIRNESS_COLUMNS
            }
        )
    return canonical


def _read_csv(path: Path, *, expected_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required report source CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path} expected {expected_rows} rows, found {len(rows)}")
    return rows


def _float(row: Mapping[str, object], key: str, *, source: Path) -> float:
    try:
        return float(str(row[key]).replace(",", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid numeric value for {key!r}") from exc


def _mean_float(rows: Sequence[Mapping[str, object]], key: str, *, source: Path) -> float:
    values = [_float(row, key, source=source) for row in rows]
    if not values:
        raise ValueError(f"{source} has no values for {key!r}")
    return sum(values) / len(values)


def _median_float(rows: Sequence[Mapping[str, object]], key: str, *, source: Path) -> float:
    values = [_float(row, key, source=source) for row in rows]
    if not values:
        raise ValueError(f"{source} has no values for {key!r}")
    return float(statistics.median(values))


def _bool(value: object, *, field: str, source: Path) -> bool:
    try:
        return bool_arg(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"{source} has invalid boolean value for {field!r}: {value!r}") from exc


def _require_all_false(rows: Sequence[Mapping[str, object]], key: str, *, source: Path) -> None:
    bad_rows = [row for row in rows if _bool(row.get(key), field=key, source=source)]
    if bad_rows:
        raise ValueError(f"{source} expected all {key} values to be false")


def _require_all_equal(
    rows: Sequence[Mapping[str, object]],
    key: str,
    expected: str,
    *,
    source: Path,
) -> None:
    bad_values = sorted({str(row.get(key, "")) for row in rows if str(row.get(key, "")) != expected})
    if bad_values:
        raise ValueError(f"{source} expected all {key} values to be {expected!r}, found {bad_values}")


def _fmt_float(value: float, digits: int, *, comma: bool = False) -> str:
    return f"{value:,.{digits}f}" if comma else f"{value:.{digits}f}"


def _fmt_int(value: object) -> str:
    return str(int(float(str(value).replace(",", ""))))


def _fmt_g(value: object) -> str:
    return f"{float(str(value).replace(',', '')):g}"


def _unbacktick(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1]
    return text


def _extract(pattern_text: str, pattern: str, *, label: str) -> str:
    match = re.search(pattern, pattern_text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"final report is missing checked prose metric: {label}")
    return match.group(1)


def _expect_metric(
    report_text: str,
    pattern: str,
    expected: str,
    *,
    label: str,
    source: Path,
) -> None:
    actual = _extract(report_text, pattern, label=label)
    if actual != expected:
        raise ValueError(
            f"final report metric mismatch for {label}: expected {expected} "
            f"from {source}, found {actual}"
        )


def _expect_contains(report_text: str, expected: str, *, label: str, source: Path) -> None:
    if expected not in report_text:
        raise ValueError(
            f"final report metric mismatch for {label}: expected text from {source}: {expected}"
        )


def _parse_markdown_rows(report_text: str, *, cells: int) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        parts = [_unbacktick(part.strip()) for part in stripped.strip("|").split("|")]
        if len(parts) == cells:
            rows[parts[0]] = parts
    return rows


def _validate_teacher_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t06_teacher_hpo_final_summary.md"
    if not source.exists():
        raise FileNotFoundError(f"required teacher summary is missing: {source}")
    source_text = source.read_text(encoding="utf-8")
    selected_val = float(
        _extract(source_text, r"Best validation accuracy: `([0-9.]+)`", label="teacher val")
    )
    final_test = float(
        _extract(source_text, r"Final test accuracy: `([0-9.]+)`", label="teacher test")
    )
    _expect_metric(
        report_text,
        r"Selected validation accuracy: `([^`]+)`",
        _fmt_float(selected_val, 4),
        label="teacher selected validation accuracy",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Final CIFAR-10 test accuracy: `([^`]+)`",
        _fmt_float(final_test, 4),
        label="teacher final CIFAR-10 test accuracy",
        source=source,
    )


def _validate_stage_c_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t13_stage_c_router_summary.csv"
    rows = _read_csv(source, expected_rows=7)
    _require_all_false(rows, "test_accessed", source=source)
    table = _parse_markdown_rows(report_text, cells=5)
    expected_names = {row["router_recipe"] for row in rows}
    actual_names = expected_names & set(table)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        raise ValueError(f"final report is missing Stage C router rows from {source}: {missing}")
    for row in rows:
        name = row["router_recipe"]
        expected = [
            name,
            _fmt_float(_float(row, "final_nmse", source=source), 6),
            _fmt_float(_float(row, "final_cosine_similarity", source=source), 6),
            _fmt_float(_float(row, "tokens_per_second", source=source), 1),
            _fmt_int(row["dead_leaves"]),
        ]
        if table[name] != expected:
            raise ValueError(
                f"final report metric mismatch for Stage C router row {name}: "
                f"expected {expected} from {source}, found {table[name]}"
            )


def _validate_stage_d_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t13_stage_d_arch_summary.csv"
    rows = _read_csv(source, expected_rows=10)
    _require_all_false(rows, "test_accessed", source=source)
    best = min(rows, key=lambda row: _float(row, "final_nmse", source=source))
    _expect_metric(
        report_text,
        r"Stage D swept ([0-9]+) one-layer architecture/recipe trials",
        str(len(rows)),
        label="Stage D trial count",
        source=source,
    )
    expected_recipe = (
        f"`{best['router_recipe']}`, `{best['route_row_role']}`, "
        f"depth `{_fmt_int(best['depth'])}`, shared fraction `{_fmt_g(best['shared_unrouted_frac'])}`,"
    )
    _expect_contains(report_text, expected_recipe, label="Stage D best recipe", source=source)
    expected_recipe_tail = (
        f"route rows `{_fmt_int(best['route_rows'])}`, leaf rows `{_fmt_int(best['leaf_rows'])}`, "
        f"LocoProp `{best['locoprop_refit']}`"
    )
    _expect_contains(
        report_text,
        expected_recipe_tail,
        label="Stage D best recipe rows",
        source=source,
    )
    _expect_metric(
        report_text,
        r"- Final NMSE: `([^`]+)`",
        _fmt_float(_float(best, "final_nmse", source=source), 6),
        label="Stage D best final NMSE",
        source=source,
    )
    _expect_metric(
        report_text,
        r"- Cosine: `([^`]+)`",
        _fmt_float(_float(best, "final_cosine_similarity", source=source), 6),
        label="Stage D best cosine similarity",
        source=source,
    )
    _expect_metric(
        report_text,
        r"- Throughput: `([^`]+)` tokens/s",
        _fmt_float(_float(best, "tokens_per_second", source=source), 1),
        label="Stage D best throughput",
        source=source,
    )
    active_rows = _extract(
        report_text,
        r"- Active rows/token: `([^`]+)`; stored rows: `([^`]+)`",
        label="Stage D active rows",
    )
    stored_rows = re.search(
        r"- Active rows/token: `[^`]+`; stored rows: `([^`]+)`", report_text
    )
    if active_rows != _fmt_float(_float(best, "active_rows_per_token", source=source), 1):
        raise ValueError(
            "final report metric mismatch for Stage D active rows/token: "
            f"expected {_fmt_float(_float(best, 'active_rows_per_token', source=source), 1)} "
            f"from {source}, found {active_rows}"
        )
    if stored_rows is None or stored_rows.group(1) != _fmt_int(best["stored_rows"]):
        found = "" if stored_rows is None else stored_rows.group(1)
        raise ValueError(
            "final report metric mismatch for Stage D stored rows: "
            f"expected {_fmt_int(best['stored_rows'])} from {source}, found {found}"
        )


def _validate_stage_f_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t13_stage_f_layerwise_summary.csv"
    rows = _read_csv(source, expected_rows=64)
    _require_all_false(rows, "test_accessed", source=source)
    _expect_metric(
        report_text,
        r"Stage F distilled all ([0-9]+) eligible Linear layers",
        str(len(rows)),
        label="Stage F eligible layer count",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Mean final normalized MSE: `([^`]+)`",
        _fmt_float(_mean_float(rows, "final_nmse", source=source), 6),
        label="Stage F mean final normalized MSE",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Median final normalized MSE: `([^`]+)`",
        _fmt_float(_median_float(rows, "final_nmse", source=source), 6),
        label="Stage F median final normalized MSE",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Mean cosine similarity: `([^`]+)`",
        _fmt_float(_mean_float(rows, "final_cosine_similarity", source=source), 6),
        label="Stage F mean cosine similarity",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Mean throughput: `([^`]+)` tokens/s",
        _fmt_float(_mean_float(rows, "tokens_per_second", source=source), 1, comma=True),
        label="Stage F mean throughput",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Mean dead leaves: `([^`]+)`",
        _fmt_float(_mean_float(rows, "dead_leaves", source=source), 2),
        label="Stage F mean dead leaves",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Mean local MSE before refit: `([^`]+)`",
        _fmt_float(_mean_float(rows, "locoprop_mse_before", source=source), 6),
        label="Stage F mean LocoProp-S MSE before",
        source=source,
    )
    _expect_metric(
        report_text,
        r"after refit: `([^`]+)`",
        _fmt_float(_mean_float(rows, "locoprop_mse_after", source=source), 6),
        label="Stage F mean LocoProp-S MSE after",
        source=source,
    )
    worst = max(rows, key=lambda row: _float(row, "final_nmse", source=source))
    _expect_metric(
        report_text,
        r"worst final NMSE was `([^`]+)`",
        _fmt_float(_float(worst, "final_nmse", source=source), 6),
        label="Stage F worst final NMSE",
        source=source,
    )
    nonincreasing = all(_bool(row.get("locoprop_nonincreasing"), field="locoprop_nonincreasing", source=source) for row in rows)
    if not nonincreasing:
        raise ValueError(f"{source} has a LocoProp-S refit that increased local MSE")


def _validate_stage_f_train_eval_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t13_stage_f_train_eval_layerwise_summary.csv"
    rows = _read_csv(source, expected_rows=64)
    _require_all_false(rows, "test_accessed", source=source)
    _require_all_equal(rows, "sample_split", "train_eval", source=source)
    _require_all_equal(rows, "metric_split", "holdout", source=source)
    _require_all_equal(rows, "scheduler_status", "succeeded", source=source)
    _require_all_equal(rows, "trial_status", "succeeded", source=source)
    eligible_indices = sorted(int(_float(row, "eligible_index", source=source)) for row in rows)
    if eligible_indices != list(range(64)):
        raise ValueError(f"{source} does not cover eligible indices 0..63 exactly once")
    for row in rows:
        fit_tokens = _float(row, "fit_tokens", source=source)
        metric_tokens = _float(row, "metric_tokens", source=source)
        if not (fit_tokens > metric_tokens > 0):
            raise ValueError(f"{source} has invalid fit/metric token split")
    _expect_metric(
        report_text,
        r"Corrected Stage F train-eval rerun distilled all ([0-9]+) eligible Linear layers",
        str(len(rows)),
        label="corrected Stage F eligible layer count",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected mean final normalized MSE: `([^`]+)`",
        _fmt_float(_mean_float(rows, "final_normalized_mse", source=source), 6),
        label="corrected Stage F mean final normalized MSE",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected median final normalized MSE: `([^`]+)`",
        _fmt_float(_median_float(rows, "final_normalized_mse", source=source), 6),
        label="corrected Stage F median final normalized MSE",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected mean cosine similarity: `([^`]+)`",
        _fmt_float(_mean_float(rows, "final_cosine_similarity", source=source), 6),
        label="corrected Stage F mean cosine similarity",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected mean throughput: `([^`]+)` tokens/s",
        _fmt_float(_mean_float(rows, "tokens_per_second", source=source), 1, comma=True),
        label="corrected Stage F mean throughput",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected mean dead leaves: `([^`]+)`",
        _fmt_float(_mean_float(rows, "dead_leaves", source=source), 2),
        label="corrected Stage F mean dead leaves",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Corrected mean local MSE before refit: `([^`]+)`",
        _fmt_float(_mean_float(rows, "locoprop_mse_before", source=source), 6),
        label="corrected Stage F mean LocoProp-S MSE before",
        source=source,
    )
    _expect_metric(
        report_text,
        r"corrected after refit: `([^`]+)`",
        _fmt_float(_mean_float(rows, "locoprop_mse_after", source=source), 6),
        label="corrected Stage F mean LocoProp-S MSE after",
        source=source,
    )
    worst = max(rows, key=lambda row: _float(row, "final_normalized_mse", source=source))
    _expect_metric(
        report_text,
        r"corrected worst final NMSE was `([^`]+)`",
        _fmt_float(_float(worst, "final_normalized_mse", source=source), 6),
        label="corrected Stage F worst final NMSE",
        source=source,
    )
    nonincreasing = all(
        _bool(row.get("locoprop_nonincreasing"), field="locoprop_nonincreasing", source=source)
        for row in rows
    )
    if not nonincreasing:
        raise ValueError(f"{source} has a LocoProp-S refit that increased local MSE")


def _validate_t20_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t20_route_row_output_ablation_results.csv"
    rows = _read_csv(source, expected_rows=7)
    _require_all_false(rows, "test_accessed", source=source)
    table = _parse_markdown_rows(report_text, cells=9)
    expected_names = {row["case_name"] for row in rows}
    actual_names = expected_names & set(table)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        raise ValueError(f"final report is missing T20 route-output rows from {source}: {missing}")
    for row in rows:
        name = row["case_name"]
        expected = [
            name,
            row["route_row_role"],
            _fmt_int(row["route_output_rows_per_node"]),
            _fmt_int(row["active_rows_per_token"]),
            _fmt_int(row["effective_stored_rows"]),
            _fmt_float(_float(row, "final_nmse", source=source), 6),
            _fmt_float(_float(row, "final_cosine_similarity", source=source), 6),
            _fmt_float(_float(row, "tokens_per_second", source=source), 1),
            _fmt_float(_float(row, "validation_accuracy_after_replacement", source=source), 4),
        ]
        if table[name] != expected:
            raise ValueError(
                f"final report metric mismatch for T20 route-output row {name}: "
                f"expected {expected} from {source}, found {table[name]}"
            )
    best = min(rows, key=lambda row: _float(row, "final_nmse", source=source))
    contributing = [row for row in rows if _bool(row["route_output_contributes"], field="route_output_contributes", source=source)]
    fastest = max(contributing, key=lambda row: _float(row, "tokens_per_second", source=source))
    _expect_metric(
        report_text,
        r"Best single-layer route-output MSE: `([^`]+)`, final NMSE `[^`]+`",
        best["case_name"],
        label="T20 best local MSE case",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Best single-layer route-output MSE: `[^`]+`, final NMSE `([^`]+)`",
        _fmt_float(_float(best, "final_nmse", source=source), 6),
        label="T20 best local MSE value",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Best single-layer route-output speed among contributing cases: `([^`]+)`, `[^`]+` tokens/s",
        fastest["case_name"],
        label="T20 fastest contributing case",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Best single-layer route-output speed among contributing cases: `[^`]+`, `([^`]+)` tokens/s",
        _fmt_float(_float(fastest, "tokens_per_second", source=source), 1),
        label="T20 fastest contributing tokens/s",
        source=source,
    )


def _validate_t14_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t14_finetune_summary.csv"
    rows = _read_csv(source, expected_rows=8)
    test_rows = [row for row in rows if _bool(row.get("test_accessed"), field="test_accessed", source=source)]
    if len(test_rows) != 1:
        raise ValueError(f"{source} expected one partial-final test row, found {len(test_rows)}")
    partial = test_rows[0]
    status = partial.get("status", "")
    match = re.search(r"partial_test_accuracy=([0-9.]+)", status)
    if match is None:
        raise ValueError(f"{source} partial test row is missing partial_test_accuracy")
    max_steps_match = re.search(r"max_test_steps=([0-9]+)", partial.get("purpose", ""))
    if max_steps_match is None:
        raise ValueError(f"{source} partial test row is missing max_test_steps in purpose")
    _expect_metric(
        report_text,
        r"Partial selected-checkpoint CIFAR-10 test evaluation: `max_test_steps=([^`]+)`",
        max_steps_match.group(1),
        label="T14 partial test max_test_steps",
        source=source,
    )
    _expect_metric(
        report_text,
        r"`test_accuracy_partial=([^`]+)`",
        _fmt_g(match.group(1)),
        label="T14 partial test accuracy",
        source=source,
    )
    _expect_metric(
        report_text,
        r"`test_accessed=([^`]+)`",
        "true",
        label="T14 partial test access",
        source=source,
    )


def _validate_stage_h_final_metrics(report_text: str, docs_dir: Path) -> None:
    validation_source = docs_dir / "stage_h_validation_full3ep_families.csv"
    validation_rows = _read_csv(validation_source, expected_rows=4)
    _require_all_false(validation_rows, "test_accessed", source=validation_source)
    validation_sorted = sorted(
        validation_rows,
        key=lambda row: _float(row, "mean_best_val_accuracy", source=validation_source),
        reverse=True,
    )
    if validation_sorted[0].get("family") != "no_balance_cosine":
        raise ValueError(f"{validation_source} does not select no_balance_cosine by validation mean")
    if _float(validation_sorted[0], "mean_best_val_accuracy", source=validation_source) <= _float(
        validation_sorted[1], "mean_best_val_accuracy", source=validation_source
    ):
        raise ValueError(f"{validation_source} selected family is not a strict validation leader")

    expected_families = {row["family"] for row in validation_rows}
    family_source = docs_dir / "stage_h_all_families_final_test_families.csv"
    family_rows = _read_csv(family_source, expected_rows=4)
    family_by_name = {str(row.get("family", "")): row for row in family_rows}
    if set(family_by_name) != expected_families:
        raise ValueError(f"{family_source} families do not match {validation_source}")
    for family_name, family_row in family_by_name.items():
        if _bool(family_row.get("partial_test_evaluation"), field="partial_test_evaluation", source=family_source):
            raise ValueError(f"{family_source} contains partial final-test rows")
        if not _bool(family_row.get("test_accessed"), field="test_accessed", source=family_source):
            raise ValueError(f"{family_source} family {family_name} did not access CIFAR-10 test")
        if _fmt_int(family_row.get("seed_count")) != "3":
            raise ValueError(f"{family_source} family {family_name} expected three final-test seeds")

    family = family_by_name["no_balance_cosine"]
    if _fmt_float(_float(family, "mean_selected_val_accuracy", source=family_source), 6) != _fmt_float(
        _float(validation_sorted[0], "mean_best_val_accuracy", source=validation_source), 6
    ):
        raise ValueError(f"{family_source} validation leader mean does not match {validation_source}")

    trial_source = docs_dir / "stage_h_all_families_final_test_trials.csv"
    trial_rows = _read_csv(trial_source, expected_rows=12)
    _require_all_equal(trial_rows, "partial_test_evaluation", "false", source=trial_source)
    _require_all_equal(trial_rows, "test_accessed", "true", source=trial_source)
    _require_all_equal(trial_rows, "allow_untracked_selection", "false", source=trial_source)
    trial_families = {str(row.get("family", "")) for row in trial_rows}
    if trial_families != expected_families:
        raise ValueError(f"{trial_source} families do not match {validation_source}")
    for family_name in expected_families:
        seeds = sorted(_fmt_int(row["seed"]) for row in trial_rows if row.get("family") == family_name)
        if seeds != ["21001", "21002", "21003"]:
            raise ValueError(f"{trial_source} family {family_name} expected seeds 21001,21002,21003")
    commits = sorted({str(row.get("remote_git_commit", "")) for row in trial_rows})
    if len(commits) != 1 or not commits[0]:
        raise ValueError(f"{trial_source} expected one non-empty final-eval remote_git_commit")
    run_ids = sorted({str(row.get("run", "")) for row in trial_rows if str(row.get("run", ""))})
    if not run_ids:
        raise ValueError(f"{trial_source} expected non-empty final-eval run ids")

    selection_source = docs_dir / "stage_h_all_families_final_selection_manifest.jsonl"
    selection_rows: list[Mapping[str, object]] = []
    with selection_source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{selection_source}:{line_number} is not a JSON object")
            selection_rows.append(payload)
    if len(selection_rows) != 12:
        raise ValueError(f"{selection_source} expected 12 all-family manifests")
    selection_by_case = {str(row.get("case", "")): row for row in selection_rows}
    if set(selection_by_case) != {str(row["case"]) for row in trial_rows}:
        raise ValueError(f"{selection_source} cases do not match {trial_source}")
    for row in trial_rows:
        selection = selection_by_case[str(row["case"])]
        if selection.get("family") != row.get("family"):
            raise ValueError(f"{selection_source} family does not match final trial row")
        if not _bool(selection.get("selected_for_final_eval"), field="selected_for_final_eval", source=selection_source):
            raise ValueError(f"{selection_source} contains unselected final manifest")
        if _bool(selection.get("test_accessed"), field="test_accessed", source=selection_source):
            raise ValueError(f"{selection_source} selection manifest accessed CIFAR-10 test")
        if str(selection.get("selection_stage")) != "stage_h_all_families_20260704":
            raise ValueError(f"{selection_source} has unexpected selection_stage")
        if "Selected every completed Stage H validation family/seed" not in str(selection.get("selection_rule", "")):
            raise ValueError(f"{selection_source} is missing the all-family selection rule")
        if _fmt_int(selection.get("seed")) != _fmt_int(row["seed"]):
            raise ValueError(f"{selection_source} seed does not match final trial row")
        if str(selection.get("checkpoint_path")) != str(row.get("checkpoint_path")):
            raise ValueError(f"{selection_source} checkpoint path does not match final trial row")
        if str(selection.get("checkpoint_sha256")) != str(row.get("checkpoint_sha256")):
            raise ValueError(f"{selection_source} checkpoint hash does not match final trial row")
        if _fmt_float(_float(selection, "best_val_accuracy", source=selection_source), 6) != _fmt_float(
            _float(row, "selected_val_accuracy", source=trial_source), 6
        ):
            raise ValueError(f"{selection_source} validation accuracy does not match final trial row")

    _expect_metric(
        report_text,
        r"Full Stage H all-family FFF student final CIFAR-10 validation-leader test mean accuracy: `([^`]+)`",
        _fmt_float(_float(family, "mean_test_accuracy", source=family_source), 6),
        label="Stage H final mean test accuracy",
        source=family_source,
    )
    _expect_metric(
        report_text,
        r"Full Stage H all-family FFF student final CIFAR-10 validation-leader test std accuracy: `([^`]+)`",
        _fmt_float(_float(family, "std_test_accuracy", source=family_source), 6),
        label="Stage H final std test accuracy",
        source=family_source,
    )
    _expect_metric(
        report_text,
        r"Full Stage H all-family FFF student final CIFAR-10 validation-leader validation mean: `([^`]+)`",
        _fmt_float(_float(family, "mean_selected_val_accuracy", source=family_source), 6),
        label="Stage H final mean selected validation accuracy",
        source=family_source,
    )
    for family_name, family_row in family_by_name.items():
        _expect_contains(
            report_text,
            f"{family_name} `{_fmt_float(_float(family_row, 'mean_test_accuracy', source=family_source), 6)}`",
            label=f"Stage H final family mean {family_name}",
            source=family_source,
        )
    _expect_contains(
        report_text,
        f"Best Stage H all-family final-test case: `{family['best_case']}`",
        label="Stage H final best case",
        source=family_source,
    )
    _expect_contains(
        report_text,
        f"seed `{_fmt_int(family['best_seed'])}`",
        label="Stage H final best seed",
        source=family_source,
    )
    _expect_contains(
        report_text,
        f"test accuracy `{_fmt_float(_float(family, 'best_test_accuracy', source=family_source), 6)}`",
        label="Stage H final best test accuracy",
        source=family_source,
    )
    _expect_contains(
        report_text,
        "Validation run: `stage_h_validation_full3ep_20260704_5b5e4e1`",
        label="Stage H final validation run id",
        source=trial_source,
    )
    _expect_contains(
        report_text,
        f"Final-eval run: `{run_ids[0]}`",
        label="Stage H final eval run id",
        source=trial_source,
    )
    _expect_contains(
        report_text,
        f"final-eval commit `{commits[0]}`",
        label="Stage H final eval commit",
        source=trial_source,
    )


def _validate_t15_metrics(
    report_text: str,
    fairness_rows: Sequence[Mapping[str, str]],
    *,
    source: Path,
) -> None:
    _expect_metric(
        report_text,
        r"The table contains ([0-9]+) rows",
        str(len(fairness_rows)),
        label="T15 fairness row count",
        source=source,
    )
    test_rows = [
        row
        for row in fairness_rows
        if str(row.get("test_accessed", "")).strip().lower() == "true"
    ]
    _expect_metric(
        report_text,
        r"Exactly ([0-9]+) fairness rows have\s+`test_accessed=true`",
        str(len(test_rows)),
        label="T15 fairness test-access row count",
        source=source,
    )
    baseline_methods = {
        "dense_teacher_copied_student": "dense_copy",
        "matched_smaller_dense_linear": "smaller_dense",
        "matched_low_rank_linear": "low_rank",
        "shared_only_rows_baseline": "shared_only",
    }
    rows_by_method = {str(row.get("method", "")): row for row in fairness_rows}
    for method, family_name in baseline_methods.items():
        row = rows_by_method.get(method)
        if row is None:
            raise ValueError(f"{source} missing baseline fairness row {method}")
        if row.get("split") != "final_test" or str(row.get("test_accessed", "")).strip().lower() != "true":
            raise ValueError(f"{source} baseline row {method} is not final-test evidence")
        _expect_contains(
            report_text,
            f"{family_name} `{_fmt_float(_float(row, 'final_test_accuracy', source=source), 6)}`",
            label=f"T15 baseline final-test mean {family_name}",
            source=source,
        )


def _validate_t19_metrics(report_text: str, docs_dir: Path) -> None:
    source = docs_dir / "t19_optimizer_ablation_summary.csv"
    rows = _read_csv(source, expected_rows=6)
    _require_all_false(rows, "test_accessed", source=source)
    steps = {_fmt_int(row["train_steps_total"]) for row in rows}
    images = {_fmt_int(row["train_images_seen"]) for row in rows}
    if len(steps) != 1 or len(images) != 1:
        raise ValueError(f"{source} T19 rows do not share a single equal smoke budget")
    _expect_metric(
        report_text,
        r"Equal budget: ([0-9]+) train steps",
        next(iter(steps)),
        label="T19 equal train-step budget",
        source=source,
    )
    _expect_metric(
        report_text,
        r"Equal budget: [0-9]+ train steps, ([0-9]+) images",
        next(iter(images)),
        label="T19 equal image budget",
        source=source,
    )
    case_names = {row["case"] for row in rows}
    expected_cases = {
        "official_muon_cosine",
        "official_muon_wsd",
        "pace_muon_ema_control",
        "pace_muon_c1e3",
        "normuon_wsd",
        "pace_normuon_c1e3",
    }
    if case_names != expected_cases:
        raise ValueError(f"{source} has unexpected T19 smoke cases: {sorted(case_names)}")
    _expect_contains(
        report_text,
        "Official Muon + cosine and Official Muon + WSD are both present.",
        label="T19 official Muon schedule summary",
        source=source,
    )
    _expect_contains(
        report_text,
        "PACE+Muon, NorMuon, and PACE+NorMuon are present as optimizer-experiments ablations.",
        label="T19 optimizer family summary",
        source=source,
    )


def _validate_gc5_metrics(report_text: str, docs_dir: Path) -> None:
    trial_source = docs_dir / "fff_gc5_optimizer_wsd_validation_trials.csv"
    family_source = docs_dir / "fff_gc5_optimizer_wsd_validation_families.csv"
    if not trial_source.exists() and not family_source.exists():
        return
    if not trial_source.exists() or not family_source.exists():
        raise FileNotFoundError(
            "GC5 validation reporting requires both "
            f"{trial_source} and {family_source}"
        )

    trial_rows = _read_csv(trial_source, expected_rows=len(GC5_EXPECTED_FAMILIES))
    family_rows = _read_csv(family_source, expected_rows=len(GC5_EXPECTED_FAMILIES))
    _require_all_false(trial_rows, "test_accessed", source=trial_source)
    _require_all_false(family_rows, "test_accessed", source=family_source)
    trial_cases = {row["case"] for row in trial_rows}
    family_names = {row["family"] for row in family_rows}
    if trial_cases != GC5_EXPECTED_FAMILIES:
        raise ValueError(f"{trial_source} has unexpected GC5 cases: {sorted(trial_cases)}")
    if family_names != GC5_EXPECTED_FAMILIES:
        raise ValueError(f"{family_source} has unexpected GC5 families: {sorted(family_names)}")
    trial_seeds = {_fmt_int(row["seed"]) for row in trial_rows}
    family_seed_lists = {row["seeds"] for row in family_rows}
    family_best_seeds = {_fmt_int(row["best_seed"]) for row in family_rows}
    family_seed_counts = {_fmt_int(row["seed_count"]) for row in family_rows}
    family_trial_counts = {_fmt_int(row["trials"]) for row in family_rows}
    if (
        trial_seeds != {"1337"}
        or family_seed_lists != {"1337"}
        or family_best_seeds != {"1337"}
        or family_seed_counts != {"1"}
        or family_trial_counts != {"1"}
    ):
        raise ValueError(
            f"{trial_source} and {family_source} must contain exactly one seed 1337 trial per GC5 family"
        )
    trial_steps = {_fmt_int(row["train_steps"]) for row in trial_rows}
    family_steps = {_fmt_int(row["mean_train_steps"]) for row in family_rows}
    if trial_steps != {"4218"} or family_steps != {"4218"}:
        raise ValueError(
            f"{trial_source} and {family_source} must use the matched 4218-step GC5 budget"
        )
    best_family = max(
        family_rows,
        key=lambda row: (
            float(row["mean_best_val_accuracy"]),
            row["family"],
        ),
    )
    _expect_metric(
        report_text,
        r"GC5 optimizer/WSD validation rows: `([0-9]+)`",
        str(len(trial_rows)),
        label="GC5 validation trial row count",
        source=trial_source,
    )
    _expect_metric(
        report_text,
        r"GC5 best validation family: `([^`]+)`",
        best_family["family"],
        label="GC5 best validation family",
        source=family_source,
    )
    _expect_metric(
        report_text,
        r"GC5 best validation accuracy: `([0-9.]+)`",
        best_family["mean_best_val_accuracy"],
        label="GC5 best validation accuracy",
        source=family_source,
    )
    _expect_contains(
        report_text,
        "WSD cells in GC5 are limited to the official Muon family",
        label="GC5 WSD coverage asymmetry",
        source=family_source,
    )
    _expect_contains(
        report_text,
        "--expect-rows 15",
        label="GC5 guarded regeneration command",
        source=trial_source,
    )


def _validate_source_metrics(
    report_text: str,
    docs_dir: Path,
    fairness_rows: Sequence[Mapping[str, str]],
) -> None:
    _validate_teacher_metrics(report_text, docs_dir)
    _validate_stage_c_metrics(report_text, docs_dir)
    _validate_stage_d_metrics(report_text, docs_dir)
    _validate_stage_f_metrics(report_text, docs_dir)
    _validate_stage_f_train_eval_metrics(report_text, docs_dir)
    _validate_t20_metrics(report_text, docs_dir)
    _validate_t14_metrics(report_text, docs_dir)
    _validate_stage_h_final_metrics(report_text, docs_dir)
    _validate_t15_metrics(report_text, fairness_rows, source=docs_dir / "t15_fairness_summary.csv")
    _validate_t19_metrics(report_text, docs_dir)
    _validate_gc5_metrics(report_text, docs_dir)


def validate_final_report(report: Path = Path("docs/final_report.md")) -> None:
    if not report.exists():
        raise FileNotFoundError("docs/final_report.md is missing")
    text = report.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        raise ValueError("final report is missing required snippets: " + ", ".join(missing))
    fairness_csv = report.parent / "t15_fairness_summary.csv"
    if not fairness_csv.exists():
        raise FileNotFoundError(f"fairness CSV is missing: {fairness_csv}")
    with fairness_csv.open("r", encoding="utf-8", newline="") as handle:
        fairness_rows = list(csv.DictReader(handle))
    expected_rows = expected_fairness_rows(report.parent)
    validate_fairness_rows(fairness_rows, expected_rows=expected_rows)
    if len(fairness_rows) != expected_rows:
        raise ValueError(
            f"fairness CSV expected {expected_rows} rows, found {len(fairness_rows)}"
        )
    test_rows = [
        row
        for row in fairness_rows
        if str(row.get("test_accessed", "")).strip().lower() == "true"
    ]
    if len(test_rows) != EXPECTED_TEST_ACCESS_ROWS:
        raise ValueError(
            f"fairness CSV expected {EXPECTED_TEST_ACCESS_ROWS} test-access rows, found {len(test_rows)}"
        )
    bad_test_rows = [
        row for row in test_rows if row.get("split") not in {"final_test", "partial_final_test"}
    ]
    if bad_test_rows:
        raise ValueError("fairness CSV has test access outside final/partial-final rows")
    regenerated_rows = _canonical_fairness_rows(build_fairness_rows(report.parent))
    committed_rows = _canonical_fairness_rows(fairness_rows)
    if committed_rows != regenerated_rows:
        raise ValueError("fairness CSV is stale relative to source evidence")
    _validate_source_metrics(text, report.parent, fairness_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    report = Path("docs/final_report.md")
    validate_final_report(report)
    print(f"report present: {report}")
    if args.quick_smoke:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
