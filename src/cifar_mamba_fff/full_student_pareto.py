from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARETO_COLUMNS = [
    "method",
    "family",
    "validation_accuracy",
    "final_test_accuracy",
    "active_rows_per_token",
    "estimated_active_flops_per_token",
    "estimated_routing_flops_per_token",
    "estimated_dense_flops_per_token",
    "stored_rows",
    "effective_stored_rows",
    "tokens_per_second",
    "seed_count",
    "test_accessed",
    "validation_source_files",
    "final_test_source_files",
    "distill_source_files",
    "source_files",
    "provenance",
]

_METHOD_COLUMNS = ("method", "family", "case_family", "case_name", "case")
_FAMILY_COLUMNS = ("family", "case_family", "method")
_VALIDATION_ACCURACY_COLUMNS = (
    "validation_accuracy",
    "mean_validation_accuracy",
    "mean_best_val_accuracy",
    "mean_selected_val_accuracy",
    "best_val_accuracy",
    "selected_val_accuracy",
    "validation_accuracy_after_replacement",
    "val_accuracy",
)
_FINAL_TEST_ACCURACY_COLUMNS = (
    "final_test_accuracy",
    "mean_final_test_accuracy",
    "mean_test_accuracy",
    "test_accuracy",
)
_SEED_SUFFIX_RE = re.compile(r"(?:[_-]seed_?\d+)$")
_TRUE_TEXT = {"1", "true", "yes", "y", "on"}
_FALSE_TEXT = {"0", "false", "no", "n", "off"}

_COST_ALIASES: dict[str, tuple[str, ...]] = {
    "active_rows_per_token": (
        "active_rows_per_token",
        "mean_active_rows_per_token",
        "active_rows",
    ),
    "estimated_active_flops_per_token": (
        "estimated_active_flops_per_token",
        "active_flops_per_token",
    ),
    "estimated_routing_flops_per_token": (
        "estimated_routing_flops_per_token",
        "routing_flops_per_token",
    ),
    "estimated_dense_flops_per_token": (
        "estimated_dense_flops_per_token",
        "dense_flops_per_token",
    ),
    "stored_rows": ("stored_rows", "total_stored_rows"),
    "effective_stored_rows": ("effective_stored_rows", "effective_trainable_rows"),
    "tokens_per_second": (
        "tokens_per_second",
        "mean_tokens_per_second",
        "throughput_tokens_per_second",
        "throughput",
    ),
}


@dataclass(frozen=True)
class _MetricValue:
    value: float
    weight: int
    source: str
    has_final_test: bool = False


@dataclass
class _FinetuneSourceRow:
    method: str
    family: str
    source: str
    validation_accuracy: float | None
    final_test_accuracy: float | None
    seed_count: int
    seeds: set[int]
    test_accessed: bool | None
    costs: dict[str, float]


@dataclass
class _MethodRows:
    method: str
    families: list[str] = field(default_factory=list)
    validation: list[_MetricValue] = field(default_factory=list)
    final_test: list[_MetricValue] = field(default_factory=list)
    seed_counts: list[int] = field(default_factory=list)
    seeds: set[int] = field(default_factory=set)
    direct_costs: dict[str, list[_MetricValue]] = field(default_factory=lambda: defaultdict(list))
    test_accessed_values: list[bool] = field(default_factory=list)
    validation_sources: set[str] = field(default_factory=set)
    final_sources: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)


@dataclass
class _DistillAggregate:
    method: str
    costs: dict[str, float]
    source_files: set[str]
    provenance: list[str]


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if math.isfinite(parsed) else None


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in _TRUE_TEXT:
        return True
    if normalized in _FALSE_TEXT:
        return False
    return None


def _first_text(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _first_float(row: Mapping[str, Any], columns: Sequence[str]) -> float | None:
    for column in columns:
        value = _as_float(row.get(column))
        if value is not None:
            return value
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV source does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strip_seed_suffix(value: str) -> str:
    return _SEED_SUFFIX_RE.sub("", value.strip())


def _looks_generic_distill_case(value: str) -> bool:
    return value.startswith("fff_distill_") or value.startswith("distill_")


def _method_from_finetune_row(row: Mapping[str, Any], *, source: str) -> str:
    method = _first_text(row, _METHOD_COLUMNS)
    if not method:
        raise ValueError(f"{source} has a row with no method/family/case column")
    return _strip_seed_suffix(method)


def _method_from_distill_row(row: Mapping[str, Any], *, source: str) -> str:
    explicit = _first_text(row, ("method", "family", "case_family"))
    if explicit:
        return _strip_seed_suffix(explicit)
    case_name = _first_text(row, ("case_name", "case"))
    router_recipe = _first_text(row, ("router_recipe",))
    if case_name and not _looks_generic_distill_case(case_name):
        return _strip_seed_suffix(case_name)
    if router_recipe:
        return _strip_seed_suffix(router_recipe)
    if case_name:
        return _strip_seed_suffix(case_name)
    raise ValueError(f"{source} has a distill row with no method/family/case/router column")


def _source_label(path: Path, row_index: int | None = None) -> str:
    label = str(path)
    return label if row_index is None else f"{label}#row{row_index}"


def _source_file(source: str) -> str:
    return source.split("#row", maxsplit=1)[0]


def _seed_values(row: Mapping[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for column in ("seeds", "comparison_seeds", "scheduler_seeds"):
        raw = row.get(column)
        if raw in (None, ""):
            continue
        for part in re.split(r"[,; ]+", str(raw)):
            seed = _as_int(part.strip())
            if seed is not None:
                seeds.add(seed)
    for column in ("seed", "best_seed"):
        seed = _as_int(row.get(column))
        if seed is not None:
            seeds.add(seed)
    return seeds


def _row_seed_count(row: Mapping[str, Any], seeds: set[int]) -> int:
    if seeds:
        return len(seeds)
    for column in ("seed_count", "trials"):
        parsed = _as_int(row.get(column))
        if parsed is not None and parsed > 0:
            return parsed
    return 1


def _cost_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric, aliases in _COST_ALIASES.items():
        parsed = _first_float(row, aliases)
        if parsed is not None:
            values[metric] = parsed
    return values


def _finetune_rows(paths: Sequence[Path]) -> list[_FinetuneSourceRow]:
    source_rows: list[_FinetuneSourceRow] = []
    for path in paths:
        rows = _read_csv(path)
        for index, row in enumerate(rows, start=2):
            source = _source_label(path, index)
            validation_accuracy = _first_float(row, _VALIDATION_ACCURACY_COLUMNS)
            final_test_accuracy = _first_float(row, _FINAL_TEST_ACCURACY_COLUMNS)
            costs = _cost_values(row)
            if validation_accuracy is None and final_test_accuracy is None and not costs:
                continue
            method = _method_from_finetune_row(row, source=source)
            family = _first_text(row, _FAMILY_COLUMNS) or method
            seeds = _seed_values(row)
            test_accessed = _parse_bool(row.get("test_accessed"))
            if final_test_accuracy is not None:
                test_accessed = True if test_accessed is None else test_accessed
            source_rows.append(
                _FinetuneSourceRow(
                    method=method,
                    family=_strip_seed_suffix(family),
                    source=source,
                    validation_accuracy=validation_accuracy,
                    final_test_accuracy=final_test_accuracy,
                    seed_count=_row_seed_count(row, seeds),
                    seeds=seeds,
                    test_accessed=test_accessed,
                    costs=costs,
                )
            )
    return source_rows


def _case_key(row: Mapping[str, Any], method: str, source: str) -> str:
    case = _first_text(row, ("case_name", "case"))
    seed = _first_text(row, ("seed",))
    run = _first_text(row, ("run_id", "run", "run_root"))
    grid = _first_text(row, ("grid_offset",))
    if case:
        return case + (f":seed{seed}" if seed else "")
    if seed:
        return f"{method}:seed{seed}"
    if run or grid:
        return f"{method}:{run}:grid{grid}"
    return source


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _distill_aggregates(paths: Sequence[Path]) -> dict[str, _DistillAggregate]:
    by_case: dict[tuple[str, str], dict[str, Any]] = {}
    source_files: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        rows = _read_csv(path)
        for index, row in enumerate(rows, start=2):
            source = _source_label(path, index)
            method = _method_from_distill_row(row, source=source)
            costs = _cost_values(row)
            if not costs:
                continue
            source_files[method].add(str(path))
            case_key = _case_key(row, method, source)
            case = by_case.setdefault(
                (method, case_key),
                {"additive": defaultdict(float), "tokens_per_second": []},
            )
            for metric, value in costs.items():
                if metric == "tokens_per_second":
                    case["tokens_per_second"].append(value)
                else:
                    case["additive"][metric] += value

    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, _case_key_value), case in by_case.items():
        for metric, value in case["additive"].items():
            by_method[method][metric].append(float(value))
        throughput = _mean(case["tokens_per_second"])
        if throughput is not None:
            by_method[method]["tokens_per_second"].append(throughput)

    aggregates: dict[str, _DistillAggregate] = {}
    for method, metric_values in by_method.items():
        costs = {
            metric: statistics.fmean(values)
            for metric, values in metric_values.items()
            if values
        }
        case_count = max((len(values) for values in metric_values.values()), default=0)
        provenance = [
            f"distill_layerwise:{path}:case_count={case_count}"
            for path in sorted(source_files.get(method, set()))
        ]
        aggregates[method] = _DistillAggregate(
            method=method,
            costs=costs,
            source_files=set(source_files.get(method, set())),
            provenance=provenance,
        )
    return aggregates


def _weighted_mean(values: Sequence[_MetricValue]) -> float | None:
    if not values:
        return None
    total_weight = sum(max(1, value.weight) for value in values)
    if total_weight <= 0:
        return None
    return sum(value.value * max(1, value.weight) for value in values) / total_weight


def _join(values: Sequence[str] | set[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def _add_finetune_row(groups: dict[str, _MethodRows], row: _FinetuneSourceRow) -> None:
    group = groups.setdefault(row.method, _MethodRows(method=row.method))
    group.families.append(row.family)
    group.seed_counts.append(row.seed_count)
    group.seeds.update(row.seeds)
    group.source_files.add(_source_file(row.source))
    if row.test_accessed is not None:
        group.test_accessed_values.append(row.test_accessed)
    if row.validation_accuracy is not None:
        group.validation.append(
            _MetricValue(
                value=row.validation_accuracy,
                weight=row.seed_count,
                source=row.source,
                has_final_test=row.final_test_accuracy is not None,
            )
        )
        group.validation_sources.add(_source_file(row.source))
    if row.final_test_accuracy is not None:
        group.final_test.append(
            _MetricValue(
                value=row.final_test_accuracy,
                weight=row.seed_count,
                source=row.source,
                has_final_test=True,
            )
        )
        group.final_sources.add(_source_file(row.source))
    for metric, value in row.costs.items():
        group.direct_costs[metric].append(
            _MetricValue(value=value, weight=row.seed_count, source=row.source)
        )


def build_full_student_pareto_rows(
    finetune_csvs: Sequence[Path | str],
    distill_layerwise_csvs: Sequence[Path | str] = (),
) -> list[dict[str, Any]]:
    """Build family-level full-student Pareto source rows from committed CSV artifacts."""
    finetune_paths = [Path(path) for path in finetune_csvs]
    distill_paths = [Path(path) for path in distill_layerwise_csvs]
    groups: dict[str, _MethodRows] = {}
    for row in _finetune_rows(finetune_paths):
        _add_finetune_row(groups, row)

    distill = _distill_aggregates(distill_paths)
    output: list[dict[str, Any]] = []
    for method, group in groups.items():
        validation_values = [
            value for value in group.validation if not value.has_final_test
        ] or group.validation
        final_test_accuracy = _weighted_mean(group.final_test)
        seed_count = (
            len(group.seeds)
            if group.seeds
            else max(group.seed_counts) if group.seed_counts else 0
        )
        test_accessed = any(group.test_accessed_values) or final_test_accuracy is not None
        distill_match = distill.get(method)
        costs: dict[str, float | None] = {}
        for metric in _COST_ALIASES:
            direct_value = _weighted_mean(group.direct_costs.get(metric, []))
            costs[metric] = direct_value
            if direct_value is None and distill_match is not None:
                costs[metric] = distill_match.costs.get(metric)

        distill_sources = distill_match.source_files if distill_match is not None else set()
        validation_sources = {_source_file(value.source) for value in validation_values}
        source_files = set(group.source_files) | set(distill_sources)
        provenance = [
            *(f"validation:{source}" for source in sorted(validation_sources)),
            *(f"final_test:{source}" for source in sorted(group.final_sources)),
            *(distill_match.provenance if distill_match is not None else []),
        ]
        output.append(
            {
                "method": method,
                "family": sorted({family for family in group.families if family})[0]
                if group.families
                else method,
                "validation_accuracy": _weighted_mean(validation_values),
                "final_test_accuracy": final_test_accuracy,
                "active_rows_per_token": costs["active_rows_per_token"],
                "estimated_active_flops_per_token": costs["estimated_active_flops_per_token"],
                "estimated_routing_flops_per_token": costs["estimated_routing_flops_per_token"],
                "estimated_dense_flops_per_token": costs["estimated_dense_flops_per_token"],
                "stored_rows": costs["stored_rows"],
                "effective_stored_rows": costs["effective_stored_rows"],
                "tokens_per_second": costs["tokens_per_second"],
                "seed_count": seed_count,
                "test_accessed": test_accessed,
                "validation_source_files": _join(validation_sources),
                "final_test_source_files": _join(group.final_sources),
                "distill_source_files": _join(distill_sources),
                "source_files": _join(source_files),
                "provenance": _join(provenance),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -float(row["validation_accuracy"] or -1.0),
            str(row["method"]),
        ),
    )


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PARETO_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in PARETO_COLUMNS})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-csv", action="append", required=True)
    parser.add_argument("--distill-layerwise-csv", action="append", default=[])
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--expect-rows", type=int)
    args = parser.parse_args(argv)

    rows = build_full_student_pareto_rows(
        [Path(path) for path in args.finetune_csv],
        [Path(path) for path in args.distill_layerwise_csv],
    )
    if not rows:
        raise RuntimeError("no full-student Pareto rows were built")
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise RuntimeError(
            f"expected {args.expect_rows} full-student Pareto rows, found {len(rows)}"
        )
    write_csv(Path(args.csv_out), rows)
    print(f"wrote {len(rows)} full-student Pareto rows to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
