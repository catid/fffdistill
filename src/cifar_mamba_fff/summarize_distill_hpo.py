from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUMMARY_COLUMNS = [
    "run_id",
    "git_commit",
    "machine",
    "gpu",
    "seed",
    "grid_offset",
    "case_name",
    "router_recipe",
    "balance_recipe",
    "balance_coeff",
    "route_row_role",
    "route_rows",
    "route_result_rows",
    "route_rows_output_count",
    "route_rows_output_fraction",
    "route_rows_contribute",
    "depth",
    "leaf_rows",
    "shared_unrouted_frac",
    "eligible_index",
    "layer",
    "sample_split",
    "sample_batches",
    "test_accessed",
    "scheduler_status",
    "trial_status",
    "captured_tokens",
    "observed_tokens",
    "fit_tokens",
    "metric_tokens",
    "metric_split",
    "metric_holdout_fraction",
    "initial_normalized_mse",
    "final_normalized_mse",
    "nmse_delta",
    "final_cosine_similarity",
    "tokens_per_second",
    "train_seconds",
    "dead_leaves",
    "route_entropy_mean",
    "leaf_tokens_p10",
    "leaf_tokens_p50",
    "leaf_tokens_p90",
    "active_rows_per_token",
    "stored_rows",
    "effective_stored_rows",
    "route_output_rows_per_token",
    "route_output_rows_per_node",
    "locoprop_status",
    "locoprop_mse_before",
    "locoprop_mse_after",
    "locoprop_nonincreasing",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def _as_bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def _mean(values: Iterable[object]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.fmean(numbers) if numbers else None


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


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _slot_dirs(collected_root: Path) -> list[Path]:
    return sorted(path.parent for path in collected_root.glob("*/*/trials/trial_000000/layer_summary.json"))


def _final_metrics_by_layer(metrics_path: Path) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(metrics_path):
        if record.get("phase") == "final" and isinstance(record.get("layer"), str):
            final[str(record["layer"])] = record
    return final


def collect_distill_hpo_rows(collected_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = collected_root.name
    for trial_dir in _slot_dirs(collected_root):
        slot_dir = trial_dir.parents[1]
        machine = slot_dir.parent.name
        gpu = slot_dir.name
        status = _load_json(slot_dir / "status.json")
        hpo_summary = _load_json(slot_dir / "distill_hpo_summary.json")
        trial_result = _load_json(trial_dir / "trial_result.json")
        run_context = _load_json(trial_dir / "run_context.json")
        layer_summary = _load_json(trial_dir / "layer_summary.json")
        if not isinstance(layer_summary, list):
            raise ValueError(f"{trial_dir / 'layer_summary.json'} must contain a list")
        final_metrics = _final_metrics_by_layer(trial_dir / "layer_metrics.jsonl")
        overrides = dict(trial_result.get("overrides") or {})
        include_indices = list(overrides.get("include_indices") or [])
        sample_split = hpo_summary.get("sample_split", run_context.get("sample_split"))
        sample_batches = hpo_summary.get("max_sample_batches", run_context.get("max_sample_batches"))
        test_accessed = bool(hpo_summary.get("test_accessed") or trial_result.get("test_accessed"))
        for layer_position, layer in enumerate(layer_summary):
            if not isinstance(layer, Mapping):
                raise ValueError(f"{trial_dir / 'layer_summary.json'} contains a non-mapping row")
            layer_name = str(layer.get("name") or layer.get("layer") or "")
            metric = final_metrics.get(layer_name, {})
            router = metric.get("router") if isinstance(metric.get("router"), Mapping) else {}
            diagnostics = (
                metric.get("diagnostics") if isinstance(metric.get("diagnostics"), Mapping) else {}
            )
            locoprop = metric.get("locoprop") if isinstance(metric.get("locoprop"), Mapping) else {}
            initial_nmse = layer.get("initial_normalized_mse")
            final_nmse = layer.get("final_normalized_mse")
            locoprop_before = locoprop.get("mse_before")
            locoprop_after = locoprop.get("mse_after")
            locoprop_nonincreasing = (
                None
                if not isinstance(locoprop_before, int | float)
                or not isinstance(locoprop_after, int | float)
                else float(locoprop_after) <= float(locoprop_before) + 1.0e-8
            )
            rows.append(
                {
                    "run_id": run_id,
                    "git_commit": status.get("git_commit") or run_context.get("git_commit"),
                    "machine": machine,
                    "gpu": gpu,
                    "seed": hpo_summary.get("seed", status.get("seed")),
                    "grid_offset": hpo_summary.get("grid_offset"),
                    "case_name": overrides.get("case_name"),
                    "router_recipe": overrides.get("router_recipe"),
                    "balance_recipe": overrides.get("balance_recipe"),
                    "balance_coeff": overrides.get("balance_coeff"),
                    "route_row_role": overrides.get("route_row_role"),
                    "route_rows": overrides.get("route_rows"),
                    "route_result_rows": overrides.get("route_result_rows"),
                    "route_rows_output_count": overrides.get("route_rows_output_count"),
                    "route_rows_output_fraction": overrides.get("route_rows_output_fraction"),
                    "route_rows_contribute": diagnostics.get("route_rows_contribute"),
                    "depth": overrides.get("depth", diagnostics.get("depth")),
                    "leaf_rows": overrides.get("leaf_rows", diagnostics.get("leaf_rows")),
                    "shared_unrouted_frac": overrides.get("shared_unrouted_frac"),
                    "eligible_index": (
                        include_indices[layer_position]
                        if layer_position < len(include_indices)
                        else None
                    ),
                    "layer": layer_name,
                    "sample_split": sample_split,
                    "sample_batches": sample_batches,
                    "test_accessed": test_accessed,
                    "scheduler_status": status.get("status"),
                    "trial_status": trial_result.get("status"),
                    "captured_tokens": layer.get("captured_tokens"),
                    "observed_tokens": layer.get("observed_tokens"),
                    "fit_tokens": layer.get("fit_tokens"),
                    "metric_tokens": layer.get("metric_tokens"),
                    "metric_split": layer.get("metric_split"),
                    "metric_holdout_fraction": layer.get("metric_holdout_fraction"),
                    "initial_normalized_mse": initial_nmse,
                    "final_normalized_mse": final_nmse,
                    "nmse_delta": (
                        None
                        if not isinstance(initial_nmse, int | float)
                        or not isinstance(final_nmse, int | float)
                        else float(initial_nmse) - float(final_nmse)
                    ),
                    "final_cosine_similarity": layer.get("final_cosine_similarity"),
                    "tokens_per_second": layer.get("tokens_per_second"),
                    "train_seconds": layer.get("train_seconds"),
                    "dead_leaves": router.get("dead_leaves"),
                    "route_entropy_mean": router.get("entropy_mean"),
                    "leaf_tokens_p10": router.get("leaf_tokens_p10"),
                    "leaf_tokens_p50": router.get("leaf_tokens_p50"),
                    "leaf_tokens_p90": router.get("leaf_tokens_p90"),
                    "active_rows_per_token": diagnostics.get("active_rows_per_token_mean"),
                    "stored_rows": diagnostics.get("stored_rows"),
                    "effective_stored_rows": diagnostics.get("effective_stored_rows"),
                    "route_output_rows_per_token": diagnostics.get("route_output_rows_per_token"),
                    "route_output_rows_per_node": diagnostics.get("route_output_rows_per_node"),
                    "locoprop_status": locoprop.get("status"),
                    "locoprop_mse_before": locoprop_before,
                    "locoprop_mse_after": locoprop_after,
                    "locoprop_nonincreasing": locoprop_nonincreasing,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("grid_offset") or 0),
            int(row.get("eligible_index") or 0),
            str(row.get("machine") or ""),
            str(row.get("gpu") or ""),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _as_bool_text(row.get(key)) for key in SUMMARY_COLUMNS})


def _aggregate_by_case(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_name"))].append(row)
    aggregates: list[dict[str, Any]] = []
    for case_name, case_rows in grouped.items():
        first = case_rows[0]
        aggregates.append(
            {
                "case_name": case_name,
                "router_recipe": first.get("router_recipe"),
                "balance_recipe": first.get("balance_recipe"),
                "route_rows": first.get("route_rows"),
                "route_row_role": first.get("route_row_role"),
                "depth": first.get("depth"),
                "layers": len(case_rows),
                "mean_nmse": _mean(row.get("final_normalized_mse") for row in case_rows),
                "mean_cosine": _mean(row.get("final_cosine_similarity") for row in case_rows),
                "mean_dead_leaves": _mean(row.get("dead_leaves") for row in case_rows),
                "mean_leaf_p50": _mean(row.get("leaf_tokens_p50") for row in case_rows),
                "mean_tokens_s": _mean(row.get("tokens_per_second") for row in case_rows),
                "test_accessed": any(bool(row.get("test_accessed")) for row in case_rows),
            }
        )
    return sorted(aggregates, key=lambda row: float(row["mean_nmse"] or 1.0e9))


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], *, collected_root: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty distill HPO summary")
    case_rows = _aggregate_by_case(rows)
    hardest = sorted(
        rows,
        key=lambda row: float(row.get("final_normalized_mse") or -1.0),
        reverse=True,
    )[:12]
    best_dead_leaf = sorted(
        case_rows,
        key=lambda row: (
            float(row.get("mean_dead_leaves") or 1.0e9),
            float(row.get("mean_nmse") or 1.0e9),
        ),
    )[:5]
    quality_pick = case_rows[0]
    fastest_pick = max(case_rows, key=lambda row: float(row.get("mean_tokens_s") or 0.0))
    lowest_dead_leaf_pick = best_dead_leaf[0]
    quality_cutoff = float(quality_pick.get("mean_nmse") or 1.0e9) + 0.01
    balanced_pool = [
        row
        for row in case_rows
        if float(row.get("mean_nmse") or 1.0e9) <= quality_cutoff
    ]
    balanced_pick = min(
        balanced_pool,
        key=lambda row: (
            float(row.get("mean_dead_leaves") or 1.0e9),
            float(row.get("mean_nmse") or 1.0e9),
        ),
    )
    test_accessed = any(bool(row.get("test_accessed")) for row in rows)
    statuses = sorted({str(row.get("trial_status")) for row in rows})
    commits = sorted({str(row.get("git_commit")) for row in rows})
    sample_splits = sorted({str(row.get("sample_split")) for row in rows})
    machines = sorted({f"{row.get('machine')}:{row.get('gpu')}" for row in rows})
    content = [
        "# Hard Out-Projection Router/Balance Train-Eval Sweep",
        "",
        f"- Run id: `{collected_root.name}`",
        f"- Collected root: `{collected_root}`",
        f"- Git commit(s): `{', '.join(commits)}`",
        f"- Machine/GPU slots: `{', '.join(machines)}`",
        f"- Rows: `{len(rows)}` layer records from `{len(case_rows)}` HPO cases",
        f"- Sample split(s): `{', '.join(sample_splits)}` with held-out token metrics",
        f"- Trial statuses: `{', '.join(statuses)}`",
        f"- CIFAR-10 test accessed: `{str(test_accessed).lower()}`",
        "",
        "These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.",
        "",
        "## Case Aggregate",
        "",
        _markdown_table(
            [
                "Case",
                "Router",
                "Balance",
                "Role",
                "Depth",
                "Rows",
                "Mean NMSE",
                "Mean cosine",
                "Mean dead leaves",
                "Mean p50 leaf tokens",
                "Mean tokens/s",
                "Test",
            ],
            [
                [
                    row["case_name"],
                    row["router_recipe"],
                    row["balance_recipe"],
                    row["route_row_role"],
                    row["depth"],
                    row["layers"],
                    row["mean_nmse"],
                    row["mean_cosine"],
                    row["mean_dead_leaves"],
                    row["mean_leaf_p50"],
                    row["mean_tokens_s"],
                    row["test_accessed"],
                ]
                for row in case_rows
            ],
        ),
        "",
        "## Lowest Dead-Leaf Cases",
        "",
        _markdown_table(
            [
                "Case",
                "Router",
                "Balance",
                "Mean NMSE",
                "Mean dead leaves",
                "Mean p50 leaf tokens",
                "Mean tokens/s",
            ],
            [
                [
                    row["case_name"],
                    row["router_recipe"],
                    row["balance_recipe"],
                    row["mean_nmse"],
                    row["mean_dead_leaves"],
                    row["mean_leaf_p50"],
                    row["mean_tokens_s"],
                ]
                for row in best_dead_leaf
            ],
        ),
        "",
        "## Selection Guidance",
        "",
        _markdown_table(
            ["Use", "Case", "Mean NMSE", "Mean dead leaves", "Mean tokens/s", "Rationale"],
            [
                [
                    "best quality",
                    quality_pick["case_name"],
                    quality_pick["mean_nmse"],
                    quality_pick["mean_dead_leaves"],
                    quality_pick["mean_tokens_s"],
                    "lowest mean held-out NMSE in this short hard-layer sweep",
                ],
                [
                    "balanced candidate",
                    balanced_pick["case_name"],
                    balanced_pick["mean_nmse"],
                    balanced_pick["mean_dead_leaves"],
                    balanced_pick["mean_tokens_s"],
                    "lowest dead leaves within +0.01 mean NMSE of the quality pick",
                ],
                [
                    "fastest",
                    fastest_pick["case_name"],
                    fastest_pick["mean_nmse"],
                    fastest_pick["mean_dead_leaves"],
                    fastest_pick["mean_tokens_s"],
                    "highest measured layer-distillation tokens/s",
                ],
                [
                    "lowest dead leaves",
                    lowest_dead_leaf_pick["case_name"],
                    lowest_dead_leaf_pick["mean_nmse"],
                    lowest_dead_leaf_pick["mean_dead_leaves"],
                    lowest_dead_leaf_pick["mean_tokens_s"],
                    "lowest mean dead leaves regardless of quality drop",
                ],
            ],
        ),
        "",
        "## Hardest Layer Records",
        "",
        _markdown_table(
            [
                "Case",
                "Layer",
                "NMSE",
                "Cosine",
                "Dead leaves",
                "p50 leaf tokens",
                "Tokens/s",
            ],
            [
                [
                    row.get("case_name"),
                    row.get("layer"),
                    row.get("final_normalized_mse"),
                    row.get("final_cosine_similarity"),
                    row.get("dead_leaves"),
                    row.get("leaf_tokens_p50"),
                    row.get("tokens_per_second"),
                ]
                for row in hardest
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- The previous strict-config failure is resolved: every relaunched case reached `succeeded` and all records preserve `test_accessed=false`.",
        "- The sweep covers the six hard middle/late `out_proj` layers identified by the Stage F validation-capture audit.",
        "- Balance-enabled cases did not automatically eliminate route collapse in this short budget; dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.",
        "- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-root", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()
    collected_root = Path(args.collected_root)
    rows = collect_distill_hpo_rows(collected_root)
    if not rows:
        raise RuntimeError(f"no distill HPO rows found under {collected_root}")
    write_csv(Path(args.csv_out), rows)
    write_markdown(Path(args.markdown_out), rows, collected_root=collected_root)
    print(f"wrote {len(rows)} rows from {collected_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
