from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SEED_SUFFIX_RE = re.compile(r"(?:[_-]seed_?\d+)$")

TRIAL_COLUMNS = [
    "run",
    "machine",
    "gpu",
    "case",
    "family",
    "seed",
    "status",
    "best_val_accuracy",
    "train_steps",
    "test_accessed",
    "student_source",
    "replacement_count",
    "eligible_linear_count",
    "replacement_parameters_total",
    "total_active_flops_per_token",
    "total_routing_flops_per_token",
    "total_dense_flops_per_token",
    "mean_active_rows_per_token",
    "mean_active_columns_per_token",
    "mean_active_flops_per_token",
    "mean_routing_flops_per_token",
    "mean_dense_flops_per_token",
    "mean_stored_rows",
    "mean_stored_columns",
    "mean_final_normalized_mse",
    "mean_final_cosine_similarity",
    "train_images_per_second",
    "train_images_per_second_train_only",
    "checkpoint_path",
    "metrics_path",
    "mse_available",
    "cosine_available",
    "notes",
]

FAMILY_COLUMNS = [
    "family",
    "trials",
    "seed_count",
    "seeds",
    "mean_best_val_accuracy",
    "std_best_val_accuracy",
    "mean_active_rows_per_token",
    "mean_active_columns_per_token",
    "mean_active_flops_per_token",
    "mean_total_active_flops_per_token",
    "mean_total_routing_flops_per_token",
    "mean_total_dense_flops_per_token",
    "mean_train_images_per_second",
    "test_accessed",
    "best_case",
    "best_seed",
    "best_val_accuracy",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_mapping(value: object, *, source: Path | str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain a JSON object")
    return value


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def infer_family(case: str, trial_result: Mapping[str, Any]) -> str:
    overrides = trial_result.get("overrides")
    if isinstance(overrides, Mapping):
        family = overrides.get("student_source")
        if isinstance(family, str) and family:
            return family
    stripped = _SEED_SUFFIX_RE.sub("", case)
    return stripped or case


def _slot_dir_from_trial_result(trial_result_path: Path) -> Path | None:
    trial_dir = trial_result_path.parent
    if len(trial_dir.parents) < 2 or trial_dir.parent.name != "trials":
        return None
    return trial_dir.parents[1]


def _path_metadata(run_root: Path, trial_result_path: Path) -> tuple[str, str, str]:
    try:
        relative = trial_result_path.relative_to(run_root)
    except ValueError:
        relative = trial_result_path
    parts = relative.parts
    for trials_index, part in enumerate(parts):
        if part != "trials":
            continue
        if trials_index >= 2:
            machine = parts[trials_index - 2]
            gpu = parts[trials_index - 1]
            run = parts[trials_index - 3] if trials_index >= 3 else run_root.name
            return run, machine, gpu
    return run_root.name, "", ""


def _slot_status(run_root: Path, trial_result_path: Path) -> tuple[str, str, str, Mapping[str, Any]]:
    run, machine, gpu = _path_metadata(run_root, trial_result_path)
    slot_dir = _slot_dir_from_trial_result(trial_result_path)
    status: Mapping[str, Any] = {}
    if slot_dir is not None and (slot_dir / "status.json").exists():
        status = _expect_mapping(_load_json(slot_dir / "status.json"), source=slot_dir / "status.json")
    return (
        run,
        str(status.get("machine") or machine),
        str(status.get("gpu_id", status.get("gpu", gpu))),
        status,
    )


def _trial_result_paths(run_root: Path) -> list[Path]:
    if not run_root.exists():
        raise FileNotFoundError(f"collected root does not exist: {run_root}")
    return sorted(path for path in run_root.rglob("trial_result.json") if path.is_file())


def _summary_mapping(trial_result: Mapping[str, Any]) -> Mapping[str, Any]:
    result = trial_result.get("result")
    if not isinstance(result, Mapping):
        return {}
    summary = result.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _descriptor_kv(descriptor: str) -> dict[str, str]:
    text = descriptor.removeprefix("generated:")
    parts = text.split(":")
    values: dict[str, str] = {}
    if parts:
        values["source"] = parts[0]
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if sep:
            values[key] = value
    return values


def _descriptor_diagnostics(descriptor: str) -> Mapping[str, Any]:
    marker = "diagnostics="
    if marker not in descriptor:
        return {}
    text = descriptor.split(marker, 1)[1]
    try:
        loaded = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _layer_budget_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = _expect_mapping(_load_json(manifest_path), source=manifest_path)
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_layers:
        if not isinstance(raw, Mapping):
            continue
        descriptor = str(raw.get("replacement_path") or "")
        kv = _descriptor_kv(descriptor)
        diagnostics = _descriptor_diagnostics(descriptor)
        in_features = _as_float(raw.get("in_features")) or 0.0
        out_features = _as_float(raw.get("out_features")) or 0.0
        active_rows = _as_float(diagnostics.get("active_rows_per_token"))
        if active_rows is None:
            active_rows = _as_float(kv.get("active_rows"))
        active_columns = _as_float(diagnostics.get("active_columns_per_token"))
        if active_columns is None and out_features > 0:
            active_columns = out_features
        stored_rows = _as_float(diagnostics.get("stored_rows"))
        if stored_rows is None:
            stored_rows = _as_float(kv.get("stored_rows"))
        stored_columns = _as_float(diagnostics.get("stored_columns"))
        dense_flops = _as_float(diagnostics.get("estimated_dense_flops_per_token"))
        if dense_flops is None and in_features > 0 and out_features > 0:
            dense_flops = 2.0 * in_features * out_features
        active_flops = _as_float(diagnostics.get("estimated_active_flops_per_token"))
        if active_flops is None and active_rows is not None and in_features > 0 and out_features > 0:
            active_flops = 2.0 * active_rows * (in_features + out_features)
        rows.append(
            {
                "parameters": _as_float(raw.get("parameters")),
                "final_normalized_mse": _as_float(raw.get("final_normalized_mse")),
                "final_cosine_similarity": _as_float(raw.get("final_cosine_similarity")),
                "active_rows_per_token": active_rows,
                "active_columns_per_token": active_columns,
                "active_flops_per_token": active_flops,
                "routing_flops_per_token": _as_float(
                    diagnostics.get("estimated_routing_flops_per_token")
                ),
                "dense_flops_per_token": dense_flops,
                "stored_rows": stored_rows,
                "stored_columns": stored_columns,
            }
        )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _as_float(row.get(key))) is not None]
    return statistics.fmean(values) if values else None


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _as_float(row.get(key))) is not None]
    return sum(values) if values else None


def _trial_row(run_root: Path, trial_result_path: Path) -> dict[str, Any] | None:
    trial_result = _expect_mapping(_load_json(trial_result_path), source=trial_result_path)
    result = trial_result.get("result") if isinstance(trial_result.get("result"), Mapping) else {}
    summary = _summary_mapping(trial_result)
    status = str(trial_result.get("status") or result.get("status") or "")
    if status != "succeeded":
        return None
    run, machine, gpu, slot_status = _slot_status(run_root, trial_result_path)
    slot_status_text = str(slot_status.get("status") or "")
    if slot_status_text and slot_status_text != "succeeded":
        return None
    case = str(trial_result.get("case") or trial_result_path.parent.name)
    overrides = trial_result.get("overrides") if isinstance(trial_result.get("overrides"), Mapping) else {}
    seed = _as_int(overrides.get("seed") or trial_result.get("seed") or summary.get("seed"))
    best_val = _as_float(summary.get("best_val_accuracy") or result.get("best_val_accuracy"))
    train_steps = _as_int(summary.get("train_steps_total") or result.get("train_steps_total"))
    manifest_path = trial_result_path.parent / "student_assembly_manifest.json"
    layer_rows = _layer_budget_rows(manifest_path)
    mse_available = any(_as_float(row.get("final_normalized_mse")) is not None for row in layer_rows)
    cosine_available = any(
        _as_float(row.get("final_cosine_similarity")) is not None for row in layer_rows
    )
    if seed is None or best_val is None or train_steps is None:
        raise ValueError(f"{trial_result_path} is missing seed, best_val_accuracy, or train_steps")
    return {
        "run": run,
        "machine": machine,
        "gpu": gpu,
        "case": case,
        "family": infer_family(case, trial_result),
        "seed": seed,
        "status": "succeeded",
        "best_val_accuracy": best_val,
        "train_steps": train_steps,
        "test_accessed": _as_bool(
            summary.get("test_accessed")
            if "test_accessed" in summary
            else result.get("test_accessed", trial_result.get("test_accessed", False))
        ),
        "student_source": summary.get("student_source") or overrides.get("student_source") or "",
        "replacement_count": summary.get("student_replacement_count") or "",
        "eligible_linear_count": summary.get("eligible_linear_count") or "",
        "replacement_parameters_total": _sum(layer_rows, "parameters"),
        "total_active_flops_per_token": _sum(layer_rows, "active_flops_per_token"),
        "total_routing_flops_per_token": _sum(layer_rows, "routing_flops_per_token"),
        "total_dense_flops_per_token": _sum(layer_rows, "dense_flops_per_token"),
        "mean_active_rows_per_token": _mean(layer_rows, "active_rows_per_token"),
        "mean_active_columns_per_token": _mean(layer_rows, "active_columns_per_token"),
        "mean_active_flops_per_token": _mean(layer_rows, "active_flops_per_token"),
        "mean_routing_flops_per_token": _mean(layer_rows, "routing_flops_per_token"),
        "mean_dense_flops_per_token": _mean(layer_rows, "dense_flops_per_token"),
        "mean_stored_rows": _mean(layer_rows, "stored_rows"),
        "mean_stored_columns": _mean(layer_rows, "stored_columns"),
        "mean_final_normalized_mse": _mean(layer_rows, "final_normalized_mse"),
        "mean_final_cosine_similarity": _mean(layer_rows, "final_cosine_similarity"),
        "train_images_per_second": _as_float(summary.get("train_images_per_second")),
        "train_images_per_second_train_only": _as_float(
            summary.get("train_images_per_second_train_only")
        ),
        "checkpoint_path": summary.get("checkpoint_path") or "",
        "metrics_path": summary.get("metrics_path") or "",
        "mse_available": mse_available,
        "cosine_available": cosine_available,
        "notes": (
            "no local distillation MSE/cosine for generated full-student baseline"
            if not mse_available and not cosine_available
            else ""
        ),
        "_slot_status": slot_status,
    }


def collect_rows(collected_roots: Sequence[Path]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_failed = 0
    for root in collected_roots:
        for trial_result_path in _trial_result_paths(root):
            row = _trial_row(root, trial_result_path)
            if row is None:
                skipped_failed += 1
                continue
            rows.append(row)
    for row in rows:
        if _as_bool(row.get("test_accessed")):
            raise ValueError(f"validation summary refuses test_accessed=true row: {row['case']}")
    return sorted(rows, key=lambda row: (str(row["family"]), int(row["seed"]))), skipped_failed


def aggregate_by_family(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family") or "")].append(row)
    family_rows: list[dict[str, Any]] = []
    for family, group in grouped.items():
        val_acc = [_as_float(row.get("best_val_accuracy")) for row in group]
        val_acc = [value for value in val_acc if value is not None]
        seeds = sorted({int(seed) for row in group if (seed := _as_int(row.get("seed"))) is not None})
        best = sorted(
            group,
            key=lambda row: (
                -float(_as_float(row.get("best_val_accuracy")) or 0.0),
                int(_as_int(row.get("seed")) or 1_000_000),
            ),
        )[0]
        family_rows.append(
            {
                "family": family,
                "trials": len(group),
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "mean_best_val_accuracy": statistics.fmean(val_acc) if val_acc else None,
                "std_best_val_accuracy": statistics.stdev(val_acc) if len(val_acc) > 1 else 0.0,
                "mean_active_rows_per_token": _mean(group, "mean_active_rows_per_token"),
                "mean_active_columns_per_token": _mean(group, "mean_active_columns_per_token"),
                "mean_active_flops_per_token": _mean(group, "mean_active_flops_per_token"),
                "mean_total_active_flops_per_token": _mean(
                    group,
                    "total_active_flops_per_token",
                ),
                "mean_total_routing_flops_per_token": _mean(
                    group,
                    "total_routing_flops_per_token",
                ),
                "mean_total_dense_flops_per_token": _mean(
                    group,
                    "total_dense_flops_per_token",
                ),
                "mean_train_images_per_second": _mean(group, "train_images_per_second"),
                "test_accessed": any(_as_bool(row.get("test_accessed")) for row in group),
                "best_case": best.get("case", ""),
                "best_seed": best.get("seed", ""),
                "best_val_accuracy": best.get("best_val_accuracy", ""),
            }
        )
    return sorted(
        family_rows,
        key=lambda row: (-float(_as_float(row.get("mean_best_val_accuracy")) or -1.0), str(row["family"])),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in columns})


def write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    collected_roots: Sequence[Path],
    skipped_failed: int,
    title: str,
) -> None:
    families = aggregate_by_family(rows)
    content = [
        f"# {title}",
        "",
        f"- Collected roots: `{', '.join(str(root) for root in collected_roots)}`",
        f"- Succeeded trial rows: `{len(rows)}`",
        f"- Skipped failed trial rows: `{skipped_failed}`",
        f"- Families: `{len(families)}`",
        "- CIFAR-10 test accessed: `false`",
        "- MSE/cosine: generated full-student baselines do not run local per-layer teacher regression, so local MSE/cosine columns are marked unavailable unless future artifacts add them.",
        "",
        "## Family Aggregate",
        "",
        _markdown_table(
            [
                "Family",
                "Trials",
                "Seeds",
                "Mean val acc",
                "Std val acc",
                "Mean active rows",
                "Mean active columns",
                "Mean active FLOPs/token",
                "Mean total active FLOPs/token",
                "Mean img/s",
                "Best case",
                "Best val acc",
            ],
            [
                [
                    row["family"],
                    row["trials"],
                    row["seeds"],
                    row["mean_best_val_accuracy"],
                    row["std_best_val_accuracy"],
                    row["mean_active_rows_per_token"],
                    row["mean_active_columns_per_token"],
                    row["mean_active_flops_per_token"],
                    row["mean_total_active_flops_per_token"],
                    row["mean_train_images_per_second"],
                    row["best_case"],
                    row["best_val_accuracy"],
                ]
                for row in families
            ],
        ),
        "",
        "## Trial Rows",
        "",
        _markdown_table(
            [
                "Case",
                "Family",
                "Seed",
                "Val acc",
                "Active rows",
                "Active columns",
                "Active FLOPs/token",
                "Total active FLOPs/token",
                "Img/s",
                "Params",
                "MSE",
                "Cosine",
            ],
            [
                [
                    row.get("case"),
                    row.get("family"),
                    row.get("seed"),
                    row.get("best_val_accuracy"),
                    row.get("mean_active_rows_per_token"),
                    row.get("mean_active_columns_per_token"),
                    row.get("mean_active_flops_per_token"),
                    row.get("total_active_flops_per_token"),
                    row.get("train_images_per_second"),
                    row.get("replacement_parameters_total"),
                    row.get("mean_final_normalized_mse"),
                    row.get("mean_final_cosine_similarity"),
                ]
                for row in rows
            ],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-root", action="append", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--family-csv-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--title", default="Generated Sublinear Baselines Validation Summary")
    parser.add_argument("--expect-rows", type=int, default=None)
    args = parser.parse_args(argv)

    roots = [Path(root) for root in args.collected_root]
    rows, skipped_failed = collect_rows(roots)
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise ValueError(f"expected {args.expect_rows} succeeded rows, found {len(rows)}")
    families = aggregate_by_family(rows)
    write_csv(Path(args.csv_out), rows, TRIAL_COLUMNS)
    write_csv(Path(args.family_csv_out), families, FAMILY_COLUMNS)
    write_markdown(
        Path(args.markdown_out),
        rows,
        collected_roots=roots,
        skipped_failed=skipped_failed,
        title=args.title,
    )
    print(f"wrote {len(rows)} rows across {len(families)} families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
