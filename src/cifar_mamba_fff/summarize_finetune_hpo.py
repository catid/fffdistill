from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TRIAL_COLUMNS = [
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
]

FAMILY_COLUMNS = [
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
]

_SEED_SUFFIX_RE = re.compile(r"(?:[_-]seed_?\d+)$")
_TRUE_TEXT = {"1", "true", "yes", "y", "on"}
_FALSE_TEXT = {"0", "false", "no", "n", "off"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_mapping(value: object, *, source: Path | str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def _parse_bool(value: object, *, field: str, source: Path | str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        raise ValueError(f"{source} missing required boolean field {field}")
    normalized = str(value).strip().lower()
    if normalized in _TRUE_TEXT:
        return True
    if normalized in _FALSE_TEXT:
        return False
    raise ValueError(f"{source} has invalid boolean value for {field}: {value!r}")


def _optional_bool(value: object, *, field: str, source: Path | str) -> bool | None:
    if value is None or value == "":
        return None
    return _parse_bool(value, field=field, source=source)


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


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
        for key in ("family", "case_family", "recipe_family"):
            family = overrides.get(key)
            if isinstance(family, str) and family:
                return family
    stripped = _SEED_SUFFIX_RE.sub("", case)
    return stripped or case


def _summary_mapping(trial_result: Mapping[str, Any]) -> Mapping[str, Any]:
    result = trial_result.get("result")
    if not isinstance(result, Mapping):
        return {}
    summary = result.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _slot_dir_from_trial_result(trial_result_path: Path) -> Path | None:
    trial_dir = trial_result_path.parent
    parents = trial_dir.parents
    if len(parents) < 2 or parents[0].name != "trials":
        return None
    return parents[1]


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
        break

    slot_dir = _slot_dir_from_trial_result(trial_result_path)
    if slot_dir is None:
        return run_root.name, "", ""
    machine = slot_dir.parent.name if slot_dir.parent != slot_dir else ""
    gpu = slot_dir.name
    run_parent = slot_dir.parent.parent if slot_dir.parent != slot_dir else None
    run = run_parent.name if run_parent is not None and run_parent != run_root.parent else run_root.name
    return run, machine, gpu


def _slot_metadata(run_root: Path, trial_result_path: Path) -> tuple[str, str, str, Mapping[str, Any]]:
    slot_dir = _slot_dir_from_trial_result(trial_result_path)
    status: Mapping[str, Any] = {}
    if slot_dir is not None and (slot_dir / "status.json").exists():
        status = _expect_mapping(_load_json(slot_dir / "status.json"), source=slot_dir / "status.json")

    path_run, path_machine, path_gpu = _path_metadata(run_root, trial_result_path)
    machine: object = status.get("machine")
    gpu: object = status.get("gpu_id", status.get("gpu"))
    machine = machine or path_machine
    gpu = gpu if gpu not in (None, "") else path_gpu
    return path_run, str(machine or ""), str(gpu if gpu not in (None, "") else ""), status


def _hpo_summary_for_trial(trial_result_path: Path) -> Mapping[str, Any]:
    slot_dir = _slot_dir_from_trial_result(trial_result_path)
    if slot_dir is None:
        return {}
    summary_path = slot_dir / "finetune_hpo_summary.json"
    if not summary_path.exists():
        return {}
    return _expect_mapping(_load_json(summary_path), source=summary_path)


def _test_accessed(
    trial_result: Mapping[str, Any],
    hpo_summary: Mapping[str, Any],
    *,
    source: Path,
) -> bool:
    result = trial_result.get("result") if isinstance(trial_result.get("result"), Mapping) else {}
    summary = _summary_mapping(trial_result)
    candidates = [
        ("finetune_hpo_summary.test_accessed", hpo_summary.get("test_accessed")),
        ("trial_result.test_accessed", trial_result.get("test_accessed")),
        ("trial_result.result.test_accessed", result.get("test_accessed")),
        ("trial_result.result.summary.test_accessed", summary.get("test_accessed")),
    ]
    parsed = [
        value
        for field, raw in candidates
        if (value := _optional_bool(raw, field=field, source=source)) is not None
    ]
    if not parsed:
        raise ValueError(f"{source} has no test_accessed field in trial, result, summary, or HPO summary")
    return any(parsed)


def _trial_result_paths(run_root: Path) -> list[Path]:
    if not run_root.exists():
        raise FileNotFoundError(f"collected root does not exist: {run_root}")
    return sorted(path for path in run_root.rglob("trial_result.json") if path.is_file())


def _require_status_succeeded(value: object, *, field: str, source: Path) -> None:
    if value in (None, ""):
        raise ValueError(f"{source} missing required status field {field}")
    if str(value) != "succeeded":
        raise ValueError(f"{source} has non-succeeded {field}: {value!r}")


def _require_value(value: object, *, field: str, source: Path) -> object:
    if value in (None, ""):
        raise ValueError(f"{source} missing required validation field {field}")
    return value


def _trial_row(run_root: Path, trial_result_path: Path) -> dict[str, Any]:
    trial_result = _expect_mapping(_load_json(trial_result_path), source=trial_result_path)
    hpo_summary = _hpo_summary_for_trial(trial_result_path)
    summary = _summary_mapping(trial_result)
    result = trial_result.get("result") if isinstance(trial_result.get("result"), Mapping) else {}
    overrides = trial_result.get("overrides") if isinstance(trial_result.get("overrides"), Mapping) else {}
    _require_status_succeeded(trial_result.get("status"), field="trial_result.status", source=trial_result_path)
    _require_status_succeeded(result.get("status"), field="trial_result.result.status", source=trial_result_path)
    run, machine, gpu, status = _slot_metadata(run_root, trial_result_path)
    case = str(
        trial_result.get("case")
        or overrides.get("case_name")
        or trial_result_path.parent.name
    )
    seed = None
    for candidate in (overrides.get("seed"), trial_result.get("seed"), summary.get("seed"), hpo_summary.get("seed")):
        seed = _as_int(candidate)
        if seed is not None:
            break
    best_val = None
    for candidate in (
        summary.get("best_val_accuracy"),
        result.get("best_val_accuracy"),
        trial_result.get("best_val_accuracy"),
    ):
        best_val = _as_float(candidate)
        if best_val is not None:
            break
    train_steps = None
    for candidate in (
        summary.get("train_steps_total"),
        summary.get("train_steps"),
        result.get("train_steps_total"),
        trial_result.get("train_steps_total"),
    ):
        train_steps = _as_int(candidate)
        if train_steps is not None:
            break
    checkpoint_path = (
        summary.get("checkpoint_path")
        or result.get("checkpoint_path")
        or trial_result.get("checkpoint_path")
        or ""
    )
    if seed is None:
        raise ValueError(f"{trial_result_path} missing required validation field seed")
    if best_val is None:
        raise ValueError(f"{trial_result_path} missing required validation field best_val_accuracy")
    if train_steps is None:
        raise ValueError(f"{trial_result_path} missing required validation field train_steps")
    _require_value(checkpoint_path, field="checkpoint_path", source=trial_result_path)
    return {
        "run": run,
        "machine": machine or status.get("machine") or "",
        "gpu": gpu,
        "case": case,
        "family": infer_family(case, trial_result),
        "seed": seed,
        "best_val_accuracy": best_val,
        "train_steps": train_steps,
        "test_accessed": _test_accessed(trial_result, hpo_summary, source=trial_result_path),
        "checkpoint_path": str(checkpoint_path),
    }


def validate_validation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_family_seeds: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if _parse_bool(row.get("test_accessed"), field="test_accessed", source=row.get("case", "row")):
            raise ValueError(
                f"validation summary refuses test_accessed=true for "
                f"{row.get('run', '')}/{row.get('machine', '')}/{row.get('gpu', '')}/"
                f"{row.get('case', '')}"
            )
        family = str(row.get("family") or "")
        seed = _as_int(row.get("seed"))
        if family and seed is not None:
            key = (family, seed)
            if key in seen_family_seeds:
                previous = seen_family_seeds[key]
                raise ValueError(
                    "validation summary refuses duplicate family/seed rows: "
                    f"{family}/seed{seed} in {previous.get('run', '')}/{previous.get('case', '')} "
                    f"and {row.get('run', '')}/{row.get('case', '')}"
                )
            seen_family_seeds[key] = row


def collect_finetune_hpo_rows(collected_roots: Path | Sequence[Path]) -> list[dict[str, Any]]:
    if isinstance(collected_roots, Path):
        roots = [collected_roots]
    else:
        roots = [Path(root) for root in collected_roots]
    rows: list[dict[str, Any]] = []
    for root in roots:
        for trial_result_path in _trial_result_paths(root):
            row = _trial_row(root, trial_result_path)
            if row["test_accessed"]:
                validate_validation_rows([row])
            rows.append(row)
    validate_validation_rows(rows)
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("run") or ""),
            str(row.get("machine") or ""),
            _as_int(row.get("gpu")) if _as_int(row.get("gpu")) is not None else 1_000_000,
            str(row.get("gpu") or ""),
            str(row.get("family") or ""),
            _as_int(row.get("seed")) if _as_int(row.get("seed")) is not None else 1_000_000,
            str(row.get("case") or ""),
        ),
    )


def aggregate_by_family(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validate_validation_rows(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family") or "")].append(row)

    aggregates: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        val_accuracies = [
            value
            for row in family_rows
            if (value := _as_float(row.get("best_val_accuracy"))) is not None
        ]
        train_steps = [
            value
            for row in family_rows
            if (value := _as_float(row.get("train_steps"))) is not None
        ]
        seeds = sorted(
            {
                value
                for row in family_rows
                if (value := _as_int(row.get("seed"))) is not None
            }
        )
        best_candidates = [row for row in family_rows if _as_float(row.get("best_val_accuracy")) is not None]
        best = (
            sorted(
                best_candidates,
                key=lambda row: (
                    -float(_as_float(row.get("best_val_accuracy")) or 0.0),
                    _as_int(row.get("seed")) if _as_int(row.get("seed")) is not None else 1_000_000,
                    str(row.get("run") or ""),
                    str(row.get("machine") or ""),
                    str(row.get("gpu") or ""),
                    str(row.get("case") or ""),
                ),
            )[0]
            if best_candidates
            else {}
        )
        aggregates.append(
            {
                "family": family,
                "trials": len(family_rows),
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "mean_best_val_accuracy": (
                    statistics.fmean(val_accuracies) if val_accuracies else None
                ),
                "std_best_val_accuracy": (
                    statistics.stdev(val_accuracies) if len(val_accuracies) > 1 else 0.0
                )
                if val_accuracies
                else None,
                "mean_train_steps": statistics.fmean(train_steps) if train_steps else None,
                "test_accessed": any(
                    _parse_bool(row.get("test_accessed"), field="test_accessed", source=row.get("case", "row"))
                    for row in family_rows
                ),
                "best_case": best.get("case", ""),
                "best_seed": best.get("seed", ""),
                "best_val_accuracy": best.get("best_val_accuracy", ""),
                "checkpoint_path": best.get("checkpoint_path", ""),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            -float(_as_float(row.get("mean_best_val_accuracy")) or -1.0),
            str(row.get("family") or ""),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    validate_validation_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in TRIAL_COLUMNS})


def write_family_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    aggregates = aggregate_by_family(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAMILY_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in aggregates:
            writer.writerow({column: _fmt(row.get(column)) for column in FAMILY_COLUMNS})


def write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    collected_roots: Sequence[Path],
    title: str = "Fine-Tune HPO Validation Summary",
) -> None:
    if not rows:
        raise ValueError("cannot write an empty fine-tune HPO summary")
    validate_validation_rows(rows)
    family_rows = aggregate_by_family(rows)
    roots_text = ", ".join(str(root) for root in collected_roots)
    content = [
        f"# {title}",
        "",
        f"- Collected roots: `{roots_text}`",
        f"- Trial rows: `{len(rows)}`",
        f"- Families: `{len(family_rows)}`",
        "- CIFAR-10 test accessed: `false`",
        "- Accuracy statistic: family standard deviation is sample standard deviation; one-trial families report `0.000000`.",
        "",
        "These are validation summaries for Stage H fine-tune HPO runs. Rows with `test_accessed=true` are rejected rather than summarized.",
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
                "Mean train steps",
                "Best case",
                "Best seed",
                "Best val acc",
            ],
            [
                [
                    row["family"],
                    row["trials"],
                    row["seeds"],
                    row["mean_best_val_accuracy"],
                    row["std_best_val_accuracy"],
                    row["mean_train_steps"],
                    row["best_case"],
                    row["best_seed"],
                    row["best_val_accuracy"],
                ]
                for row in family_rows
            ],
        ),
        "",
        "## Trial Rows",
        "",
        _markdown_table(
            [
                "Run",
                "Machine",
                "GPU",
                "Case",
                "Family",
                "Seed",
                "Best val acc",
                "Train steps",
                "Checkpoint",
            ],
            [
                [
                    row.get("run"),
                    row.get("machine"),
                    row.get("gpu"),
                    row.get("case"),
                    row.get("family"),
                    row.get("seed"),
                    row.get("best_val_accuracy"),
                    row.get("train_steps"),
                    row.get("checkpoint_path"),
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
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--family-csv-out", default=None)
    parser.add_argument("--title", default="Fine-Tune HPO Validation Summary")
    parser.add_argument("--expect-rows", type=int, default=None)
    args = parser.parse_args(argv)

    roots = [Path(root) for root in args.collected_root]
    rows = collect_finetune_hpo_rows(roots)
    if not rows:
        raise RuntimeError(f"no fine-tune HPO trial_result.json files found under {roots}")
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise RuntimeError(f"expected {args.expect_rows} fine-tune HPO rows, found {len(rows)}")
    write_csv(Path(args.csv_out), rows)
    if args.family_csv_out:
        write_family_csv(Path(args.family_csv_out), rows)
    write_markdown(Path(args.markdown_out), rows, collected_roots=roots, title=args.title)
    print(f"wrote {len(rows)} fine-tune HPO validation rows from {len(roots)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
