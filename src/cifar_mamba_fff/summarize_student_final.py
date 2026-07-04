from __future__ import annotations

import argparse
import csv
import json
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
    "selected_val_accuracy",
    "test_accuracy",
    "test_loss",
    "test_steps",
    "elapsed_seconds",
    "partial_test_evaluation",
    "test_accessed",
    "checkpoint_path",
    "checkpoint_sha256",
    "selection_record",
    "allow_untracked_selection",
    "remote_git_commit",
]

FAMILY_COLUMNS = [
    "family",
    "trials",
    "seed_count",
    "seeds",
    "mean_selected_val_accuracy",
    "std_selected_val_accuracy",
    "mean_test_accuracy",
    "std_test_accuracy",
    "mean_test_steps",
    "partial_test_evaluation",
    "test_accessed",
    "best_case",
    "best_seed",
    "best_test_accuracy",
    "checkpoint_path",
]

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
    if value in (None, ""):
        raise ValueError(f"{source} missing required boolean field {field}")
    normalized = str(value).strip().lower()
    if normalized in _TRUE_TEXT:
        return True
    if normalized in _FALSE_TEXT:
        return False
    raise ValueError(f"{source} has invalid boolean value for {field}: {value!r}")


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number)


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


def _slot_metadata(metrics_path: Path, run_root: Path) -> tuple[str, str, str, Mapping[str, Any]]:
    slot_dir = metrics_path.parent
    status_path = slot_dir / "status.json"
    status: Mapping[str, Any] = {}
    if status_path.exists():
        status = _expect_mapping(_load_json(status_path), source=status_path)
    try:
        rel = metrics_path.relative_to(run_root)
        parts = rel.parts
        machine = parts[0] if len(parts) >= 3 else str(status.get("machine") or "")
        gpu = parts[1] if len(parts) >= 3 else str(status.get("gpu_id") or "")
    except ValueError:
        machine = str(status.get("machine") or "")
        gpu = str(status.get("gpu_id") or "")
    run = run_root.name
    return run, machine, gpu, status


def _selection_records(root: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    selection_dir = root / "selection_records"
    if not selection_dir.exists():
        return records
    for path in sorted(selection_dir.glob("*.json")):
        payload = _expect_mapping(_load_json(path), source=path)
        case = str(payload.get("case") or path.stem.removesuffix("_selection"))
        records[case] = payload
    return records


def _require_number(value: object, *, field: str, source: Path) -> float:
    parsed = _as_float(value)
    if parsed is None:
        raise ValueError(f"{source} missing required numeric field {field}")
    return parsed


def _trial_row(run_root: Path, metrics_path: Path, selection_by_case: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _expect_mapping(_load_json(metrics_path), source=metrics_path)
    run, machine, gpu, status = _slot_metadata(metrics_path, run_root)
    if str(status.get("status")) != "succeeded":
        raise ValueError(f"{metrics_path} slot status is not succeeded: {status.get('status')!r}")
    if status.get("returncode") not in (0, "0"):
        raise ValueError(f"{metrics_path} slot returncode is not zero: {status.get('returncode')!r}")
    if metrics.get("phase") != "student_final_test":
        raise ValueError(f"{metrics_path} is not a student_final_test result")
    if _parse_bool(metrics.get("partial_test_evaluation"), field="partial_test_evaluation", source=metrics_path):
        raise ValueError(f"{metrics_path} is partial; full Stage H final summaries require full test")
    if not _parse_bool(metrics.get("test_accessed"), field="test_accessed", source=metrics_path):
        raise ValueError(f"{metrics_path} did not access CIFAR-10 test")
    selection = metrics.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{metrics_path} missing selection metadata")
    if _parse_bool(selection.get("allow_untracked_selection"), field="allow_untracked_selection", source=metrics_path):
        raise ValueError(f"{metrics_path} used untracked selection override")
    checkpoint_path = str(metrics.get("checkpoint_path") or "")
    checkpoint_sha = str(metrics.get("checkpoint_sha256") or selection.get("checkpoint_sha256") or "")
    if not checkpoint_path:
        raise ValueError(f"{metrics_path} missing checkpoint_path")
    if not checkpoint_sha:
        raise ValueError(f"{metrics_path} missing checkpoint_sha256")
    metadata = status.get("metadata") if isinstance(status.get("metadata"), Mapping) else {}
    case = str(metadata.get("case") or metrics_path.parent.name.removesuffix("_final_test"))
    family = str(metadata.get("family") or selection_by_case.get(case, {}).get("family") or case)
    seed = _as_int(status.get("seed") or selection_by_case.get(case, {}).get("seed"))
    if seed is None:
        raise ValueError(f"{metrics_path} missing seed")
    selection_record_path = str(selection.get("selection_record") or "")
    selection_record = selection_by_case.get(case)
    if selection_record is None:
        raise ValueError(f"{metrics_path} missing copied selection manifest for case {case}")
    if not _parse_bool(selection_record.get("selected_for_final_eval"), field="selected_for_final_eval", source=case):
        raise ValueError(f"{case} selection manifest is not selected_for_final_eval")
    if _parse_bool(selection_record.get("test_accessed"), field="test_accessed", source=case):
        raise ValueError(f"{case} selection manifest accessed CIFAR-10 test")
    if str(selection_record.get("checkpoint_path")) != checkpoint_path:
        raise ValueError(f"{case} selection checkpoint does not match final metrics")
    if str(selection_record.get("checkpoint_sha256")) != checkpoint_sha:
        raise ValueError(f"{case} selection hash does not match final metrics")
    selected_val = _require_number(metrics.get("selected_val_accuracy"), field="selected_val_accuracy", source=metrics_path)
    selection_val = _require_number(selection_record.get("best_val_accuracy"), field="best_val_accuracy", source=metrics_path)
    if abs(selected_val - selection_val) > 1e-8:
        raise ValueError(f"{case} selection validation accuracy does not match final metrics")
    return {
        "run": run,
        "machine": machine,
        "gpu": gpu,
        "case": case,
        "family": family,
        "seed": seed,
        "selected_val_accuracy": selected_val,
        "test_accuracy": _require_number(metrics.get("test_accuracy"), field="test_accuracy", source=metrics_path),
        "test_loss": _require_number(metrics.get("test_loss"), field="test_loss", source=metrics_path),
        "test_steps": _as_int(metrics.get("test_steps")),
        "elapsed_seconds": _require_number(metrics.get("elapsed_seconds"), field="elapsed_seconds", source=metrics_path),
        "partial_test_evaluation": False,
        "test_accessed": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "selection_record": selection_record_path,
        "allow_untracked_selection": False,
        "remote_git_commit": str(status.get("remote_git_commit") or ""),
    }


def collect_student_final_rows(collected_roots: Path | Sequence[Path]) -> list[dict[str, Any]]:
    roots = [collected_roots] if isinstance(collected_roots, Path) else [Path(root) for root in collected_roots]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"collected root does not exist: {root}")
        selection_by_case = _selection_records(root)
        paths = sorted(root.rglob("student_final_test_metrics.json"))
        if not paths:
            raise ValueError(f"{root} has no student_final_test_metrics.json files")
        for metrics_path in paths:
            rows.append(_trial_row(root, metrics_path, selection_by_case))
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["family"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate final-test family/seed row: {key}")
        seen.add(key)
    return sorted(rows, key=lambda row: (str(row["family"]), int(row["seed"]), str(row["case"])))


def aggregate_by_family(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family") or "")].append(row)
    out: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        test_acc = [float(row["test_accuracy"]) for row in family_rows]
        selected_val = [float(row["selected_val_accuracy"]) for row in family_rows]
        test_steps = [float(row["test_steps"]) for row in family_rows if row.get("test_steps") not in (None, "")]
        seeds = sorted(int(row["seed"]) for row in family_rows)
        best = max(family_rows, key=lambda row: (float(row["test_accuracy"]), -int(row["seed"])))
        out.append(
            {
                "family": family,
                "trials": len(family_rows),
                "seed_count": len(set(seeds)),
                "seeds": ",".join(str(seed) for seed in seeds),
                "mean_selected_val_accuracy": statistics.fmean(selected_val),
                "std_selected_val_accuracy": statistics.stdev(selected_val) if len(selected_val) > 1 else 0.0,
                "mean_test_accuracy": statistics.fmean(test_acc),
                "std_test_accuracy": statistics.stdev(test_acc) if len(test_acc) > 1 else 0.0,
                "mean_test_steps": statistics.fmean(test_steps) if test_steps else None,
                "partial_test_evaluation": any(bool(row["partial_test_evaluation"]) for row in family_rows),
                "test_accessed": any(bool(row["test_accessed"]) for row in family_rows),
                "best_case": best["case"],
                "best_seed": best["seed"],
                "best_test_accuracy": best["test_accuracy"],
                "checkpoint_path": best["checkpoint_path"],
            }
        )
    return sorted(out, key=lambda row: (-float(row["mean_test_accuracy"]), str(row["family"])))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in TRIAL_COLUMNS})


def write_family_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    families = aggregate_by_family(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAMILY_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in families:
            writer.writerow({column: _fmt(row.get(column)) for column in FAMILY_COLUMNS})


def write_selection_manifest_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], collected_roots: Sequence[Path]) -> None:
    records: list[Mapping[str, Any]] = []
    for root in collected_roots:
        records.extend(_selection_records(root).values())
    records_by_case = {str(record.get("case")): record for record in records}
    missing = sorted(str(row["case"]) for row in rows if str(row["case"]) not in records_by_case)
    if missing:
        raise ValueError(f"missing selection manifests for final rows: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(records_by_case[str(row["case"])], sort_keys=True) + "\n")


def write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    collected_roots: Sequence[Path],
    title: str = "Stage H Final Student Test Summary",
) -> None:
    families = aggregate_by_family(rows)
    roots_text = ", ".join(str(root) for root in collected_roots)
    content = [
        f"# {title}",
        "",
        f"- Collected roots: `{roots_text}`",
        f"- Trial rows: `{len(rows)}`",
        f"- Families: `{len(families)}`",
        "- CIFAR-10 test accessed: `true`",
        "- Partial test evaluation: `false`",
        "- Selection rule: validation-selected `no_balance_cosine` family from Stage H full 3-epoch validation.",
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
                "Mean test acc",
                "Std test acc",
                "Best case",
                "Best seed",
                "Best test acc",
            ],
            [
                [
                    row["family"],
                    row["trials"],
                    row["seeds"],
                    row["mean_selected_val_accuracy"],
                    row["std_selected_val_accuracy"],
                    row["mean_test_accuracy"],
                    row["std_test_accuracy"],
                    row["best_case"],
                    row["best_seed"],
                    row["best_test_accuracy"],
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
                "Machine",
                "GPU",
                "Seed",
                "Val acc",
                "Test acc",
                "Test steps",
                "Checkpoint SHA256",
            ],
            [
                [
                    row["case"],
                    row["machine"],
                    row["gpu"],
                    row["seed"],
                    row["selected_val_accuracy"],
                    row["test_accuracy"],
                    row["test_steps"],
                    row["checkpoint_sha256"],
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
    parser.add_argument("--selection-jsonl-out", required=True)
    parser.add_argument("--title", default="Stage H Final Student Test Summary")
    args = parser.parse_args(argv)

    roots = [Path(root) for root in args.collected_root]
    rows = collect_student_final_rows(roots)
    if not rows:
        raise RuntimeError("no student final rows collected")
    write_csv(Path(args.csv_out), rows)
    write_family_csv(Path(args.family_csv_out), rows)
    write_selection_manifest_jsonl(Path(args.selection_jsonl_out), rows, roots)
    write_markdown(Path(args.markdown_out), rows, collected_roots=roots, title=args.title)
    print(f"wrote {len(rows)} student final-test rows from {len(roots)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
