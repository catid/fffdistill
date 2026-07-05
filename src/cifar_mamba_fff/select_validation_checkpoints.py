from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

from .cluster import MachineSpec, load_machines
from .utils import write_json

REQUIRED_COLUMNS = [
    "machine",
    "gpu",
    "case",
    "family",
    "seed",
    "best_val_accuracy",
    "test_accessed",
    "checkpoint_path",
]

MANIFEST_COLUMNS = [
    "machine",
    "gpu",
    "case",
    "family",
    "seed",
    "best_val_accuracy",
    "checkpoint_path",
    "checkpoint_sha256",
    "selection_record",
]

SKIPPED_SHA256 = "sha256-skipped"

_TRUE_TEXT = {"1", "true", "yes", "y", "on"}
_FALSE_TEXT = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class ValidationTrial:
    run: str
    machine: str
    gpu: int
    case: str
    family: str
    seed: int
    best_val_accuracy: float
    test_accessed: bool
    checkpoint_path: str
    row_index: int


@dataclass(frozen=True)
class SelectionRecord:
    trial: ValidationTrial
    checkpoint_sha256: str
    selection_record: str


def _parse_bool(value: object, *, field: str, source: str) -> bool:
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


def _parse_int(value: object, *, field: str, source: str) -> int:
    if value in (None, ""):
        raise ValueError(f"{source} missing required integer field {field}")
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{source} has invalid integer value for {field}: {value!r}") from exc


def _parse_float(value: object, *, field: str, source: str) -> float:
    if value in (None, ""):
        raise ValueError(f"{source} missing required numeric field {field}")
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{source} has invalid numeric value for {field}: {value!r}") from exc


def _require_text(row: Mapping[str, str], field: str, *, source: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"{source} missing required field {field}")
    return value.strip()


def _validate_columns(fieldnames: Sequence[str] | None, *, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} has no CSV header")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def read_validation_trials(path: str | Path) -> list[ValidationTrial]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, path=csv_path)
        rows = list(reader)

    trials: list[ValidationTrial] = []
    for row_index, row in enumerate(rows, start=2):
        source = f"{csv_path}:{row_index}"
        test_accessed = _parse_bool(row.get("test_accessed"), field="test_accessed", source=source)
        if test_accessed:
            case = row.get("case") or f"row {row_index}"
            raise ValueError(f"{source} refuses test_accessed=true for validation trial {case!r}")
        trials.append(
            ValidationTrial(
                run=str(row.get("run") or "").strip(),
                machine=_require_text(row, "machine", source=source),
                gpu=_parse_int(row.get("gpu"), field="gpu", source=source),
                case=_require_text(row, "case", source=source),
                family=_require_text(row, "family", source=source),
                seed=_parse_int(row.get("seed"), field="seed", source=source),
                best_val_accuracy=_parse_float(
                    row.get("best_val_accuracy"),
                    field="best_val_accuracy",
                    source=source,
                ),
                test_accessed=test_accessed,
                checkpoint_path=_require_text(row, "checkpoint_path", source=source),
                row_index=row_index,
            )
        )
    if not trials:
        raise ValueError(f"{csv_path} contains no validation trials")
    return trials


def _split_family_values(values: Sequence[str]) -> list[str]:
    families: list[str] = []
    for value in values:
        families.extend(part.strip() for part in value.split(",") if part.strip())
    return families


def family_scores(
    trials: Sequence[ValidationTrial],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ValidationTrial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.family].append(trial)
    return {
        family: {
            "mean": statistics.fmean(trial.best_val_accuracy for trial in family_trials),
            "best": max(trial.best_val_accuracy for trial in family_trials),
            "trials": len(family_trials),
        }
        for family, family_trials in grouped.items()
    }


def top_families(
    trials: Sequence[ValidationTrial],
    n: int,
    *,
    metric: str = "mean",
) -> list[str]:
    if n <= 0:
        raise ValueError("top family count must be positive")
    if metric not in {"mean", "best"}:
        raise ValueError("top family metric must be 'mean' or 'best'")
    scores = family_scores(trials)
    ranked = sorted(
        scores,
        key=lambda family: (
            -float(scores[family][metric]),
            -float(scores[family]["best"]),
            family,
        ),
    )
    return ranked[:n]


def select_trials(
    trials: Sequence[ValidationTrial],
    *,
    families: Sequence[str] | None = None,
    top_family_count: int | None = None,
    top_family_metric: str = "mean",
) -> list[ValidationTrial]:
    requested_families = _split_family_values(families or [])
    if bool(requested_families) == (top_family_count is not None):
        raise ValueError("provide exactly one of families or top_family_count")

    available = {trial.family for trial in trials}
    if top_family_count is not None:
        requested_families = top_families(trials, top_family_count, metric=top_family_metric)
    else:
        missing = sorted(set(requested_families) - available)
        if missing:
            raise ValueError(f"requested families not found in validation trials: {missing}")

    family_set = set(requested_families)
    selected = [trial for trial in trials if trial.family in family_set]
    if not selected:
        raise ValueError("selection produced no validation trials")
    _require_unique_cases(selected)
    return selected


def _require_unique_cases(trials: Sequence[ValidationTrial]) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for trial in trials:
        prior = seen.setdefault(trial.case, trial.row_index)
        if prior != trial.row_index:
            duplicates.append(trial.case)
    if duplicates:
        raise ValueError(f"duplicate selected case rows: {sorted(set(duplicates))}")


def _safe_selection_filename(case: str) -> str:
    if "/" in case or "\\" in case:
        raise ValueError(f"case cannot contain path separators: {case!r}")
    return f"{case}_selection.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_local_machine(spec: MachineSpec) -> bool:
    return spec.role == "local" or spec.host in {"localhost", "127.0.0.1"}


def _local_checkpoint_path(spec: MachineSpec, checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    return path if path.is_absolute() else Path(spec.workdir) / path


def _validate_sha256(value: str, *, source: str) -> str:
    stripped = value.strip()
    if len(stripped) != 64 or any(char not in "0123456789abcdefABCDEF" for char in stripped):
        raise ValueError(f"{source} did not return a SHA256 hex digest: {value!r}")
    return stripped.lower()


def _remote_hash_command(spec: MachineSpec, checkpoint_path: str) -> str:
    script = (
        "import hashlib, sys\n"
        "from pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        "digest = hashlib.sha256()\n"
        "with path.open('rb') as handle:\n"
        "    for chunk in iter(lambda: handle.read(1024 * 1024), b''):\n"
        "        digest.update(chunk)\n"
        "print(digest.hexdigest())\n"
    )
    return f"cd {quote(spec.workdir)} && python3 -c {quote(script)} {quote(checkpoint_path)}"


def checkpoint_sha256(
    trial: ValidationTrial,
    machines: Mapping[str, MachineSpec],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: int = 300,
) -> str:
    spec = machines.get(trial.machine)
    if spec is None:
        raise ValueError(f"{trial.case} uses unknown machine {trial.machine!r}")
    if _is_local_machine(spec):
        path = _local_checkpoint_path(spec, trial.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"{trial.case} checkpoint not found: {path}")
        return _sha256_file(path)

    command = _remote_hash_command(spec, trial.checkpoint_path)
    completed = runner(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            spec.host,
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{trial.case} remote SHA256 failed on {trial.machine}: "
            f"{str(completed.stderr).strip() or str(completed.stdout).strip()}"
        )
    return _validate_sha256(str(completed.stdout).strip().splitlines()[-1], source=trial.case)


def _selection_rule(
    *,
    families: Sequence[str] | None,
    top_family_count: int | None,
    top_family_metric: str,
) -> str:
    requested_families = _split_family_values(families or [])
    if requested_families:
        return (
            "Selected validation fine-tune families "
            f"{', '.join(requested_families)} before final CIFAR-10 test access."
        )
    return (
        f"Selected top {top_family_count} validation fine-tune family/families by "
        f"{top_family_metric} best_val_accuracy before final CIFAR-10 test access."
    )


def _selection_payload(
    record: SelectionRecord,
    *,
    trials_csv: Path,
    selection_stage: str,
    selection_rule: str,
    source_validation_summary: str | None = None,
) -> dict[str, Any]:
    trial = record.trial
    summary = {
        "best_val_accuracy": trial.best_val_accuracy,
        "checkpoint_path": trial.checkpoint_path,
        "checkpoint_sha256": record.checkpoint_sha256,
        "selected_for_final_eval": True,
        "status": "succeeded",
        "test_accessed": False,
    }
    payload: dict[str, Any] = {
        "status": "succeeded",
        "selected_for_final_eval": True,
        "selection_stage": selection_stage,
        "selection_rule": selection_rule,
        "source_trials_csv": str(trials_csv),
        "run": trial.run,
        "machine": trial.machine,
        "gpu": trial.gpu,
        "case": trial.case,
        "family": trial.family,
        "seed": trial.seed,
        "best_val_accuracy": trial.best_val_accuracy,
        "checkpoint_path": trial.checkpoint_path,
        "checkpoint_sha256": record.checkpoint_sha256,
        "selection_record": record.selection_record,
        "summary": summary,
        "test_accessed": False,
    }
    if source_validation_summary:
        payload["source_validation_summary"] = source_validation_summary
    return payload


def _manifest_payload(record: SelectionRecord) -> dict[str, Any]:
    trial = record.trial
    return {
        "machine": trial.machine,
        "gpu": trial.gpu,
        "case": trial.case,
        "family": trial.family,
        "seed": trial.seed,
        "best_val_accuracy": trial.best_val_accuracy,
        "checkpoint_path": trial.checkpoint_path,
        "checkpoint_sha256": record.checkpoint_sha256,
        "selection_record": record.selection_record,
    }


def build_selection_records(
    trials: Sequence[ValidationTrial],
    *,
    selection_dir: Path,
    machines: Mapping[str, MachineSpec],
    skip_hash: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: int = 300,
) -> list[SelectionRecord]:
    records: list[SelectionRecord] = []
    for trial in trials:
        selection_path = selection_dir / _safe_selection_filename(trial.case)
        sha256 = (
            SKIPPED_SHA256
            if skip_hash
            else checkpoint_sha256(trial, machines, runner=runner, timeout_s=timeout_s)
        )
        records.append(
            SelectionRecord(
                trial=trial,
                checkpoint_sha256=sha256,
                selection_record=str(selection_path),
            )
        )
    return records


def write_selection_outputs(
    *,
    records: Sequence[SelectionRecord],
    trials_csv: Path,
    selection_dir: Path,
    manifest_out: Path,
    selection_stage: str,
    selection_rule: str,
    source_validation_summary: str | None = None,
) -> None:
    selection_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_json(
            selection_dir / _safe_selection_filename(record.trial.case),
            _selection_payload(
                record,
                trials_csv=trials_csv,
                selection_stage=selection_stage,
                selection_rule=selection_rule,
                source_validation_summary=source_validation_summary,
            ),
        )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_manifest_payload(record), sort_keys=True) + "\n")


def select_validation_checkpoints(
    *,
    trials_csv: Path,
    selection_dir: Path,
    manifest_out: Path,
    machines_path: Path,
    families: Sequence[str] | None = None,
    top_family_count: int | None = None,
    top_family_metric: str = "mean",
    selection_stage: str | None = None,
    selection_rule: str | None = None,
    source_validation_summary: str | None = None,
    skip_hash: bool = False,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: int = 300,
) -> list[SelectionRecord]:
    trials = read_validation_trials(trials_csv)
    selected = select_trials(
        trials,
        families=families,
        top_family_count=top_family_count,
        top_family_metric=top_family_metric,
    )
    machines = {machine.name: machine for machine in load_machines(machines_path)}
    records = build_selection_records(
        selected,
        selection_dir=selection_dir,
        machines=machines,
        skip_hash=skip_hash or dry_run,
        runner=runner,
        timeout_s=timeout_s,
    )
    if not dry_run:
        write_selection_outputs(
            records=records,
            trials_csv=trials_csv,
            selection_dir=selection_dir,
            manifest_out=manifest_out,
            selection_stage=selection_stage or selection_dir.name,
            selection_rule=selection_rule
            or _selection_rule(
                families=families,
                top_family_count=top_family_count,
                top_family_metric=top_family_metric,
            ),
            source_validation_summary=source_validation_summary,
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-csv", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--machines", default="configs/machines.yaml")
    parser.add_argument(
        "--family",
        action="append",
        nargs="+",
        default=[],
        help="Family to select. May be repeated, space-separated, or comma-separated.",
    )
    parser.add_argument("--top-families", type=int, default=None)
    parser.add_argument(
        "--top-family-metric",
        choices=("mean", "best"),
        default="mean",
        help="Metric used to rank families for --top-families.",
    )
    parser.add_argument("--selection-stage", default=None)
    parser.add_argument("--selection-rule", default=None)
    parser.add_argument("--source-validation-summary", default=None)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and print records without hashing or writing outputs.",
    )
    parser.add_argument("--hash-timeout-s", type=int, default=300)
    args = parser.parse_args(argv)
    families = [family for group in args.family for family in group]

    records = select_validation_checkpoints(
        trials_csv=Path(args.trials_csv),
        selection_dir=Path(args.selection_dir),
        manifest_out=Path(args.manifest_out),
        machines_path=Path(args.machines),
        families=families,
        top_family_count=args.top_families,
        top_family_metric=args.top_family_metric,
        selection_stage=args.selection_stage,
        selection_rule=args.selection_rule,
        source_validation_summary=args.source_validation_summary,
        skip_hash=args.skip_hash,
        dry_run=args.dry_run,
        timeout_s=args.hash_timeout_s,
    )
    if args.dry_run:
        for record in records:
            print(json.dumps(_manifest_payload(record), sort_keys=True))
    else:
        print(f"wrote {len(records)} selection record(s) to {args.selection_dir}")
        print(f"wrote scheduler launch manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
