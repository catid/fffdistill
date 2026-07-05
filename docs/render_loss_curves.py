#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["svg.hashsalt"] = "cifar-mamba-fff-loss-curves"

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"
OUT_DIR = DOCS / "loss_curves"

BUILTIN_SOURCES = (
    (
        "Stage H selected FFF students",
        DOCS / "stage_h_all_families_final_test_trials.csv",
        DOCS / "stage_h_all_families_final_test_families.csv",
    ),
    (
        "T15 selected baselines",
        DOCS / "t15_baselines_final_test_trials.csv",
        DOCS / "t15_baselines_final_test_families.csv",
    ),
)

CURVE_FIELDS = ("train_loss", "val_loss", "val_accuracy")
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass(frozen=True)
class LossCurveSource:
    suite: str
    trials_csv: Path
    families_csv: Path


@dataclass(frozen=True)
class TrialCurve:
    suite: str
    source_csv: Path
    family: str
    case: str
    seed: str
    checkpoint_path: str
    metrics_path: Path | None
    records: tuple[dict[str, Any], ...]
    selected_val_accuracy: float | None
    test_accuracy: float | None
    test_loss: float | None
    notes: str

    @property
    def has_curve(self) -> bool:
        return self.metrics_path is not None and bool(self.records)


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required loss-curve source CSV is missing: {_rel(path)}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _first_float(row: dict[str, str], keys: Sequence[str]) -> float | None:
    for key in keys:
        parsed = _as_float(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _curve_records(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for record in _read_jsonl(path):
        epoch = _as_float(record.get("epoch"))
        if epoch is None:
            continue
        if not any(_as_float(record.get(field)) is not None for field in CURVE_FIELDS):
            continue
        records.append(record)
    records.sort(key=lambda item: float(item["epoch"]))
    return tuple(records)


def _case_from_trial_dir_name(trial_name: str) -> str:
    if not trial_name.startswith("trial_"):
        return trial_name
    parts = trial_name.split("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[2]
    return trial_name


def _indexed_metric_paths(outputs_root: Path) -> dict[str, list[Path]]:
    by_case: dict[str, list[Path]] = defaultdict(list)
    if not outputs_root.exists():
        return by_case
    for path in outputs_root.glob("**/metrics.jsonl"):
        trial_name = path.parent.name
        case = _case_from_trial_dir_name(trial_name)
        by_case[case].append(path)
    for paths in by_case.values():
        paths.sort()
    return by_case


def _score_metric_candidate(path: Path, checkpoint_path: str, suite: str) -> tuple[int, str]:
    text = str(path)
    score = 0
    checkpoint_parts = {part for part in Path(checkpoint_path).parts if part}
    for part in checkpoint_parts:
        if part in text:
            score += 5
    if path.parent.name in checkpoint_path:
        score += 30
    for part in Path(checkpoint_path).parts:
        if (part.endswith("_wave0") or part.endswith("_wave1") or "full3ep" in part) and part in text:
            score += 25
    if "Stage H" in suite and "stage_h_validation_full3ep" in text:
        score += 40
    if "T15" in suite and "baseline_validation" in text:
        score += 20
    if "scheduler_collected" in text:
        score += 3
    return score, text


def _suite_short_name(suite: str) -> str:
    lowered = suite.lower()
    if "stage h" in lowered:
        return "H"
    if "t15" in lowered:
        return "T15"
    if "optimizer" in lowered or "wsd" in lowered:
        return "opt"
    if "bank" in lowered and "muon" in lowered:
        return "bank"
    if "generated" in lowered or "sublinear" in lowered:
        return "sublinear"
    words = [word for word in suite.replace("_", " ").replace("-", " ").split() if word]
    if not words:
        return "suite"
    initials = "".join(word[0] for word in words[:3]).upper()
    return initials[:8]


def _find_metrics_path(
    row: dict[str, str],
    suite: str,
    outputs_root: Path,
    metric_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    checkpoint_path = row.get("checkpoint_path", "")
    case = row.get("case", "")
    if checkpoint_path:
        checkpoint_parts = Path(checkpoint_path).parts
        if checkpoint_parts and checkpoint_parts[0] == "outputs":
            direct = outputs_root.joinpath(*checkpoint_parts[1:])
        else:
            direct = ROOT / checkpoint_path
        candidate = direct.parent / "metrics.jsonl"
        if candidate.exists():
            return candidate, "checkpoint-relative metrics.jsonl"

    candidates = metric_index.get(case, [])
    readable_candidates: list[tuple[tuple[int, str], Path]] = []
    for candidate in candidates:
        try:
            records = _curve_records(candidate)
        except ValueError:
            continue
        if records:
            readable_candidates.append((_score_metric_candidate(candidate, checkpoint_path, suite), candidate))
    if readable_candidates:
        readable_candidates.sort(reverse=True)
        return readable_candidates[0][1], "indexed outputs metrics.jsonl"

    if candidates:
        return None, f"found {len(candidates)} metrics.jsonl candidate(s), but no epoch curve fields"
    return None, "no metrics.jsonl found for case"


def _source_label_from_trials_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_final_test_trials", "_validation_trials", "_trials"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ").replace("-", " ").strip().title()


def _parse_source_spec(spec: str) -> LossCurveSource:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--source must use the format 'Label:docs/trials.csv:docs/families.csv'"
        )
    label, trials_csv, families_csv = parts
    if not label.strip():
        raise ValueError("--source label must not be empty")
    trials_path = Path(trials_csv)
    families_path = Path(families_csv)
    if not trials_path.is_absolute():
        trials_path = ROOT / trials_path
    if not families_path.is_absolute():
        families_path = ROOT / families_path
    return LossCurveSource(label.strip(), trials_path, families_path)


def _discover_source_pairs(
    docs_dir: Path,
    *,
    include_validation: bool,
) -> list[LossCurveSource]:
    patterns = ["*_final_test_trials.csv"]
    if include_validation:
        patterns.append("*_validation_trials.csv")
    discovered: list[LossCurveSource] = []
    for pattern in patterns:
        for trials_csv in sorted(docs_dir.glob(pattern)):
            families_csv = trials_csv.with_name(
                trials_csv.name.replace("_trials.csv", "_families.csv")
            )
            if not families_csv.exists():
                continue
            discovered.append(
                LossCurveSource(
                    _source_label_from_trials_path(trials_csv),
                    trials_csv,
                    families_csv,
                )
            )
    return discovered


def _resolve_sources(
    *,
    docs_dir: Path,
    source_specs: Sequence[str],
    discover_extra_sources: bool,
    include_validation: bool,
) -> tuple[LossCurveSource, ...]:
    sources: list[LossCurveSource] = [
        LossCurveSource(label, trials_csv, families_csv)
        for label, trials_csv, families_csv in BUILTIN_SOURCES
    ]
    sources.extend(_parse_source_spec(spec) for spec in source_specs)
    if discover_extra_sources:
        sources.extend(
            _discover_source_pairs(
                docs_dir,
                include_validation=include_validation,
            )
        )
    deduped: list[LossCurveSource] = []
    seen: set[tuple[Path, Path]] = set()
    for source in sources:
        key = (source.trials_csv.resolve(), source.families_csv.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return tuple(deduped)


def _load_trials(outputs_root: Path, sources: Sequence[LossCurveSource]) -> list[TrialCurve]:
    metric_index = _indexed_metric_paths(outputs_root)
    trials: list[TrialCurve] = []
    for source in sources:
        rows = _read_csv(source.trials_csv)
        family_rows = _read_csv(source.families_csv)
        if not rows:
            raise ValueError(
                f"required loss-curve source CSV has no rows: {_rel(source.trials_csv)}"
            )
        if not family_rows:
            raise ValueError(
                f"required loss-curve family CSV has no rows: {_rel(source.families_csv)}"
            )
        for row in rows:
            metrics_path, source_note = _find_metrics_path(
                row,
                source.suite,
                outputs_root,
                metric_index,
            )
            records: tuple[dict[str, Any], ...] = ()
            notes = source_note
            if metrics_path is not None:
                try:
                    records = _curve_records(metrics_path)
                except ValueError as exc:
                    notes = str(exc)
                    metrics_path = None
            if metrics_path is not None and not records:
                notes = "metrics file has no epoch records with curve fields"
                metrics_path = None
            trials.append(
                TrialCurve(
                    suite=source.suite,
                    source_csv=source.trials_csv,
                    family=row.get("family", ""),
                    case=row.get("case", ""),
                    seed=row.get("seed", ""),
                    checkpoint_path=row.get("checkpoint_path", ""),
                    metrics_path=metrics_path,
                    records=records,
                    selected_val_accuracy=_first_float(
                        row,
                        ("selected_val_accuracy", "best_val_accuracy", "val_accuracy"),
                    ),
                    test_accuracy=_first_float(
                        row,
                        ("test_accuracy", "final_test_accuracy"),
                    ),
                    test_loss=_as_float(row.get("test_loss")),
                    notes=notes,
                )
            )
    return trials


def _family_values(trials: list[TrialCurve], field: str) -> dict[tuple[str, str], dict[float, list[float]]]:
    values: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        if not trial.has_curve:
            continue
        key = (trial.suite, trial.family)
        for record in trial.records:
            epoch = _as_float(record.get("epoch"))
            value = _as_float(record.get(field))
            if epoch is not None and value is not None:
                values[key][epoch].append(value)
    return values


def _plot_mean_curve(ax: Any, trials: list[TrialCurve], field: str, ylabel: str) -> None:
    values = _family_values(trials, field)
    keys = sorted(values)
    colors = {key: PALETTE[index % len(PALETTE)] for index, key in enumerate(keys)}
    for key in keys:
        suite, family = key
        epochs = sorted(values[key])
        means = [mean(values[key][epoch]) for epoch in epochs]
        stds = [stdev(values[key][epoch]) if len(values[key][epoch]) > 1 else 0.0 for epoch in epochs]
        label = f"{family} ({_suite_short_name(suite)})"
        linestyle = "-" if "baseline" in suite.lower() or suite.startswith("T15") else "--"
        color = colors[key]
        ax.plot(epochs, means, marker="o", linewidth=2.4, linestyle=linestyle, color=color, label=label)
        if any(stds):
            lower = [m - s for m, s in zip(means, stds, strict=True)]
            upper = [m + s for m, s in zip(means, stds, strict=True)]
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_final_accuracy(ax: Any, trials: list[TrialCurve]) -> None:
    by_family: dict[tuple[str, str], list[TrialCurve]] = defaultdict(list)
    for trial in trials:
        by_family[(trial.suite, trial.family)].append(trial)
    keys = sorted(by_family)
    labels = [f"{family}\n{_suite_short_name(suite)}" for suite, family in keys]
    val_means: list[float | None] = []
    test_means: list[float | None] = []
    for key in keys:
        val_values = [v for v in (trial.selected_val_accuracy for trial in by_family[key]) if v is not None]
        test_values = [v for v in (trial.test_accuracy for trial in by_family[key]) if v is not None]
        val_means.append(mean(val_values) if val_values else None)
        test_means.append(mean(test_values) if test_values else None)
    x = list(range(len(keys)))
    width = 0.36
    ax.bar(
        [pos - width / 2 for pos in x],
        [value if value is not None else math.nan for value in val_means],
        width=width,
        label="selected/best val",
        color="#6b7280",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        [value if value is not None else math.nan for value in test_means],
        width=width,
        label="final test",
        color="#2563eb",
    )
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    finite_values = [value for value in [*val_means, *test_means] if value is not None]
    if finite_values:
        ax.set_ylim(max(0.0, min(finite_values) - 0.04), min(1.0, max(finite_values) + 0.02))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")


def _render_svg(trials: list[TrialCurve], output_path: Path) -> None:
    available = [trial for trial in trials if trial.has_curve]
    suite_count = len({trial.suite for trial in trials})
    family_count = len({(trial.suite, trial.family) for trial in trials})
    width = max(20, min(34, 14 + family_count * 0.45))
    height = max(13, min(24, 10 + suite_count * 0.9))
    fig, axes = plt.subplots(2, 2, figsize=(width, height), constrained_layout=True)
    fig.suptitle(
        "Final-selected CIFAR-10 Mamba/FFF loss curves from real experiment logs",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        (
            "Curves are per-family means across available seeds; shaded bands are one sample std. "
            "Validation-only suites contribute validation bars but no final-test bars."
        ),
        ha="center",
        va="top",
        fontsize=10,
    )
    _plot_mean_curve(axes[0][0], available, "val_loss", "Validation loss")
    axes[0][0].set_title("Validation loss")
    _plot_mean_curve(axes[0][1], available, "train_loss", "Training loss")
    axes[0][1].set_title("Training loss")
    _plot_mean_curve(axes[1][0], available, "val_accuracy", "Validation accuracy")
    axes[1][0].set_title("Validation accuracy")
    _plot_final_accuracy(axes[1][1], trials)
    axes[1][1].set_title("Validation-selected final test points")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside right center", title="Family")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    text = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def _write_manifest(trials: list[TrialCurve], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite",
        "family",
        "case",
        "seed",
        "curve_status",
        "epochs",
        "curve_fields",
        "metrics_path",
        "source_csv",
        "selected_val_accuracy",
        "test_accuracy",
        "test_loss",
        "checkpoint_path",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in sorted(trials, key=lambda item: (item.suite, item.family, item.seed, item.case)):
            epochs = sorted({_as_float(record.get("epoch")) for record in trial.records})
            fields = [
                field
                for field in CURVE_FIELDS
                if any(_as_float(record.get(field)) is not None for record in trial.records)
            ]
            writer.writerow(
                {
                    "suite": trial.suite,
                    "family": trial.family,
                    "case": trial.case,
                    "seed": trial.seed,
                    "curve_status": "available" if trial.has_curve else "unavailable",
                    "epochs": ";".join(f"{epoch:g}" for epoch in epochs if epoch is not None),
                    "curve_fields": ";".join(fields),
                    "metrics_path": _rel(trial.metrics_path),
                    "source_csv": _rel(trial.source_csv),
                    "selected_val_accuracy": "" if trial.selected_val_accuracy is None else f"{trial.selected_val_accuracy:.6f}",
                    "test_accuracy": "" if trial.test_accuracy is None else f"{trial.test_accuracy:.6f}",
                    "test_loss": "" if trial.test_loss is None else f"{trial.test_loss:.6f}",
                    "checkpoint_path": trial.checkpoint_path,
                    "notes": trial.notes,
                }
            )


def _family_summary(trials: list[TrialCurve]) -> list[tuple[str, str, int, int, str]]:
    by_family: dict[tuple[str, str], list[TrialCurve]] = defaultdict(list)
    for trial in trials:
        by_family[(trial.suite, trial.family)].append(trial)
    rows = []
    for (suite, family), family_trials in sorted(by_family.items()):
        total = len(family_trials)
        available = sum(trial.has_curve for trial in family_trials)
        missing = [trial.case for trial in family_trials if not trial.has_curve]
        rows.append((suite, family, available, total, ", ".join(missing) if missing else "none"))
    return rows


def _write_readme(
    trials: list[TrialCurve],
    output_path: Path,
    outputs_root: Path,
    sources: Sequence[LossCurveSource],
) -> None:
    available = sum(trial.has_curve for trial in trials)
    metric_count = len(list(outputs_root.glob("**/metrics.jsonl"))) if outputs_root.exists() else 0
    layer_metric_count = len(list(outputs_root.glob("**/layer_metrics.jsonl"))) if outputs_root.exists() else 0
    lines = [
        "# Loss Curves",
        "",
        "Generated by `python docs/render_loss_curves.py` from committed final/validation summary CSVs and ignored real experiment logs under `outputs/`.",
        "No synthetic or fallback data is added; rows without a usable metrics file are marked unavailable in `manifest.csv`.",
        "",
        "## Files",
        "",
        "- `loss_curves.svg`: large matplotlib SVG with validation loss, training loss, validation accuracy, and validation/final-test accuracy panels.",
        "- `manifest.csv`: per compared row, including metrics source path and availability.",
        "",
        "## Scan Summary",
        "",
        f"- Compared rows: {len(trials)}",
        f"- Rows with epoch curves: {available}",
        f"- Rows without epoch curves: {len(trials) - available}",
        f"- Discovered `metrics.jsonl` files under `{_rel(outputs_root)}`: {metric_count}",
        f"- Discovered `layer_metrics.jsonl` files under `{_rel(outputs_root)}`: {layer_metric_count}",
        "",
        "## Compared Options",
        "",
        "| Suite | Family | Curves | Unavailable cases |",
        "| --- | --- | ---: | --- |",
    ]
    for suite, family, available_count, total, missing in _family_summary(trials):
        lines.append(f"| {suite} | `{family}` | {available_count}/{total} | {missing} |")
    lines.extend(
        [
            "",
            "## Source CSVs",
            "",
        ]
    )
    for source in sources:
        lines.append(
            f"- {source.suite}: `{_rel(source.trials_csv)}` and `{_rel(source.families_csv)}`"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="LABEL:TRIALS_CSV:FAMILIES_CSV",
        help="Additional explicit source pair. May be repeated.",
    )
    parser.add_argument(
        "--discover-extra-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-include docs/*_final_test_trials.csv and validation trial/family pairs.",
    )
    parser.add_argument(
        "--include-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When discovering sources, include docs/*_validation_trials.csv pairs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed loss-curve artifacts are stale",
    )
    return parser.parse_args()


def _write_outputs(
    *,
    trials: list[TrialCurve],
    out_dir: Path,
    outputs_root: Path,
    sources: Sequence[LossCurveSource],
) -> None:
    _render_svg(trials, out_dir / "loss_curves.svg")
    _write_manifest(trials, out_dir / "manifest.csv")
    _write_readme(trials, out_dir / "README.md", outputs_root, sources)


def _check_outputs(
    *,
    trials: list[TrialCurve],
    out_dir: Path,
    outputs_root: Path,
    sources: Sequence[LossCurveSource],
) -> int:
    with tempfile.TemporaryDirectory(prefix="loss-curves-check-") as tmp:
        tmp_dir = Path(tmp)
        _write_outputs(
            trials=trials,
            out_dir=tmp_dir,
            outputs_root=outputs_root,
            sources=sources,
        )
        stale: list[str] = []
        for name in ("loss_curves.svg", "manifest.csv", "README.md"):
            path = out_dir / name
            expected_path = tmp_dir / name
            if not path.exists():
                stale.append(f"missing: {_rel(path)}")
                continue
            if path.read_text(encoding="utf-8") != expected_path.read_text(encoding="utf-8"):
                stale.append(f"stale: {_rel(path)}")
        if stale:
            for item in stale:
                print(item)
            return 1
    print(f"loss-curve artifacts are current: {_rel(out_dir)}")
    return 0


def main() -> int:
    args = parse_args()
    sources = _resolve_sources(
        docs_dir=DOCS,
        source_specs=args.source,
        discover_extra_sources=args.discover_extra_sources,
        include_validation=args.include_validation,
    )
    trials = _load_trials(args.outputs_root, sources)
    out_dir = args.out_dir
    if args.check:
        return _check_outputs(
            trials=trials,
            out_dir=out_dir,
            outputs_root=args.outputs_root,
            sources=sources,
        )
    _write_outputs(
        trials=trials,
        out_dir=out_dir,
        outputs_root=args.outputs_root,
        sources=sources,
    )
    available = sum(trial.has_curve for trial in trials)
    print(f"Rendered {available}/{len(trials)} available comparison trial curves to {_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
