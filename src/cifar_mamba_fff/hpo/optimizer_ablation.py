from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

from cifar_mamba_fff.train_teacher import (
    TeacherRunConfig,
    TeacherTrainConfig,
    load_teacher_run_config,
    run_teacher_training,
)
from cifar_mamba_fff.utils import RunContext, bool_arg, load_yaml, write_json

NORMUON_ABLATION_OPTIMIZERS = frozenset({"normuon_adamw", "muon_normuon", "pace_normuon"})


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


def _expect_sequence(value: object, section: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{section} must be a sequence")
    return value


def _resolve_case_run_config(
    base_run: TeacherRunConfig,
    case: Mapping[str, object],
    *,
    seed: int,
    quick_smoke: bool,
) -> TeacherRunConfig:
    train_raw = _expect_mapping(case.get("train", {}), "case.train")
    unknown = sorted(set(train_raw) - set(TeacherTrainConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"case.train has unknown keys: {', '.join(unknown)}")
    train_config = replace(base_run.train, **dict(train_raw))
    train_config.validate()
    data_config = replace(
        base_run.data,
        batch_size=train_config.batch_size_per_gpu,
        num_workers=train_config.num_workers,
        seed=seed,
        quick_smoke=quick_smoke,
        label_smoothing=train_config.label_smoothing,
        mixup=train_config.mixup,
        cutmix=train_config.cutmix,
        use_test=False,
    )
    run_config = replace(base_run, seed=seed, data=data_config, train=train_config)
    run_config.validate()
    if run_config.data.use_test:
        raise ValueError("optimizer ablation may not access CIFAR-10 test data")
    return run_config


def _case_name(case: Mapping[str, object], index: int) -> str:
    name = case.get("name", f"case_{index:03d}")
    if not isinstance(name, str) or not name:
        raise ValueError("case.name must be a non-empty string")
    return name


def _resolve_seed_list(
    raw_config: Mapping[str, object],
    *,
    base_seed: int,
    seed_override: int | None,
) -> list[int]:
    if "seeds" in raw_config:
        raw_seeds = _expect_sequence(raw_config["seeds"], "seeds")
        seeds = [int(item) for item in raw_seeds]
    else:
        seeds = [int(raw_config.get("seed", base_seed) if seed_override is None else seed_override)]
    if not seeds:
        raise ValueError("optimizer ablation seed list must be non-empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("optimizer ablation seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("optimizer ablation seeds must be unique")
    return seeds


def _normuon_update_rms_note(
    raw_config: Mapping[str, object],
    cases: Sequence[object],
) -> str:
    has_normuon_case = False
    for index, raw_case in enumerate(cases):
        case = _expect_mapping(raw_case, f"cases[{index}]")
        train = _expect_mapping(case.get("train", {}), f"cases[{index}].train")
        optimizer = train.get("optimizer")
        if isinstance(optimizer, str) and optimizer in NORMUON_ABLATION_OPTIMIZERS:
            has_normuon_case = True
            break
    note = raw_config.get("update_rms_calibration_note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise ValueError("update_rms_calibration_note must be a string when set")
    note = note.strip()
    if has_normuon_case and not note:
        raise ValueError(
            "NorMuon/PACE+NorMuon ablations require update_rms_calibration_note "
            "unless a separate LR-sweep protocol is added"
        )
    return note


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _csv_row(summary: Mapping[str, object]) -> dict[str, object]:
    optimizer_summary = summary.get("optimizer_summary", {})
    if not isinstance(optimizer_summary, Mapping):
        optimizer_summary = {}
    seed_list = summary.get("seed_list", [summary["seed"]])
    return {
        "case": summary["case"],
        "seed": summary["seed"],
        "seed_list": json.dumps(list(seed_list), separators=(",", ":")),
        "seed_count": summary.get("seed_count", 1),
        "optimizer": summary["optimizer"],
        "schedule": summary["schedule"],
        "update_rms_calibration_note": summary.get("update_rms_calibration_note", ""),
        "best_val_accuracy": summary.get("best_val_accuracy"),
        "train_steps_total": summary.get("train_steps_total"),
        "train_images_seen": summary.get("train_images_seen"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "train_images_per_second": summary.get("train_images_per_second"),
        "uses_ema_eval": optimizer_summary.get("uses_ema_eval"),
        "optimizer_source": optimizer_summary.get("optimizer_source"),
        "test_accessed": False,
    }


def _mean_finite(values: Sequence[object]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, int | float) and math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _aggregate_case_results(
    *,
    name: str,
    global_index: int,
    seed_results: Sequence[Mapping[str, object]],
    update_rms_calibration_note: str,
) -> dict[str, object]:
    first = seed_results[0]
    seed_list = [int(result["seed"]) for result in seed_results]
    row: dict[str, object] = {
        "case": name,
        "case_index": global_index,
        "seed": seed_list[0],
        "seed_list": seed_list,
        "seed_count": len(seed_list),
        "optimizer": first["optimizer"],
        "schedule": first["schedule"],
        "test_accessed": False,
        "per_seed_results": list(seed_results),
        "optimizer_summary": first.get("optimizer_summary", {}),
    }
    if first["optimizer"] in NORMUON_ABLATION_OPTIMIZERS:
        row["update_rms_calibration_note"] = update_rms_calibration_note
    for metric in (
        "best_val_accuracy",
        "train_steps_total",
        "train_images_seen",
        "elapsed_seconds",
        "train_images_per_second",
        "train_images_per_second_train_only",
        "peak_cuda_memory_reserved_bytes",
        "parameter_count",
    ):
        mean = _mean_finite([result.get(metric) for result in seed_results])
        if mean is not None:
            row[metric] = mean
    return row


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_csv_row(rows[0]).keys()) if rows else ["case", "test_accessed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def run_optimizer_ablation(
    *,
    base_config_path: Path,
    ablation_config_path: Path,
    output_dir: Path,
    quick_smoke: bool,
    max_train_steps: int | None,
    max_val_steps: int | None,
    case_offset: int = 0,
    case_limit: int | None = None,
    seed: int | None = None,
    training_fn=run_teacher_training,
) -> dict[str, object]:
    base_run = load_teacher_run_config(base_config_path, quick_smoke=quick_smoke)
    raw = load_yaml(ablation_config_path)
    cases = list(_expect_sequence(raw.get("cases"), "cases"))
    if case_offset < 0:
        raise ValueError("case_offset must be non-negative")
    if case_limit is not None and case_limit <= 0:
        raise ValueError("case_limit must be positive when set")
    selected_cases = cases[case_offset:] if case_limit is None else cases[case_offset:case_offset + case_limit]
    if not selected_cases:
        raise ValueError("selected optimizer ablation case list is empty")

    seed_list = _resolve_seed_list(raw, base_seed=base_run.seed, seed_override=seed)
    update_rms_calibration_note = _normuon_update_rms_note(raw, cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for local_index, raw_case in enumerate(selected_cases):
        global_index = case_offset + local_index
        case = _expect_mapping(raw_case, f"cases[{global_index}]")
        name = _case_name(case, global_index)
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            case_dir / "case_config.json",
            _jsonable({
                "case": name,
                "case_index": global_index,
                "raw_case": case,
                "seed_list": seed_list,
                "update_rms_calibration_note": update_rms_calibration_note,
                "test_accessed": False,
            }),
        )
        seed_results: list[dict[str, object]] = []
        case_failed = False
        for case_seed in seed_list:
            run_config = _resolve_case_run_config(
                base_run,
                case,
                seed=case_seed,
                quick_smoke=quick_smoke,
            )
            seed_dir = case_dir if len(seed_list) == 1 else case_dir / f"seed_{case_seed}"
            if seed_dir != case_dir:
                seed_dir.mkdir(parents=True, exist_ok=False)
            write_json(
                seed_dir / "resolved_case_config.json",
                _jsonable({
                    "case": name,
                    "case_index": global_index,
                    "seed": case_seed,
                    "seed_list": seed_list,
                    "resolved_train": run_config.train,
                    "test_accessed": False,
                }),
            )
            try:
                result = training_fn(
                    run_config,
                    output_dir=seed_dir,
                    quick_smoke=quick_smoke,
                    max_train_steps=max_train_steps,
                    max_val_steps=max_val_steps,
                    save_checkpoint=False,
                    epoch_callback=None,
                )
            except Exception as exc:
                failures.append(
                    {
                        "case": name,
                        "case_index": global_index,
                        "seed": case_seed,
                        "seed_list": seed_list,
                        "status": "failed_logic",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "test_accessed": False,
                    }
                )
                case_failed = True
                break
            best_val = float(result.get("best_val_accuracy", float("nan")))
            if not math.isfinite(best_val):
                failures.append(
                    {
                        "case": name,
                        "case_index": global_index,
                        "seed": case_seed,
                        "seed_list": seed_list,
                        "status": "failed_logic",
                        "reason": "best_val_accuracy was not finite",
                        "test_accessed": False,
                    }
                )
                case_failed = True
                break
            seed_results.append(
                {
                    "case": name,
                    "case_index": global_index,
                    "seed": run_config.seed,
                    "seed_list": seed_list,
                    "optimizer": run_config.train.optimizer,
                    "schedule": run_config.train.schedule,
                    "test_accessed": False,
                    **result,
                }
            )
        if case_failed:
            continue
        rows.append(
            _aggregate_case_results(
                name=name,
                global_index=global_index,
                seed_results=seed_results,
                update_rms_calibration_note=update_rms_calibration_note,
            )
        )

    summary = {
        "mode": "optimizer_ablation_t19",
        "status": "completed" if rows else "failed_zero_successes",
        "base_config": str(base_config_path),
        "ablation_config": str(ablation_config_path),
        "output_dir": str(output_dir),
        "case_offset": case_offset,
        "case_limit": case_limit,
        "seed_list": seed_list,
        "seed_count": len(seed_list),
        "update_rms_calibration_note": update_rms_calibration_note,
        "requested_cases": len(selected_cases),
        "succeeded": len(rows),
        "failed_logic": len(failures),
        "failures": failures,
        "results": rows,
        "summary_csv": str(output_dir / "optimizer_ablation_summary.csv"),
        "test_accessed": False,
    }
    write_json(output_dir / "optimizer_ablation_summary.json", _jsonable(summary))
    if rows:
        _write_summary_csv(output_dir / "optimizer_ablation_summary.csv", rows)
    if not rows:
        raise RuntimeError("optimizer ablation completed with zero successful cases")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/teacher_default.yaml")
    parser.add_argument("--ablation-config", "--config", default="configs/teacher_optimizer_ablation_t19.yaml")
    parser.add_argument("--output-dir", default="outputs/t19_optimizer_ablation")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    raw = load_yaml(args.ablation_config)
    base_run = load_teacher_run_config(args.base_config, quick_smoke=args.quick_smoke)
    seed_list = _resolve_seed_list(raw, base_seed=base_run.seed, seed_override=args.seed)
    run_seed = seed_list[0]
    context = RunContext(Path(args.output_dir), seed=run_seed, quick_smoke=args.quick_smoke)
    context.prepare()
    write_json(
        context.output_dir / "run_context.json",
        context.metadata()
        | {
            "base_config": str(args.base_config),
            "ablation_config": str(args.ablation_config),
            "max_train_steps": args.max_train_steps,
            "max_val_steps": args.max_val_steps,
            "case_offset": args.case_offset,
            "case_limit": args.case_limit,
            "seed_list": seed_list,
            "test_accessed": False,
        },
    )
    summary = run_optimizer_ablation(
        base_config_path=Path(args.base_config),
        ablation_config_path=Path(args.ablation_config),
        output_dir=context.output_dir,
        quick_smoke=args.quick_smoke,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        case_offset=args.case_offset,
        case_limit=args.case_limit,
        seed=args.seed,
    )
    print(f"optimizer ablation complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
