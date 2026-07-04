from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from cifar_mamba_fff.finetune_student import _expect_mapping, _expect_sequence, _jsonable
from cifar_mamba_fff.utils import bool_arg, load_yaml, write_json

FineTuneTrialRunner = Callable[..., dict[str, object]]


TRAIN_OVERRIDE_KEYS = {
    "fine_tune_epochs": "epochs",
    "epochs": "epochs",
    "lr_muon": "lr_muon",
    "lr_adamw": "lr_adamw",
    "weight_decay_muon": "weight_decay_muon",
    "weight_decay_adamw": "weight_decay_adamw",
    "schedule": "schedule",
    "optimizer": "optimizer",
    "optimizer_family": "optimizer",
    "batch_size_per_gpu": "batch_size_per_gpu",
    "num_workers": "num_workers",
}
LOSS_OVERRIDE_KEYS = {
    "kd_temperature": "kd_temperature",
    "lambda_kd": "lambda_kd",
    "lambda_ce": "lambda_ce",
    "lambda_hidden": "lambda_hidden",
    "lambda_balance": "lambda_balance",
    "balance_recipe": "balance_recipe",
    "min_leaf_tokens": "min_leaf_tokens",
}
IGNORED_RECORDED_KEYS = {"keep_locoprop_refits"}
KNOWN_OVERRIDE_KEYS = set(TRAIN_OVERRIDE_KEYS) | set(LOSS_OVERRIDE_KEYS) | IGNORED_RECORDED_KEYS | {"case_name"}


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=True), encoding="utf-8")


def _case_name(case: Mapping[str, object], index: int) -> str:
    name = case.get("name", case.get("case_name", f"trial_{index:06d}"))
    if not isinstance(name, str) or not name:
        raise ValueError("fine-tune case name must be a non-empty string")
    return name


def _apply_overrides(base_config: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(overrides) - KNOWN_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown fine-tune HPO override keys: {', '.join(unknown)}")
    config = dict(base_config)
    train = dict(_expect_mapping(config.get("train", {}), "base.train"))
    losses = dict(_expect_mapping(config.get("losses", {}), "base.losses"))
    recorded: dict[str, object] = {}
    for key, value in overrides.items():
        if key in TRAIN_OVERRIDE_KEYS:
            train[TRAIN_OVERRIDE_KEYS[key]] = value
        elif key in LOSS_OVERRIDE_KEYS:
            losses[LOSS_OVERRIDE_KEYS[key]] = value
        elif key in IGNORED_RECORDED_KEYS:
            recorded[key] = value
    config["train"] = train
    config["losses"] = losses
    if recorded:
        config["hpo_recorded_options"] = recorded
    return config


def _grid_cases(search_space: Mapping[str, object], *, max_trials: int, offset: int) -> list[dict[str, object]]:
    keys = list(search_space)
    values = [_expect_sequence(search_space[key], f"search_space.{key}") for key in keys]
    cases: list[dict[str, object]] = []
    for combo_index, combo in enumerate(itertools.product(*values)):
        if combo_index < offset:
            continue
        if len(cases) >= max_trials:
            break
        cases.append(dict(zip(keys, combo, strict=True)))
    return cases


def _configured_cases(hpo_config: Mapping[str, object], *, max_trials: int, offset: int) -> list[dict[str, object]]:
    if hpo_config.get("cases") is not None:
        raw_cases = _expect_sequence(hpo_config["cases"], "cases")
        selected = raw_cases[offset : offset + max_trials]
        return [dict(_expect_mapping(case, f"cases[{offset + idx}]")) for idx, case in enumerate(selected)]
    search_space = _expect_mapping(hpo_config.get("search_space"), "search_space")
    return _grid_cases(search_space, max_trials=max_trials, offset=offset)


def write_finetune_hpo_trial_plan(
    *,
    base_config: Mapping[str, object],
    hpo_config: Mapping[str, object],
    output_dir: Path,
    max_trials: int,
    grid_offset: int = 0,
) -> dict[str, object]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if grid_offset < 0:
        raise ValueError("grid_offset must be non-negative")
    cases = _configured_cases(hpo_config, max_trials=max_trials, offset=grid_offset)
    if not cases:
        raise RuntimeError("fine-tune HPO produced zero cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, object]] = []
    for local_index, case in enumerate(cases):
        trial_index = grid_offset + local_index
        name = _case_name(case, trial_index)
        overrides = {key: value for key, value in case.items() if key not in {"name"}}
        trial_config = _apply_overrides(base_config, overrides)
        trial_dir = output_dir / "trials" / f"trial_{trial_index:06d}_{name}"
        config_path = trial_dir / "finetune_config.yaml"
        _write_yaml(config_path, trial_config)
        record = {
            "trial_index": trial_index,
            "case": name,
            "overrides": overrides,
            "config_path": str(config_path),
            "output_dir": str(trial_dir),
            "test_accessed": False,
        }
        write_json(trial_dir / "trial_config.json", record | {"config": trial_config})
        trials.append(record)
    summary = {
        "mode": "finetune_hpo_plan",
        "status": "planned",
        "study_name": hpo_config.get("study_name", "finetune_hpo"),
        "requested_trials": max_trials,
        "accepted_trials": len(trials),
        "grid_offset": grid_offset,
        "test_accessed": False,
        "trials": trials,
    }
    write_json(output_dir / "finetune_hpo_summary.json", summary)
    return summary


def run_finetune_trial_command(
    *,
    config_path: Path,
    output_dir: Path,
    quick_smoke: bool,
    max_train_steps: int | None,
    max_val_steps: int | None,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "cifar_mamba_fff.finetune_student",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--quick-smoke",
        str(quick_smoke).lower(),
        "--smoke-mode",
        "metadata",
    ]
    if max_train_steps is not None:
        command.extend(["--max-train-steps", str(max_train_steps)])
    if max_val_steps is not None:
        command.extend(["--max-val-steps", str(max_val_steps)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    summary_path = output_dir / "metrics_summary.json"
    payload: dict[str, object] = {
        "status": "succeeded" if completed.returncode == 0 else "failed_logic",
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout[-8192:],
        "stderr": completed.stderr[-8192:],
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "test_accessed": False,
    }
    if summary_path.exists():
        payload["summary"] = load_yaml(summary_path)
    return payload


def _csv_row(record: Mapping[str, object]) -> dict[str, object]:
    result = _expect_mapping(record.get("result", {}), "trial.result")
    summary = result.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "trial_index": record.get("trial_index"),
        "case": record.get("case"),
        "status": record.get("status"),
        "best_val_accuracy": summary.get("best_val_accuracy"),
        "train_steps_total": summary.get("train_steps_total"),
        "student_replacement_count": summary.get("student_replacement_count"),
        "optimizer": summary.get("optimizer"),
        "schedule": summary.get("schedule"),
        "test_accessed": record.get("test_accessed", False),
    }


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(_csv_row(rows[0]).keys()) if rows else ["trial_index", "test_accessed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def run_finetune_hpo_trials(
    *,
    base_config: Mapping[str, object],
    hpo_config: Mapping[str, object],
    output_dir: Path,
    max_trials: int,
    grid_offset: int = 0,
    quick_smoke: bool = False,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    trial_runner: FineTuneTrialRunner = run_finetune_trial_command,
) -> dict[str, object]:
    plan = write_finetune_hpo_trial_plan(
        base_config=base_config,
        hpo_config=hpo_config,
        output_dir=output_dir,
        max_trials=max_trials,
        grid_offset=grid_offset,
    )
    succeeded = 0
    failed_logic = 0
    trial_results: list[dict[str, object]] = []
    for trial in plan["trials"]:
        trial_record = _expect_mapping(trial, "trial")
        trial_dir = Path(str(trial_record["output_dir"]))
        config_path = Path(str(trial_record["config_path"]))
        try:
            runner_result = trial_runner(
                config_path=config_path,
                output_dir=trial_dir,
                quick_smoke=quick_smoke,
                max_train_steps=max_train_steps,
                max_val_steps=max_val_steps,
            )
        except Exception as exc:
            runner_result = {
                "status": "failed_logic",
                "reason": f"{type(exc).__name__}: {exc}",
                "test_accessed": False,
            }
        status = str(runner_result.get("status", "failed_logic"))
        test_accessed = bool(runner_result.get("test_accessed", False))
        if test_accessed:
            status = "failed_logic"
            runner_result = dict(runner_result) | {"reason": "trial reported CIFAR-10 test access"}
        if status == "succeeded":
            succeeded += 1
        else:
            failed_logic += 1
        record = {
            **dict(trial_record),
            "status": status,
            "result": runner_result,
            "test_accessed": test_accessed,
        }
        write_json(trial_dir / "trial_result.json", record)
        trial_results.append(record)

    summary = {
        **{key: value for key, value in plan.items() if key != "trials"},
        "mode": "finetune_hpo_execute",
        "status": "completed" if succeeded > 0 else "failed_zero_successes",
        "succeeded": succeeded,
        "failed_logic": failed_logic,
        "quick_smoke": quick_smoke,
        "max_train_steps": max_train_steps,
        "max_val_steps": max_val_steps,
        "test_accessed": any(bool(result["test_accessed"]) for result in trial_results),
        "summary_csv": str(output_dir / "finetune_hpo_summary.csv"),
        "trials": trial_results,
    }
    write_json(output_dir / "finetune_hpo_summary.json", summary)
    _write_summary_csv(output_dir / "finetune_hpo_summary.csv", trial_results)
    if succeeded == 0:
        raise RuntimeError("fine-tune HPO completed with zero successful trials")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", "--config", default="configs/finetune_default.yaml")
    parser.add_argument("--hpo-config", default="configs/finetune_hpo.yaml")
    parser.add_argument("--output-dir", default="outputs/finetune_hpo")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--grid-offset", type=int, default=0)
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--plan-only", type=bool_arg, default=False)
    args = parser.parse_args(argv)

    base_config = load_yaml(args.base_config)
    hpo_config = load_yaml(args.hpo_config)
    if args.plan_only:
        summary = write_finetune_hpo_trial_plan(
            base_config=base_config,
            hpo_config=hpo_config,
            output_dir=Path(args.output_dir),
            max_trials=args.max_trials,
            grid_offset=args.grid_offset,
        )
    else:
        summary = run_finetune_hpo_trials(
            base_config=base_config,
            hpo_config=hpo_config,
            output_dir=Path(args.output_dir),
            max_trials=args.max_trials,
            grid_offset=args.grid_offset,
            quick_smoke=args.quick_smoke,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
        )
    print(f"fine-tune HPO complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
