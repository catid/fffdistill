from __future__ import annotations

import argparse
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from filelock import FileLock

from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig
from cifar_mamba_fff.train_teacher import (
    TeacherCandidateResult,
    TeacherRunConfig,
    TeacherTrainConfig,
    evaluate_teacher_candidate,
    load_teacher_run_config,
    run_teacher_training,
)
from cifar_mamba_fff.utils import RunContext, append_jsonl, bool_arg, load_yaml, write_json


@dataclass(frozen=True)
class HpoCandidate:
    trial_index: int
    attempt_index: int
    overrides: dict[str, object]
    run_config: TeacherRunConfig
    parameter_count: int


class HpoTrialPruned(RuntimeError):
    pass


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


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


def _sample_loguniform(rng: random.Random, bounds: Sequence[object]) -> float:
    if len(bounds) != 2:
        raise ValueError("loguniform bounds must contain exactly two values")
    low, high = float(bounds[0]), float(bounds[1])
    if low <= 0.0 or high <= low:
        raise ValueError("loguniform bounds must be positive and increasing")
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_teacher_overrides(
    search_space: Mapping[str, object],
    *,
    rng: random.Random,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key, values in search_space.items():
        if key.endswith("_loguniform"):
            overrides[key.removesuffix("_loguniform")] = _sample_loguniform(rng, values)  # type: ignore[arg-type]
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"search_space.{key} must be a non-empty sequence")
        overrides[key] = rng.choice(list(values))
    return overrides


def split_teacher_overrides(overrides: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    model_fields = set(Mamba3CifarConfig.__dataclass_fields__)
    train_fields = set(TeacherTrainConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - model_fields - train_fields)
    if unknown:
        raise ValueError(f"teacher HPO overrides contain unknown keys: {', '.join(unknown)}")
    model_overrides = {key: value for key, value in overrides.items() if key in model_fields}
    train_overrides = {key: value for key, value in overrides.items() if key in train_fields}
    return model_overrides, train_overrides


def resolve_hpo_run_config(
    base_run: TeacherRunConfig,
    overrides: Mapping[str, object],
    *,
    seed: int,
    quick_smoke: bool,
) -> TeacherRunConfig:
    model_overrides, train_overrides = split_teacher_overrides(overrides)
    model_config = replace(base_run.model, **model_overrides)
    train_config = replace(base_run.train, **train_overrides)
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
    )
    run_config = TeacherRunConfig(
        seed=seed,
        dataset_name=base_run.dataset_name,
        data=data_config,
        model=model_config,
        train=train_config,
    )
    run_config.validate()
    return run_config


def evaluate_hpo_candidate(
    base_model: Mamba3CifarConfig,
    overrides: Mapping[str, object],
) -> tuple[Mamba3CifarConfig, TeacherCandidateResult]:
    model_overrides, _ = split_teacher_overrides(overrides)
    candidate_config = replace(base_model, **model_overrides)
    result = evaluate_teacher_candidate(candidate_config)
    return candidate_config, result


def _locked_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock")
    with lock:
        append_jsonl(path, payload)


def sample_valid_hpo_candidates(
    base_run: TeacherRunConfig,
    search_space: Mapping[str, object],
    *,
    quick_smoke: bool,
    max_trials: int,
    max_attempts: int,
    rng: random.Random,
    event_log_path: Path | None = None,
) -> list[HpoCandidate]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if max_attempts < max_trials:
        raise ValueError("max_attempts must be >= max_trials")

    candidates: list[HpoCandidate] = []
    for attempt_index in range(max_attempts):
        if len(candidates) >= max_trials:
            break
        overrides = sample_teacher_overrides(search_space, rng=rng)
        candidate_model, result = evaluate_hpo_candidate(base_run.model, overrides)
        if not result.accepted:
            if event_log_path is not None:
                _locked_append_jsonl(
                    event_log_path,
                    {
                        "event": "rejected_pretrial",
                        "attempt_index": attempt_index,
                        "parameter_count": result.parameter_count,
                        "reason": result.reason,
                        "model": _jsonable(candidate_model),
                        "overrides": _jsonable(overrides),
                    },
                )
            continue

        trial_index = len(candidates)
        run_config = resolve_hpo_run_config(
            base_run,
            overrides,
            seed=base_run.seed + trial_index,
            quick_smoke=quick_smoke,
        )
        candidates.append(
            HpoCandidate(
                trial_index=trial_index,
                attempt_index=attempt_index,
                overrides=dict(overrides),
                run_config=run_config,
                parameter_count=int(result.parameter_count),
            )
        )
    return candidates


def write_candidate_filter_smoke(
    *,
    base_config_path: Path,
    hpo_config_path: Path,
    output_dir: Path,
    quick_smoke: bool,
    max_candidates: int,
) -> dict[str, object]:
    base_run = load_teacher_run_config(base_config_path, quick_smoke=quick_smoke)
    hpo_config = load_yaml(hpo_config_path)
    search_space = _expect_mapping(hpo_config.get("search_space"), "search_space")
    rng = random.Random(base_run.seed)
    output_path = output_dir / "teacher_hpo_candidate_filter.jsonl"
    accepted = 0
    rejected = 0
    for trial_index in range(max_candidates):
        overrides = sample_teacher_overrides(search_space, rng=rng)
        candidate_config, result = evaluate_hpo_candidate(base_run.model, overrides)
        if result.accepted:
            accepted += 1
        else:
            rejected += 1
        append_jsonl(
            output_path,
            {
                "trial_index": trial_index,
                "accepted": result.accepted,
                "parameter_count": result.parameter_count,
                "reason": result.reason,
                "model": _jsonable(candidate_config),
                "overrides": _jsonable(overrides),
            },
        )
    summary = {
        "candidate_filter_path": str(output_path),
        "max_candidates": max_candidates,
        "accepted": accepted,
        "rejected": rejected,
        "quick_smoke": quick_smoke,
    }
    write_json(output_dir / "teacher_hpo_summary.json", summary)
    return summary


def _make_prune_callback(
    *,
    trial_index: int,
    prune_on: str,
    prune_min_value: float | None,
    event_log_path: Path,
):
    if prune_min_value is None:
        return None

    def callback(metrics: dict[str, object]) -> None:
        metric = float(metrics[prune_on])
        if metric < prune_min_value:
            _locked_append_jsonl(
                event_log_path,
                {
                    "event": "trial_pruned",
                    "trial_index": trial_index,
                    "epoch": metrics.get("epoch"),
                    "metric": prune_on,
                    "value": metric,
                    "threshold": prune_min_value,
                },
            )
            raise HpoTrialPruned(
                f"trial {trial_index} pruned: {prune_on}={metric} < {prune_min_value}"
            )

    return callback


def run_teacher_hpo(
    *,
    base_config_path: Path,
    hpo_config_path: Path,
    output_dir: Path,
    quick_smoke: bool,
    max_trials: int,
    max_attempts: int,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    prune_min_value: float | None = None,
    training_fn=run_teacher_training,
) -> dict[str, object]:
    base_run = load_teacher_run_config(base_config_path, quick_smoke=quick_smoke)
    hpo_config = load_yaml(hpo_config_path)
    search_space = _expect_mapping(hpo_config.get("search_space"), "search_space")
    prune_on = str(hpo_config.get("prune_on", "val_accuracy"))
    event_log_path = output_dir / "teacher_hpo_events.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(base_run.seed)
    candidates = sample_valid_hpo_candidates(
        base_run,
        search_space,
        quick_smoke=quick_smoke,
        max_trials=max_trials,
        max_attempts=max_attempts,
        rng=rng,
        event_log_path=event_log_path,
    )

    succeeded = 0
    pruned = 0
    failed_logic = 0
    best_metric: float | None = None
    best_trial: int | None = None
    for candidate in candidates:
        trial_dir = output_dir / "trials" / f"trial_{candidate.trial_index:06d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        trial_config_path = trial_dir / "trial_config.json"
        write_json(
            trial_config_path,
            _jsonable(
                {
                    "trial_index": candidate.trial_index,
                    "attempt_index": candidate.attempt_index,
                    "parameter_count": candidate.parameter_count,
                    "overrides": candidate.overrides,
                    "run_config": candidate.run_config,
                }
            ),
        )
        _locked_append_jsonl(
            event_log_path,
            {
                "event": "trial_started",
                "trial_index": candidate.trial_index,
                "attempt_index": candidate.attempt_index,
                "parameter_count": candidate.parameter_count,
                "output_dir": str(trial_dir),
            },
        )
        try:
            summary = training_fn(
                candidate.run_config,
                output_dir=trial_dir,
                quick_smoke=quick_smoke,
                max_train_steps=max_train_steps,
                max_val_steps=max_val_steps,
                save_checkpoint=not quick_smoke,
                epoch_callback=_make_prune_callback(
                    trial_index=candidate.trial_index,
                    prune_on=prune_on,
                    prune_min_value=prune_min_value,
                    event_log_path=event_log_path,
                ),
            )
        except HpoTrialPruned as exc:
            pruned += 1
            trial_summary = {
                "trial_index": candidate.trial_index,
                "status": "pruned",
                "reason": str(exc),
            }
        except Exception as exc:
            failed_logic += 1
            trial_summary = {
                "trial_index": candidate.trial_index,
                "status": "failed_logic",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        else:
            succeeded += 1
            metric = float(summary.get("best_val_accuracy", float("nan")))
            if math.isfinite(metric) and (best_metric is None or metric > best_metric):
                best_metric = metric
                best_trial = candidate.trial_index
            trial_summary = {
                "trial_index": candidate.trial_index,
                "status": "succeeded",
                "summary": summary,
            }
        write_json(trial_dir / "trial_summary.json", _jsonable(trial_summary))
        _locked_append_jsonl(event_log_path, {"event": "trial_finished", **_jsonable(trial_summary)})

    summary = {
        "mode": "teacher_hpo",
        "storage": "jsonl_filelock",
        "event_log_path": str(event_log_path),
        "requested_trials": max_trials,
        "accepted_trials": len(candidates),
        "max_attempts": max_attempts,
        "succeeded": succeeded,
        "pruned": pruned,
        "failed_logic": failed_logic,
        "best_trial": best_trial,
        "best_val_accuracy": best_metric,
        "quick_smoke": quick_smoke,
    }
    write_json(output_dir / "teacher_hpo_summary.json", _jsonable(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/teacher_default.yaml")
    parser.add_argument("--hpo-config", "--config", default="configs/teacher_hpo.yaml")
    parser.add_argument("--output-dir", default="outputs/teacher_hpo")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--execute-trials", type=bool_arg, default=False)
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--prune-min-value", type=float, default=None)
    args = parser.parse_args(argv)

    if args.max_candidates <= 0:
        raise ValueError("max-candidates must be positive")
    if args.max_trials <= 0:
        raise ValueError("max-trials must be positive")
    if args.max_attempts < args.max_trials:
        raise ValueError("max-attempts must be >= max-trials")

    base_run = load_teacher_run_config(args.base_config, quick_smoke=args.quick_smoke)
    context = RunContext(Path(args.output_dir), seed=base_run.seed, quick_smoke=args.quick_smoke)
    context.prepare()
    write_json(
        context.output_dir / "run_context.json",
        context.metadata()
        | {
            "base_config": str(args.base_config),
            "hpo_config": str(args.hpo_config),
            "max_candidates": args.max_candidates,
            "execute_trials": args.execute_trials,
            "max_trials": args.max_trials,
            "max_attempts": args.max_attempts,
        },
    )
    if args.execute_trials:
        summary = run_teacher_hpo(
            base_config_path=Path(args.base_config),
            hpo_config_path=Path(args.hpo_config),
            output_dir=context.output_dir,
            quick_smoke=args.quick_smoke,
            max_trials=args.max_trials,
            max_attempts=args.max_attempts,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            prune_min_value=args.prune_min_value,
        )
        print(f"teacher HPO run complete: {summary}")
        return 0

    summary = write_candidate_filter_smoke(
        base_config_path=Path(args.base_config),
        hpo_config_path=Path(args.hpo_config),
        output_dir=context.output_dir,
        quick_smoke=args.quick_smoke,
        max_candidates=args.max_candidates,
    )
    if not args.quick_smoke:
        raise RuntimeError(
            "teacher HPO training launch is still gated behind bounded trainer smoke and scheduler launch sanity; "
            f"candidate filtering completed: {summary}"
        )
    print(f"teacher HPO candidate filter smoke complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
