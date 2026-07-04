from __future__ import annotations

import argparse
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig
from cifar_mamba_fff.train_teacher import (
    TeacherCandidateResult,
    TeacherTrainConfig,
    evaluate_teacher_candidate,
    load_teacher_run_config,
)
from cifar_mamba_fff.utils import RunContext, append_jsonl, bool_arg, load_yaml, write_json


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
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


def evaluate_hpo_candidate(
    base_model: Mamba3CifarConfig,
    overrides: Mapping[str, object],
) -> tuple[Mamba3CifarConfig, TeacherCandidateResult]:
    model_overrides, _ = split_teacher_overrides(overrides)
    candidate_config = replace(base_model, **model_overrides)
    result = evaluate_teacher_candidate(candidate_config)
    return candidate_config, result


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
                "model": candidate_config.__dict__,
                "overrides": overrides,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/teacher_default.yaml")
    parser.add_argument("--hpo-config", "--config", default="configs/teacher_hpo.yaml")
    parser.add_argument("--output-dir", default="outputs/teacher_hpo")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args(argv)

    if args.max_candidates <= 0:
        raise ValueError("max-candidates must be positive")
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
        },
    )
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
