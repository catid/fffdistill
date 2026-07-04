from __future__ import annotations

import argparse
import itertools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig
from cifar_mamba_fff.train_teacher import (
    TeacherCandidateResult,
    TeacherRunConfig,
    TeacherTrainConfig,
    build_teacher_model,
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


class CudaKernelSmokeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateFilterConfig:
    cuda_kernel_smoke: bool = False
    kernel_smoke_batch_size: int = 1
    parameter_count_prefilter: bool = False

    def validate(self) -> None:
        if not isinstance(self.cuda_kernel_smoke, bool):
            raise ValueError("candidate_filter.cuda_kernel_smoke must be a bool")
        if (
            isinstance(self.kernel_smoke_batch_size, bool)
            or not isinstance(self.kernel_smoke_batch_size, int)
            or self.kernel_smoke_batch_size <= 0
        ):
            raise ValueError("candidate_filter.kernel_smoke_batch_size must be a positive integer")
        if not isinstance(self.parameter_count_prefilter, bool):
            raise ValueError("candidate_filter.parameter_count_prefilter must be a bool")


def _parse_optional_bool(value: object, *, key: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"candidate_filter.{key} must be a bool")
    return value


def _parse_optional_positive_int(value: object, *, key: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"candidate_filter.{key} must be a positive integer")
    return value


def parse_candidate_filter_config(raw: Mapping[str, object]) -> CandidateFilterConfig:
    value = raw.get("candidate_filter", {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("candidate_filter must be a mapping")
    unknown = sorted(
        set(value)
        - {"cuda_kernel_smoke", "kernel_smoke_batch_size", "parameter_count_prefilter"}
    )
    if unknown:
        raise ValueError(f"candidate_filter contains unknown keys: {', '.join(unknown)}")
    config = CandidateFilterConfig(
        cuda_kernel_smoke=_parse_optional_bool(
            value.get("cuda_kernel_smoke"),
            key="cuda_kernel_smoke",
            default=False,
        ),
        kernel_smoke_batch_size=_parse_optional_positive_int(
            value.get("kernel_smoke_batch_size"),
            key="kernel_smoke_batch_size",
            default=1,
        ),
        parameter_count_prefilter=_parse_optional_bool(
            value.get("parameter_count_prefilter"),
            key="parameter_count_prefilter",
            default=False,
        ),
    )
    config.validate()
    return config


def _resolve_hpo_seed(
    hpo_config: Mapping[str, object],
    *,
    base_seed: int,
    seed_override: int | None,
) -> int:
    seed = base_seed if seed_override is None else seed_override
    if seed_override is None and "seed" in hpo_config:
        seed = int(hpo_config["seed"])
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("HPO seed must be a non-negative integer")
    return int(seed)


def _with_run_seed(base_run: TeacherRunConfig, seed: int) -> TeacherRunConfig:
    return replace(base_run, seed=seed, data=replace(base_run.data, seed=seed))


def _is_cuda_oom(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        message = str(current).lower()
        if isinstance(current, RuntimeError) and "cuda" in message and "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _cleanup_cuda_after_oom() -> None:
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        return


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


def _search_space_sequence(search_space: Mapping[str, object], key: str) -> list[object]:
    values = search_space[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"search_space.{key} must be a non-empty sequence")
    return list(values)


def build_parameter_count_prefilter_pool(
    base_model: Mamba3CifarConfig,
    search_space: Mapping[str, object],
) -> list[dict[str, object]]:
    model_fields = set(Mamba3CifarConfig.__dataclass_fields__)
    model_keys = [
        key
        for key in search_space
        if key in model_fields and not key.endswith("_loguniform")
    ]
    if not model_keys:
        return [{}]

    pool: list[dict[str, object]] = []
    value_lists = [_search_space_sequence(search_space, key) for key in model_keys]
    for values in itertools.product(*value_lists):
        overrides = dict(zip(model_keys, values, strict=True))
        _, result = evaluate_hpo_candidate(base_model, overrides)
        if result.accepted:
            pool.append(overrides)
    return pool


def _sample_with_prefiltered_model_overrides(
    search_space: Mapping[str, object],
    *,
    rng: random.Random,
    model_pool: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    overrides = sample_teacher_overrides(search_space, rng=rng)
    if model_pool:
        overrides.update(dict(rng.choice(list(model_pool))))
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


def cuda_kernel_smoke_candidate(
    run_config: TeacherRunConfig,
    *,
    batch_size: int,
) -> TeacherCandidateResult:
    if not torch.cuda.is_available():
        raise CudaKernelSmokeUnavailable("candidate_filter.cuda_kernel_smoke requires CUDA")
    device = torch.device("cuda")
    model: torch.nn.Module | None = None
    image: torch.Tensor | None = None
    try:
        torch.manual_seed(run_config.seed)
        model, parameter_count = build_teacher_model(run_config.model, device=device, enforce_target_params=True)
        model.train()
        model.zero_grad(set_to_none=True)
        image = torch.randn(batch_size, 3, 32, 32, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(image)
            expected_shape = (batch_size, run_config.model.num_classes)
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f"teacher candidate smoke produced logits shape {tuple(logits.shape)}, "
                    f"expected {expected_shape}"
                )
            loss = logits.float().square().mean()
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError("non-finite teacher candidate smoke loss")
        loss.backward()
        torch.cuda.synchronize(device)
        return TeacherCandidateResult(True, parameter_count, "accepted")
    except CudaKernelSmokeUnavailable:
        raise
    except Exception as exc:
        _cleanup_cuda_after_oom()
        count: int | None = None
        if model is not None:
            count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return TeacherCandidateResult(
            False,
            count,
            f"cuda_kernel_smoke_failed: {type(exc).__name__}: {exc}",
        )
    finally:
        del image
        del model
        _cleanup_cuda_after_oom()


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
    candidate_filter: CandidateFilterConfig | None = None,
    kernel_smoke_fn: Callable[[TeacherRunConfig], TeacherCandidateResult] | None = None,
) -> list[HpoCandidate]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if max_attempts < max_trials:
        raise ValueError("max_attempts must be >= max_trials")
    filter_config = candidate_filter or CandidateFilterConfig()
    if kernel_smoke_fn is None:
        def kernel_smoke_fn(run_config: TeacherRunConfig) -> TeacherCandidateResult:
            return cuda_kernel_smoke_candidate(
                run_config,
                batch_size=filter_config.kernel_smoke_batch_size,
            )

    candidates: list[HpoCandidate] = []
    model_pool: list[dict[str, object]] | None = None
    if filter_config.parameter_count_prefilter:
        model_pool = build_parameter_count_prefilter_pool(base_run.model, search_space)
        if event_log_path is not None:
            _locked_append_jsonl(
                event_log_path,
                {
                    "event": "parameter_count_prefilter_pool",
                    "accepted_model_shapes": len(model_pool),
                },
            )
        if not model_pool:
            return []

    for attempt_index in range(max_attempts):
        if len(candidates) >= max_trials:
            break
        overrides = _sample_with_prefiltered_model_overrides(
            search_space,
            rng=rng,
            model_pool=model_pool,
        )
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
        if filter_config.cuda_kernel_smoke:
            try:
                smoke_result = kernel_smoke_fn(run_config)
            except CudaKernelSmokeUnavailable:
                raise
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _cleanup_cuda_after_oom()
                smoke_result = TeacherCandidateResult(
                    False,
                    result.parameter_count,
                    f"cuda_kernel_smoke_failed: {type(exc).__name__}: {exc}",
                )
            if not smoke_result.accepted:
                if event_log_path is not None:
                    _locked_append_jsonl(
                        event_log_path,
                        {
                            "event": "rejected_pretrial",
                            "attempt_index": attempt_index,
                            "parameter_count": smoke_result.parameter_count or result.parameter_count,
                            "reason": smoke_result.reason,
                            "model": _jsonable(candidate_model),
                            "overrides": _jsonable(overrides),
                        },
                    )
                continue
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
    seed: int | None = None,
) -> dict[str, object]:
    base_run = load_teacher_run_config(base_config_path, quick_smoke=quick_smoke)
    hpo_config = load_yaml(hpo_config_path)
    search_space = _expect_mapping(hpo_config.get("search_space"), "search_space")
    candidate_filter = parse_candidate_filter_config(hpo_config)
    hpo_seed = _resolve_hpo_seed(hpo_config, base_seed=base_run.seed, seed_override=seed)
    rng = random.Random(hpo_seed)
    output_path = output_dir / "teacher_hpo_candidate_filter.jsonl"
    accepted = 0
    rejected = 0
    for trial_index in range(max_candidates):
        overrides = sample_teacher_overrides(search_space, rng=rng)
        candidate_config, result = evaluate_hpo_candidate(base_run.model, overrides)
        if result.accepted and candidate_filter.cuda_kernel_smoke:
            run_config = resolve_hpo_run_config(
                _with_run_seed(base_run, hpo_seed),
                overrides,
                seed=hpo_seed + trial_index,
                quick_smoke=quick_smoke,
            )
            result = cuda_kernel_smoke_candidate(
                run_config,
                batch_size=candidate_filter.kernel_smoke_batch_size,
            )
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
        "seed": hpo_seed,
        "quick_smoke": quick_smoke,
        "candidate_filter": _jsonable(candidate_filter),
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
    seed: int | None = None,
    training_fn=run_teacher_training,
) -> dict[str, object]:
    base_run = load_teacher_run_config(base_config_path, quick_smoke=quick_smoke)
    hpo_config = load_yaml(hpo_config_path)
    search_space = _expect_mapping(hpo_config.get("search_space"), "search_space")
    candidate_filter = parse_candidate_filter_config(hpo_config)
    prune_on = str(hpo_config.get("prune_on", "val_accuracy"))
    event_log_path = output_dir / "teacher_hpo_events.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    hpo_seed = _resolve_hpo_seed(hpo_config, base_seed=base_run.seed, seed_override=seed)
    seeded_base_run = _with_run_seed(base_run, hpo_seed)
    rng = random.Random(hpo_seed)
    candidates = sample_valid_hpo_candidates(
        seeded_base_run,
        search_space,
        quick_smoke=quick_smoke,
        max_trials=max_trials,
        max_attempts=max_attempts,
        rng=rng,
        event_log_path=event_log_path,
        candidate_filter=candidate_filter,
    )
    if not candidates:
        summary = {
            "mode": "teacher_hpo",
            "status": "failed_zero_candidates",
            "storage": "jsonl_filelock",
            "event_log_path": str(event_log_path),
            "requested_trials": max_trials,
            "accepted_trials": 0,
            "max_attempts": max_attempts,
            "succeeded": 0,
            "pruned": 0,
            "failed_logic": 0,
            "failed_oom": 0,
            "best_trial": None,
            "best_val_accuracy": None,
            "seed": hpo_seed,
            "quick_smoke": quick_smoke,
            "candidate_filter": _jsonable(candidate_filter),
        }
        write_json(output_dir / "teacher_hpo_summary.json", _jsonable(summary))
        _locked_append_jsonl(
            event_log_path,
            {"event": "hpo_failed_zero_candidates", **_jsonable(summary)},
        )
        raise RuntimeError("teacher HPO produced zero valid candidates; refusing to report success")

    succeeded = 0
    pruned = 0
    failed_logic = 0
    failed_oom = 0
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
            if _is_cuda_oom(exc):
                failed_oom += 1
                _cleanup_cuda_after_oom()
                trial_summary = {
                    "trial_index": candidate.trial_index,
                    "status": "failed_oom",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            else:
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
        "status": "completed" if succeeded > 0 else "failed_zero_successes",
        "storage": "jsonl_filelock",
        "event_log_path": str(event_log_path),
        "requested_trials": max_trials,
        "accepted_trials": len(candidates),
        "max_attempts": max_attempts,
        "succeeded": succeeded,
        "pruned": pruned,
        "failed_logic": failed_logic,
        "failed_oom": failed_oom,
        "best_trial": best_trial,
        "best_val_accuracy": best_metric,
        "seed": hpo_seed,
        "quick_smoke": quick_smoke,
        "candidate_filter": _jsonable(candidate_filter),
    }
    write_json(output_dir / "teacher_hpo_summary.json", _jsonable(summary))
    if succeeded == 0:
        _locked_append_jsonl(event_log_path, {"event": "hpo_failed_zero_successes", **_jsonable(summary)})
        raise RuntimeError("teacher HPO completed with zero successful trials; refusing to report success")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the HPO sampler seed and first trial run seed.",
    )
    args = parser.parse_args(argv)

    if args.max_candidates <= 0:
        raise ValueError("max-candidates must be positive")
    if args.max_trials <= 0:
        raise ValueError("max-trials must be positive")
    if args.max_attempts < args.max_trials:
        raise ValueError("max-attempts must be >= max-trials")

    base_run = load_teacher_run_config(args.base_config, quick_smoke=args.quick_smoke)
    hpo_config = load_yaml(args.hpo_config)
    hpo_seed = _resolve_hpo_seed(hpo_config, base_seed=base_run.seed, seed_override=args.seed)
    context = RunContext(Path(args.output_dir), seed=hpo_seed, quick_smoke=args.quick_smoke)
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
            "seed": hpo_seed,
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
            seed=hpo_seed,
        )
        print(f"teacher HPO run complete: {summary}")
        return 0

    summary = write_candidate_filter_smoke(
        base_config_path=Path(args.base_config),
        hpo_config_path=Path(args.hpo_config),
        output_dir=context.output_dir,
        quick_smoke=args.quick_smoke,
        max_candidates=args.max_candidates,
        seed=hpo_seed,
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
