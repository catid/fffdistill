from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from .data import Cifar10DataConfig, build_cifar10_loaders
from .metrics import accuracy
from .models.mamba3_cifar import Mamba3CifarConfig, Mamba3CifarTeacher
from .optim.muon_groups import format_param_assignments, split_muon_adamw_parameters
from .optim.normuon import MuonNorMuon, build_normuon_param_groups
from .optim.pace import (
    OPTIMIZER_EXPERIMENTS_COMMIT,
    OPTIMIZER_EXPERIMENTS_SOURCE,
    PaceOptimizer,
)
from .utils import RunContext, append_jsonl, bool_arg, load_yaml, seed_everything, write_json

OptimizerName = Literal[
    "muon_adamw",
    "official_muon",
    "pace_muon",
    "normuon_adamw",
    "muon_normuon",
    "pace_normuon",
]
PrecisionName = Literal["bf16"]
ScheduleName = Literal["cosine", "wsd"]
PacePrecondName = Literal["adam", "scalar", "row"]
SmokeMode = Literal["metadata", "train"]

OFFICIAL_MUON_OPTIMIZERS = frozenset({"muon_adamw", "official_muon"})
NORMUON_OPTIMIZERS = frozenset({"normuon_adamw", "muon_normuon"})
PACE_OPTIMIZERS = frozenset({"pace_muon", "pace_normuon"})
ALL_OPTIMIZERS = OFFICIAL_MUON_OPTIMIZERS | NORMUON_OPTIMIZERS | PACE_OPTIMIZERS


@dataclass(frozen=True)
class TeacherTrainConfig:
    epochs: int = 200
    batch_size_per_gpu: int = 512
    num_workers: int = 8
    precision: PrecisionName = "bf16"
    optimizer: OptimizerName = "muon_adamw"
    schedule: ScheduleName = "cosine"
    warmup_epochs: int = 10
    lr_muon: float = 0.02
    lr_adamw: float = 0.001
    weight_decay_muon: float = 0.03
    weight_decay_adamw: float = 0.03
    label_smoothing: float = 0.1
    mixup: float = 0.2
    cutmix: float = 1.0
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-10
    muon_momentum: float = 0.95
    wsd_stable_fraction: float = 0.8
    pace_pullback_c: float = 1e-3
    pace_kappa: float = 0.5
    pace_precond: PacePrecondName = "adam"
    pace_beta2: float = 0.999
    pace_eps: float = 1e-8
    pace_update_freq: int = 1
    normuon_beta2: float = 0.93
    normuon_eps: float = 1e-10
    grad_clip_norm: float | None = None

    def validate(self) -> None:
        _positive_int("epochs", self.epochs)
        _positive_int("batch_size_per_gpu", self.batch_size_per_gpu)
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.precision != "bf16":
            raise ValueError("teacher precision must be bf16 for this project")
        if self.optimizer not in ALL_OPTIMIZERS:
            raise ValueError(f"teacher optimizer must be one of: {', '.join(sorted(ALL_OPTIMIZERS))}")
        if self.schedule not in {"cosine", "wsd"}:
            raise ValueError("schedule must be one of: cosine, wsd")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        for name in (
            "lr_muon",
            "lr_adamw",
            "weight_decay_muon",
            "weight_decay_adamw",
            "adamw_eps",
            "muon_momentum",
            "wsd_stable_fraction",
            "pace_pullback_c",
            "pace_kappa",
            "pace_beta2",
            "pace_eps",
            "normuon_beta2",
            "normuon_eps",
        ):
            _finite_nonnegative(name, float(getattr(self, name)))
        if self.pace_precond not in {"adam", "scalar", "row"}:
            raise ValueError("pace_precond must be one of: adam, scalar, row")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.adamw_betas[0] < 1.0 or not 0.0 <= self.adamw_betas[1] < 1.0:
            raise ValueError("adamw_betas must be in [0, 1)")
        if not 0.0 < self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must be in (0, 1)")
        if not 0.0 < self.wsd_stable_fraction <= 1.0:
            raise ValueError("wsd_stable_fraction must be in (0, 1]")
        if not 0.0 < self.pace_kappa <= 1.0:
            raise ValueError("pace_kappa must be in (0, 1]")
        if not 0.0 <= self.pace_beta2 < 1.0:
            raise ValueError("pace_beta2 must be in [0, 1)")
        if self.pace_eps <= 0.0:
            raise ValueError("pace_eps must be positive")
        _positive_int("pace_update_freq", self.pace_update_freq)
        if not 0.0 <= self.normuon_beta2 < 1.0:
            raise ValueError("normuon_beta2 must be in [0, 1)")
        if self.normuon_eps <= 0.0:
            raise ValueError("normuon_eps must be positive")
        if self.grad_clip_norm is not None:
            _finite_nonnegative("grad_clip_norm", self.grad_clip_norm)
            if self.grad_clip_norm == 0.0:
                raise ValueError("grad_clip_norm must be positive when set")


@dataclass(frozen=True)
class TeacherRunConfig:
    seed: int
    dataset_name: Literal["cifar10"]
    data: Cifar10DataConfig
    model: Mamba3CifarConfig
    train: TeacherTrainConfig

    def validate(self) -> None:
        _positive_int("seed", self.seed)
        if self.dataset_name != "cifar10":
            raise ValueError("only CIFAR-10 is allowed for teacher training")
        self.data.validate()
        self.model.validate()
        self.train.validate()


@dataclass(frozen=True)
class TeacherCandidateResult:
    accepted: bool
    parameter_count: int | None
    reason: str


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


def _reject_unknown(section: str, raw: Mapping[str, object], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{section} has unknown keys: {', '.join(unknown)}")


def _parse_tuple2(value: object, *, section: str, key: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{section}.{key} must be a two-element sequence")
    return float(value[0]), float(value[1])


def _coerce_dataclass_kwargs(
    cls: type,
    raw: Mapping[str, object],
    *,
    section: str,
    extra_allowed: set[str] | None = None,
) -> dict[str, object]:
    field_names = {field.name for field in fields(cls)}
    _reject_unknown(section, raw, field_names | (extra_allowed or set()))
    return {name: raw[name] for name in field_names if name in raw}


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


def parse_teacher_run_config(
    raw_config: Mapping[str, object],
    *,
    quick_smoke: bool,
) -> TeacherRunConfig:
    top_allowed = {"seed", "dataset", "model", "train"}
    _reject_unknown("teacher config", raw_config, top_allowed)
    seed = int(raw_config.get("seed", 1337))

    dataset_raw = _expect_mapping(raw_config.get("dataset", {}), "dataset")
    dataset_allowed = {field.name for field in fields(Cifar10DataConfig)} | {"name"}
    _reject_unknown("dataset", dataset_raw, dataset_allowed)
    dataset_name = str(dataset_raw.get("name", "cifar10"))
    if dataset_name != "cifar10":
        raise ValueError("dataset.name must be cifar10")

    train_raw = _expect_mapping(raw_config.get("train", {}), "train")
    train_kwargs = _coerce_dataclass_kwargs(TeacherTrainConfig, train_raw, section="train")
    if "adamw_betas" in train_kwargs:
        train_kwargs["adamw_betas"] = _parse_tuple2(
            train_kwargs["adamw_betas"], section="train", key="adamw_betas"
        )
    train_config = TeacherTrainConfig(**train_kwargs)
    train_config.validate()

    train_owned_dataset_keys = {
        "batch_size",
        "num_workers",
        "seed",
        "quick_smoke",
        "label_smoothing",
        "mixup",
        "cutmix",
    }
    duplicated = sorted(set(dataset_raw).intersection(train_owned_dataset_keys))
    if duplicated:
        raise ValueError(
            "dataset section may not define train-owned keys: " + ", ".join(duplicated)
        )

    data_kwargs = _coerce_dataclass_kwargs(
        Cifar10DataConfig,
        dataset_raw,
        section="dataset",
        extra_allowed={"name"},
    )
    data_kwargs.pop("name", None)
    if "data_dir" in data_kwargs:
        data_kwargs["data_dir"] = Path(str(data_kwargs["data_dir"]))
    data_kwargs["batch_size"] = train_config.batch_size_per_gpu
    data_kwargs["num_workers"] = train_config.num_workers
    data_kwargs["seed"] = seed
    data_kwargs["quick_smoke"] = quick_smoke
    data_kwargs["label_smoothing"] = train_config.label_smoothing
    data_kwargs["mixup"] = train_config.mixup
    data_kwargs["cutmix"] = train_config.cutmix
    data_config = Cifar10DataConfig(**data_kwargs)
    data_config.validate()

    model_raw = _expect_mapping(raw_config.get("model", {}), "model")
    model_kwargs = _coerce_dataclass_kwargs(Mamba3CifarConfig, model_raw, section="model")
    model_config = Mamba3CifarConfig(**model_kwargs)
    model_config.validate()

    run_config = TeacherRunConfig(
        seed=seed,
        dataset_name="cifar10",
        data=data_config,
        model=model_config,
        train=train_config,
    )
    run_config.validate()
    return run_config


def load_teacher_run_config(path: str | Path, *, quick_smoke: bool) -> TeacherRunConfig:
    return parse_teacher_run_config(load_yaml(path), quick_smoke=quick_smoke)


def build_teacher_model(
    model_config: Mamba3CifarConfig,
    *,
    device: torch.device | str | None = None,
    enforce_target_params: bool = True,
) -> tuple[Mamba3CifarTeacher, int]:
    model = Mamba3CifarTeacher(model_config)
    parameter_count = model.assert_target_parameter_count() if enforce_target_params else sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if device is not None:
        model = model.to(device)
    return model, parameter_count


def evaluate_teacher_candidate(model_config: Mamba3CifarConfig) -> TeacherCandidateResult:
    try:
        _, parameter_count = build_teacher_model(model_config, enforce_target_params=True)
    except Exception as exc:
        count: int | None = None
        try:
            model, count = build_teacher_model(model_config, enforce_target_params=False)
            del model
        except Exception:
            pass
        return TeacherCandidateResult(False, count, str(exc))
    return TeacherCandidateResult(True, parameter_count, "accepted")


def _import_official_muon_class() -> type[torch.optim.Optimizer]:
    errors: list[str] = []
    for module_name in ("muon", "Muon", "muon_optimizer"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - dependency-specific
            errors.append(f"{module_name}: {exc}")
            continue
        optimizer_cls = getattr(module, "SingleDeviceMuonWithAuxAdam", None)
        if optimizer_cls is not None:
            return optimizer_cls
        errors.append(f"{module_name}: missing SingleDeviceMuonWithAuxAdam")
    raise RuntimeError(
        "Official Muon SingleDeviceMuonWithAuxAdam is required for the baseline trainer. "
        "Do not fall back to AdamW-only. " + "; ".join(errors)
    )


def _split_and_log_optimizer_parameters(
    model: nn.Module,
    *,
    assignment_log_path: Path | None,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, int]]:
    muon_params, adamw_params, assignments = split_muon_adamw_parameters(model)
    if not muon_params:
        raise ValueError("Muon parameter group is empty; refusing AdamW-only teacher training")
    if not adamw_params:
        raise ValueError("AdamW fallback parameter group is empty; grouping is suspicious")

    if assignment_log_path is not None:
        assignment_log_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_log_path.write_text(
            "\n".join(format_param_assignments(assignments)) + "\n",
            encoding="utf-8",
        )

    return muon_params, adamw_params, {
        "muon_tensors": len(muon_params),
        "adamw_tensors": len(adamw_params),
        "muon_parameters": sum(parameter.numel() for parameter in muon_params),
        "adamw_parameters": sum(parameter.numel() for parameter in adamw_params),
    }


def build_muon_adamw_optimizer(
    model: nn.Module,
    train_config: TeacherTrainConfig,
    *,
    assignment_log_path: Path | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    muon_params, adamw_params, parameter_summary = _split_and_log_optimizer_parameters(
        model,
        assignment_log_path=assignment_log_path,
    )
    optimizer_cls = _import_official_muon_class()
    optimizer = optimizer_cls(
        [
            {
                "params": muon_params,
                "lr": train_config.lr_muon,
                "momentum": train_config.muon_momentum,
                "weight_decay": train_config.weight_decay_muon,
                "use_muon": True,
            },
            {
                "params": adamw_params,
                "lr": train_config.lr_adamw,
                "betas": train_config.adamw_betas,
                "eps": train_config.adamw_eps,
                "weight_decay": train_config.weight_decay_adamw,
                "use_muon": False,
            },
        ]
    )
    return optimizer, parameter_summary | {
        "optimizer_family": "muon_adamw",
        "optimizer_source": "official KellerJordan/Muon SingleDeviceMuonWithAuxAdam",
        "uses_ema_eval": False,
    }


def build_normuon_adamw_optimizer(
    model: nn.Module,
    train_config: TeacherTrainConfig,
    *,
    assignment_log_path: Path | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    muon_params, adamw_params, parameter_summary = _split_and_log_optimizer_parameters(
        model,
        assignment_log_path=assignment_log_path,
    )
    optimizer = MuonNorMuon(
        build_normuon_param_groups(
            muon_params,
            adamw_params,
            lr_muon=train_config.lr_muon,
            lr_adamw=train_config.lr_adamw,
            weight_decay_muon=train_config.weight_decay_muon,
            weight_decay_adamw=train_config.weight_decay_adamw,
            muon_momentum=train_config.muon_momentum,
            adamw_betas=train_config.adamw_betas,
            adamw_eps=train_config.adamw_eps,
        ),
        normuon_beta2=train_config.normuon_beta2,
        normuon_eps=train_config.normuon_eps,
    )
    return optimizer, parameter_summary | {
        "optimizer_family": "normuon_adamw",
        "optimizer_source": "vendored optimizer_experiments Muon+NorMuon ablation",
        "optimizer_experiments_commit": OPTIMIZER_EXPERIMENTS_COMMIT,
        "optimizer_experiments_source": OPTIMIZER_EXPERIMENTS_SOURCE,
        "normuon_beta2": train_config.normuon_beta2,
        "normuon_eps": train_config.normuon_eps,
        "uses_ema_eval": False,
    }


def _wrap_with_pace(
    optimizer: torch.optim.Optimizer,
    train_config: TeacherTrainConfig,
) -> PaceOptimizer:
    return PaceOptimizer(
        optimizer,
        pullback_c=train_config.pace_pullback_c,
        kappa=train_config.pace_kappa,
        precond=train_config.pace_precond,
        beta2=train_config.pace_beta2,
        eps=train_config.pace_eps,
        update_freq=train_config.pace_update_freq,
    )


def build_training_optimizer(
    model: nn.Module,
    train_config: TeacherTrainConfig,
    *,
    assignment_log_path: Path | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    if train_config.optimizer in OFFICIAL_MUON_OPTIMIZERS:
        optimizer, summary = build_muon_adamw_optimizer(
            model,
            train_config,
            assignment_log_path=assignment_log_path,
        )
        return optimizer, summary | {
            "optimizer_family": train_config.optimizer,
            "uses_ema_eval": False,
        }
    if train_config.optimizer == "pace_muon":
        base_optimizer, summary = build_muon_adamw_optimizer(
            model,
            train_config,
            assignment_log_path=assignment_log_path,
        )
        return _wrap_with_pace(base_optimizer, train_config), summary | {
            "optimizer_family": "pace_muon",
            "base_optimizer_family": "muon_adamw",
            "optimizer_source": "official Muon wrapped by vendored optimizer_experiments PACE",
            "optimizer_experiments_commit": OPTIMIZER_EXPERIMENTS_COMMIT,
            "optimizer_experiments_source": OPTIMIZER_EXPERIMENTS_SOURCE,
            "pace_pullback_c": train_config.pace_pullback_c,
            "pace_kappa": train_config.pace_kappa,
            "pace_precond": train_config.pace_precond,
            "pace_beta2": train_config.pace_beta2,
            "pace_update_freq": train_config.pace_update_freq,
            "uses_ema_eval": True,
        }
    if train_config.optimizer in NORMUON_OPTIMIZERS:
        optimizer, summary = build_normuon_adamw_optimizer(
            model,
            train_config,
            assignment_log_path=assignment_log_path,
        )
        return optimizer, summary | {
            "optimizer_family": train_config.optimizer,
            "uses_ema_eval": False,
        }
    if train_config.optimizer == "pace_normuon":
        base_optimizer, summary = build_normuon_adamw_optimizer(
            model,
            train_config,
            assignment_log_path=assignment_log_path,
        )
        return _wrap_with_pace(base_optimizer, train_config), summary | {
            "optimizer_family": "pace_normuon",
            "base_optimizer_family": "normuon_adamw",
            "optimizer_source": "vendored optimizer_experiments Muon+NorMuon wrapped by PACE",
            "optimizer_experiments_commit": OPTIMIZER_EXPERIMENTS_COMMIT,
            "optimizer_experiments_source": OPTIMIZER_EXPERIMENTS_SOURCE,
            "pace_pullback_c": train_config.pace_pullback_c,
            "pace_kappa": train_config.pace_kappa,
            "pace_precond": train_config.pace_precond,
            "pace_beta2": train_config.pace_beta2,
            "pace_update_freq": train_config.pace_update_freq,
            "uses_ema_eval": True,
        }
    raise ValueError(f"unsupported optimizer family: {train_config.optimizer}")


@contextmanager
def optimizer_eval_context(optimizer: torch.optim.Optimizer):
    use_ema_weights = getattr(optimizer, "use_ema_weights", None)
    if callable(use_ema_weights):
        with use_ema_weights():
            yield
    else:
        yield


def _optimizer_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    if not lrs:
        return {}
    metrics = {
        "lr_min": min(lrs),
        "lr_max": max(lrs),
        "lr_group_0": lrs[0],
        "lr_muon": lrs[0],
        "lr_adamw": lrs[1] if len(lrs) > 1 else lrs[0],
    }
    if len(lrs) > 1:
        metrics["lr_group_1"] = lrs[1]
    return metrics


def _scheduler_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    base_optimizer = getattr(optimizer, "base_optimizer", optimizer)
    if not isinstance(base_optimizer, torch.optim.Optimizer):
        raise TypeError("LR scheduler target must be a torch.optim.Optimizer")
    return base_optimizer


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TeacherTrainConfig,
    *,
    steps_per_epoch: int,
) -> LambdaLR:
    _positive_int("steps_per_epoch", steps_per_epoch)
    total_steps = max(1, train_config.epochs * steps_per_epoch)
    warmup_steps = min(total_steps, train_config.warmup_epochs * steps_per_epoch)
    stable_end = int(total_steps * train_config.wsd_stable_fraction)

    def lr_factor(step: int) -> float:
        current = step + 1
        if warmup_steps > 0 and current <= warmup_steps:
            return current / float(warmup_steps)
        if train_config.schedule == "wsd" and current <= stable_end:
            return 1.0
        if train_config.schedule == "wsd":
            decay_start = max(warmup_steps, stable_end)
        else:
            decay_start = warmup_steps
        decay_steps = max(1, total_steps - decay_start)
        progress = min(1.0, max(0.0, (current - decay_start) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_factor)


def _label_distribution(
    target: torch.Tensor,
    *,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    confidence = 1.0 - label_smoothing
    off_value = label_smoothing / float(num_classes - 1) if num_classes > 1 else 0.0
    distribution = torch.full(
        (target.shape[0], num_classes),
        off_value,
        device=target.device,
        dtype=torch.float32,
    )
    distribution.scatter_(1, target[:, None], confidence)
    return distribution


def _soft_cross_entropy(logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    return -(target_distribution * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_height = max(1, int(height * cut_ratio))
    cut_width = max(1, int(width * cut_ratio))
    center_y = random.randrange(height)
    center_x = random.randrange(width)
    y1 = max(0, center_y - cut_height // 2)
    y2 = min(height, center_y + cut_height // 2)
    x1 = max(0, center_x - cut_width // 2)
    x2 = min(width, center_x + cut_width // 2)
    return y1, y2, x1, x2


def _augment_batch(
    image: torch.Tensor,
    target: torch.Tensor,
    train_config: TeacherTrainConfig,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_distribution = _label_distribution(
        target,
        num_classes=num_classes,
        label_smoothing=train_config.label_smoothing,
    )
    enabled: list[str] = []
    if train_config.mixup > 0.0:
        enabled.append("mixup")
    if train_config.cutmix > 0.0:
        enabled.append("cutmix")
    if not enabled:
        return image, target_distribution

    choice = random.choice(enabled)
    permutation = torch.randperm(image.shape[0], device=image.device)
    if choice == "mixup":
        lam = random.betavariate(train_config.mixup, train_config.mixup)
        mixed = image.mul(lam).add(image[permutation], alpha=1.0 - lam)
        mixed_target = target_distribution.mul(lam).add(target_distribution[permutation], alpha=1.0 - lam)
        return mixed, mixed_target

    lam = random.betavariate(train_config.cutmix, train_config.cutmix)
    y1, y2, x1, x2 = _rand_bbox(image.shape[-2], image.shape[-1], lam)
    mixed = image.clone()
    mixed[:, :, y1:y2, x1:x2] = image[permutation, :, y1:y2, x1:x2]
    box_area = float((y2 - y1) * (x2 - x1))
    adjusted_lam = 1.0 - box_area / float(image.shape[-2] * image.shape[-1])
    mixed_target = target_distribution.mul(adjusted_lam).add(
        target_distribution[permutation],
        alpha=1.0 - adjusted_lam,
    )
    return mixed, mixed_target


def _autocast_context(device: torch.device, precision: PrecisionName):
    if device.type != "cuda":
        raise RuntimeError("teacher training requires CUDA; do not run CPU-only smoke or training")
    if precision != "bf16":
        raise ValueError("only BF16 precision is supported")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _move_batch(batch: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise ValueError("CIFAR-10 batch must be a (image, target) pair")
    image, target = batch
    if not isinstance(image, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("CIFAR-10 loader must return tensor images and tensor targets")
    return image.to(device, non_blocking=True), target.to(device, non_blocking=True)


def _train_one_step(
    model: nn.Module,
    batch: object,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    run_config: TeacherRunConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    image, target = _move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    image, soft_target = _augment_batch(
        image,
        target,
        run_config.train,
        num_classes=run_config.model.num_classes,
    )
    with _autocast_context(device, run_config.train.precision):
        logits = model(image)
        loss = _soft_cross_entropy(logits, soft_target)
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError("non-finite teacher training loss")
    loss.backward()
    if run_config.train.grad_clip_norm is not None:
        clip_grad_norm_(model.parameters(), max_norm=run_config.train.grad_clip_norm)
    optimizer.step()
    scheduler.step()
    return {
        "train_batch_size": float(target.numel()),
        "train_loss": float(loss.detach().float().item()),
        "train_accuracy_hard_labels": accuracy(logits.detach().float(), target.detach()),
        **_optimizer_lr_metrics(optimizer),
    }


@torch.no_grad()
def _evaluate_steps(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    run_config: TeacherRunConfig,
    device: torch.device,
    *,
    max_steps: int | None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    steps = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        image, target = _move_batch(batch, device)
        with _autocast_context(device, run_config.train.precision):
            logits = model(image)
        loss = F.cross_entropy(logits.float(), target)
        batch_size = int(target.numel())
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.float().argmax(dim=1) == target).sum().item())
        total += batch_size
        steps += 1
    if total == 0:
        raise RuntimeError("validation loader produced no batches")
    return {
        "val_loss": loss_sum / total,
        "val_accuracy": correct / total,
        "val_steps": float(steps),
    }


def run_teacher_training(
    run_config: TeacherRunConfig,
    *,
    output_dir: Path,
    quick_smoke: bool,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    save_checkpoint: bool = False,
    epoch_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    run_config.validate()
    seed_everything(run_config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 official Mamba-3 teacher training")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    train_loader, val_loader = build_cifar10_loaders(run_config.data)
    model, parameter_count = build_teacher_model(run_config.model, device=device)
    optimizer, optimizer_summary = build_training_optimizer(
        model,
        run_config.train,
        assignment_log_path=output_dir / "param_assignments.txt",
    )
    steps_per_epoch = len(train_loader) if max_train_steps is None else min(len(train_loader), max_train_steps)
    scheduler = build_lr_scheduler(
        _scheduler_optimizer(optimizer),
        run_config.train,
        steps_per_epoch=max(1, steps_per_epoch),
    )
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "teacher_best.pt"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to append to existing metrics file: {metrics_path}")
    if save_checkpoint and checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {checkpoint_path}")

    best_val_accuracy = -1.0
    total_train_steps = 0
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    train_step_limit = 1 if quick_smoke else max_train_steps
    val_step_limit = 1 if quick_smoke else max_val_steps
    epochs = 1 if quick_smoke else run_config.train.epochs
    train_images_seen = 0
    train_elapsed_seconds = 0.0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_metrics: dict[str, float] | None = None
        epoch_train_images_seen = 0
        epoch_train_elapsed_seconds = 0.0
        for step, batch in enumerate(train_loader):
            if train_step_limit is not None and step >= train_step_limit:
                break
            train_step_start = time.perf_counter()
            train_metrics = _train_one_step(model, batch, optimizer, scheduler, run_config, device)
            torch.cuda.synchronize(device)
            train_step_seconds = time.perf_counter() - train_step_start
            train_elapsed_seconds += train_step_seconds
            epoch_train_elapsed_seconds += train_step_seconds
            total_train_steps += 1
            step_batch_size = int(train_metrics.pop("train_batch_size"))
            train_images_seen += step_batch_size
            epoch_train_images_seen += step_batch_size
        if train_metrics is None:
            raise RuntimeError("train loader produced no batches")
        with optimizer_eval_context(optimizer):
            val_metrics = _evaluate_steps(model, val_loader, run_config, device, max_steps=val_step_limit)
            val_accuracy = val_metrics["val_accuracy"]
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
            metrics = {
                "phase": "teacher_train",
                "epoch": epoch,
                "parameter_count": parameter_count,
                "train_steps_total": total_train_steps,
                "train_images_seen": train_images_seen,
                "epoch_train_images_seen": epoch_train_images_seen,
                "epoch_train_elapsed_seconds": epoch_train_elapsed_seconds,
                "epoch_train_images_per_second": epoch_train_images_seen
                / max(epoch_train_elapsed_seconds, 1e-9),
                "epoch_seconds": time.perf_counter() - epoch_start,
                **train_metrics,
                **val_metrics,
            }
            append_jsonl(metrics_path, metrics)
            if epoch_callback is not None:
                epoch_callback(metrics)
            if save_checkpoint and is_best:
                save_teacher_checkpoint_atomic(
                    checkpoint_path,
                    {
                        "model": model.state_dict(),
                        "config": _jsonable(run_config),
                        "parameter_count": parameter_count,
                        "metrics": metrics,
                    },
                )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "parameter_count": parameter_count,
        "best_val_accuracy": best_val_accuracy,
        "train_steps_total": total_train_steps,
        "elapsed_seconds": elapsed_seconds,
        "metrics_path": str(metrics_path),
        "batch_size_per_gpu": run_config.train.batch_size_per_gpu,
        "train_images_seen": train_images_seen,
        "train_images_per_second": train_images_seen / max(elapsed_seconds, 1e-9),
        "train_elapsed_seconds": train_elapsed_seconds,
        "train_images_per_second_train_only": train_images_seen / max(train_elapsed_seconds, 1e-9),
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "optimizer": run_config.train.optimizer,
        "optimizer_summary": optimizer_summary,
        "quick_smoke": quick_smoke,
    }
    write_json(output_dir / "metrics_summary.json", _jsonable(summary))
    return summary


def save_teacher_checkpoint_atomic(checkpoint_path: Path, payload: Mapping[str, object]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def write_run_context(
    *,
    output_dir: Path,
    context: RunContext,
    config_path: Path,
    raw_config: Mapping[str, object],
    run_config: TeacherRunConfig,
    smoke_mode: SmokeMode,
) -> None:
    write_json(
        output_dir / "run_context.json",
        _jsonable(
            context.metadata()
            | {
                "argv": sys.argv,
                "config_path": str(config_path),
                "smoke_mode": smoke_mode,
                "config": raw_config,
                "resolved_config": run_config,
            }
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher_default.yaml")
    parser.add_argument("--output-dir", default="outputs/teacher")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument(
        "--smoke-mode",
        choices=("metadata", "train"),
        default="train",
        help="metadata keeps entrypoint tests cheap; train runs the one-batch CUDA BF16 smoke.",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--save-checkpoint", type=bool_arg, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the run and augmentation seed while preserving the configured train/val split seed.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    raw_config = load_yaml(config_path)
    run_config = load_teacher_run_config(config_path, quick_smoke=args.quick_smoke)
    if args.seed is not None:
        run_config = replace(
            run_config,
            seed=args.seed,
            data=replace(run_config.data, seed=args.seed),
        )
    context = RunContext(Path(args.output_dir), seed=run_config.seed, quick_smoke=args.quick_smoke)
    context.prepare()
    write_run_context(
        output_dir=context.output_dir,
        context=context,
        config_path=config_path,
        raw_config=raw_config,
        run_config=run_config,
        smoke_mode=args.smoke_mode,
    )
    if args.quick_smoke and args.smoke_mode == "metadata":
        print("teacher quick smoke metadata written")
        return 0

    summary = run_teacher_training(
        run_config,
        output_dir=context.output_dir,
        quick_smoke=args.quick_smoke,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        save_checkpoint=(not args.quick_smoke) if args.save_checkpoint is None else args.save_checkpoint,
    )
    print(f"teacher training complete: {_jsonable(summary)}")
    return 0


def model_config_with_overrides(
    base: Mamba3CifarConfig,
    overrides: Mapping[str, object],
) -> Mamba3CifarConfig:
    allowed = {field.name for field in fields(Mamba3CifarConfig)}
    _reject_unknown("model overrides", overrides, allowed)
    return replace(base, **dict(overrides))


if __name__ == "__main__":
    raise SystemExit(main())
