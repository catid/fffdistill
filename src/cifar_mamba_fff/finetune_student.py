from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .data import Cifar10DataConfig, build_cifar10_loaders
from .distill_linears import load_teacher_for_distillation
from .losses.balance import min_leaf_occupancy_loss, split_balance_loss, uniform_leaf_balance_loss
from .metrics import accuracy, normalized_mse
from .models.fff_linear import FFFLinear
from .models.replacement import (
    discover_linear_layers,
    get_module,
    make_fff_replacement,
    replace_module,
)
from .train_teacher import (
    ALL_OPTIMIZERS,
    PacePrecondName,
    PrecisionName,
    ScheduleName,
    TeacherTrainConfig,
    _autocast_context,
    _move_batch,
    _scheduler_optimizer,
    build_lr_scheduler,
    build_training_optimizer,
    optimizer_eval_context,
    resolve_scheduler_total_steps,
    save_teacher_checkpoint_atomic,
)
from .utils import RunContext, append_jsonl, bool_arg, load_yaml, seed_everything, write_json


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


def _expect_sequence(value: object, section: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{section} must be a sequence")
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


@dataclass(frozen=True)
class StudentAssemblyConfig:
    source: str = "distill_artifacts"
    distill_artifact_root: Path | None = None
    distill_config: Path = Path("configs/fff_distill_stage_f.yaml")
    student_checkpoint: Path | None = None
    min_in_features: int = 64
    min_out_features: int = 64
    require_full_replacement: bool = True
    allow_dense_copy: bool = False

    def validate(self, *, quick_smoke: bool) -> None:
        if self.source not in {"distill_artifacts", "student_checkpoint", "dense_copy"}:
            raise ValueError("student.source must be one of: distill_artifacts, student_checkpoint, dense_copy")
        _positive_int("student.min_in_features", self.min_in_features)
        _positive_int("student.min_out_features", self.min_out_features)
        if self.source == "distill_artifacts" and not quick_smoke and self.distill_artifact_root is None:
            raise ValueError("student.distill_artifact_root is required for distill_artifacts")
        if self.source == "student_checkpoint" and not quick_smoke and self.student_checkpoint is None:
            raise ValueError("student.student_checkpoint is required for student_checkpoint")
        if self.source == "dense_copy" and not self.allow_dense_copy:
            raise ValueError("student.allow_dense_copy must be true to run dense-copy sanity fine-tuning")


@dataclass(frozen=True)
class FineTuneTrainConfig:
    epochs: int = 50
    batch_size_per_gpu: int = 32
    num_workers: int = 8
    precision: PrecisionName = "bf16"
    optimizer: str = "muon_adamw"
    schedule: ScheduleName = "cosine"
    warmup_epochs: int = 2
    lr_muon: float = 1e-3
    lr_adamw: float = 1e-4
    weight_decay_muon: float = 0.03
    weight_decay_adamw: float = 0.03
    label_smoothing: float = 0.0
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
        _positive_int("train.epochs", self.epochs)
        _positive_int("train.batch_size_per_gpu", self.batch_size_per_gpu)
        _nonnegative_int("train.num_workers", self.num_workers)
        if self.precision != "bf16":
            raise ValueError("fine-tune precision must be bf16")
        if self.optimizer not in ALL_OPTIMIZERS:
            raise ValueError(f"train.optimizer must be one of: {', '.join(sorted(ALL_OPTIMIZERS))}")
        if self.schedule not in {"cosine", "wsd"}:
            raise ValueError("train.schedule must be one of: cosine, wsd")
        _nonnegative_int("train.warmup_epochs", self.warmup_epochs)
        for name in (
            "lr_muon",
            "lr_adamw",
            "weight_decay_muon",
            "weight_decay_adamw",
            "label_smoothing",
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
            _finite_nonnegative(f"train.{name}", float(getattr(self, name)))
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("train.label_smoothing must be in [0, 1)")
        if not 0.0 <= self.adamw_betas[0] < 1.0 or not 0.0 <= self.adamw_betas[1] < 1.0:
            raise ValueError("train.adamw_betas must be in [0, 1)")
        if not 0.0 < self.muon_momentum < 1.0:
            raise ValueError("train.muon_momentum must be in (0, 1)")
        if not 0.0 < self.wsd_stable_fraction <= 1.0:
            raise ValueError("train.wsd_stable_fraction must be in (0, 1]")
        if self.pace_precond not in {"adam", "scalar", "row"}:
            raise ValueError("train.pace_precond must be one of: adam, scalar, row")
        if not 0.0 < self.pace_kappa <= 1.0:
            raise ValueError("train.pace_kappa must be in (0, 1]")
        if not 0.0 <= self.pace_beta2 < 1.0:
            raise ValueError("train.pace_beta2 must be in [0, 1)")
        if self.pace_eps <= 0.0:
            raise ValueError("train.pace_eps must be positive")
        _positive_int("train.pace_update_freq", self.pace_update_freq)
        if not 0.0 <= self.normuon_beta2 < 1.0:
            raise ValueError("train.normuon_beta2 must be in [0, 1)")
        if self.normuon_eps <= 0.0:
            raise ValueError("train.normuon_eps must be positive")
        if self.grad_clip_norm is not None:
            _finite_nonnegative("train.grad_clip_norm", self.grad_clip_norm)
            if self.grad_clip_norm == 0.0:
                raise ValueError("train.grad_clip_norm must be positive when set")

    def to_teacher_train_config(self) -> TeacherTrainConfig:
        return TeacherTrainConfig(
            epochs=self.epochs,
            batch_size_per_gpu=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            precision=self.precision,
            optimizer=self.optimizer,  # type: ignore[arg-type]
            schedule=self.schedule,
            warmup_epochs=self.warmup_epochs,
            lr_muon=self.lr_muon,
            lr_adamw=self.lr_adamw,
            weight_decay_muon=self.weight_decay_muon,
            weight_decay_adamw=self.weight_decay_adamw,
            label_smoothing=self.label_smoothing,
            mixup=0.0,
            cutmix=0.0,
            adamw_betas=self.adamw_betas,
            adamw_eps=self.adamw_eps,
            muon_momentum=self.muon_momentum,
            wsd_stable_fraction=self.wsd_stable_fraction,
            pace_pullback_c=self.pace_pullback_c,
            pace_kappa=self.pace_kappa,
            pace_precond=self.pace_precond,
            pace_beta2=self.pace_beta2,
            pace_eps=self.pace_eps,
            pace_update_freq=self.pace_update_freq,
            normuon_beta2=self.normuon_beta2,
            normuon_eps=self.normuon_eps,
            grad_clip_norm=self.grad_clip_norm,
        )


@dataclass(frozen=True)
class FineTuneLossConfig:
    kd_temperature: float = 4.0
    lambda_kd: float = 0.7
    lambda_ce: float = 0.3
    lambda_hidden: float = 0.05
    lambda_balance: float = 0.001
    balance_recipe: str = "split_minleaf"
    min_leaf_tokens: int = 64

    def validate(self) -> None:
        if self.kd_temperature <= 0.0 or not math.isfinite(self.kd_temperature):
            raise ValueError("losses.kd_temperature must be positive and finite")
        for name in ("lambda_kd", "lambda_ce", "lambda_hidden", "lambda_balance"):
            _finite_nonnegative(f"losses.{name}", float(getattr(self, name)))
        if self.lambda_kd == 0.0 and self.lambda_ce == 0.0 and self.lambda_hidden == 0.0:
            raise ValueError("at least one supervised fine-tune loss coefficient must be non-zero")
        if self.balance_recipe not in {"none", "split", "split_minleaf", "split_minleaf_uniform"}:
            raise ValueError(
                "losses.balance_recipe must be one of: none, split, split_minleaf, split_minleaf_uniform"
            )
        _nonnegative_int("losses.min_leaf_tokens", self.min_leaf_tokens)


@dataclass(frozen=True)
class FineTuneRunConfig:
    seed: int
    teacher_checkpoint: Path | None
    data: Cifar10DataConfig
    student: StudentAssemblyConfig
    train: FineTuneTrainConfig
    losses: FineTuneLossConfig

    def validate(self, *, quick_smoke: bool) -> None:
        _positive_int("seed", self.seed)
        if self.teacher_checkpoint is None and not quick_smoke:
            raise ValueError("teacher_checkpoint is required for non-smoke fine-tuning")
        self.data.validate()
        if self.data.use_test:
            raise ValueError("fine-tuning must not access CIFAR-10 test data")
        self.student.validate(quick_smoke=quick_smoke)
        self.train.validate()
        self.losses.validate()


@dataclass(frozen=True)
class AssemblyRecord:
    name: str
    replacement_path: str
    in_features: int
    out_features: int
    parameters: int
    final_normalized_mse: float | None
    final_cosine_similarity: float | None

    def log_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StudentAssemblyResult:
    model: nn.Module
    source: str
    replacement_count: int
    eligible_count: int
    manifest: list[AssemblyRecord]
    checkpoint_path: Path | None = None

    def log_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "replacement_count": self.replacement_count,
            "eligible_count": self.eligible_count,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            "layers": [record.log_record() for record in self.manifest],
        }


def _parse_path(value: object, *, section: str, allow_none: bool = True) -> Path | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{section} is required")
    if not isinstance(value, str | os.PathLike):
        raise ValueError(f"{section} must be a path string")
    return Path(value)


def _parse_data_config(raw_config: Mapping[str, object], train: FineTuneTrainConfig, *, quick_smoke: bool, seed: int) -> Cifar10DataConfig:
    dataset_raw = _expect_mapping(raw_config.get("dataset", {}), "dataset")
    dataset_allowed = {field.name for field in fields(Cifar10DataConfig)} | {"name"}
    _reject_unknown("dataset", dataset_raw, dataset_allowed)
    if str(dataset_raw.get("name", "cifar10")) != "cifar10":
        raise ValueError("dataset.name must be cifar10")
    train_owned_dataset_keys = {
        "batch_size",
        "num_workers",
        "seed",
        "quick_smoke",
        "label_smoothing",
        "mixup",
        "cutmix",
        "use_test",
    }
    duplicated = sorted(set(dataset_raw).intersection(train_owned_dataset_keys))
    if duplicated:
        raise ValueError("dataset section may not define train-owned keys: " + ", ".join(duplicated))
    data_kwargs = _coerce_dataclass_kwargs(
        Cifar10DataConfig,
        dataset_raw,
        section="dataset",
        extra_allowed={"name"},
    )
    data_kwargs.pop("name", None)
    if "data_dir" in data_kwargs:
        data_kwargs["data_dir"] = Path(str(data_kwargs["data_dir"]))
    data_kwargs.update(
        {
            "batch_size": train.batch_size_per_gpu,
            "num_workers": train.num_workers,
            "seed": seed,
            "quick_smoke": quick_smoke,
            "label_smoothing": train.label_smoothing,
            "mixup": 0.0,
            "cutmix": 0.0,
            "use_test": False,
        }
    )
    data_config = Cifar10DataConfig(**data_kwargs)
    data_config.validate()
    return data_config


def parse_finetune_run_config(raw_config: Mapping[str, object], *, quick_smoke: bool) -> FineTuneRunConfig:
    top_allowed = {
        "seed",
        "dataset",
        "teacher_checkpoint",
        "student_checkpoint",
        "distill_artifact_root",
        "distill_config",
        "train",
        "losses",
        "student",
    }
    _reject_unknown("fine-tune config", raw_config, top_allowed)
    seed = int(raw_config.get("seed", 1337))

    train_raw = dict(_expect_mapping(raw_config.get("train", {}), "train"))
    legacy_loss_keys = {
        "kd_temperature",
        "lambda_kd",
        "lambda_ce",
        "lambda_hidden",
        "lambda_balance",
        "keep_locoprop_refits",
    }
    losses_raw = dict(_expect_mapping(raw_config.get("losses", {}), "losses"))
    for key in sorted(set(train_raw).intersection(legacy_loss_keys)):
        if key != "keep_locoprop_refits":
            losses_raw.setdefault(key, train_raw.pop(key))
        else:
            train_raw.pop(key)
    train_kwargs = _coerce_dataclass_kwargs(FineTuneTrainConfig, train_raw, section="train")
    if "adamw_betas" in train_kwargs:
        train_kwargs["adamw_betas"] = _parse_tuple2(
            train_kwargs["adamw_betas"], section="train", key="adamw_betas"
        )
    train = FineTuneTrainConfig(**train_kwargs)
    train.validate()

    student_raw = dict(_expect_mapping(raw_config.get("student", {}), "student"))
    if raw_config.get("student_checkpoint") is not None:
        student_raw.setdefault("student_checkpoint", raw_config["student_checkpoint"])
        student_raw.setdefault("source", "student_checkpoint")
    if raw_config.get("distill_artifact_root") is not None:
        student_raw.setdefault("distill_artifact_root", raw_config["distill_artifact_root"])
        student_raw.setdefault("source", "distill_artifacts")
    if raw_config.get("distill_config") is not None:
        student_raw.setdefault("distill_config", raw_config["distill_config"])
    student_kwargs = _coerce_dataclass_kwargs(StudentAssemblyConfig, student_raw, section="student")
    for key in ("distill_artifact_root", "distill_config", "student_checkpoint"):
        if key in student_kwargs and student_kwargs[key] is not None:
            student_kwargs[key] = Path(str(student_kwargs[key]))
    student = StudentAssemblyConfig(**student_kwargs)

    losses_kwargs = _coerce_dataclass_kwargs(FineTuneLossConfig, losses_raw, section="losses")
    losses = FineTuneLossConfig(**losses_kwargs)
    data = _parse_data_config(raw_config, train, quick_smoke=quick_smoke, seed=seed)
    config = FineTuneRunConfig(
        seed=seed,
        teacher_checkpoint=_parse_path(raw_config.get("teacher_checkpoint"), section="teacher_checkpoint"),
        data=data,
        student=student,
        train=train,
        losses=losses,
    )
    config.validate(quick_smoke=quick_smoke)
    return config


def load_finetune_run_config(path: str | Path, *, quick_smoke: bool) -> FineTuneRunConfig:
    return parse_finetune_run_config(load_yaml(path), quick_smoke=quick_smoke)


def _load_layer_summaries(artifact_root: Path) -> dict[str, dict[str, object]]:
    if not artifact_root.exists():
        raise FileNotFoundError(f"distillation artifact root not found: {artifact_root}")
    records: dict[str, dict[str, object]] = {}
    summary_paths = sorted(artifact_root.glob("**/layer_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"no layer_summary.json files found under {artifact_root}")
    for summary_path in summary_paths:
        loaded = torch.load(summary_path, map_location="cpu", weights_only=False) if summary_path.suffix == ".pt" else None
        if loaded is not None:
            raw_rows = loaded
        else:
            import json

            raw_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"{summary_path} must contain a JSON list")
        for raw in raw_rows:
            row = dict(_expect_mapping(raw, f"{summary_path} row"))
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{summary_path} row is missing non-empty name")
            if name in records:
                raise ValueError(f"duplicate layer summary for {name!r}")
            row["_summary_path"] = str(summary_path)
            records[name] = row
    return records


def _resolve_replacement_path(row: Mapping[str, object], *, artifact_root: Path) -> Path:
    name = row.get("name")
    raw_path = row.get("replacement_path")
    candidates: list[Path] = []
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
    summary_path_raw = row.get("_summary_path")
    if isinstance(summary_path_raw, str):
        summary_path = Path(summary_path_raw)
        if isinstance(name, str):
            candidates.append(summary_path.parent / "layers" / name.replace(".", "__") / "fff_state.pt")
    if isinstance(name, str):
        candidates.extend(artifact_root.glob(f"**/layers/{name.replace('.', '__')}/fff_state.pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"replacement state for {name!r} not found; tried {[str(c) for c in candidates]}")


def assemble_fff_student_from_artifacts(
    model: nn.Module,
    *,
    artifact_root: Path,
    distill_config_path: Path,
    min_in_features: int,
    min_out_features: int,
    require_full_replacement: bool,
    device: torch.device,
) -> StudentAssemblyResult:
    distill_config = load_yaml(distill_config_path)
    raw_fff = _expect_mapping(distill_config.get("fff", {}), "distill_config.fff")
    fff_config = dict(raw_fff)
    reports = discover_linear_layers(
        model,
        min_in_features=min_in_features,
        min_out_features=min_out_features,
    )
    eligible = [report for report in reports if report.included]
    summaries = _load_layer_summaries(artifact_root)
    eligible_names = {report.name for report in eligible}
    summary_names = set(summaries)
    missing = sorted(eligible_names - summary_names)
    extra = sorted(summary_names - eligible_names)
    if require_full_replacement and (missing or extra):
        raise RuntimeError(
            "distillation artifacts do not match eligible Linear set: "
            f"missing={missing[:8]} extra={extra[:8]} missing_count={len(missing)} extra_count={len(extra)}"
        )

    manifest: list[AssemblyRecord] = []
    for report in eligible:
        row = summaries.get(report.name)
        if row is None:
            continue
        original = get_module(model, report.name)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{report.name!r} is {type(original).__name__}, not nn.Linear")
        replacement = make_fff_replacement(original, config=fff_config)
        replacement_path = _resolve_replacement_path(row, artifact_root=artifact_root)
        state = torch.load(replacement_path, map_location=device, weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError(f"{replacement_path} did not contain a state_dict mapping")
        replacement.load_state_dict(state, strict=True)
        replacement.train(original.training)
        replace_module(model, report.name, replacement)
        manifest.append(
            AssemblyRecord(
                name=report.name,
                replacement_path=str(replacement_path),
                in_features=report.in_features,
                out_features=report.out_features,
                parameters=sum(parameter.numel() for parameter in replacement.parameters()),
                final_normalized_mse=_optional_float(row.get("final_normalized_mse")),
                final_cosine_similarity=_optional_float(row.get("final_cosine_similarity")),
            )
        )

    if require_full_replacement and len(manifest) != len(eligible):
        raise RuntimeError(f"assembled {len(manifest)} replacements for {len(eligible)} eligible layers")
    return StudentAssemblyResult(
        model=model,
        source="distill_artifacts",
        replacement_count=len(manifest),
        eligible_count=len(eligible),
        manifest=manifest,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    return converted


def load_student_checkpoint(model: nn.Module, checkpoint_path: Path) -> StudentAssemblyResult:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("student checkpoint must be a mapping")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("student checkpoint is missing model state_dict")
    model.load_state_dict(state, strict=True)
    manifest_raw = checkpoint.get("assembly_manifest", [])
    manifest: list[AssemblyRecord] = []
    if isinstance(manifest_raw, Sequence) and not isinstance(manifest_raw, (str, bytes)):
        for raw in manifest_raw:
            row = _expect_mapping(raw, "assembly_manifest row")
            manifest.append(
                AssemblyRecord(
                    name=str(row["name"]),
                    replacement_path=str(row.get("replacement_path", "")),
                    in_features=int(row.get("in_features", 0)),
                    out_features=int(row.get("out_features", 0)),
                    parameters=int(row.get("parameters", 0)),
                    final_normalized_mse=_optional_float(row.get("final_normalized_mse")),
                    final_cosine_similarity=_optional_float(row.get("final_cosine_similarity")),
                )
            )
    return StudentAssemblyResult(
        model=model,
        source="student_checkpoint",
        replacement_count=sum(1 for module in model.modules() if isinstance(module, FFFLinear)),
        eligible_count=sum(1 for report in discover_linear_layers(model) if report.included),
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )


def build_student_model(
    *,
    loaded_teacher_model: nn.Module,
    config: FineTuneRunConfig,
    device: torch.device,
) -> StudentAssemblyResult:
    student = type(loaded_teacher_model)(loaded_teacher_model.config).to(device)  # type: ignore[attr-defined,call-arg]
    student.load_state_dict(loaded_teacher_model.state_dict(), strict=True)
    if config.student.source == "dense_copy":
        return StudentAssemblyResult(
            model=student,
            source="dense_copy",
            replacement_count=0,
            eligible_count=sum(
                report.included
                for report in discover_linear_layers(
                    student,
                    min_in_features=config.student.min_in_features,
                    min_out_features=config.student.min_out_features,
                )
            ),
            manifest=[],
        )
    if config.student.source == "student_checkpoint":
        if config.student.student_checkpoint is None:
            raise RuntimeError("student checkpoint source selected without a checkpoint path")
        return load_student_checkpoint(student, config.student.student_checkpoint)
    if config.student.distill_artifact_root is None:
        raise RuntimeError("distill artifact source selected without artifact root")
    return assemble_fff_student_from_artifacts(
        student,
        artifact_root=config.student.distill_artifact_root,
        distill_config_path=config.student.distill_config,
        min_in_features=config.student.min_in_features,
        min_out_features=config.student.min_out_features,
        require_full_replacement=config.student.require_full_replacement,
        device=device,
    )


class FFFBalanceAccumulator:
    def __init__(self, model: nn.Module, losses: FineTuneLossConfig) -> None:
        self.model = model
        self.losses = losses
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.values: list[torch.Tensor] = []

    def install(self) -> None:
        if self.handles:
            return
        for module in self.model.modules():
            if isinstance(module, FFFLinear):
                self.handles.append(module.register_forward_pre_hook(self._hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.values.clear()

    def _hook(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        if self.losses.lambda_balance == 0.0 or self.losses.balance_recipe == "none":
            return
        if not isinstance(module, FFFLinear) or not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        x = inputs[0]
        route_info = module.route(x, hard=False)
        route_probs = F.softmax(route_info.route_logits.float(), dim=-1)
        components = [split_balance_loss(route_probs, pair_probs=True)]
        if self.losses.balance_recipe in {"split_minleaf", "split_minleaf_uniform"}:
            token_count = max(1, route_info.leaf_probs.reshape(-1, module.leaves).shape[0])
            min_leaf_occupancy = float(self.losses.min_leaf_tokens) / float(token_count)
            components.append(
                min_leaf_occupancy_loss(
                    route_info.leaf_probs.float(),
                    min_occupancy=min_leaf_occupancy,
                )
            )
        if self.losses.balance_recipe == "split_minleaf_uniform":
            components.append(uniform_leaf_balance_loss(route_info.leaf_probs.float()))
        self.values.append(sum(components))

    def loss(self, device: torch.device) -> torch.Tensor:
        if not self.values:
            return torch.zeros((), device=device)
        return torch.stack([value.float() for value in self.values]).mean()

    def diagnostics(self) -> dict[str, object]:
        return {
            "balance_modules": len(self.handles),
            "balance_terms": len(self.values),
            "balance_recipe": self.losses.balance_recipe,
        }


def _forward_logits_features(model: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    forward_features = getattr(model, "forward_features", None)
    head = getattr(model, "head", None)
    if callable(forward_features) and callable(head):
        features = forward_features(image)
        return head(features), features
    return model(image), None


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def _fine_tune_step(
    *,
    teacher: nn.Module,
    student: nn.Module,
    batch: object,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: FineTuneRunConfig,
    balance: FFFBalanceAccumulator,
    device: torch.device,
) -> dict[str, float]:
    teacher.eval()
    student.train()
    image, target = _move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    balance.reset()
    with torch.no_grad(), _autocast_context(device, config.train.precision):
        teacher_logits, teacher_features = _forward_logits_features(teacher, image)
    with _autocast_context(device, config.train.precision):
        student_logits, student_features = _forward_logits_features(student, image)
        ce = F.cross_entropy(
            student_logits.float(),
            target,
            label_smoothing=config.train.label_smoothing,
        )
        kd = kd_loss(student_logits, teacher_logits, temperature=config.losses.kd_temperature)
        if (
            config.losses.lambda_hidden
            and student_features is not None
            and teacher_features is not None
        ):
            hidden = normalized_mse(student_features, teacher_features)
        else:
            hidden = student_logits.float().new_zeros(())
        balance_raw = balance.loss(device)
        loss = (
            config.losses.lambda_ce * ce
            + config.losses.lambda_kd * kd
            + config.losses.lambda_hidden * hidden
            + config.losses.lambda_balance * balance_raw
        )
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError("non-finite fine-tune loss")
    loss.backward()
    if config.train.grad_clip_norm is not None:
        clip_grad_norm_(student.parameters(), max_norm=config.train.grad_clip_norm)
    optimizer.step()
    scheduler.step()
    return {
        "train_batch_size": float(target.numel()),
        "train_loss": float(loss.detach().float().item()),
        "train_ce": float(ce.detach().float().item()),
        "train_kd": float(kd.detach().float().item()),
        "train_hidden": float(hidden.detach().float().item()),
        "train_balance": float(balance_raw.detach().float().item()),
        "train_accuracy_hard_labels": accuracy(student_logits.detach().float(), target.detach()),
    }


@torch.no_grad()
def _evaluate_student_steps(
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: FineTuneRunConfig,
    device: torch.device,
    *,
    max_steps: int | None,
) -> dict[str, float]:
    student.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    steps = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        image, target = _move_batch(batch, device)
        with _autocast_context(device, config.train.precision):
            logits, _features = _forward_logits_features(student, image)
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


def _write_manifest_csv(path: Path, rows: Sequence[AssemblyRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AssemblyRecord("", "", 0, 0, 0, None, None).log_record())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.log_record())


def run_finetune_training(
    run_config: FineTuneRunConfig,
    *,
    output_dir: Path,
    quick_smoke: bool,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    save_checkpoint: bool = False,
    epoch_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    run_config.validate(quick_smoke=quick_smoke)
    if quick_smoke:
        raise RuntimeError("run_finetune_training is not used for metadata-only quick smoke")
    seed_everything(run_config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 fine-tuning; do not run CPU-only training")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    if run_config.teacher_checkpoint is None:
        raise RuntimeError("teacher_checkpoint is required for fine-tuning")

    teacher_loaded = load_teacher_for_distillation(
        checkpoint_path=run_config.teacher_checkpoint,
        quick_smoke=quick_smoke,
        device=device,
        batch_size=run_config.train.batch_size_per_gpu,
        num_workers=run_config.train.num_workers,
    )
    teacher = teacher_loaded.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_loader, val_loader = build_cifar10_loaders(run_config.data)
    student_result = build_student_model(
        loaded_teacher_model=teacher,
        config=run_config,
        device=device,
    )
    student = student_result.model
    student.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "student_assembly_manifest.json", student_result.log_record())
    _write_manifest_csv(output_dir / "student_assembly_manifest.csv", student_result.manifest)

    teacher_train_config = run_config.train.to_teacher_train_config()
    optimizer, optimizer_summary = build_training_optimizer(
        student,
        teacher_train_config,
        assignment_log_path=output_dir / "param_assignments.txt",
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler_total_steps = resolve_scheduler_total_steps(
        teacher_train_config,
        steps_per_epoch=steps_per_epoch,
        max_train_steps=max_train_steps,
        quick_smoke=quick_smoke,
    )
    scheduler = build_lr_scheduler(
        _scheduler_optimizer(optimizer),
        teacher_train_config,
        steps_per_epoch=steps_per_epoch,
        total_steps=scheduler_total_steps,
    )
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "student_best.pt"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to append to existing metrics file: {metrics_path}")
    if save_checkpoint and checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {checkpoint_path}")

    balance = FFFBalanceAccumulator(student, run_config.losses)
    balance.install()
    best_val_accuracy = -1.0
    total_train_steps = 0
    train_images_seen = 0
    train_elapsed_seconds = 0.0
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    train_step_limit = 1 if quick_smoke else max_train_steps
    val_step_limit = 1 if quick_smoke else max_val_steps
    epochs = 1 if quick_smoke else run_config.train.epochs
    try:
        for epoch in range(epochs):
            if _reached_train_step_limit(total_train_steps, train_step_limit):
                break
            epoch_start = time.perf_counter()
            train_metrics: dict[str, float] | None = None
            epoch_train_images_seen = 0
            epoch_train_elapsed_seconds = 0.0
            for batch in train_loader:
                if _reached_train_step_limit(total_train_steps, train_step_limit):
                    break
                train_step_start = time.perf_counter()
                train_metrics = _fine_tune_step(
                    teacher=teacher,
                    student=student,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=run_config,
                    balance=balance,
                    device=device,
                )
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
                val_metrics = _evaluate_student_steps(
                    student,
                    val_loader,
                    run_config,
                    device,
                    max_steps=val_step_limit,
                )
            val_accuracy = val_metrics["val_accuracy"]
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
            metrics = {
                "phase": "student_finetune",
                "epoch": epoch,
                "teacher_checkpoint": str(run_config.teacher_checkpoint),
                "teacher_selected_val_accuracy": teacher_loaded.selected_val_accuracy,
                "student_source": student_result.source,
                "student_replacement_count": student_result.replacement_count,
                "eligible_linear_count": student_result.eligible_count,
                "train_steps_total": total_train_steps,
                "train_images_seen": train_images_seen,
                "epoch_train_images_seen": epoch_train_images_seen,
                "epoch_train_elapsed_seconds": epoch_train_elapsed_seconds,
                "epoch_train_images_per_second": epoch_train_images_seen
                / max(epoch_train_elapsed_seconds, 1e-9),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "test_accessed": False,
                **train_metrics,
                **val_metrics,
                **balance.diagnostics(),
            }
            append_jsonl(metrics_path, metrics)
            if epoch_callback is not None:
                epoch_callback(metrics)
            if save_checkpoint and is_best:
                save_teacher_checkpoint_atomic(
                    checkpoint_path,
                    {
                        "model": student.state_dict(),
                        "config": _jsonable(run_config),
                        "metrics": metrics,
                        "assembly_manifest": [record.log_record() for record in student_result.manifest],
                        "test_accessed": False,
                    },
                )
    finally:
        balance.close()

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "mode": "student_finetune_t14",
        "status": "completed",
        "teacher_checkpoint": str(run_config.teacher_checkpoint),
        "teacher_selected_val_accuracy": teacher_loaded.selected_val_accuracy,
        "teacher_parameter_count": teacher_loaded.parameter_count,
        "student_source": student_result.source,
        "student_replacement_count": student_result.replacement_count,
        "eligible_linear_count": student_result.eligible_count,
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
        "schedule": run_config.train.schedule,
        "optimizer_summary": optimizer_summary,
        "quick_smoke": quick_smoke,
        "test_accessed": False,
        "checkpoint_path": str(checkpoint_path) if save_checkpoint and checkpoint_path.exists() else None,
    }
    write_json(output_dir / "metrics_summary.json", _jsonable(summary))
    return summary


def write_run_context(
    *,
    output_dir: Path,
    context: RunContext,
    config_path: Path,
    raw_config: Mapping[str, object],
    run_config: FineTuneRunConfig,
    smoke_mode: str,
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
                "test_accessed": False,
            }
        ),
    )


def _context_for_smoke(smoke_mode: str) -> AbstractContextManager[None]:
    if smoke_mode == "metadata":
        return nullcontext()
    return nullcontext()


def _reached_train_step_limit(total_train_steps: int, train_step_limit: int | None) -> bool:
    return train_step_limit is not None and total_train_steps >= train_step_limit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_default.yaml")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument(
        "--smoke-mode",
        choices=("metadata", "train"),
        default="metadata",
        help="metadata keeps entrypoint tests cheap; train runs one CUDA BF16 KD step.",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--save-checkpoint", type=bool_arg, default=None)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    raw_config = load_yaml(config_path)
    run_config = parse_finetune_run_config(raw_config, quick_smoke=args.quick_smoke)
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
        print("fine-tune quick smoke metadata written")
        return 0
    with _context_for_smoke(args.smoke_mode):
        summary = run_finetune_training(
            run_config,
            output_dir=context.output_dir,
            quick_smoke=args.quick_smoke,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            save_checkpoint=(not args.quick_smoke) if args.save_checkpoint is None else args.save_checkpoint,
        )
    print(f"student fine-tune complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
