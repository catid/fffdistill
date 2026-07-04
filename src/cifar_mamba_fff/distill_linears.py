from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .data import build_cifar10_loaders
from .evaluate_teacher import run_config_from_checkpoint, selected_val_accuracy
from .locoprop.ridge_refit import ridge_refit
from .losses.balance import (
    min_leaf_occupancy_loss,
    route_margin_loss,
    split_balance_loss,
    uniform_leaf_balance_loss,
)
from .losses.distill import distillation_loss
from .losses.router_ste import (
    clipped_ste,
    expert_choice_imitation,
    hard_em_utility_ste,
    no_ste_soft_router,
    router_recipe_diagnostics,
    sigmoid_surrogate_ste,
    st_gumbel,
    utility_targeted_ce,
    utility_targeted_ste,
    vanilla_ste,
)
from .models.fff_linear import FFFLinear
from .models.replacement import (
    DEFAULT_LINEAR_CAPTURE_MAX_BYTES,
    DEFAULT_LINEAR_CAPTURE_MAX_TOKENS,
    LinearCaptureSet,
    LinearReport,
    discover_linear_layers,
    get_module,
    linear_reports_as_log_records,
    make_fff_replacement,
    select_progressive_reports,
)
from .train_teacher import TeacherRunConfig, build_teacher_model
from .utils import RunContext, append_jsonl, bool_arg, load_yaml, write_json

SampleSplit = Literal["train", "train_eval", "val"]
RouterRecipe = Literal[
    "none",
    "no_ste_soft_router",
    "vanilla_ste",
    "clipped_ste",
    "sigmoid_surrogate_ste",
    "st_gumbel",
    "utility_targeted_ste",
    "hard_em_utility_ste",
    "expert_choice_imitation",
]
ROUTER_RECIPES: tuple[str, ...] = (
    "none",
    "no_ste_soft_router",
    "vanilla_ste",
    "clipped_ste",
    "sigmoid_surrogate_ste",
    "st_gumbel",
    "utility_targeted_ste",
    "hard_em_utility_ste",
    "expert_choice_imitation",
)
BalanceRecipe = Literal[
    "none",
    "split",
    "split_minleaf",
    "split_minleaf_uniform",
    "split_minleaf_margin",
]
BALANCE_RECIPES: tuple[str, ...] = (
    "none",
    "split",
    "split_minleaf",
    "split_minleaf_uniform",
    "split_minleaf_margin",
)
MAIN_LOSS_ROUTER_RECIPES: tuple[str, ...] = (
    "no_ste_soft_router",
    "vanilla_ste",
    "clipped_ste",
    "sigmoid_surrogate_ste",
    "st_gumbel",
    "utility_targeted_ste",
    "hard_em_utility_ste",
    "expert_choice_imitation",
)
DISTILL_CONFIG_TOP_LEVEL_KEYS = {
    "seed",
    "teacher_checkpoint",
    "mode",
    "eligible_linear",
    "distill",
    "fff",
    "balance",
    "router",
    "locoprop",
}


def _parse_distill_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool_arg(value)
    raise ValueError(f"distill.{key} must be a bool")


def reject_unknown_distill_config_keys(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - DISTILL_CONFIG_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"distill config has unknown top-level keys: {', '.join(unknown)}")


@dataclass(frozen=True)
class LinearDistillConfig:
    steps: int = 100
    lr: float = 1.0e-3
    batch_size: int = 256
    normalized_mse_weight: float = 1.0
    cosine_weight: float = 0.0
    variance_weight: float = 0.0
    max_layers: int | None = None
    max_capture_tokens_per_layer: int | None = DEFAULT_LINEAR_CAPTURE_MAX_TOKENS
    max_capture_bytes_per_layer: int | None = DEFAULT_LINEAR_CAPTURE_MAX_BYTES
    device: str = "cpu"
    capture_autocast_bf16: bool = True
    metric_holdout_fraction: float = 0.10
    metric_split_seed: int = 1337

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> LinearDistillConfig:
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("distill config must be a mapping")
        raw = dict(raw or {})
        max_layers = raw.get("max_layers", cls.max_layers)
        max_capture_tokens = raw.get(
            "max_capture_tokens_per_layer",
            cls.max_capture_tokens_per_layer,
        )
        max_capture_bytes = raw.get(
            "max_capture_bytes_per_layer",
            cls.max_capture_bytes_per_layer,
        )
        config = cls(
            steps=int(raw.get("steps", cls.steps)),
            lr=float(raw.get("lr", cls.lr)),
            batch_size=int(raw.get("batch_size", cls.batch_size)),
            normalized_mse_weight=float(
                raw.get("normalized_mse_weight", cls.normalized_mse_weight)
            ),
            cosine_weight=float(raw.get("cosine_weight", cls.cosine_weight)),
            variance_weight=float(raw.get("variance_weight", cls.variance_weight)),
            max_layers=None if max_layers is None else int(max_layers),
            max_capture_tokens_per_layer=(
                None if max_capture_tokens is None else int(max_capture_tokens)
            ),
            max_capture_bytes_per_layer=(
                None if max_capture_bytes is None else int(max_capture_bytes)
            ),
            device=str(raw.get("device", cls.device)),
            capture_autocast_bf16=_parse_distill_bool(
                raw.get("capture_autocast_bf16", cls.capture_autocast_bf16),
                key="capture_autocast_bf16",
            ),
            metric_holdout_fraction=float(
                raw.get("metric_holdout_fraction", cls.metric_holdout_fraction)
            ),
            metric_split_seed=int(raw.get("metric_split_seed", cls.metric_split_seed)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("distill.steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("distill.lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("distill.batch_size must be positive")
        if self.max_layers is not None and self.max_layers < 0:
            raise ValueError("distill.max_layers must be non-negative or null")
        if (
            self.max_capture_tokens_per_layer is not None
            and self.max_capture_tokens_per_layer < 0
        ):
            raise ValueError("distill.max_capture_tokens_per_layer must be non-negative or null")
        if (
            self.max_capture_bytes_per_layer is not None
            and self.max_capture_bytes_per_layer < 0
        ):
            raise ValueError("distill.max_capture_bytes_per_layer must be non-negative or null")
        if not isinstance(self.capture_autocast_bf16, bool):
            raise ValueError("distill.capture_autocast_bf16 must be a bool")
        if (
            not math.isfinite(self.metric_holdout_fraction)
            or self.metric_holdout_fraction < 0.0
            or self.metric_holdout_fraction >= 1.0
        ):
            raise ValueError("distill.metric_holdout_fraction must be finite and in [0, 1)")
        if self.metric_split_seed < 0:
            raise ValueError("distill.metric_split_seed must be non-negative")


@dataclass(frozen=True)
class RouterDistillConfig:
    recipe: RouterRecipe = "none"
    loss_coeff: float = 0.0
    temperature: float = 1.0
    utility_temperature: float = 1.0
    clip: float = 1.0
    expert_choice_capacity_factor: float = 1.25

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> RouterDistillConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("router config must be a mapping")
        recipe = str(raw.get("recipe", "none"))
        default_loss_coeff = 0.0 if recipe == "none" else 1.0e-3
        config = cls(
            recipe=recipe,  # type: ignore[arg-type]
            loss_coeff=float(raw.get("loss_coeff", default_loss_coeff)),
            temperature=float(raw.get("temperature", raw.get("tau", cls.temperature))),
            utility_temperature=float(
                raw.get("utility_temperature", cls.utility_temperature)
            ),
            clip=float(raw.get("clip", cls.clip)),
            expert_choice_capacity_factor=float(
                raw.get(
                    "expert_choice_capacity_factor",
                    raw.get("capacity_factor", cls.expert_choice_capacity_factor),
                )
            ),
        )
        config.validate()
        return config

    @property
    def enabled(self) -> bool:
        return self.recipe != "none" and self.loss_coeff > 0.0

    def validate(self) -> None:
        if self.recipe not in ROUTER_RECIPES:
            raise ValueError(f"router.recipe must be one of: {', '.join(ROUTER_RECIPES)}")
        if self.loss_coeff < 0.0 or not math.isfinite(self.loss_coeff):
            raise ValueError("router.loss_coeff must be a non-negative finite value")
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("router.temperature must be a positive finite value")
        if self.utility_temperature <= 0.0 or not math.isfinite(self.utility_temperature):
            raise ValueError("router.utility_temperature must be a positive finite value")
        if self.clip <= 0.0 or not math.isfinite(self.clip):
            raise ValueError("router.clip must be a positive finite value")
        if (
            self.expert_choice_capacity_factor <= 0.0
            or not math.isfinite(self.expert_choice_capacity_factor)
        ):
            raise ValueError(
                "router.expert_choice_capacity_factor must be a positive finite value"
            )


@dataclass(frozen=True)
class BalanceDistillConfig:
    recipe: BalanceRecipe = "none"
    coeff: float = 0.0
    min_leaf_tokens: int = 0
    margin: float = 1.0
    margin_coeff: float = 1.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> BalanceDistillConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("balance config must be a mapping")
        recipe = str(raw.get("recipe", raw.get("balance_recipe", cls.recipe)))
        config = cls(
            recipe=recipe,  # type: ignore[arg-type]
            coeff=float(raw.get("coeff", raw.get("balance_coeff", cls.coeff))),
            min_leaf_tokens=int(raw.get("min_leaf_tokens", cls.min_leaf_tokens)),
            margin=float(raw.get("margin", cls.margin)),
            margin_coeff=float(raw.get("margin_coeff", cls.margin_coeff)),
        )
        config.validate()
        return config

    @property
    def enabled(self) -> bool:
        return self.recipe != "none" and self.coeff > 0.0

    def validate(self) -> None:
        if self.recipe not in BALANCE_RECIPES:
            raise ValueError(f"balance.recipe must be one of: {', '.join(BALANCE_RECIPES)}")
        if self.coeff < 0.0 or not math.isfinite(self.coeff):
            raise ValueError("balance.coeff must be a non-negative finite value")
        if self.min_leaf_tokens < 0:
            raise ValueError("balance.min_leaf_tokens must be non-negative")
        if self.margin < 0.0 or not math.isfinite(self.margin):
            raise ValueError("balance.margin must be a non-negative finite value")
        if self.margin_coeff < 0.0 or not math.isfinite(self.margin_coeff):
            raise ValueError("balance.margin_coeff must be a non-negative finite value")


@dataclass(frozen=True)
class LocoPropDistillConfig:
    enabled: bool = False
    interval_steps: int = 500
    ridge_lambda: float = 1.0e-4
    blend_alpha: float = 0.5
    damp_optimizer_state_after_refit: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> LocoPropDistillConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("locoprop config must be a mapping")
        config = cls(
            enabled=_parse_distill_bool(raw.get("enabled", cls.enabled), key="locoprop.enabled"),
            interval_steps=int(raw.get("interval_steps", cls.interval_steps)),
            ridge_lambda=float(raw.get("ridge_lambda", cls.ridge_lambda)),
            blend_alpha=float(
                raw.get("blend_alpha", raw.get("locoprop_blend_alpha", cls.blend_alpha))
            ),
            damp_optimizer_state_after_refit=_parse_distill_bool(
                raw.get(
                    "damp_optimizer_state_after_refit",
                    cls.damp_optimizer_state_after_refit,
                ),
                key="locoprop.damp_optimizer_state_after_refit",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.interval_steps <= 0:
            raise ValueError("locoprop.interval_steps must be positive")
        if self.ridge_lambda < 0.0 or not math.isfinite(self.ridge_lambda):
            raise ValueError("locoprop.ridge_lambda must be a non-negative finite value")
        if not 0.0 <= self.blend_alpha <= 1.0 or not math.isfinite(self.blend_alpha):
            raise ValueError("locoprop.blend_alpha must be in [0, 1]")


@dataclass(frozen=True)
class LayerDistillResult:
    name: str
    initial_loss: float
    final_loss: float
    initial_normalized_mse: float
    final_normalized_mse: float
    final_cosine_similarity: float
    final_cosine_loss: float
    train_seconds: float
    tokens_per_second: float
    captured_tokens: int
    observed_tokens: int
    dropped_tokens: int
    replacement_path: str
    fit_tokens: int = 0
    metric_tokens: int = 0
    metric_split: str = "train"
    metric_holdout_fraction: float = 0.0

    def log_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedTeacher:
    model: nn.Module
    run_config: TeacherRunConfig
    checkpoint_path: Path
    selected_val_accuracy: float
    parameter_count: int


def _load_checkpoint_mapping(checkpoint_path: Path) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("teacher checkpoint must be a mapping")
    return checkpoint


def load_teacher_for_distillation(
    *,
    checkpoint_path: Path,
    quick_smoke: bool,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> LoadedTeacher:
    checkpoint = _load_checkpoint_mapping(checkpoint_path)
    selected_val = selected_val_accuracy(checkpoint)
    run_config = run_config_from_checkpoint(
        checkpoint,
        quick_smoke=quick_smoke,
        batch_size=batch_size,
        num_workers=num_workers,
        use_test=False,
    )
    if run_config.data.use_test:
        raise RuntimeError("distillation must not access CIFAR-10 test data")
    model, parameter_count = build_teacher_model(run_config.model, device=device)
    expected_count = int(checkpoint.get("parameter_count", parameter_count))
    if parameter_count != expected_count:
        raise ValueError(
            f"checkpoint parameter_count={expected_count} does not match rebuilt model "
            f"parameter_count={parameter_count}"
        )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("teacher checkpoint is missing model state_dict")
    model.load_state_dict(state_dict)
    return LoadedTeacher(
        model=model,
        run_config=run_config,
        checkpoint_path=checkpoint_path,
        selected_val_accuracy=selected_val,
        parameter_count=parameter_count,
    )


def _sample_batches_from_run_config(
    run_config: TeacherRunConfig,
    *,
    split: SampleSplit,
    max_batches: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if max_batches <= 0:
        raise ValueError("max_sample_batches must be positive")
    if run_config.data.use_test:
        raise RuntimeError("distillation sample batches must not use CIFAR-10 test data")
    if split == "train_eval":
        train_loader, val_loader = build_cifar10_loaders(
            run_config.data,
            train_eval_transform=True,
        )
    else:
        train_loader, val_loader = build_cifar10_loaders(run_config.data)
    loader = val_loader if split == "val" else train_loader
    batches: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = batch[0] if isinstance(batch, list | tuple) else batch
        if not isinstance(images, torch.Tensor):
            raise TypeError("CIFAR loader must return image tensors")
        batches.append(images.to(device=device, non_blocking=device.type == "cuda"))
    if not batches:
        raise RuntimeError(f"{split} loader produced no sample batches")
    return batches


def linear_replacement_plan(
    model: nn.Module,
    config: dict[str, Any],
    *,
    progressive_step: int | None = None,
    progressive_step_size: int = 1,
) -> dict[str, Any]:
    eligible_config = config.get("eligible_linear", {})
    if not isinstance(eligible_config, dict):
        raise ValueError("eligible_linear config must be a mapping")

    reports = _linear_reports(model, eligible_config)
    selected = _selected_reports_from_reports(
        reports,
        eligible_config=eligible_config,
        distill_config=LinearDistillConfig.from_mapping(config.get("distill")),
        progressive_step=progressive_step,
        progressive_step_size=progressive_step_size,
    )
    return {
        "linear_layers": linear_reports_as_log_records(reports),
        "selected_replacements": [report.name for report in selected],
        "progressive": {
            "step": progressive_step,
            "step_size": progressive_step_size,
            "selected_count": len(selected),
            "eligible_count": sum(report.included for report in reports),
        },
    }


def _linear_reports(model: nn.Module, eligible_config: dict[str, Any]) -> list[LinearReport]:
    return discover_linear_layers(
        model,
        min_in_features=int(eligible_config.get("min_in_features", 64)),
        min_out_features=int(eligible_config.get("min_out_features", 64)),
    )


def _as_name_list(value: object, *, key: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"eligible_linear.{key} must be a list of non-empty strings")
    return value


def _as_index_list(value: object, *, key: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value
    ):
        raise ValueError(f"eligible_linear.{key} must be a list of non-negative integers")
    return value


def _apply_explicit_layer_selection(
    eligible: list[LinearReport],
    eligible_config: dict[str, Any],
) -> list[LinearReport]:
    include_names = _as_name_list(
        eligible_config.get("include_names", eligible_config.get("names")),
        key="include_names",
    )
    include_indices = _as_index_list(
        eligible_config.get("include_indices", eligible_config.get("indices")),
        key="include_indices",
    )
    if include_names is not None and include_indices is not None:
        raise ValueError("eligible_linear include_names and include_indices are mutually exclusive")
    if include_names is not None:
        by_name = {report.name: report for report in eligible}
        missing = [name for name in include_names if name not in by_name]
        if missing:
            raise ValueError("eligible_linear.include_names contains unknown eligible layers: " + ", ".join(missing))
        return [by_name[name] for name in include_names]
    if include_indices is not None:
        out_of_range = [index for index in include_indices if index >= len(eligible)]
        if out_of_range:
            raise ValueError(
                "eligible_linear.include_indices out of range for "
                f"{len(eligible)} eligible layers: {out_of_range}"
            )
        return [eligible[index] for index in include_indices]
    return eligible


def _selected_reports_from_reports(
    reports: list[LinearReport],
    *,
    eligible_config: dict[str, Any],
    distill_config: LinearDistillConfig,
    progressive_step: int | None,
    progressive_step_size: int,
) -> list[LinearReport]:
    selected = select_progressive_reports(
        reports,
        step=progressive_step,
        step_size=progressive_step_size,
        max_replacements=None,
    )
    selected = _apply_explicit_layer_selection(selected, eligible_config)
    if distill_config.max_layers is not None:
        selected = selected[: distill_config.max_layers]
    return selected


def _selected_reports(
    model: nn.Module,
    config: dict[str, Any],
    *,
    progressive_step: int | None = None,
    progressive_step_size: int = 1,
) -> list[LinearReport]:
    eligible_config = config.get("eligible_linear", {})
    if not isinstance(eligible_config, dict):
        raise ValueError("eligible_linear config must be a mapping")
    reports = _linear_reports(model, eligible_config)
    return _selected_reports_from_reports(
        reports,
        eligible_config=eligible_config,
        distill_config=LinearDistillConfig.from_mapping(config.get("distill")),
        progressive_step=progressive_step,
        progressive_step_size=progressive_step_size,
    )


def _diagnostics_record(layer: FFFLinear, x: torch.Tensor) -> dict[str, object]:
    diagnostics = layer.diagnostics(x)
    active_rows = diagnostics.get("active_rows_per_token")
    if isinstance(active_rows, torch.Tensor):
        active_float = active_rows.float()
        diagnostics["active_rows_per_token_mean"] = float(active_float.mean().item())
        diagnostics["active_rows_per_token_min"] = int(active_rows.min().item())
        diagnostics["active_rows_per_token_max"] = int(active_rows.max().item())
        diagnostics.pop("active_rows_per_token", None)
    return diagnostics


def _float_diagnostics(values: dict[str, torch.Tensor]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in values.items():
        detached = value.detach().float().cpu()
        if detached.ndim == 0:
            converted[key] = float(detached.item())
        else:
            converted[key] = [float(item) for item in detached.reshape(-1).tolist()]
    return converted


def _full_branch_logits(layer: FFFLinear, x: torch.Tensor) -> torch.Tensor:
    route_preacts = torch.einsum("ni,mri->nmr", x, layer.route_weight) + layer.route_bias
    return layer._branch_logits(route_preacts)


def _leaf_teacher_utility(layer: FFFLinear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    flat = x.float()
    target = y.float()
    leaf_values = layer._activation(
        torch.einsum("ni,lri->nlr", flat, layer.leaf_weight[: layer.leaves].float())
        + layer.leaf_bias[: layer.leaves].float()
    )
    leaf_outputs = torch.einsum(
        "nlr,lro->nlo",
        leaf_values,
        layer.leaf_output[: layer.leaves].float(),
    )
    shared = leaf_outputs.new_zeros(flat.shape[0], layer.out_features)
    if layer.shared_rows > 0:
        if layer.shared_weight is None or layer.shared_bias is None or layer.shared_output is None:
            raise RuntimeError("shared rows are partially initialized")
        shared_values = layer._activation(
            F.linear(flat, layer.shared_weight.float(), layer.shared_bias.float())
        )
        shared = shared + shared_values @ layer.shared_output.float()
    if layer.bias is not None:
        shared = shared + layer.bias.float()
    squared_error = (leaf_outputs + shared[:, None, :] - target[:, None, :]).square().mean(dim=-1)
    return -squared_error.detach()


def _branch_utility_from_leaf_utility(layer: FFFLinear, leaf_utility: torch.Tensor) -> torch.Tensor:
    branch_utilities: list[torch.Tensor] = []
    for node_idx in range(layer.internal_nodes):
        depth_idx = math.floor(math.log2(node_idx + 1))
        level_offset = (1 << depth_idx) - 1
        position = node_idx - level_offset
        descendant_count = 1 << (layer.depth - depth_idx - 1)
        left_start = (position * 2) * descendant_count
        right_start = (position * 2 + 1) * descendant_count
        left = leaf_utility[:, left_start : left_start + descendant_count].max(dim=1).values
        right = leaf_utility[:, right_start : right_start + descendant_count].max(dim=1).values
        branch_utilities.append(torch.stack((left, right), dim=-1))
    return torch.stack(branch_utilities, dim=1)


def _leaf_probs_from_branch_routes(layer: FFFLinear, branch_routes: torch.Tensor) -> torch.Tensor:
    batch_size = branch_routes.shape[0]
    frontier = [(0, branch_routes.new_ones(batch_size))]
    leaf_probs: list[torch.Tensor | None] = [None] * layer.leaves

    for depth_idx in range(layer.depth):
        next_frontier: list[tuple[int, torch.Tensor]] = []
        for node_idx, prob_here in frontier:
            left_prob = prob_here * branch_routes[:, node_idx, 0]
            right_prob = prob_here * branch_routes[:, node_idx, 1]
            left_child = 2 * node_idx + 1
            right_child = left_child + 1
            if depth_idx == layer.depth - 1:
                leaf_probs[left_child - layer.internal_nodes] = left_prob
                leaf_probs[right_child - layer.internal_nodes] = right_prob
            else:
                next_frontier.append((left_child, left_prob))
                next_frontier.append((right_child, right_prob))
        frontier = next_frontier

    return torch.stack([prob for prob in leaf_probs if prob is not None], dim=-1)


def _branch_routes_for_recipe(
    layer: FFFLinear,
    x: torch.Tensor,
    y: torch.Tensor,
    config: RouterDistillConfig,
) -> torch.Tensor:
    branch_logits = _full_branch_logits(layer, x).float()
    if config.recipe == "no_ste_soft_router":
        return no_ste_soft_router(branch_logits, temperature=config.temperature)
    if config.recipe == "vanilla_ste":
        return vanilla_ste(branch_logits, temperature=config.temperature)
    if config.recipe == "clipped_ste":
        return clipped_ste(branch_logits, clip=config.clip)
    if config.recipe == "sigmoid_surrogate_ste":
        return sigmoid_surrogate_ste(branch_logits, temperature=config.temperature)
    if config.recipe == "st_gumbel":
        return st_gumbel(
            branch_logits,
            tau=config.temperature,
            hard=True,
            training=layer.training,
        )

    if config.recipe in {
        "utility_targeted_ste",
        "hard_em_utility_ste",
        "expert_choice_imitation",
    }:
        utility = _branch_utility_from_leaf_utility(layer, _leaf_teacher_utility(layer, x, y))
        if config.recipe == "utility_targeted_ste":
            routed, _ = utility_targeted_ste(
                branch_logits,
                utility,
                temperature=config.temperature,
                utility_temperature=config.utility_temperature,
                return_diagnostics=True,
            )
            return routed
        if config.recipe == "hard_em_utility_ste":
            routed, _, _ = hard_em_utility_ste(
                branch_logits,
                utility,
                temperature=config.temperature,
                return_targets=True,
                return_diagnostics=True,
            )
            return routed
        return no_ste_soft_router(branch_logits, temperature=config.temperature)

    raise RuntimeError(f"validated router recipe became invalid: {config.recipe}")


def _forward_with_leaf_weights(
    layer: FFFLinear,
    x: torch.Tensor,
    leaf_weights: torch.Tensor,
) -> torch.Tensor:
    flat = x.float()
    hard_route_info = layer._route_flat(flat, hard=True)
    out = flat.new_zeros(flat.shape[0], layer.out_features)

    if layer.shared_weight is not None:
        shared_values = layer._activation(
            F.linear(flat, layer.shared_weight.float(), layer.shared_bias.float())
        )
        out = out + shared_values @ layer.shared_output.float()

    leaf_values = layer._activation(
        torch.einsum("ni,lri->nlr", flat, layer.leaf_weight[: layer.leaves].float())
        + layer.leaf_bias[: layer.leaves].float()
    )
    leaf_outputs = torch.einsum(
        "nlr,lro->nlo",
        leaf_values,
        layer.leaf_output[: layer.leaves].float(),
    )
    out = out + (leaf_outputs * leaf_weights.unsqueeze(-1)).sum(dim=1)
    out = out + layer._extra_leaf_output_grouped(flat, fallback_weight=layer._fallback_weight())
    out = out + layer._route_output_grouped(hard_route_info)

    if layer.bias is not None:
        out = out + layer.bias.float()
    return out


def _replacement_prediction(
    layer: FFFLinear,
    x: torch.Tensor,
    y: torch.Tensor,
    router_config: RouterDistillConfig,
) -> torch.Tensor:
    if router_config.recipe == "none":
        return layer(x)
    if router_config.recipe not in MAIN_LOSS_ROUTER_RECIPES:
        raise RuntimeError(f"router recipe does not support main-loss routing: {router_config.recipe}")
    branch_routes = _branch_routes_for_recipe(layer, x, y, router_config)
    leaf_probs = _leaf_probs_from_branch_routes(layer, branch_routes.to(dtype=x.dtype))
    leaf_weights, _fallback_weight = layer._regular_leaf_weights(leaf_probs)
    return _forward_with_leaf_weights(layer, x, leaf_weights)


def _leaf_occupancy_stats(route_info: Any, *, leaves: int) -> dict[str, object]:
    leaf_ids = route_info.leaf_ids.reshape(-1).detach().cpu()
    counts = torch.bincount(leaf_ids, minlength=leaves).float()
    return {
        "dead_leaves": int((counts == 0).sum().item()),
        "leaf_tokens_p10": float(torch.quantile(counts, 0.10).item()),
        "leaf_tokens_p50": float(torch.quantile(counts, 0.50).item()),
        "leaf_tokens_p90": float(torch.quantile(counts, 0.90).item()),
    }


def _router_auxiliary_loss(
    layer: FFFLinear,
    x: torch.Tensor,
    y: torch.Tensor,
    config: RouterDistillConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    zero = x.new_zeros(())
    if not config.enabled:
        return zero, {"recipe": config.recipe, "loss": 0.0, "loss_coeff": config.loss_coeff}

    branch_logits = _full_branch_logits(layer, x).float()
    route_info = layer.route(x, hard=True)
    utility: torch.Tensor | None = None
    loss_name = "split_balance_loss"

    if config.recipe == "no_ste_soft_router":
        routed = no_ste_soft_router(branch_logits, temperature=config.temperature)
        raw_loss = split_balance_loss(routed, pair_probs=True)
    elif config.recipe == "vanilla_ste":
        routed = vanilla_ste(branch_logits, temperature=config.temperature)
        raw_loss = split_balance_loss(routed, pair_probs=True)
    elif config.recipe == "clipped_ste":
        routed = clipped_ste(branch_logits, clip=config.clip)
        raw_loss = split_balance_loss(routed, pair_probs=True)
    elif config.recipe == "sigmoid_surrogate_ste":
        routed = sigmoid_surrogate_ste(branch_logits, temperature=config.temperature)
        raw_loss = split_balance_loss(routed, pair_probs=True)
    elif config.recipe == "st_gumbel":
        routed = st_gumbel(
            branch_logits,
            tau=config.temperature,
            hard=True,
            training=layer.training,
        )
        raw_loss = split_balance_loss(routed, pair_probs=True)
    else:
        utility = _branch_utility_from_leaf_utility(layer, _leaf_teacher_utility(layer, x, y))
        flat_logits = branch_logits.reshape(-1, 2)
        flat_utility = utility.reshape(-1, 2)
        if config.recipe == "utility_targeted_ste":
            routed, _ = utility_targeted_ste(
                branch_logits,
                utility,
                temperature=config.temperature,
                utility_temperature=config.utility_temperature,
                return_diagnostics=True,
            )
            raw_loss = utility_targeted_ce(
                flat_logits,
                flat_utility,
                temperature=config.temperature,
            )
            loss_name = "utility_targeted_ce"
        elif config.recipe == "hard_em_utility_ste":
            routed, _, _ = hard_em_utility_ste(
                branch_logits,
                utility,
                temperature=config.temperature,
                return_targets=True,
                return_diagnostics=True,
            )
            raw_loss = utility_targeted_ce(
                flat_logits,
                flat_utility,
                temperature=config.temperature,
            )
            loss_name = "hard_em_route_ce"
        elif config.recipe == "expert_choice_imitation":
            routed = no_ste_soft_router(branch_logits, temperature=config.temperature)
            capacity = max(
                1,
                math.ceil(
                    flat_logits.shape[0]
                    / float(flat_logits.shape[1])
                    * config.expert_choice_capacity_factor
                ),
            )
            raw_loss = expert_choice_imitation(
                flat_logits,
                flat_utility,
                capacity=capacity,
                temperature=config.temperature,
            )
            loss_name = "expert_choice_bce"
        else:
            raise RuntimeError(f"validated router recipe became invalid: {config.recipe}")

    diagnostics = _float_diagnostics(
        router_recipe_diagnostics(
            routed,
            logits=branch_logits,
            utility=utility,
            dim=-1,
        )
    )
    diagnostics.update(
        {
            "recipe": config.recipe,
            "loss_name": loss_name,
            "loss": float(raw_loss.detach().float().item()),
            "loss_coeff": config.loss_coeff,
            **_leaf_occupancy_stats(route_info, leaves=layer.leaves),
        }
    )
    return raw_loss * config.loss_coeff, diagnostics


def _balance_auxiliary_loss(
    layer: FFFLinear,
    x: torch.Tensor,
    config: BalanceDistillConfig,
    *,
    total_tokens: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    zero = x.new_zeros(())
    diagnostics: dict[str, object] = {
        "recipe": config.recipe,
        "coeff": config.coeff,
        "enabled": config.enabled,
        "loss": 0.0,
        "weighted_loss": 0.0,
        "min_leaf_tokens": config.min_leaf_tokens,
        "min_leaf_occupancy": 0.0,
        "margin": config.margin,
        "margin_coeff": config.margin_coeff,
        "components": {},
    }
    if not config.enabled:
        return zero, diagnostics
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive for balance loss")

    branch_logits = _full_branch_logits(layer, x).float()
    branch_probs = branch_logits.softmax(dim=-1)
    leaf_probs = _leaf_probs_from_branch_routes(layer, branch_probs.to(dtype=x.dtype)).float()
    min_leaf_occupancy = float(config.min_leaf_tokens) / float(total_tokens)

    components: dict[str, torch.Tensor] = {
        "split": split_balance_loss(branch_probs, pair_probs=True),
    }
    if config.recipe in {
        "split_minleaf",
        "split_minleaf_uniform",
        "split_minleaf_margin",
    }:
        components["min_leaf"] = min_leaf_occupancy_loss(
            leaf_probs,
            min_occupancy=min_leaf_occupancy,
        )
    if config.recipe == "split_minleaf_uniform":
        components["uniform_leaf"] = uniform_leaf_balance_loss(leaf_probs)
    if config.recipe == "split_minleaf_margin":
        components["margin"] = route_margin_loss(
            branch_logits,
            target_margin=config.margin,
        ) * config.margin_coeff

    raw_loss = sum(components.values(), start=zero)
    weighted_loss = raw_loss * config.coeff
    component_values = {
        key: float(value.detach().float().item()) for key, value in components.items()
    }
    diagnostics.update(
        {
            "loss": float(raw_loss.detach().float().item()),
            "weighted_loss": float(weighted_loss.detach().float().item()),
            "min_leaf_occupancy": min_leaf_occupancy,
            "components": component_values,
        }
    )
    return weighted_loss, diagnostics


def _locoprop_basis_and_prior(
    layer: FFFLinear,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    flat = x.detach().to(dtype=torch.float32)
    route_info = layer._route_flat(flat, hard=True)
    basis_parts: list[torch.Tensor] = []
    prior_parts: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []

    if layer.shared_weight is not None and layer.shared_output is not None:
        shared_values = layer._activation(
            F.linear(flat, layer.shared_weight.float(), layer.shared_bias.float())
        )
        basis_parts.append(shared_values)
        prior_parts.append(layer.shared_output.detach().float())
        layout.append({"kind": "shared", "rows": layer.shared_rows})

    route_rows_per_node = layer._route_output_rows_per_node_selected()
    if route_rows_per_node > 0:
        if layer.route_row_role == "shared_routing_and_output":
            if layer.route_output is None:
                raise RuntimeError("route_output is required for LocoProp route refit")
            values = route_info.route_values[:, :, :route_rows_per_node].detach().float()
            prior = layer.route_output[:, :route_rows_per_node].detach().float()
            kind = "route_output"
        elif layer.route_row_role == "split_routing_output":
            if layer.route_result_output is None:
                raise RuntimeError("route_result_output is required for LocoProp route refit")
            values = route_info.route_result_values[:, :, :route_rows_per_node].detach().float()
            prior = layer.route_result_output[:, :route_rows_per_node].detach().float()
            kind = "route_result_output"
        else:
            raise RuntimeError("routing_only cannot have route output rows")
        route_basis = flat.new_zeros(flat.shape[0], layer.internal_nodes * route_rows_per_node)
        offsets = torch.arange(route_rows_per_node, device=flat.device)
        cols = route_info.node_ids.unsqueeze(-1) * route_rows_per_node + offsets
        route_basis.scatter_add_(1, cols.reshape(flat.shape[0], -1), values.reshape(flat.shape[0], -1))
        basis_parts.append(route_basis)
        prior_parts.append(prior.reshape(-1, layer.out_features))
        layout.append(
            {
                "kind": kind,
                "rows_per_node": route_rows_per_node,
                "rows": layer.internal_nodes * route_rows_per_node,
            }
        )

    leaf_values = layer._activation(
        torch.einsum("ni,lri->nlr", flat, layer.leaf_weight[: layer.leaves].float())
        + layer.leaf_bias[: layer.leaves].float()
    )
    leaf_basis = flat.new_zeros(flat.shape[0], layer.leaf_banks, layer.leaf_rows)
    leaf_basis[:, : layer.leaves] = leaf_values * route_info.leaf_weights.float().unsqueeze(-1)

    master_idx = layer.master_leaf_index
    if master_idx is not None:
        leaf_basis[:, master_idx] = layer._activation(
            F.linear(flat, layer.leaf_weight[master_idx].float(), layer.leaf_bias[master_idx].float())
        )

    fallback_idx = layer.fallback_leaf_index
    fallback_weight = layer._fallback_weight()
    if fallback_idx is not None and fallback_weight != 0.0:
        fallback_values = layer._activation(
            F.linear(
                flat,
                layer.leaf_weight[fallback_idx].float(),
                layer.leaf_bias[fallback_idx].float(),
            )
        )
        leaf_basis[:, fallback_idx] = fallback_values * fallback_weight

    basis_parts.append(leaf_basis.reshape(flat.shape[0], -1))
    prior_parts.append(layer.leaf_output.detach().float().reshape(-1, layer.out_features))
    layout.append({"kind": "leaf_output", "rows": layer.leaf_banks * layer.leaf_rows})

    return torch.cat(basis_parts, dim=1), torch.cat(prior_parts, dim=0), layout


def _copy_locoprop_weights(
    layer: FFFLinear,
    weights: torch.Tensor,
    layout: list[dict[str, object]],
) -> list[nn.Parameter]:
    touched: list[nn.Parameter] = []
    offset = 0
    with torch.no_grad():
        for entry in layout:
            rows = int(entry["rows"])
            block = weights[offset : offset + rows].to(device=layer.leaf_output.device)
            offset += rows
            kind = str(entry["kind"])
            if kind == "shared":
                if layer.shared_output is None:
                    raise RuntimeError("shared_output disappeared during LocoProp copy")
                layer.shared_output.copy_(block.to(dtype=layer.shared_output.dtype))
                touched.append(layer.shared_output)
            elif kind == "route_output":
                if layer.route_output is None:
                    raise RuntimeError("route_output disappeared during LocoProp copy")
                rows_per_node = int(entry["rows_per_node"])
                reshaped = block.reshape(layer.internal_nodes, rows_per_node, layer.out_features)
                layer.route_output[:, :rows_per_node].copy_(reshaped.to(dtype=layer.route_output.dtype))
                touched.append(layer.route_output)
            elif kind == "route_result_output":
                if layer.route_result_output is None:
                    raise RuntimeError("route_result_output disappeared during LocoProp copy")
                rows_per_node = int(entry["rows_per_node"])
                reshaped = block.reshape(layer.internal_nodes, rows_per_node, layer.out_features)
                layer.route_result_output[:, :rows_per_node].copy_(
                    reshaped.to(dtype=layer.route_result_output.dtype)
                )
                touched.append(layer.route_result_output)
            elif kind == "leaf_output":
                reshaped = block.reshape(layer.leaf_banks, layer.leaf_rows, layer.out_features)
                layer.leaf_output.copy_(reshaped.to(dtype=layer.leaf_output.dtype))
                touched.append(layer.leaf_output)
            else:
                raise RuntimeError(f"unknown LocoProp layout kind: {kind}")
    if offset != weights.shape[0]:
        raise RuntimeError("LocoProp layout did not consume all solved rows")
    return touched


def _damp_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> int:
    damped = 0
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
                damped += 1
    return damped


def _apply_locoprop_refit(
    layer: FFFLinear,
    x: torch.Tensor,
    y: torch.Tensor,
    config: LocoPropDistillConfig,
    *,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    if not config.enabled:
        return {"enabled": False, "status": "skipped"}
    with torch.no_grad():
        pred_before = layer(x).float()
        actual_mse_before = torch.mean((pred_before - y.float()).square()).item()
        basis, prior, layout = _locoprop_basis_and_prior(layer, x)
        target = y.float()
        if layer.bias is not None:
            target = target - layer.bias.detach().float()
        result = ridge_refit(
            basis,
            target,
            ridge_lambda=config.ridge_lambda,
            v0=prior,
            alpha=config.blend_alpha,
            out_dtype=prior.dtype,
        )
        touched = _copy_locoprop_weights(layer, result.weights, layout)
        damped_state_tensors = (
            _damp_optimizer_state(optimizer, touched)
            if config.damp_optimizer_state_after_refit
            else 0
        )
        pred_after = layer(x).float()
        actual_mse_after = torch.mean((pred_after - y.float()).square()).item()
    return {
        "enabled": True,
        "status": "succeeded",
        "basis_rows": int(basis.shape[1]),
        "tokens": int(basis.shape[0]),
        "ridge_lambda": config.ridge_lambda,
        "blend_alpha": config.blend_alpha,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "actual_mse_before": float(actual_mse_before),
        "actual_mse_after": float(actual_mse_after),
        "jitter_used": result.jitter_used,
        "used_fallback": result.used_fallback,
        "attempts": result.attempts,
        "cholesky_info": result.cholesky_info,
        "method": result.method,
        "damped_optimizer_state_tensors": damped_state_tensors,
    }


def _guard_router_training(
    layer: FFFLinear,
    router_config: RouterDistillConfig,
    balance_config: BalanceDistillConfig,
) -> None:
    diagnostics = layer.diagnostics()
    if (
        layer.config.hard_routing
        and not bool(diagnostics["route_output_contributes"])
        and router_config.recipe == "none"
        and not balance_config.enabled
    ):
        raise ValueError(
            "hard-routed FFFLinear without route output contribution needs a positive "
            "router or balance recipe; otherwise route parameters are not trained"
        )


def _batch_indices(total: int, batch_size: int, *, device: torch.device) -> torch.Tensor:
    return torch.randperm(total, device=device)[: min(batch_size, total)]


def _fit_metric_split_indices(
    total: int,
    *,
    holdout_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if total <= 0:
        raise ValueError("total tokens must be positive")
    if holdout_fraction <= 0.0 or total < 2:
        indices = torch.arange(total, device=device)
        return indices, indices, "train"

    holdout_tokens = max(1, round(float(total) * holdout_fraction))
    holdout_tokens = min(holdout_tokens, total - 1)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator).to(device=device)
    metric_indices = perm[:holdout_tokens]
    fit_indices = perm[holdout_tokens:]
    return fit_indices, metric_indices, "holdout"


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cosine_alignment_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_f = pred.detach().float().flatten(0, -2)
    target_f = target.detach().float().flatten(0, -2)
    similarity = F.cosine_similarity(pred_f, target_f, dim=-1, eps=1.0e-8).mean()
    cosine_loss = 1.0 - similarity
    return {
        "final_cosine_similarity": float(similarity.item()),
        "final_cosine_loss": float(cosine_loss.item()),
    }


def _capture_autocast_context(distill_config: LinearDistillConfig):
    device = torch.device(distill_config.device)
    if distill_config.capture_autocast_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _evaluate_replacement_report(
    replacement: FFFLinear,
    x_metric: torch.Tensor,
    y_metric: torch.Tensor,
    *,
    distill_config: LinearDistillConfig,
    router_config: RouterDistillConfig,
    balance_config: BalanceDistillConfig,
) -> dict[str, object]:
    previous_training = replacement.training
    replacement.eval()
    try:
        with torch.no_grad():
            pred = replacement(x_metric)
            loss = distillation_loss(
                pred,
                y_metric,
                normalized_mse_weight=distill_config.normalized_mse_weight,
                cosine_weight=distill_config.cosine_weight,
                variance_weight=distill_config.variance_weight,
            )
            normalized_mse = distillation_loss(
                pred,
                y_metric,
                normalized_mse_weight=1.0,
                cosine_weight=0.0,
                variance_weight=0.0,
            )
            alignment = _cosine_alignment_metrics(pred, y_metric)
            diag_tokens = min(int(x_metric.shape[0]), distill_config.batch_size)
            diag_x = x_metric[:diag_tokens]
            diag_y = y_metric[:diag_tokens]
            router = _router_auxiliary_loss(
                replacement,
                diag_x,
                diag_y,
                router_config,
            )[1]
            balance = _balance_auxiliary_loss(
                replacement,
                diag_x,
                balance_config,
                total_tokens=diag_x.shape[0],
            )[1]
            diagnostics = _diagnostics_record(replacement, diag_x)
    finally:
        replacement.train(previous_training)

    return {
        "loss": float(loss.item()),
        "normalized_mse": float(normalized_mse.item()),
        "final_cosine_similarity": alignment["final_cosine_similarity"],
        "final_cosine_loss": alignment["final_cosine_loss"],
        "diagnostics": diagnostics,
        "router": router,
        "balance": balance,
        "metric_mode": "eval",
        "metric_hard_routing": True,
        "metric_loss_includes_auxiliary": False,
        "diagnostics_tokens": diag_tokens,
    }


def distill_linear_from_tensors(
    name: str,
    linear: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fff_config: dict[str, Any],
    distill_config: LinearDistillConfig,
    router_config: RouterDistillConfig | None = None,
    balance_config: BalanceDistillConfig | None = None,
    locoprop_config: LocoPropDistillConfig | None = None,
    output_dir: Path,
) -> LayerDistillResult:
    if x.ndim != 2 or x.shape[1] != linear.in_features:
        raise ValueError(f"x for {name} must be [N, {linear.in_features}]")
    if y.ndim != 2 or y.shape[1] != linear.out_features:
        raise ValueError(f"y for {name} must be [N, {linear.out_features}]")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y token count mismatch for {name}: {x.shape[0]} != {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError(f"no captured tokens for {name}")

    device = torch.device(distill_config.device)
    x_all = x.to(device=device, dtype=torch.float32)
    y_all = y.to(device=device, dtype=torch.float32)
    fit_indices, metric_indices, metric_split = _fit_metric_split_indices(
        x_all.shape[0],
        holdout_fraction=distill_config.metric_holdout_fraction,
        seed=distill_config.metric_split_seed,
        device=device,
    )
    x_train = x_all[fit_indices]
    y_train = y_all[fit_indices]
    x_metric = x_all[metric_indices]
    y_metric = y_all[metric_indices]
    replacement = make_fff_replacement(linear, config=fff_config).to(
        device=device,
        dtype=torch.float32,
    )
    replacement.train()
    router_config = router_config or RouterDistillConfig()
    balance_config = balance_config or BalanceDistillConfig()
    locoprop_config = locoprop_config or LocoPropDistillConfig()
    _guard_router_training(replacement, router_config, balance_config)
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=distill_config.lr)
    metrics_path = output_dir / "layer_metrics.jsonl"

    def compute_loss(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        pred = _replacement_prediction(replacement, batch_x, batch_y, router_config)
        main_loss = distillation_loss(
            pred,
            batch_y,
            normalized_mse_weight=distill_config.normalized_mse_weight,
            cosine_weight=distill_config.cosine_weight,
            variance_weight=distill_config.variance_weight,
        )
        router_loss, _ = _router_auxiliary_loss(replacement, batch_x, batch_y, router_config)
        balance_loss, _ = _balance_auxiliary_loss(
            replacement,
            batch_x,
            balance_config,
            total_tokens=x_train.shape[0],
        )
        return main_loss + router_loss + balance_loss

    initial_report = _evaluate_replacement_report(
        replacement,
        x_metric,
        y_metric,
        distill_config=distill_config,
        router_config=router_config,
        balance_config=balance_config,
    )
    append_jsonl(
        metrics_path,
        {
            "layer": name,
            "phase": "initial",
            "loss": initial_report["loss"],
            "normalized_mse": initial_report["normalized_mse"],
            "tokens": int(x_metric.shape[0]),
            "fit_tokens": int(x_train.shape[0]),
            "metric_tokens": int(x_metric.shape[0]),
            "metric_split": metric_split,
            "metric_holdout_fraction": distill_config.metric_holdout_fraction,
            "metric_mode": initial_report["metric_mode"],
            "metric_hard_routing": initial_report["metric_hard_routing"],
            "metric_loss_includes_auxiliary": initial_report["metric_loss_includes_auxiliary"],
            "diagnostics_tokens": initial_report["diagnostics_tokens"],
            "diagnostics": initial_report["diagnostics"],
            "router": initial_report["router"],
            "balance": initial_report["balance"],
        },
    )

    train_tokens = distill_config.steps * min(distill_config.batch_size, x_train.shape[0])
    _synchronize_if_cuda(device)
    train_start = time.perf_counter()
    for step in range(distill_config.steps):
        indices = _batch_indices(x_train.shape[0], distill_config.batch_size, device=device)
        batch_x = x_train[indices]
        batch_y = y_train[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(batch_x, batch_y)
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError(f"non-finite distillation loss for {name} at step {step}")
        loss.backward()
        optimizer.step()
    _synchronize_if_cuda(device)
    train_seconds = max(time.perf_counter() - train_start, sys.float_info.epsilon)
    tokens_per_second = float(train_tokens) / train_seconds

    locoprop_record: dict[str, object] = {
        "enabled": locoprop_config.enabled,
        "status": "skipped",
    }
    if locoprop_config.enabled:
        try:
            locoprop_record = _apply_locoprop_refit(
                replacement,
                x_train,
                y_train,
                locoprop_config,
                optimizer=optimizer,
            )
        except Exception as exc:
            locoprop_record = {
                "enabled": True,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(
            metrics_path,
            {
                "layer": name,
                "phase": "locoprop_refit",
                "tokens": int(x_train.shape[0]),
                "fit_tokens": int(x_train.shape[0]),
                "metric_tokens": int(x_metric.shape[0]),
                "metric_split": metric_split,
                "locoprop": locoprop_record,
            },
        )

    final_report = _evaluate_replacement_report(
        replacement,
        x_metric,
        y_metric,
        distill_config=distill_config,
        router_config=router_config,
        balance_config=balance_config,
    )
    layer_dir = output_dir / "layers" / name.replace(".", "__")
    layer_dir.mkdir(parents=True, exist_ok=True)
    state_path = layer_dir / "fff_state.pt"
    torch.save(replacement.state_dict(), state_path)
    result = LayerDistillResult(
        name=name,
        initial_loss=float(initial_report["loss"]),
        final_loss=float(final_report["loss"]),
        initial_normalized_mse=float(initial_report["normalized_mse"]),
        final_normalized_mse=float(final_report["normalized_mse"]),
        final_cosine_similarity=float(final_report["final_cosine_similarity"]),
        final_cosine_loss=float(final_report["final_cosine_loss"]),
        train_seconds=float(train_seconds),
        tokens_per_second=float(tokens_per_second),
        captured_tokens=int(x_all.shape[0]),
        observed_tokens=int(x_all.shape[0]),
        dropped_tokens=0,
        replacement_path=str(state_path),
        fit_tokens=int(x_train.shape[0]),
        metric_tokens=int(x_metric.shape[0]),
        metric_split=metric_split,
        metric_holdout_fraction=distill_config.metric_holdout_fraction,
    )
    append_jsonl(
        metrics_path,
        {
            "layer": name,
            "phase": "final",
            "loss": result.final_loss,
            "normalized_mse": result.final_normalized_mse,
            "final_cosine_similarity": result.final_cosine_similarity,
            "final_cosine_loss": result.final_cosine_loss,
            "train_seconds": result.train_seconds,
            "tokens_per_second": result.tokens_per_second,
            "tokens": result.captured_tokens,
            "fit_tokens": result.fit_tokens,
            "metric_tokens": result.metric_tokens,
            "metric_split": result.metric_split,
            "metric_holdout_fraction": result.metric_holdout_fraction,
            "metric_mode": final_report["metric_mode"],
            "metric_hard_routing": final_report["metric_hard_routing"],
            "metric_loss_includes_auxiliary": final_report["metric_loss_includes_auxiliary"],
            "diagnostics_tokens": final_report["diagnostics_tokens"],
            "diagnostics": final_report["diagnostics"],
            "router": final_report["router"],
            "balance": final_report["balance"],
            "locoprop": locoprop_record,
            "replacement_path": result.replacement_path,
        },
    )
    return result


def run_layerwise_distillation(
    model: nn.Module,
    sample_batches: list[torch.Tensor],
    config: dict[str, Any],
    *,
    output_dir: Path,
    progressive_step: int | None = None,
    progressive_step_size: int = 1,
) -> list[LayerDistillResult]:
    reject_unknown_distill_config_keys(config)
    distill_config = LinearDistillConfig.from_mapping(config.get("distill"))
    router_config = RouterDistillConfig.from_mapping(config.get("router"))
    balance_config = BalanceDistillConfig.from_mapping(config.get("balance"))
    locoprop_config = LocoPropDistillConfig.from_mapping(config.get("locoprop"))
    raw_fff_config = config.get("fff") or {}
    if not isinstance(raw_fff_config, dict):
        raise ValueError("fff config must be a mapping")
    fff_config = dict(raw_fff_config)
    if not sample_batches:
        raise ValueError("sample_batches must not be empty")
    selected = _selected_reports(
        model,
        config,
        progressive_step=progressive_step,
        progressive_step_size=progressive_step_size,
    )
    if not selected:
        raise ValueError("no eligible Linear layers selected for distillation")

    model.eval()
    with LinearCaptureSet(
        model,
        selected,
        max_tokens_per_layer=distill_config.max_capture_tokens_per_layer,
        max_bytes_per_layer=distill_config.max_capture_bytes_per_layer,
    ) as captures, torch.no_grad():
        for batch in sample_batches:
            with _capture_autocast_context(distill_config):
                model(batch)

    results: list[LayerDistillResult] = []
    for report in selected:
        linear = get_module(model, report.name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{report.name!r} is {type(linear).__name__}, not nn.Linear")
        x, y = captures.tensors(report.name)
        capture = captures.captures[report.name]
        result = distill_linear_from_tensors(
            report.name,
            linear,
            x,
            y,
            fff_config=fff_config,
            distill_config=distill_config,
            router_config=router_config,
            balance_config=balance_config,
            locoprop_config=locoprop_config,
            output_dir=output_dir,
        )
        results.append(
            LayerDistillResult(
                name=result.name,
                initial_loss=result.initial_loss,
                final_loss=result.final_loss,
                initial_normalized_mse=result.initial_normalized_mse,
                final_normalized_mse=result.final_normalized_mse,
                final_cosine_similarity=result.final_cosine_similarity,
                final_cosine_loss=result.final_cosine_loss,
                train_seconds=result.train_seconds,
                tokens_per_second=result.tokens_per_second,
                captured_tokens=capture.captured_tokens,
                observed_tokens=capture.observed_tokens,
                dropped_tokens=capture.dropped_tokens,
                replacement_path=result.replacement_path,
                fit_tokens=result.fit_tokens,
                metric_tokens=result.metric_tokens,
                metric_split=result.metric_split,
                metric_holdout_fraction=result.metric_holdout_fraction,
            )
        )
    write_json(output_dir / "layer_summary.json", [result.log_record() for result in results])
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fff_distill_default.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override config.teacher_checkpoint for validation-selected teacher distillation.",
    )
    parser.add_argument("--output-dir", default="outputs/distill")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--progressive-step", type=int, default=None)
    parser.add_argument("--progressive-step-size", type=int, default=1)
    parser.add_argument("--sample-split", choices=("train", "train_eval", "val"), default="train_eval")
    parser.add_argument("--max-sample-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    select_progressive_reports(
        [],
        step=args.progressive_step,
        step_size=args.progressive_step_size,
    )
    config = load_yaml(args.config)
    reject_unknown_distill_config_keys(config)
    context = RunContext(
        Path(args.output_dir),
        seed=int(config.get("seed", 1337)),
        quick_smoke=args.quick_smoke,
    )
    context.prepare()
    write_json(
        context.output_dir / "run_context.json",
        context.metadata()
        | {
            "config": config,
            "argv": sys.argv,
            "progressive_step": args.progressive_step,
            "progressive_step_size": args.progressive_step_size,
            "progressive_args_validated": True,
            "sample_split": args.sample_split,
            "max_sample_batches": args.max_sample_batches,
        },
    )
    if args.quick_smoke:
        print("distillation quick smoke metadata written")
        return 0

    raw_checkpoint = args.checkpoint or config.get("teacher_checkpoint")
    if raw_checkpoint is None:
        raise RuntimeError("teacher_checkpoint is required for non-smoke layerwise distillation")
    distill_config = LinearDistillConfig.from_mapping(config.get("distill"))
    device = torch.device(distill_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for layerwise distillation, but CUDA is unavailable")
    loaded = load_teacher_for_distillation(
        checkpoint_path=Path(str(raw_checkpoint)),
        quick_smoke=args.quick_smoke,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    sample_batches = _sample_batches_from_run_config(
        loaded.run_config,
        split=args.sample_split,
        max_batches=args.max_sample_batches,
        device=device,
    )
    results = run_layerwise_distillation(
        loaded.model,
        sample_batches,
        config,
        output_dir=context.output_dir,
        progressive_step=args.progressive_step,
        progressive_step_size=args.progressive_step_size,
    )
    write_json(
        context.output_dir / "distill_summary.json",
        {
            "teacher_checkpoint": str(loaded.checkpoint_path),
            "selected_val_accuracy": loaded.selected_val_accuracy,
            "parameter_count": loaded.parameter_count,
            "sample_split": args.sample_split,
            "sample_batches": len(sample_batches),
            "test_accessed": False,
            "layers": [result.log_record() for result in results],
        },
    )
    print(f"layerwise distillation complete: {len(results)} layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
