from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .data import build_cifar10_loaders
from .evaluate_teacher import run_config_from_checkpoint, selected_val_accuracy
from .losses.balance import split_balance_loss
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
class LayerDistillResult:
    name: str
    initial_loss: float
    final_loss: float
    initial_normalized_mse: float
    final_normalized_mse: float
    captured_tokens: int
    observed_tokens: int
    dropped_tokens: int
    replacement_path: str

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
    split: Literal["train", "val"],
    max_batches: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if max_batches <= 0:
        raise ValueError("max_sample_batches must be positive")
    if run_config.data.use_test:
        raise RuntimeError("distillation sample batches must not use CIFAR-10 test data")
    train_loader, val_loader = build_cifar10_loaders(run_config.data)
    loader = train_loader if split == "train" else val_loader
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

    reports = discover_linear_layers(
        model,
        min_in_features=int(eligible_config.get("min_in_features", 64)),
        min_out_features=int(eligible_config.get("min_out_features", 64)),
    )
    selected = select_progressive_reports(
        reports,
        step=progressive_step,
        step_size=progressive_step_size,
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
    reports = discover_linear_layers(
        model,
        min_in_features=int(eligible_config.get("min_in_features", 64)),
        min_out_features=int(eligible_config.get("min_out_features", 64)),
    )
    return select_progressive_reports(
        reports,
        step=progressive_step,
        step_size=progressive_step_size,
        max_replacements=LinearDistillConfig.from_mapping(config.get("distill")).max_layers,
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


def _guard_router_training(layer: FFFLinear, config: RouterDistillConfig) -> None:
    diagnostics = layer.diagnostics()
    if (
        layer.config.hard_routing
        and not bool(diagnostics["route_output_contributes"])
        and not config.enabled
    ):
        raise ValueError(
            "hard-routed FFFLinear without route output contribution needs a positive "
            "router auxiliary recipe; otherwise route parameters are not trained"
        )


def _batch_indices(total: int, batch_size: int, *, device: torch.device) -> torch.Tensor:
    return torch.randperm(total, device=device)[: min(batch_size, total)]


def distill_linear_from_tensors(
    name: str,
    linear: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fff_config: dict[str, Any],
    distill_config: LinearDistillConfig,
    router_config: RouterDistillConfig | None = None,
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
    x_train = x.to(device=device, dtype=torch.float32)
    y_train = y.to(device=device, dtype=torch.float32)
    replacement = make_fff_replacement(linear, config=fff_config).to(
        device=device,
        dtype=torch.float32,
    )
    replacement.train()
    router_config = router_config or RouterDistillConfig()
    _guard_router_training(replacement, router_config)
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=distill_config.lr)
    metrics_path = output_dir / "layer_metrics.jsonl"

    def compute_loss(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        pred = replacement(batch_x)
        main_loss = distillation_loss(
            pred,
            batch_y,
            normalized_mse_weight=distill_config.normalized_mse_weight,
            cosine_weight=distill_config.cosine_weight,
            variance_weight=distill_config.variance_weight,
        )
        router_loss, _ = _router_auxiliary_loss(replacement, batch_x, batch_y, router_config)
        return main_loss + router_loss

    with torch.no_grad():
        initial_loss_tensor = compute_loss(x_train, y_train)
        initial_mse = distillation_loss(
            replacement(x_train),
            y_train,
            normalized_mse_weight=1.0,
            cosine_weight=0.0,
            variance_weight=0.0,
        )
    append_jsonl(
        metrics_path,
        {
            "layer": name,
            "phase": "initial",
            "loss": float(initial_loss_tensor.item()),
            "normalized_mse": float(initial_mse.item()),
            "tokens": int(x_train.shape[0]),
            "diagnostics": _diagnostics_record(replacement, x_train[: distill_config.batch_size]),
            "router": _router_auxiliary_loss(
                replacement,
                x_train[: distill_config.batch_size],
                y_train[: distill_config.batch_size],
                router_config,
            )[1],
        },
    )

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

    with torch.no_grad():
        final_loss_tensor = compute_loss(x_train, y_train)
        final_mse = distillation_loss(
            replacement(x_train),
            y_train,
            normalized_mse_weight=1.0,
            cosine_weight=0.0,
            variance_weight=0.0,
        )
    layer_dir = output_dir / "layers" / name.replace(".", "__")
    layer_dir.mkdir(parents=True, exist_ok=True)
    state_path = layer_dir / "fff_state.pt"
    torch.save(replacement.state_dict(), state_path)
    result = LayerDistillResult(
        name=name,
        initial_loss=float(initial_loss_tensor.item()),
        final_loss=float(final_loss_tensor.item()),
        initial_normalized_mse=float(initial_mse.item()),
        final_normalized_mse=float(final_mse.item()),
        captured_tokens=int(x_train.shape[0]),
        observed_tokens=int(x_train.shape[0]),
        dropped_tokens=0,
        replacement_path=str(state_path),
    )
    append_jsonl(
        metrics_path,
        {
            "layer": name,
            "phase": "final",
            "loss": result.final_loss,
            "normalized_mse": result.final_normalized_mse,
            "tokens": result.captured_tokens,
            "diagnostics": _diagnostics_record(replacement, x_train[: distill_config.batch_size]),
            "router": _router_auxiliary_loss(
                replacement,
                x_train[: distill_config.batch_size],
                y_train[: distill_config.batch_size],
                router_config,
            )[1],
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
    distill_config = LinearDistillConfig.from_mapping(config.get("distill"))
    router_config = RouterDistillConfig.from_mapping(config.get("router"))
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
            output_dir=output_dir,
        )
        results.append(
            LayerDistillResult(
                name=result.name,
                initial_loss=result.initial_loss,
                final_loss=result.final_loss,
                initial_normalized_mse=result.initial_normalized_mse,
                final_normalized_mse=result.final_normalized_mse,
                captured_tokens=capture.captured_tokens,
                observed_tokens=capture.observed_tokens,
                dropped_tokens=capture.dropped_tokens,
                replacement_path=result.replacement_path,
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
    parser.add_argument("--sample-split", choices=("train", "val"), default="train")
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
