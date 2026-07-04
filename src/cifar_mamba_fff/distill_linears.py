from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .losses.distill import distillation_loss
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
from .utils import RunContext, append_jsonl, bool_arg, load_yaml, write_json


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


def _selected_reports(model: nn.Module, config: dict[str, Any]) -> list[LinearReport]:
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
        step=None,
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
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=distill_config.lr)
    metrics_path = output_dir / "layer_metrics.jsonl"

    def compute_loss(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        pred = replacement(batch_x)
        return distillation_loss(
            pred,
            batch_y,
            normalized_mse_weight=distill_config.normalized_mse_weight,
            cosine_weight=distill_config.cosine_weight,
            variance_weight=distill_config.variance_weight,
        )

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
) -> list[LayerDistillResult]:
    distill_config = LinearDistillConfig.from_mapping(config.get("distill"))
    raw_fff_config = config.get("fff") or {}
    if not isinstance(raw_fff_config, dict):
        raise ValueError("fff config must be a mapping")
    fff_config = dict(raw_fff_config)
    if not sample_batches:
        raise ValueError("sample_batches must not be empty")
    selected = _selected_reports(model, config)
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
    parser.add_argument("--output-dir", default="outputs/distill")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--progressive-step", type=int, default=None)
    parser.add_argument("--progressive-step-size", type=int, default=1)
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
            "progressive_step": args.progressive_step,
            "progressive_step_size": args.progressive_step_size,
            "progressive_args_validated": True,
        },
    )
    if args.quick_smoke:
        print("distillation quick smoke metadata written")
        return 0
    if config.get("teacher_checkpoint") is None:
        raise RuntimeError("teacher_checkpoint is required for non-smoke layerwise distillation")
    raise RuntimeError("checkpoint loading for layerwise distillation is not wired yet")


if __name__ == "__main__":
    raise SystemExit(main())
