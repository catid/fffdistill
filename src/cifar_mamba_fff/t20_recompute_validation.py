from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .data import build_cifar10_loaders
from .distill_linears import load_teacher_for_distillation, reject_unknown_distill_config_keys
from .models.replacement import (
    discover_linear_layers,
    get_module,
    make_fff_replacement,
    replace_module,
)
from .utils import RunContext, bool_arg, load_yaml, write_json


@dataclass(frozen=True)
class SingleLayerReplacementSpec:
    layer_name: str
    eligible_index: int
    replacement_state: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute T20 single-layer FFF replacement validation accuracy"
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--distill-config", default="configs/fff_distill_default.yaml")
    parser.add_argument("--replacement-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-name", default=None)
    parser.add_argument("--eligible-index", type=int, default=None)
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser


def _autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.no_grad()


def _expect_mapping(value: object, section: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


def _resolve_target_layer(
    model: nn.Module,
    *,
    eligible_config: Mapping[str, object],
    layer_name: str | None,
    eligible_index: int | None,
) -> tuple[str, int]:
    if (layer_name is None) == (eligible_index is None):
        raise ValueError("exactly one of --layer-name or --eligible-index is required")
    min_in = int(eligible_config.get("min_in_features", 64))
    min_out = int(eligible_config.get("min_out_features", 64))
    eligible = [
        report
        for report in discover_linear_layers(
            model,
            min_in_features=min_in,
            min_out_features=min_out,
        )
        if report.included
    ]
    if not eligible:
        raise ValueError("no eligible Linear layers are available for replacement")
    if eligible_index is not None:
        if eligible_index < 0 or eligible_index >= len(eligible):
            raise IndexError(
                f"eligible_index={eligible_index} is outside [0, {len(eligible) - 1}]"
            )
        return eligible[eligible_index].name, eligible_index
    assert layer_name is not None
    for index, report in enumerate(eligible):
        if report.name == layer_name:
            return report.name, index
    raise KeyError(f"{layer_name!r} is not an eligible Linear layer")


def _replace_single_layer(
    model: nn.Module,
    *,
    distill_config: Mapping[str, object],
    spec: SingleLayerReplacementSpec,
    device: torch.device,
) -> dict[str, object]:
    raw_fff = _expect_mapping(distill_config.get("fff", {}), "distill_config.fff")
    original = get_module(model, spec.layer_name)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"{spec.layer_name!r} is {type(original).__name__}, not nn.Linear")
    replacement = make_fff_replacement(original, config=dict(raw_fff)).to(device)
    state = torch.load(spec.replacement_state, map_location=device, weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError(f"{spec.replacement_state} did not contain a state_dict mapping")
    replacement.load_state_dict(state, strict=True)
    replacement.train(original.training)
    replace_module(model, spec.layer_name, replacement)
    return {
        "layer": spec.layer_name,
        "eligible_index": spec.eligible_index,
        "replacement_state": str(spec.replacement_state),
        "replacement_parameters": sum(parameter.numel() for parameter in replacement.parameters()),
    }


@torch.no_grad()
def _evaluate_validation(
    model: nn.Module,
    val_loader,
    *,
    device: torch.device,
    max_val_steps: int | None,
    precision: str,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    steps = 0
    for step, batch in enumerate(val_loader):
        if max_val_steps is not None and step >= max_val_steps:
            break
        image, target = batch
        image = image.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        with _autocast_context(device, precision):
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
        "validation_loss_after_replacement": loss_sum / total,
        "validation_accuracy_after_replacement": correct / total,
        "validation_steps_after_replacement": float(steps),
    }


def recompute_single_layer_validation(
    *,
    teacher_checkpoint: Path,
    distill_config_path: Path,
    replacement_state: Path,
    output_dir: Path,
    layer_name: str | None = None,
    eligible_index: int | None = None,
    quick_smoke: bool = False,
    device: torch.device | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_val_steps: int | None = None,
    precision: str = "bf16",
) -> dict[str, object]:
    device = device or torch.device("cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for T20 validation recompute, but CUDA is unavailable")
    distill_config = load_yaml(distill_config_path)
    reject_unknown_distill_config_keys(dict(distill_config))
    loaded = load_teacher_for_distillation(
        checkpoint_path=teacher_checkpoint,
        quick_smoke=quick_smoke,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    eligible_config = _expect_mapping(
        distill_config.get("eligible_linear", {}),
        "distill_config.eligible_linear",
    )
    resolved_layer, resolved_index = _resolve_target_layer(
        loaded.model,
        eligible_config=eligible_config,
        layer_name=layer_name,
        eligible_index=eligible_index,
    )
    context = RunContext(output_dir, seed=loaded.run_config.seed, quick_smoke=quick_smoke)
    context.prepare()
    spec = SingleLayerReplacementSpec(
        layer_name=resolved_layer,
        eligible_index=resolved_index,
        replacement_state=replacement_state,
    )
    replacement_record = _replace_single_layer(
        loaded.model,
        distill_config=distill_config,
        spec=spec,
        device=device,
    )
    _train_loader, val_loader = build_cifar10_loaders(loaded.run_config.data)
    start = time.perf_counter()
    metrics = _evaluate_validation(
        loaded.model,
        val_loader,
        device=device,
        max_val_steps=max_val_steps,
        precision=precision,
    )
    elapsed = time.perf_counter() - start
    result: dict[str, object] = {
        "phase": "t20_single_layer_validation_recompute",
        "teacher_checkpoint": str(teacher_checkpoint),
        "distill_config": str(distill_config_path),
        "selected_val_accuracy": loaded.selected_val_accuracy,
        "parameter_count": loaded.parameter_count,
        "quick_smoke": quick_smoke,
        "test_accessed": False,
        "max_val_steps": max_val_steps,
        "precision": precision,
        "elapsed_seconds": elapsed,
        **replacement_record,
        **metrics,
    }
    write_json(context.output_dir / "t20_validation_recompute.json", result)
    write_json(
        context.output_dir / "run_context.json",
        context.metadata()
        | {
            "argv": sys.argv,
            "test_accessed": False,
            "sample_split": "val",
            "distill_config": distill_config,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = recompute_single_layer_validation(
        teacher_checkpoint=Path(args.teacher_checkpoint),
        distill_config_path=Path(args.distill_config),
        replacement_state=Path(args.replacement_state),
        output_dir=Path(args.output_dir),
        layer_name=args.layer_name,
        eligible_index=args.eligible_index,
        quick_smoke=args.quick_smoke,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_steps=args.max_val_steps,
        precision=args.precision,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
