from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from .data import build_cifar10_test_loader
from .distill_linears import load_teacher_for_distillation
from .evaluate_teacher import _jsonable
from .finetune_student import (
    _evaluate_student_steps,
    build_student_model,
    load_student_checkpoint,
    parse_finetune_run_config,
)
from .utils import RunContext, append_jsonl, bool_arg, write_json


def _checkpoint_config_to_finetune_input(raw_config: Mapping[str, object]) -> dict[str, object]:
    config = dict(raw_config)
    if "dataset" not in config and "data" in config:
        data = config.pop("data")
        if not isinstance(data, Mapping):
            raise ValueError("student checkpoint config.data must be a mapping")
        dataset = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "batch_size",
                "num_workers",
                "seed",
                "quick_smoke",
                "label_smoothing",
                "mixup",
                "cutmix",
                "use_test",
            }
        }
        dataset.setdefault("name", "cifar10")
        config["dataset"] = dataset
    return config


def selected_student_val_accuracy(checkpoint: Mapping[str, object]) -> float:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping) or "val_accuracy" not in metrics:
        raise ValueError("student checkpoint is missing validation metrics; refusing final test access")
    return float(metrics["val_accuracy"])


def validate_student_final_eval_threshold(
    *,
    selected_val_accuracy: float,
    min_selected_val_accuracy: float,
    allow_below_target: bool,
) -> None:
    if selected_val_accuracy < min_selected_val_accuracy and not allow_below_target:
        raise ValueError(
            f"selected checkpoint val_accuracy={selected_val_accuracy:.4f} is below "
            f"required {min_selected_val_accuracy:.4f}; pass --allow-below-target true only "
            "for explicitly labeled failure analysis"
        )


def format_student_final_result(
    *,
    checkpoint_path: Path,
    selected_val_accuracy_value: float,
    metrics: Mapping[str, float],
    elapsed_seconds: float,
    replacement_count: int,
    eligible_count: int,
    quick_smoke: bool,
    max_test_steps: int | None,
) -> dict[str, object]:
    partial = quick_smoke or max_test_steps is not None
    result: dict[str, object] = {
        "phase": "student_final_test",
        "checkpoint_path": str(checkpoint_path),
        "selected_val_accuracy": selected_val_accuracy_value,
        "test_steps": metrics["val_steps"],
        "elapsed_seconds": elapsed_seconds,
        "student_replacement_count": replacement_count,
        "eligible_linear_count": eligible_count,
        "quick_smoke": quick_smoke,
        "max_test_steps": max_test_steps,
        "partial_test_evaluation": partial,
        "test_accessed": True,
    }
    if partial:
        result["test_accuracy_partial"] = metrics["val_accuracy"]
        result["test_loss_partial"] = metrics["val_loss"]
    else:
        result["test_accuracy"] = metrics["val_accuracy"]
        result["test_loss"] = metrics["val_loss"]
    return result


def evaluate_student_checkpoint(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    quick_smoke: bool = False,
    max_test_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    min_selected_val_accuracy: float = 0.0,
    allow_below_target: bool = False,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final FFF student evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("student checkpoint must be a mapping")
    selected_val = selected_student_val_accuracy(checkpoint)
    validate_student_final_eval_threshold(
        selected_val_accuracy=selected_val,
        min_selected_val_accuracy=min_selected_val_accuracy,
        allow_below_target=allow_below_target,
    )
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("student checkpoint is missing resolved config")
    run_config = parse_finetune_run_config(
        _checkpoint_config_to_finetune_input(raw_config),
        quick_smoke=quick_smoke,
    )
    if run_config.teacher_checkpoint is None:
        raise ValueError("student checkpoint config is missing teacher_checkpoint")

    data_config = run_config.data
    if batch_size is not None:
        data_config = replace(data_config, batch_size=batch_size)
    if num_workers is not None:
        data_config = replace(data_config, num_workers=num_workers)
    data_config = replace(data_config, use_test=True, quick_smoke=quick_smoke)
    data_config.validate(allow_test=True)

    context = RunContext(output_dir, seed=run_config.seed, quick_smoke=quick_smoke)
    context.prepare()
    metrics_path = context.output_dir / "student_final_test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite final metrics: {metrics_path}")

    write_json(
        context.output_dir / "run_context.json",
        _jsonable(
            context.metadata()
            | {
                "argv": sys.argv,
                "checkpoint_path": str(checkpoint_path),
                "selected_val_accuracy": selected_val,
                "test_accessed": True,
                "run_config": run_config,
                "checkpoint_metrics": checkpoint.get("metrics", {}),
            }
        ),
    )

    device = torch.device("cuda")
    teacher_loaded = load_teacher_for_distillation(
        checkpoint_path=run_config.teacher_checkpoint,
        quick_smoke=quick_smoke,
        device=device,
        batch_size=run_config.train.batch_size_per_gpu,
        num_workers=run_config.train.num_workers,
    )
    teacher = teacher_loaded.model.eval()
    assembled_result = build_student_model(
        loaded_teacher_model=teacher,
        config=run_config,
        device=device,
    )
    student_result = load_student_checkpoint(assembled_result.model, checkpoint_path)
    eligible_count = student_result.eligible_count or assembled_result.eligible_count
    test_loader = build_cifar10_test_loader(data_config)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    metrics = _evaluate_student_steps(
        student_result.model,
        test_loader,
        run_config,
        device,
        max_steps=1 if quick_smoke else max_test_steps,
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    result = format_student_final_result(
        checkpoint_path=checkpoint_path,
        selected_val_accuracy_value=selected_val,
        metrics=metrics,
        elapsed_seconds=elapsed_seconds,
        replacement_count=student_result.replacement_count,
        eligible_count=eligible_count,
        quick_smoke=quick_smoke,
        max_test_steps=max_test_steps,
    )
    write_json(metrics_path, result)
    append_jsonl(context.output_dir / "student_final_test_metrics.jsonl", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/student_final")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-test-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--min-selected-val-accuracy", type=float, default=0.0)
    parser.add_argument("--allow-below-target", type=bool_arg, default=False)
    args = parser.parse_args(argv)
    result = evaluate_student_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        quick_smoke=args.quick_smoke,
        max_test_steps=args.max_test_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_selected_val_accuracy=args.min_selected_val_accuracy,
        allow_below_target=args.allow_below_target,
    )
    print(f"student final test complete: {_jsonable(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
