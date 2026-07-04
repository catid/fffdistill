from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

import torch

from .data import Cifar10DataConfig, build_cifar10_test_loader
from .models.mamba3_cifar import Mamba3CifarConfig
from .train_teacher import (
    TeacherRunConfig,
    TeacherTrainConfig,
    _evaluate_steps,
    build_teacher_model,
)
from .utils import RunContext, append_jsonl, bool_arg, write_json


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


def _dataclass_kwargs(cls: type, raw: Mapping[str, object], *, section: str) -> dict[str, object]:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"checkpoint {section} contains unknown keys: {', '.join(unknown)}")
    return {name: raw[name] for name in allowed if name in raw}


def selected_val_accuracy(checkpoint: Mapping[str, object]) -> float:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping) or "val_accuracy" not in metrics:
        raise ValueError("checkpoint is missing validation metrics; refusing final test access")
    return float(metrics["val_accuracy"])


def run_config_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    quick_smoke: bool,
    batch_size: int | None = None,
    num_workers: int | None = None,
    use_test: bool = True,
) -> TeacherRunConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint is missing resolved config")
    data_raw = raw_config.get("data")
    model_raw = raw_config.get("model")
    train_raw = raw_config.get("train")
    if not isinstance(data_raw, Mapping):
        raise ValueError("checkpoint config.data must be a mapping")
    if not isinstance(model_raw, Mapping):
        raise ValueError("checkpoint config.model must be a mapping")
    if not isinstance(train_raw, Mapping):
        raise ValueError("checkpoint config.train must be a mapping")

    data_kwargs = _dataclass_kwargs(Cifar10DataConfig, data_raw, section="config.data")
    if "data_dir" in data_kwargs:
        data_kwargs["data_dir"] = Path(str(data_kwargs["data_dir"]))
    train_kwargs = _dataclass_kwargs(TeacherTrainConfig, train_raw, section="config.train")
    if "adamw_betas" in train_kwargs:
        betas = train_kwargs["adamw_betas"]
        if not isinstance(betas, list | tuple) or len(betas) != 2:
            raise ValueError("checkpoint config.train.adamw_betas must have two values")
        train_kwargs["adamw_betas"] = (float(betas[0]), float(betas[1]))

    data_config = Cifar10DataConfig(**data_kwargs)
    if batch_size is not None:
        data_config = replace(data_config, batch_size=batch_size)
    if num_workers is not None:
        data_config = replace(data_config, num_workers=num_workers)
    data_config = replace(data_config, use_test=use_test, quick_smoke=quick_smoke)
    data_config.validate(allow_test=use_test)

    model_config = Mamba3CifarConfig(
        **_dataclass_kwargs(Mamba3CifarConfig, model_raw, section="config.model")
    )
    model_config.validate()
    train_config = TeacherTrainConfig(**train_kwargs)
    train_config.validate()

    return TeacherRunConfig(
        seed=int(raw_config.get("seed", 1337)),
        dataset_name="cifar10",
        data=data_config,
        model=model_config,
        train=train_config,
    )


def evaluate_teacher_checkpoint(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    quick_smoke: bool = False,
    max_test_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    min_selected_val_accuracy: float = 0.90,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final official Mamba-3 teacher evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("teacher checkpoint must be a mapping")
    selected_val = selected_val_accuracy(checkpoint)
    if selected_val < min_selected_val_accuracy:
        raise ValueError(
            f"selected checkpoint val_accuracy={selected_val:.4f} is below "
            f"required {min_selected_val_accuracy:.4f}; refusing final test access"
        )
    run_config = run_config_from_checkpoint(
        checkpoint,
        quick_smoke=quick_smoke,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    context = RunContext(output_dir, seed=run_config.seed, quick_smoke=quick_smoke)
    context.prepare()
    metrics_path = context.output_dir / "teacher_final_test_metrics.json"
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
    model, parameter_count = build_teacher_model(run_config.model, device=device)
    expected_count = int(checkpoint.get("parameter_count", parameter_count))
    if parameter_count != expected_count:
        raise ValueError(
            f"checkpoint parameter_count={expected_count} does not match rebuilt model "
            f"parameter_count={parameter_count}"
        )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError("teacher checkpoint is missing model state_dict")
    model.load_state_dict(state_dict)
    test_loader = build_cifar10_test_loader(run_config.data)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    metrics = _evaluate_steps(
        model,
        test_loader,
        run_config,
        device,
        max_steps=1 if quick_smoke else max_test_steps,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "phase": "teacher_final_test",
        "checkpoint_path": str(checkpoint_path),
        "selected_val_accuracy": selected_val,
        "test_accuracy": metrics["val_accuracy"],
        "test_loss": metrics["val_loss"],
        "test_steps": metrics["val_steps"],
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count,
        "quick_smoke": quick_smoke,
        "test_accessed": True,
    }
    write_json(metrics_path, result)
    append_jsonl(context.output_dir / "teacher_final_test_metrics.jsonl", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/teacher_final")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--max-test-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--min-selected-val-accuracy", type=float, default=0.90)
    args = parser.parse_args(argv)
    result = evaluate_teacher_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        quick_smoke=args.quick_smoke,
        max_test_steps=args.max_test_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_selected_val_accuracy=args.min_selected_val_accuracy,
    )
    print(f"teacher final test complete: {_jsonable(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
