from __future__ import annotations

from pathlib import Path

import pytest

from cifar_mamba_fff.evaluate_teacher import (
    run_config_from_checkpoint,
    selected_val_accuracy,
)


def _checkpoint_config() -> dict[str, object]:
    return {
        "config": {
            "seed": 2037,
            "dataset_name": "cifar10",
            "data": {
                "data_dir": "data/cifar10",
                "batch_size": 2048,
                "num_workers": 8,
                "seed": 2037,
                "split_seed": 1337,
                "train_size": 45000,
                "val_size": 5000,
                "download": True,
                "quick_smoke": False,
                "smoke_train_size": 1024,
                "smoke_val_size": 256,
                "smoke_test_size": 256,
                "randaugment": False,
                "label_smoothing": 0.1,
                "mixup": 0.2,
                "cutmix": 1.0,
                "use_test": False,
            },
            "model": {
                "d_model": 256,
                "depth": 20,
                "patch_size": 4,
                "d_state": 64,
                "expand": 2,
                "headdim": 64,
                "is_mimo": True,
                "mimo_rank": 2,
                "chunk_size": 16,
                "bidirectional": False,
                "drop_path": 0.1,
                "norm_epsilon": 1e-5,
                "residual_in_fp32": True,
                "num_classes": 10,
                "target_min_params": 9_000_000,
                "target_max_params": 11_000_000,
            },
            "train": {
                "epochs": 200,
                "batch_size_per_gpu": 2048,
                "num_workers": 8,
                "precision": "bf16",
                "optimizer": "muon_adamw",
                "schedule": "cosine",
                "warmup_epochs": 10,
                "lr_muon": 0.01,
                "lr_adamw": 0.001,
                "weight_decay_muon": 0.03,
                "weight_decay_adamw": 0.03,
                "label_smoothing": 0.1,
                "mixup": 0.2,
                "cutmix": 1.0,
                "adamw_betas": [0.9, 0.95],
                "adamw_eps": 1e-10,
                "muon_momentum": 0.95,
                "wsd_stable_fraction": 0.8,
                "grad_clip_norm": None,
            },
        },
        "metrics": {"val_accuracy": 0.9048, "epoch": 199},
        "parameter_count": 9_772_554,
        "model": {},
    }


def test_selected_val_accuracy_requires_checkpoint_metrics() -> None:
    assert selected_val_accuracy(_checkpoint_config()) == pytest.approx(0.9048)
    with pytest.raises(ValueError, match="validation metrics"):
        selected_val_accuracy({"config": {}})


def test_final_eval_checkpoint_config_enables_test_only_for_selected_eval() -> None:
    run_config = run_config_from_checkpoint(
        _checkpoint_config(),
        quick_smoke=True,
        batch_size=128,
        num_workers=0,
    )

    assert run_config.seed == 2037
    assert run_config.dataset_name == "cifar10"
    assert run_config.data.data_dir == Path("data/cifar10")
    assert run_config.data.use_test is True
    assert run_config.data.quick_smoke is True
    assert run_config.data.batch_size == 128
    assert run_config.data.num_workers == 0
    assert run_config.model.d_model == 256
    assert run_config.model.depth == 20
    assert run_config.train.precision == "bf16"
    assert run_config.train.adamw_betas == (0.9, 0.95)


def test_checkpoint_config_can_stay_train_val_only_for_distillation() -> None:
    run_config = run_config_from_checkpoint(
        _checkpoint_config(),
        quick_smoke=False,
        batch_size=64,
        num_workers=0,
        use_test=False,
    )

    assert run_config.data.use_test is False
    assert run_config.data.quick_smoke is False
    assert run_config.data.batch_size == 64
    assert run_config.data.num_workers == 0
    run_config.validate()


def test_final_eval_checkpoint_config_rejects_unknown_keys() -> None:
    checkpoint = _checkpoint_config()
    data = dict(checkpoint["config"]["data"])  # type: ignore[index]
    data["surprise"] = True
    checkpoint["config"] = {**checkpoint["config"], "data": data}  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown keys"):
        run_config_from_checkpoint(checkpoint, quick_smoke=False)
