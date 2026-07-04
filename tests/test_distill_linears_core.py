from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import cifar_mamba_fff.distill_linears as distill_linears
from cifar_mamba_fff.distill_linears import (
    LinearDistillConfig,
    RouterDistillConfig,
    _capture_autocast_context,
    _replacement_prediction,
    _router_auxiliary_loss,
    distill_linear_from_tensors,
    load_teacher_for_distillation,
    run_layerwise_distillation,
)
from cifar_mamba_fff.models.fff_linear import FFFLinear


def _small_distill_config() -> dict[str, object]:
    return {
        "eligible_linear": {"min_in_features": 6, "min_out_features": 4},
        "fff": {
            "shared_rows": 24,
            "depth": 1,
            "route_rows": 1,
            "leaf_rows": 4,
            "activation": "gelu",
            "route_row_role": "routing_only",
            "route_rows_output_count": 0,
            "hard_routing": True,
        },
        "distill": {
            "steps": 80,
            "lr": 0.02,
            "batch_size": 32,
            "max_layers": 1,
            "max_capture_tokens_per_layer": 96,
            "max_capture_bytes_per_layer": None,
            "device": "cpu",
        },
        "router": {
            "recipe": "vanilla_ste",
            "loss_coeff": 0.0001,
        },
    }


class _TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 4)
        self.loaded_state: dict[str, object] | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(image.reshape(image.shape[0], -1))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        self.loaded_state = dict(state_dict)
        return None


def _teacher_checkpoint_payload(parameter_count: int = 1234) -> dict[str, object]:
    return {
        "config": {
            "seed": 2037,
            "dataset_name": "cifar10",
            "data": {
                "data_dir": "data/cifar10",
                "batch_size": 8,
                "num_workers": 0,
                "seed": 2037,
                "split_seed": 1337,
                "train_size": 45000,
                "val_size": 5000,
                "download": False,
                "quick_smoke": False,
                "smoke_train_size": 1024,
                "smoke_val_size": 256,
                "smoke_test_size": 256,
                "randaugment": False,
                "label_smoothing": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
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
                "batch_size_per_gpu": 8,
                "num_workers": 0,
                "precision": "bf16",
                "optimizer": "muon_adamw",
                "schedule": "cosine",
                "warmup_epochs": 10,
                "lr_muon": 0.01,
                "lr_adamw": 0.001,
                "weight_decay_muon": 0.03,
                "weight_decay_adamw": 0.03,
                "label_smoothing": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "adamw_betas": [0.9, 0.95],
                "adamw_eps": 1e-10,
                "muon_momentum": 0.95,
                "wsd_stable_fraction": 0.8,
                "grad_clip_norm": None,
            },
        },
        "metrics": {"val_accuracy": 0.9234, "epoch": 118},
        "parameter_count": parameter_count,
        "model": {"fake": torch.ones(1)},
    }


def _write_teacher_checkpoint(path: Path, *, parameter_count: int = 1234) -> None:
    torch.save(_teacher_checkpoint_payload(parameter_count), path)


def test_linear_distill_config_defaults_do_not_require_all_keys() -> None:
    config = LinearDistillConfig.from_mapping({"steps": 3, "max_capture_bytes_per_layer": None})

    assert config.steps == 3
    assert config.max_capture_tokens_per_layer is not None
    assert config.max_capture_bytes_per_layer is None
    assert config.capture_autocast_bf16 is True


def test_linear_distill_config_parses_capture_autocast_bool() -> None:
    config = LinearDistillConfig.from_mapping({"capture_autocast_bf16": "false"})

    assert config.capture_autocast_bf16 is False

    with pytest.raises(ValueError, match="capture_autocast_bf16"):
        LinearDistillConfig.from_mapping({"capture_autocast_bf16": 1})


def test_capture_autocast_context_uses_bf16_only_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return FakeAutocast()

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with _capture_autocast_context(
        LinearDistillConfig.from_mapping({"device": "cpu", "capture_autocast_bf16": True})
    ):
        pass
    assert calls == []

    with _capture_autocast_context(
        LinearDistillConfig.from_mapping({"device": "cuda", "capture_autocast_bf16": True})
    ):
        pass
    assert calls == [("cuda", torch.bfloat16)]

    with _capture_autocast_context(
        LinearDistillConfig.from_mapping({"device": "cuda", "capture_autocast_bf16": False})
    ):
        pass
    assert calls == [("cuda", torch.bfloat16)]


def test_load_teacher_for_distillation_keeps_checkpoint_train_val_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "teacher_best.pt"
    _write_teacher_checkpoint(checkpoint_path, parameter_count=1234)

    def fake_build_teacher_model(_model_config, *, device):
        return _TinyTeacher().to(device), 1234

    monkeypatch.setattr(distill_linears, "build_teacher_model", fake_build_teacher_model)

    loaded = load_teacher_for_distillation(
        checkpoint_path=checkpoint_path,
        quick_smoke=True,
        device=torch.device("cpu"),
        batch_size=4,
        num_workers=0,
    )

    assert loaded.selected_val_accuracy == pytest.approx(0.9234)
    assert loaded.parameter_count == 1234
    assert loaded.run_config.data.use_test is False
    assert loaded.run_config.data.quick_smoke is True
    assert loaded.run_config.data.batch_size == 4
    assert isinstance(loaded.model, _TinyTeacher)
    assert loaded.model.loaded_state is not None
    assert torch.equal(loaded.model.loaded_state["fake"], torch.ones(1))


def test_load_teacher_for_distillation_rejects_parameter_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "teacher_best.pt"
    _write_teacher_checkpoint(checkpoint_path, parameter_count=1234)

    def fake_build_teacher_model(_model_config, *, device):
        return _TinyTeacher().to(device), 999

    monkeypatch.setattr(distill_linears, "build_teacher_model", fake_build_teacher_model)

    with pytest.raises(ValueError, match="parameter_count"):
        load_teacher_for_distillation(
            checkpoint_path=checkpoint_path,
            quick_smoke=False,
            device=torch.device("cpu"),
        )


def test_distill_cli_loads_checkpoint_and_uses_train_val_batches_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "teacher_best.pt"
    config_path = tmp_path / "distill.yaml"
    output_dir = tmp_path / "out"
    _write_teacher_checkpoint(checkpoint_path, parameter_count=4321)
    config_path.write_text(
        "\n".join(
            [
                f"teacher_checkpoint: {checkpoint_path}",
                "seed: 2037",
                "eligible_linear:",
                "  min_in_features: 6",
                "  min_out_features: 4",
                "distill:",
                "  steps: 2",
                "  lr: 0.01",
                "  batch_size: 4",
                "  max_layers: 1",
                "  max_capture_tokens_per_layer: 8",
                "  max_capture_bytes_per_layer: null",
                "  device: cpu",
                "fff:",
                "  shared_rows: 4",
                "  depth: 1",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  hard_routing: true",
                "  route_row_role: routing_only",
                "  route_rows_output_count: 0",
                "router:",
                "  recipe: vanilla_ste",
                "  loss_coeff: 0.0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    seen_data_configs: list[object] = []

    def fake_build_teacher_model(_model_config, *, device):
        return _TinyTeacher().to(device), 4321

    def fake_build_cifar10_loaders(data_config):
        seen_data_configs.append(data_config)
        images = torch.randn(10, 1, 2, 3)
        labels = torch.zeros(10, dtype=torch.long)
        return [(images, labels)], [(images + 1.0, labels)]

    monkeypatch.setattr(distill_linears, "build_teacher_model", fake_build_teacher_model)
    monkeypatch.setattr(distill_linears, "build_cifar10_loaders", fake_build_cifar10_loaders)
    monkeypatch.setattr(
        "sys.argv",
        [
            "distill_linears",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--sample-split",
            "val",
            "--max-sample-batches",
            "1",
            "--progressive-step",
            "1",
            "--progressive-step-size",
            "1",
        ],
    )

    assert distill_linears.main() == 0

    assert seen_data_configs
    assert all(not data_config.use_test for data_config in seen_data_configs)
    summary = json.loads((output_dir / "distill_summary.json").read_text(encoding="utf-8"))
    assert summary["teacher_checkpoint"] == str(checkpoint_path)
    assert summary["selected_val_accuracy"] == pytest.approx(0.9234)
    assert summary["test_accessed"] is False
    assert summary["sample_split"] == "val"
    assert summary["sample_batches"] == 1
    assert summary["layers"][0]["name"] == "linear"
    run_context = json.loads((output_dir / "run_context.json").read_text(encoding="utf-8"))
    assert run_context["progressive_step"] == 1
    assert run_context["progressive_step_size"] == 1


def test_layerwise_distillation_decreases_mse_and_writes_artifacts(tmp_path) -> None:
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(6, 4))
    x = torch.randn(128, 6)

    results = run_layerwise_distillation(
        model,
        [x],
        _small_distill_config(),
        output_dir=tmp_path,
    )

    assert len(results) == 1
    result = results[0]
    assert result.name == "0"
    assert result.captured_tokens == 96
    assert result.observed_tokens == 128
    assert result.dropped_tokens == 32
    assert result.final_normalized_mse < 0.01 * result.initial_normalized_mse
    assert (tmp_path / "layers" / "0" / "fff_state.pt").exists()

    summary = json.loads((tmp_path / "layer_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["final_normalized_mse"] == pytest.approx(result.final_normalized_mse)

    records = [
        json.loads(line)
        for line in (tmp_path / "layer_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["phase"] for record in records] == ["initial", "final"]
    assert records[-1]["diagnostics"]["route_row_role"] == "routing_only"
    assert records[-1]["diagnostics"]["route_output_contributes"] is False
    assert "active_rows_per_token_mean" in records[-1]["diagnostics"]
    assert records[-1]["router"]["recipe"] == "vanilla_ste"
    assert records[-1]["router"]["loss"] >= 0.0
    assert "entropy_mean" in records[-1]["router"]
    assert "dead_leaves" in records[-1]["router"]


def test_distill_linear_from_bfloat16_tensors_uses_fp32_losses(tmp_path) -> None:
    torch.manual_seed(12)
    linear = nn.Linear(6, 4)
    x = torch.randn(64, 6).bfloat16()
    with torch.no_grad():
        y = linear(x.float()).bfloat16()

    result = distill_linear_from_tensors(
        "layer",
        linear,
        x,
        y,
        fff_config={
            "shared_rows": 16,
            "depth": 1,
            "route_rows": 1,
            "leaf_rows": 2,
            "activation": "gelu",
        },
        distill_config=LinearDistillConfig.from_mapping(
            {
                "steps": 50,
                "lr": 0.02,
                "batch_size": 32,
                "max_capture_bytes_per_layer": None,
            }
        ),
        router_config=RouterDistillConfig(recipe="vanilla_ste", loss_coeff=0.0001),
        output_dir=tmp_path,
    )

    assert result.final_loss < result.initial_loss
    metrics = [
        json.loads(line)
        for line in (tmp_path / "layer_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(isinstance(record["loss"], float) for record in metrics)


def test_hard_routing_without_route_output_or_router_loss_is_rejected(tmp_path) -> None:
    linear = nn.Linear(6, 4)
    x = torch.randn(32, 6)
    y = linear(x).detach()

    with pytest.raises(ValueError, match="route parameters are not trained"):
        distill_linear_from_tensors(
            "layer",
            linear,
            x,
            y,
            fff_config={
                "shared_rows": 4,
                "depth": 1,
                "route_rows": 1,
                "leaf_rows": 1,
                "hard_routing": True,
                "route_row_role": "routing_only",
                "route_rows_output_count": 0,
            },
            distill_config=LinearDistillConfig.from_mapping(
                {"steps": 1, "batch_size": 8, "max_capture_bytes_per_layer": None}
            ),
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "recipe",
    [
        "no_ste_soft_router",
        "vanilla_ste",
        "clipped_ste",
        "sigmoid_surrogate_ste",
        "st_gumbel",
    ],
)
def test_main_distillation_loss_sends_recipe_gradients_to_route_weights(
    recipe: str,
) -> None:
    torch.manual_seed(14)
    layer = FFFLinear(
        6,
        4,
        depth=2,
        shared_rows=2,
        route_rows=2,
        leaf_rows=2,
        hard_routing=True,
        route_row_role="routing_only",
        route_rows_output_count=0,
        bias=False,
    )
    x = torch.randn(32, 6)
    y = torch.randn(32, 4)
    router_config = RouterDistillConfig(recipe=recipe, loss_coeff=0.0)

    pred = _replacement_prediction(layer, x, y, router_config)
    loss = (pred - y).square().mean()
    loss.backward()

    assert torch.isfinite(loss.detach())
    assert layer.route_weight.grad is not None
    assert torch.isfinite(layer.route_weight.grad).all()
    assert layer.route_weight.grad.abs().sum() > 0.0


@pytest.mark.parametrize(
    "recipe",
    [
        "no_ste_soft_router",
        "vanilla_ste",
        "clipped_ste",
        "sigmoid_surrogate_ste",
        "st_gumbel",
        "utility_targeted_ste",
        "hard_em_utility_ste",
        "expert_choice_imitation",
    ],
)
def test_router_auxiliary_recipes_send_gradients_to_route_weights(recipe: str) -> None:
    torch.manual_seed(13)
    layer = FFFLinear(
        6,
        4,
        depth=2,
        shared_rows=2,
        route_rows=2,
        leaf_rows=2,
        hard_routing=True,
        route_row_role="routing_only",
        route_rows_output_count=0,
        bias=False,
    )
    x = torch.randn(24, 6)
    y = torch.randn(24, 4)

    loss, diagnostics = _router_auxiliary_loss(
        layer,
        x,
        y,
        RouterDistillConfig(recipe=recipe, loss_coeff=1.0, temperature=1.0),
    )
    loss.backward()

    assert torch.isfinite(loss.detach())
    assert diagnostics["recipe"] == recipe
    assert diagnostics["loss"] >= 0.0
    assert "entropy_mean" in diagnostics
    assert "dead_leaves" in diagnostics
    assert layer.route_weight.grad is not None
    assert torch.isfinite(layer.route_weight.grad).all()
    assert layer.route_weight.grad.abs().sum() > 0.0
