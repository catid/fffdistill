from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from cifar_mamba_fff.distill_linears import (
    LinearDistillConfig,
    RouterDistillConfig,
    _router_auxiliary_loss,
    distill_linear_from_tensors,
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


def test_linear_distill_config_defaults_do_not_require_all_keys() -> None:
    config = LinearDistillConfig.from_mapping({"steps": 3, "max_capture_bytes_per_layer": None})

    assert config.steps == 3
    assert config.max_capture_tokens_per_layer is not None
    assert config.max_capture_bytes_per_layer is None


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
