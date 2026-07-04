from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from cifar_mamba_fff.distill_linears import (
    LinearDistillConfig,
    distill_linear_from_tensors,
    run_layerwise_distillation,
)


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
        output_dir=tmp_path,
    )

    assert result.final_loss < result.initial_loss
    metrics = [
        json.loads(line)
        for line in (tmp_path / "layer_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(isinstance(record["loss"], float) for record in metrics)
