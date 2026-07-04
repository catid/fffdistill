from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from cifar_mamba_fff.distill_linears import (
    _route_churn_diagnostics,
    run_layerwise_distillation,
)
from cifar_mamba_fff.summarize_distill_hpo import collect_distill_hpo_rows, write_csv


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_route_churn_diagnostics_counts_changed_hard_leaf_ids() -> None:
    record = _route_churn_diagnostics(
        torch.tensor([0, 1, 1, 3]),
        torch.tensor([0, 2, 1, 0]),
    )

    assert record["route_churn_available"] is True
    assert record["route_churn_fraction"] == pytest.approx(0.5)
    assert record["route_churn_changed_tokens"] == 2
    assert record["route_churn_tokens"] == 4
    assert record["route_churn_initial_unique_leaves"] == 3
    assert record["route_churn_final_unique_leaves"] == 3
    assert record["route_churn_reason"] is None


def test_route_churn_diagnostics_is_optional_when_leaf_ids_are_unavailable() -> None:
    record = _route_churn_diagnostics(
        None,
        torch.tensor([0, 1]),
        initial_reason="leaf_ids_unavailable",
    )

    assert record["route_churn_available"] is False
    assert record["route_churn_fraction"] is None
    assert record["route_churn_changed_tokens"] is None
    assert record["route_churn_reason"] == "leaf_ids_unavailable"
    json.dumps(record)


def test_distill_linear_records_heldout_route_churn_in_json_outputs(tmp_path) -> None:
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(4, 3))
    x = torch.randn(24, 4)

    result = run_layerwise_distillation(
        model,
        [x],
        {
            "eligible_linear": {"min_in_features": 1, "min_out_features": 1},
            "fff": {
                "depth": 2,
                "shared_rows": 0,
                "route_rows": 1,
                "leaf_rows": 1,
                "activation": "gelu",
                "hard_routing": True,
                "route_row_role": "routing_only",
                "route_rows_output_count": 0,
            },
            "distill": {
                "steps": 2,
                "lr": 0.01,
                "batch_size": 4,
                "device": "cpu",
                "metric_holdout_fraction": 0.25,
                "metric_split_seed": 17,
            },
            "router": {"recipe": "vanilla_ste", "loss_coeff": 0.0001},
        },
        output_dir=tmp_path,
    )[0]

    assert result.metric_split == "holdout"
    assert result.metric_tokens == 6
    assert result.route_churn_available is True
    assert result.route_churn_tokens == result.metric_tokens
    assert result.route_churn_changed_tokens is not None
    assert 0 <= result.route_churn_changed_tokens <= result.route_churn_tokens
    assert result.route_churn_fraction is not None
    assert math.isfinite(result.route_churn_fraction)
    assert 0.0 <= result.route_churn_fraction <= 1.0

    records = [
        json.loads(line)
        for line in (tmp_path / "layer_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_record = next(record for record in records if record["phase"] == "final")
    assert final_record["route_churn_available"] is True
    assert final_record["route_churn_fraction"] == pytest.approx(result.route_churn_fraction)
    assert final_record["route_churn_tokens"] == result.metric_tokens
    assert final_record["router"]["route_churn_fraction"] == pytest.approx(
        result.route_churn_fraction
    )

    summary = json.loads((tmp_path / "layer_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["route_churn_fraction"] == pytest.approx(result.route_churn_fraction)
    assert summary[0]["route_churn_tokens"] == result.metric_tokens


def test_summarize_distill_hpo_collects_route_churn_for_csv(tmp_path) -> None:
    slot = tmp_path / "run_x" / "work" / "0"
    trial = slot / "trials" / "trial_000000"
    layer_name = "linear"
    _write_json(slot / "status.json", {"status": "succeeded", "git_commit": "abc123"})
    _write_json(
        slot / "distill_hpo_summary.json",
        {"grid_offset": 3, "seed": 7, "sample_split": "train_eval", "test_accessed": False},
    )
    _write_json(
        trial / "trial_result.json",
        {
            "status": "succeeded",
            "test_accessed": False,
            "overrides": {
                "case_name": "case_a",
                "include_indices": [0],
                "router_recipe": "vanilla_ste",
                "route_row_role": "routing_only",
            },
        },
    )
    _write_json(trial / "run_context.json", {"git_commit": "abc123"})
    _write_json(
        trial / "layer_summary.json",
        [
            {
                "name": layer_name,
                "initial_normalized_mse": 1.0,
                "final_normalized_mse": 0.5,
                "route_churn_fraction": 0.25,
                "route_churn_tokens": 8,
            }
        ],
    )
    (trial / "layer_metrics.jsonl").write_text(
        json.dumps(
            {
                "phase": "final",
                "layer": layer_name,
                "router": {
                    "route_churn_available": True,
                    "route_churn_fraction": 0.25,
                    "route_churn_changed_tokens": 2,
                    "route_churn_tokens": 8,
                    "route_churn_initial_unique_leaves": 4,
                    "route_churn_final_unique_leaves": 3,
                    "route_churn_reason": None,
                },
                "diagnostics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_distill_hpo_rows(tmp_path / "run_x")

    assert rows[0]["route_churn_available"] is True
    assert rows[0]["route_churn_fraction"] == 0.25
    assert rows[0]["route_churn_changed_tokens"] == 2
    assert rows[0]["route_churn_tokens"] == 8
    assert rows[0]["route_churn_initial_unique_leaves"] == 4
    assert rows[0]["route_churn_final_unique_leaves"] == 3

    csv_out = tmp_path / "summary.csv"
    write_csv(csv_out, rows)
    csv_text = csv_out.read_text(encoding="utf-8")
    assert "route_churn_fraction" in csv_text
    assert "0.25" in csv_text
