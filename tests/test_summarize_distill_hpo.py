from __future__ import annotations

import json
from pathlib import Path

import pytest

from cifar_mamba_fff.summarize_distill_hpo import (
    collect_distill_hpo_rows,
    write_csv,
    write_markdown,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_distill_hpo_collects_layer_and_router_metrics(tmp_path) -> None:
    slot = tmp_path / "run_x" / "work" / "0"
    trial = slot / "trials" / "trial_000000"
    _write_json(
        slot / "status.json",
        {
            "status": "succeeded",
            "git_commit": "abc123",
            "machine": "work",
            "gpu_id": 0,
            "seed": 7,
        },
    )
    _write_json(
        slot / "distill_hpo_summary.json",
        {
            "grid_offset": 3,
            "seed": 7,
            "sample_split": "train_eval",
            "max_sample_batches": 2,
            "test_accessed": False,
        },
    )
    _write_json(
        trial / "trial_result.json",
        {
            "status": "succeeded",
            "test_accessed": False,
            "overrides": {
                "case_name": "case_a",
                "include_indices": [35],
                "router_recipe": "vanilla_ste",
                "balance_recipe": "split_minleaf",
                "balance_coeff": 0.001,
                "route_row_role": "split_routing_output",
                "route_rows": 1,
                "route_result_rows": 2,
                "route_rows_output_count": "all",
                "route_rows_output_fraction": 0.5,
                "depth": 5,
                "leaf_rows": 4,
                "shared_unrouted_frac": 0.2,
            },
        },
    )
    _write_json(
        trial / "run_context.json",
        {"sample_split": "train_eval", "max_sample_batches": 2, "git_commit": "abc123"},
    )
    _write_json(
        trial / "layer_summary.json",
        [
            {
                "name": "blocks.8.reverse_block.mixer.mixer.out_proj",
                "initial_normalized_mse": 1.0,
                "final_normalized_mse": 0.25,
                "final_cosine_similarity": 0.8,
                "tokens_per_second": 1234.5,
                "train_seconds": 0.5,
                "captured_tokens": 100,
                "observed_tokens": 200,
                "fit_tokens": 90,
                "metric_tokens": 10,
                "metric_split": "holdout",
                "metric_holdout_fraction": 0.1,
            }
        ],
    )
    (trial / "layer_metrics.jsonl").write_text(
        json.dumps(
            {
                "phase": "final",
                "layer": "blocks.8.reverse_block.mixer.mixer.out_proj",
                "router": {
                    "dead_leaves": 2,
                    "entropy_mean": 0.3,
                    "leaf_tokens_p10": 1,
                    "leaf_tokens_p50": 4,
                    "leaf_tokens_p90": 8,
                },
                "diagnostics": {
                    "active_rows_per_token_mean": 47,
                    "stored_rows": 259,
                    "effective_stored_rows": 228,
                    "route_output_rows_per_token": 5,
                    "route_output_rows_per_node": 1,
                    "route_rows_contribute": True,
                },
                "locoprop": {
                    "status": "succeeded",
                    "mse_before": 0.4,
                    "mse_after": 0.2,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_distill_hpo_rows(tmp_path / "run_x")

    assert len(rows) == 1
    row = rows[0]
    assert row["case_name"] == "case_a"
    assert row["eligible_index"] == 35
    assert row["sample_split"] == "train_eval"
    assert row["test_accessed"] is False
    assert row["final_normalized_mse"] == 0.25
    assert row["nmse_delta"] == 0.75
    assert row["dead_leaves"] == 2
    assert row["leaf_tokens_p50"] == 4
    assert row["locoprop_nonincreasing"] is True

    csv_out = tmp_path / "summary.csv"
    md_out = tmp_path / "summary.md"
    write_csv(csv_out, rows)
    write_markdown(md_out, rows, collected_root=tmp_path / "run_x")
    assert "case_a" in csv_out.read_text(encoding="utf-8")
    assert "CIFAR-10 test accessed: `false`" in md_out.read_text(encoding="utf-8")


def test_summarize_distill_hpo_rejects_test_accessed_child_summary(tmp_path) -> None:
    slot = tmp_path / "run_x" / "work" / "0"
    trial = slot / "trials" / "trial_000000"
    _write_json(slot / "status.json", {"status": "succeeded", "git_commit": "abc123"})
    _write_json(
        slot / "distill_hpo_summary.json",
        {"sample_split": "train_eval", "max_sample_batches": 1, "test_accessed": False},
    )
    _write_json(
        trial / "trial_result.json",
        {
            "status": "failed_logic",
            "test_accessed": False,
            "result": {"test_accessed": False, "summary": {"test_accessed": True}},
            "overrides": {"case_name": "case_a"},
        },
    )
    _write_json(trial / "run_context.json", {"sample_split": "train_eval", "max_sample_batches": 1})
    _write_json(trial / "layer_summary.json", [{"name": "layer", "final_normalized_mse": 1.0}])
    (trial / "layer_metrics.jsonl").write_text(
        json.dumps({"phase": "final", "layer": "layer"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a completed successful distill trial"):
        collect_distill_hpo_rows(tmp_path / "run_x")

    _write_json(
        trial / "trial_result.json",
        {
            "status": "succeeded",
            "test_accessed": False,
            "result": {"test_accessed": False, "summary": {"test_accessed": True}},
            "overrides": {"case_name": "case_a"},
        },
    )

    with pytest.raises(ValueError, match="test_accessed=true"):
        collect_distill_hpo_rows(tmp_path / "run_x")
