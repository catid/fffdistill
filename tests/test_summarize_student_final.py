from __future__ import annotations

import json
from pathlib import Path

import pytest

from cifar_mamba_fff.summarize_student_final import (
    aggregate_by_family,
    collect_student_final_rows,
    write_csv,
    write_family_csv,
    write_markdown,
    write_selection_manifest_jsonl,
)


def _write_case(root: Path, *, gpu: int, case: str, seed: int, val: float, test: float) -> None:
    slot = root / "foureyes" / str(gpu)
    slot.mkdir(parents=True)
    ckpt = f"outputs/stage_h/{case}/student_best.pt"
    sha = f"sha-{seed}"
    selection_record = f"outputs/final_eval_selection/{case}_selection.json"
    (slot / "status.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "returncode": 0,
                "remote_git_commit": "commit",
                "seed": seed,
                "metadata": {"case": case, "family": "no_balance_cosine"},
            }
        ),
        encoding="utf-8",
    )
    (slot / "student_final_test_metrics.json").write_text(
        json.dumps(
            {
                "phase": "student_final_test",
                "checkpoint_path": ckpt,
                "checkpoint_sha256": sha,
                "selected_val_accuracy": val,
                "test_accuracy": test,
                "test_loss": 0.25,
                "test_steps": 313,
                "elapsed_seconds": 12.0,
                "partial_test_evaluation": False,
                "test_accessed": True,
                "selection": {
                    "selection_record": selection_record,
                    "allow_untracked_selection": False,
                    "checkpoint_sha256": sha,
                },
            }
        ),
        encoding="utf-8",
    )
    selection_dir = root / "selection_records"
    selection_dir.mkdir(exist_ok=True)
    (selection_dir / f"{case}_selection.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "selected_for_final_eval": True,
                "family": "no_balance_cosine",
                "case": case,
                "seed": seed,
                "test_accessed": False,
                "checkpoint_path": ckpt,
                "checkpoint_sha256": sha,
                "best_val_accuracy": val,
            }
        ),
        encoding="utf-8",
    )


def test_collect_student_final_rows_and_writers(tmp_path: Path) -> None:
    root = tmp_path / "collected"
    _write_case(root, gpu=0, case="no_balance_cosine_seed21001", seed=21001, val=0.91, test=0.913)
    _write_case(root, gpu=1, case="no_balance_cosine_seed21002", seed=21002, val=0.92, test=0.919)
    rows = collect_student_final_rows(root)
    families = aggregate_by_family(rows)

    assert len(rows) == 2
    assert families[0]["family"] == "no_balance_cosine"
    assert families[0]["mean_test_accuracy"] == pytest.approx(0.916)

    write_csv(tmp_path / "trials.csv", rows)
    write_family_csv(tmp_path / "families.csv", rows)
    write_selection_manifest_jsonl(tmp_path / "selection.jsonl", rows, [root])
    write_markdown(tmp_path / "summary.md", rows, collected_roots=[root])

    assert "no_balance_cosine_seed21002" in (tmp_path / "trials.csv").read_text(encoding="utf-8")
    assert "mean_test_accuracy" in (tmp_path / "families.csv").read_text(encoding="utf-8")
    assert (tmp_path / "selection.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert "Partial test evaluation: `false`" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_collect_student_final_rows_rejects_partial_or_untracked(tmp_path: Path) -> None:
    root = tmp_path / "collected"
    _write_case(root, gpu=0, case="no_balance_cosine_seed21001", seed=21001, val=0.91, test=0.913)
    metrics = json.loads((root / "foureyes" / "0" / "student_final_test_metrics.json").read_text())
    metrics["partial_test_evaluation"] = True
    (root / "foureyes" / "0" / "student_final_test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="partial"):
        collect_student_final_rows(root)


def test_collect_student_final_rows_labels_below_target_failure_analysis(tmp_path: Path) -> None:
    root = tmp_path / "collected"
    _write_case(root, gpu=0, case="low_lr_cosine_seed21001", seed=21001, val=0.8892, test=0.888)

    with pytest.raises(ValueError, match="without allow_below_target"):
        collect_student_final_rows(root)

    run_context = root / "foureyes" / "0" / "run_context.json"
    run_context.write_text(
        json.dumps(
            {
                "argv": [
                    "evaluate_student.py",
                    "--min-selected-val-accuracy",
                    "0.9",
                    "--allow-below-target",
                    "true",
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = collect_student_final_rows(root)

    assert rows[0]["min_selected_val_accuracy"] == pytest.approx(0.9)
    assert rows[0]["allow_below_target"] is True
    assert aggregate_by_family(rows)[0]["below_target_trial_count"] == 1
