from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cifar_mamba_fff.summarize_finetune_hpo import (
    TRIAL_COLUMNS,
    aggregate_by_family,
    collect_finetune_hpo_rows,
    infer_family,
    main,
    write_csv,
    write_family_csv,
    write_markdown,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trial(
    root: Path,
    *,
    machine: str,
    gpu: int,
    trial_index: int,
    case: str,
    seed: int | None,
    best_val_accuracy: float | None,
    train_steps: int | None = 100,
    checkpoint_path: str | None = None,
    status: str = "succeeded",
    result_status: str = "succeeded",
    test_accessed: bool | str | None = False,
    result_test_accessed: bool | str | None = None,
    summary_test_accessed: bool | None = None,
    hpo_test_accessed: bool | None = False,
    family: str | None = None,
) -> Path:
    slot = root / machine / str(gpu)
    trial = slot / "trials" / f"trial_{trial_index:06d}_{case}"
    _write_json(
        slot / "status.json",
        {
            "status": "succeeded",
            "machine": machine,
            "gpu_id": gpu,
            "git_commit": "abc123",
        },
    )
    _write_json(
        slot / "finetune_hpo_summary.json",
        {
            "mode": "finetune_hpo_execute",
            "status": "completed",
            "seed": 9999,
        }
        | ({"test_accessed": hpo_test_accessed} if hpo_test_accessed is not None else {}),
    )
    summary: dict[str, object] = {}
    if best_val_accuracy is not None:
        summary["best_val_accuracy"] = best_val_accuracy
    if train_steps is not None:
        summary["train_steps_total"] = train_steps
    if checkpoint_path is not None:
        summary["checkpoint_path"] = checkpoint_path
    elif checkpoint_path is None:
        summary["checkpoint_path"] = f"outputs/{case}/student_best.pt"
    if summary_test_accessed is not None:
        summary["test_accessed"] = summary_test_accessed
    overrides: dict[str, object] = {}
    if seed is not None:
        overrides["seed"] = seed
    if family is not None:
        overrides["family"] = family
    result_payload: dict[str, object] = {
        "status": result_status,
        "summary": summary,
    }
    if result_test_accessed is None:
        result_test_accessed = test_accessed
    if result_test_accessed is not None:
        result_payload["test_accessed"] = result_test_accessed
    trial_payload: dict[str, object] = {
        "trial_index": trial_index,
        "case": case,
        "overrides": overrides,
        "status": status,
        "result": result_payload,
    }
    if test_accessed is not None:
        trial_payload["test_accessed"] = test_accessed
    _write_json(
        trial / "trial_result.json",
        trial_payload,
    )
    return trial


def test_collect_finetune_hpo_rows_and_aggregate_across_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "stage_h_wave_a"
    root_b = tmp_path / "stage_h_wave_b"
    _write_trial(
        root_a,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
    )
    _write_trial(
        root_a,
        machine="work",
        gpu=1,
        trial_index=1,
        case="baseline_cosine_seed21002",
        seed=21002,
        best_val_accuracy=0.75,
    )
    _write_trial(
        root_b,
        machine="ai",
        gpu=0,
        trial_index=2,
        case="wsd_seed21001",
        seed=21001,
        best_val_accuracy=0.65,
    )

    rows = collect_finetune_hpo_rows([root_a, root_b])

    assert [row["case"] for row in rows] == [
        "baseline_cosine_seed21001",
        "baseline_cosine_seed21002",
        "wsd_seed21001",
    ]
    baseline = [row for row in rows if row["family"] == "baseline_cosine"]
    assert [row["seed"] for row in baseline] == [21001, 21002]
    assert baseline[0]["machine"] == "work"
    assert baseline[0]["gpu"] == "0"
    assert baseline[0]["checkpoint_path"].endswith("student_best.pt")

    aggregates = {row["family"]: row for row in aggregate_by_family(rows)}
    assert aggregates["baseline_cosine"]["trials"] == 2
    assert aggregates["baseline_cosine"]["seed_count"] == 2
    assert aggregates["baseline_cosine"]["mean_best_val_accuracy"] == pytest.approx(0.725)
    assert aggregates["baseline_cosine"]["std_best_val_accuracy"] == pytest.approx(
        0.035355339,
    )
    assert aggregates["baseline_cosine"]["best_case"] == "baseline_cosine_seed21002"
    assert aggregates["wsd"]["std_best_val_accuracy"] == 0.0


def test_collect_accepts_common_scheduler_collected_parent_and_preserves_run(tmp_path: Path) -> None:
    common_root = tmp_path / "scheduler_collected"
    run_root = common_root / "stage_h_wave"
    _write_trial(
        run_root,
        machine="ripper",
        gpu=2,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
    )

    rows = collect_finetune_hpo_rows(common_root)

    assert rows[0]["run"] == "stage_h_wave"
    assert rows[0]["machine"] == "ripper"
    assert rows[0]["gpu"] == "2"


def test_validation_summary_rejects_failed_or_partial_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "stage_h_wave"
    _write_trial(
        root,
        machine="work",
        gpu=0,
        trial_index=0,
        case="failed_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        status="failed_logic",
    )
    with pytest.raises(ValueError, match=r"non-succeeded trial_result\.status"):
        collect_finetune_hpo_rows(root)

    root_missing = tmp_path / "stage_h_missing"
    _write_trial(
        root_missing,
        machine="work",
        gpu=0,
        trial_index=0,
        case="missing_val_seed21001",
        seed=21001,
        best_val_accuracy=None,
    )
    with pytest.raises(ValueError, match="best_val_accuracy"):
        collect_finetune_hpo_rows(root_missing)

    root_no_checkpoint = tmp_path / "stage_h_no_checkpoint"
    _write_trial(
        root_no_checkpoint,
        machine="work",
        gpu=0,
        trial_index=0,
        case="missing_checkpoint_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        checkpoint_path="",
    )
    with pytest.raises(ValueError, match="checkpoint_path"):
        collect_finetune_hpo_rows(root_no_checkpoint)


def test_validation_summary_rejects_duplicate_family_seed_rows(tmp_path: Path) -> None:
    root_a = tmp_path / "stage_h_wave_a"
    root_b = tmp_path / "stage_h_wave_b"
    for root, value in [(root_a, 0.70), (root_b, 0.75)]:
        _write_trial(
            root,
            machine="work",
            gpu=0,
            trial_index=0,
            case="baseline_cosine_seed21001",
            seed=21001,
            best_val_accuracy=value,
        )

    with pytest.raises(ValueError, match="duplicate family/seed"):
        collect_finetune_hpo_rows([root_a, root_b])


def test_write_csv_markdown_and_cli_outputs(tmp_path: Path) -> None:
    root = tmp_path / "stage_h_wave"
    _write_trial(
        root,
        machine="foureyes",
        gpu=3,
        trial_index=4,
        case="low_lr_cosine_seed21003",
        seed=21003,
        best_val_accuracy=0.625,
    )
    rows = collect_finetune_hpo_rows(root)
    csv_out = tmp_path / "summary.csv"
    family_csv = tmp_path / "families.csv"
    md_out = tmp_path / "summary.md"

    write_csv(csv_out, rows)
    write_family_csv(family_csv, rows)
    write_markdown(md_out, rows, collected_roots=[root])

    with csv_out.open("r", encoding="utf-8", newline="") as handle:
        trial_rows = list(csv.DictReader(handle))
    assert trial_rows[0] == {
        "run": "stage_h_wave",
        "machine": "foureyes",
        "gpu": "3",
        "case": "low_lr_cosine_seed21003",
        "family": "low_lr_cosine",
        "seed": "21003",
        "best_val_accuracy": "0.625000",
        "train_steps": "100",
        "test_accessed": "false",
        "checkpoint_path": "outputs/low_lr_cosine_seed21003/student_best.pt",
    }
    assert list(trial_rows[0]) == TRIAL_COLUMNS

    with family_csv.open("r", encoding="utf-8", newline="") as handle:
        family_rows = list(csv.DictReader(handle))
    assert family_rows[0]["family"] == "low_lr_cosine"
    assert family_rows[0]["mean_best_val_accuracy"] == "0.625000"
    assert family_rows[0]["std_best_val_accuracy"] == "0.000000"
    markdown = md_out.read_text(encoding="utf-8")
    assert "CIFAR-10 test accessed: `false`" in markdown
    assert "low_lr_cosine" in markdown

    cli_csv = tmp_path / "cli_summary.csv"
    cli_family_csv = tmp_path / "cli_families.csv"
    cli_md = tmp_path / "cli_summary.md"
    assert (
        main(
            [
                "--collected-root",
                str(root),
                "--csv-out",
                str(cli_csv),
                "--family-csv-out",
                str(cli_family_csv),
                "--markdown-out",
                str(cli_md),
            ],
        )
        == 0
    )
    assert cli_csv.exists()
    assert cli_family_csv.exists()
    assert cli_md.exists()

    cli_expected_csv = tmp_path / "cli_expected_summary.csv"
    cli_expected_md = tmp_path / "cli_expected_summary.md"
    assert (
        main(
            [
                "--collected-root",
                str(root),
                "--csv-out",
                str(cli_expected_csv),
                "--markdown-out",
                str(cli_expected_md),
                "--expect-rows",
                "1",
            ],
        )
        == 0
    )
    assert cli_expected_csv.exists()
    assert cli_expected_md.exists()

    with pytest.raises(RuntimeError, match="expected 2 fine-tune HPO rows, found 1"):
        main(
            [
                "--collected-root",
                str(root),
                "--csv-out",
                str(tmp_path / "cli_bad_summary.csv"),
                "--markdown-out",
                str(tmp_path / "cli_bad_summary.md"),
                "--expect-rows",
                "2",
            ],
        )


def test_cli_expect_rows_supports_multi_root_family_summary(tmp_path: Path) -> None:
    root_a = tmp_path / "gc5_wave_a"
    root_b = tmp_path / "gc5_wave_b"
    _write_trial(
        root_a,
        machine="work",
        gpu=0,
        trial_index=0,
        case="official_muon_cosine_lr_base",
        seed=1337,
        best_val_accuracy=0.9148,
        train_steps=4218,
    )
    _write_trial(
        root_b,
        machine="ripper",
        gpu=1,
        trial_index=1,
        case="pace_muon_cosine_lr_base",
        seed=1337,
        best_val_accuracy=0.9136,
        train_steps=4218,
    )
    csv_out = tmp_path / "gc5_trials.csv"
    family_csv = tmp_path / "gc5_families.csv"
    md_out = tmp_path / "gc5_summary.md"

    assert (
        main(
            [
                "--collected-root",
                str(root_a),
                "--collected-root",
                str(root_b),
                "--csv-out",
                str(csv_out),
                "--family-csv-out",
                str(family_csv),
                "--markdown-out",
                str(md_out),
                "--expect-rows",
                "2",
            ],
        )
        == 0
    )

    with csv_out.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with family_csv.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert "Trial rows: `2`" in md_out.read_text(encoding="utf-8")


def test_validation_summary_rejects_test_accessed_true(tmp_path: Path) -> None:
    root = tmp_path / "stage_h_wave"
    _write_trial(
        root,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        summary_test_accessed=True,
    )

    with pytest.raises(ValueError, match="test_accessed=true"):
        collect_finetune_hpo_rows(root)


def test_validation_summary_rejects_top_level_and_result_test_access(tmp_path: Path) -> None:
    root_top = tmp_path / "stage_h_top"
    _write_trial(
        root_top,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        test_accessed=True,
        result_test_accessed=False,
    )
    with pytest.raises(ValueError, match="test_accessed=true"):
        collect_finetune_hpo_rows(root_top)

    root_result = tmp_path / "stage_h_result"
    _write_trial(
        root_result,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        test_accessed=False,
        result_test_accessed=True,
    )
    with pytest.raises(ValueError, match="test_accessed=true"):
        collect_finetune_hpo_rows(root_result)


def test_validation_summary_rejects_invalid_or_missing_test_accessed(tmp_path: Path) -> None:
    root_invalid = tmp_path / "stage_h_invalid"
    _write_trial(
        root_invalid,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        test_accessed="maybe",
        result_test_accessed=False,
    )
    with pytest.raises(ValueError, match="invalid boolean"):
        collect_finetune_hpo_rows(root_invalid)

    root_missing = tmp_path / "stage_h_missing_access"
    _write_trial(
        root_missing,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        test_accessed=None,
        result_test_accessed=None,
        hpo_test_accessed=None,
    )
    with pytest.raises(ValueError, match="no test_accessed field"):
        collect_finetune_hpo_rows(root_missing)


def test_validation_summary_rejects_hpo_summary_test_accessed_true(tmp_path: Path) -> None:
    root = tmp_path / "stage_h_wave"
    _write_trial(
        root,
        machine="work",
        gpu=0,
        trial_index=0,
        case="baseline_cosine_seed21001",
        seed=21001,
        best_val_accuracy=0.70,
        hpo_test_accessed=True,
    )

    with pytest.raises(ValueError, match="test_accessed=true"):
        collect_finetune_hpo_rows(root)


def test_infer_family_prefers_explicit_override() -> None:
    assert infer_family(
        "trial_name_seed123",
        {"overrides": {"family": "explicit_family"}},
    ) == "explicit_family"
    assert infer_family("trial_name_seed123", {"overrides": {}}) == "trial_name"
