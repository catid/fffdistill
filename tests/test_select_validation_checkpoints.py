from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cifar_mamba_fff.cluster import MachineSpec
from cifar_mamba_fff.select_validation_checkpoints import (
    SKIPPED_SHA256,
    ValidationTrial,
    checkpoint_sha256,
    main,
    read_validation_trials,
    select_trials,
)


def _write_machines(path: Path, *, workdir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "machines:",
                "  work:",
                "    host: localhost",
                f"    workdir: {workdir}",
                "    gpus: 2",
                "    role: local",
                "  remote:",
                "    host: remote-box",
                "    workdir: /srv/fff",
                "    gpus: 1",
                "    role: remote",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_trials(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run",
        "machine",
        "gpu",
        "case",
        "family",
        "seed",
        "best_val_accuracy",
        "test_accessed",
        "checkpoint_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_family_selection_writes_local_hash_records_and_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    checkpoint_a = repo / "outputs/a/student_best.pt"
    checkpoint_b = repo / "outputs/b/student_best.pt"
    checkpoint_a.parent.mkdir(parents=True)
    checkpoint_b.parent.mkdir(parents=True)
    checkpoint_a.write_bytes(b"checkpoint-a")
    checkpoint_b.write_bytes(b"checkpoint-b")
    machines = tmp_path / "machines.yaml"
    _write_machines(machines, workdir=repo)
    trials_csv = tmp_path / "validation_trials.csv"
    _write_trials(
        trials_csv,
        [
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "baseline_seed1",
                "family": "baseline",
                "seed": 1,
                "best_val_accuracy": 0.91,
                "test_accessed": "false",
                "checkpoint_path": "outputs/a/student_best.pt",
            },
            {
                "run": "wave",
                "machine": "work",
                "gpu": 1,
                "case": "baseline_seed2",
                "family": "baseline",
                "seed": 2,
                "best_val_accuracy": 0.92,
                "test_accessed": "false",
                "checkpoint_path": "outputs/b/student_best.pt",
            },
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "other_seed1",
                "family": "other",
                "seed": 1,
                "best_val_accuracy": 0.99,
                "test_accessed": "false",
                "checkpoint_path": "outputs/a/student_best.pt",
            },
        ],
    )
    selection_dir = tmp_path / "selection"
    manifest = tmp_path / "launch_manifest.jsonl"

    assert (
        main(
            [
                "--trials-csv",
                str(trials_csv),
                "--selection-dir",
                str(selection_dir),
                "--manifest-out",
                str(manifest),
                "--machines",
                str(machines),
                "--family",
                "baseline",
            ],
        )
        == 0
    )

    rows = _jsonl(manifest)
    assert [row["case"] for row in rows] == ["baseline_seed1", "baseline_seed2"]
    expected_sha = hashlib.sha256(b"checkpoint-a").hexdigest()
    assert rows[0] == {
        "machine": "work",
        "gpu": 0,
        "case": "baseline_seed1",
        "family": "baseline",
        "seed": 1,
        "best_val_accuracy": 0.91,
        "checkpoint_path": "outputs/a/student_best.pt",
        "checkpoint_sha256": expected_sha,
        "selection_record": str(selection_dir / "baseline_seed1_selection.json"),
    }
    record = json.loads((selection_dir / "baseline_seed1_selection.json").read_text(encoding="utf-8"))
    assert record["status"] == "succeeded"
    assert record["selected_for_final_eval"] is True
    assert record["test_accessed"] is False
    assert record["summary"]["checkpoint_sha256"] == expected_sha
    assert record["summary"]["best_val_accuracy"] == 0.91


def test_top_families_can_rank_by_mean_or_best_validation_accuracy(tmp_path: Path) -> None:
    trials_csv = tmp_path / "validation_trials.csv"
    _write_trials(
        trials_csv,
        [
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "spiky_seed1",
                "family": "spiky",
                "seed": 1,
                "best_val_accuracy": 0.99,
                "test_accessed": "false",
                "checkpoint_path": "outputs/spiky1.pt",
            },
            {
                "run": "wave",
                "machine": "work",
                "gpu": 1,
                "case": "spiky_seed2",
                "family": "spiky",
                "seed": 2,
                "best_val_accuracy": 0.10,
                "test_accessed": "false",
                "checkpoint_path": "outputs/spiky2.pt",
            },
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "steady_seed1",
                "family": "steady",
                "seed": 1,
                "best_val_accuracy": 0.80,
                "test_accessed": "false",
                "checkpoint_path": "outputs/steady1.pt",
            },
            {
                "run": "wave",
                "machine": "work",
                "gpu": 1,
                "case": "steady_seed2",
                "family": "steady",
                "seed": 2,
                "best_val_accuracy": 0.81,
                "test_accessed": "false",
                "checkpoint_path": "outputs/steady2.pt",
            },
        ],
    )
    trials = read_validation_trials(trials_csv)

    by_mean = select_trials(trials, top_family_count=1, top_family_metric="mean")
    by_best = select_trials(trials, top_family_count=1, top_family_metric="best")

    assert {trial.family for trial in by_mean} == {"steady"}
    assert {trial.family for trial in by_best} == {"spiky"}


def test_rejects_validation_csv_with_test_accessed_true(tmp_path: Path) -> None:
    trials_csv = tmp_path / "validation_trials.csv"
    _write_trials(
        trials_csv,
        [
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "leaky",
                "family": "baseline",
                "seed": 1,
                "best_val_accuracy": 0.91,
                "test_accessed": "true",
                "checkpoint_path": "outputs/a.pt",
            }
        ],
    )

    with pytest.raises(ValueError, match="test_accessed=true"):
        read_validation_trials(trials_csv)


def test_skip_hash_writes_explicit_placeholder_without_checkpoint_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    machines = tmp_path / "machines.yaml"
    _write_machines(machines, workdir=repo)
    trials_csv = tmp_path / "validation_trials.csv"
    _write_trials(
        trials_csv,
        [
            {
                "run": "wave",
                "machine": "work",
                "gpu": 0,
                "case": "baseline_seed1",
                "family": "baseline",
                "seed": 1,
                "best_val_accuracy": 0.91,
                "test_accessed": "false",
                "checkpoint_path": "outputs/missing.pt",
            }
        ],
    )
    selection_dir = tmp_path / "selection"
    manifest = tmp_path / "launch_manifest.jsonl"

    assert (
        main(
            [
                "--trials-csv",
                str(trials_csv),
                "--selection-dir",
                str(selection_dir),
                "--manifest-out",
                str(manifest),
                "--machines",
                str(machines),
                "--family",
                "baseline",
                "--skip-hash",
            ],
        )
        == 0
    )

    assert _jsonl(manifest)[0]["checkpoint_sha256"] == SKIPPED_SHA256
    record = json.loads((selection_dir / "baseline_seed1_selection.json").read_text(encoding="utf-8"))
    assert record["summary"]["checkpoint_sha256"] == SKIPPED_SHA256


def test_remote_hash_uses_ssh_host_and_configured_workdir() -> None:
    trial = ValidationTrial(
        run="wave",
        machine="remote",
        gpu=0,
        case="remote_seed1",
        family="remote_family",
        seed=1,
        best_val_accuracy=0.9,
        test_accessed=False,
        checkpoint_path="outputs/remote/student_best.pt",
        row_index=2,
    )
    spec = MachineSpec(
        name="remote",
        host="remote-box",
        gpus=1,
        role="remote",
        workdir="/srv/fff",
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0, stdout="a" * 64 + "\n", stderr="")

    assert checkpoint_sha256(trial, {"remote": spec}, runner=fake_run, timeout_s=7) == "a" * 64
    assert calls
    command = calls[0]
    assert command[:2] == ["ssh", "-o"]
    assert "remote-box" in command
    remote_command = command[-1]
    assert "cd /srv/fff" in remote_command
    assert "outputs/remote/student_best.pt" in remote_command
