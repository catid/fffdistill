from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import cifar_mamba_fff.hpo.teacher_hpo as teacher_hpo
import cifar_mamba_fff.train_teacher as train_teacher
from cifar_mamba_fff.data import Cifar10DataConfig
from cifar_mamba_fff.hpo.teacher_hpo import evaluate_hpo_candidate
from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig
from cifar_mamba_fff.train_teacher import (
    TeacherCandidateResult,
    TeacherRunConfig,
    TeacherTrainConfig,
    build_lr_scheduler,
    evaluate_teacher_candidate,
    load_teacher_run_config,
    parse_teacher_run_config,
)


def test_default_teacher_config_parses_strict_sections() -> None:
    run_config = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)

    assert run_config.seed == 1337
    assert run_config.dataset_name == "cifar10"
    assert run_config.data.quick_smoke is True
    assert run_config.data.use_test is False
    assert run_config.data.batch_size == run_config.train.batch_size_per_gpu
    assert run_config.train.optimizer == "muon_adamw"
    assert run_config.train.precision == "bf16"
    assert run_config.model.d_model == 224


def test_teacher_config_rejects_unknown_keys_and_test_access() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        parse_teacher_run_config({"seed": 1, "model": {"d_model": 224, "surprise": 1}}, quick_smoke=True)

    with pytest.raises(ValueError, match="test access"):
        parse_teacher_run_config(
            {
                "seed": 1,
                "dataset": {"name": "cifar10", "use_test": True},
                "model": {},
                "train": {},
            },
            quick_smoke=True,
        )

    with pytest.raises(ValueError, match="train-owned keys"):
        parse_teacher_run_config(
            {
                "seed": 1,
                "dataset": {"name": "cifar10", "batch_size": 8},
                "model": {},
                "train": {},
            },
            quick_smoke=True,
        )


def test_teacher_train_config_validates_grad_clip() -> None:
    with pytest.raises(ValueError, match="grad_clip_norm"):
        TeacherTrainConfig(grad_clip_norm=0.0).validate()
    TeacherTrainConfig(grad_clip_norm=1.0).validate()


def test_default_teacher_config_parameter_count_is_accepted() -> None:
    pytest.importorskip("mamba_ssm")
    run_config = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)

    result = evaluate_teacher_candidate(run_config.model)

    assert result.accepted is True
    assert result.parameter_count is not None
    assert 9_000_000 <= result.parameter_count <= 11_000_000


def test_bidirectional_default_shape_is_filtered_before_hpo_training() -> None:
    pytest.importorskip("mamba_ssm")
    run_config = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)

    candidate, result = evaluate_hpo_candidate(run_config.model, {"bidirectional": True})

    assert candidate.bidirectional is True
    assert result.accepted is False
    assert result.parameter_count is not None
    assert result.parameter_count > candidate.target_max_params
    assert "outside" in result.reason


def test_hpo_candidate_filter_rejects_out_of_band_sampled_configs_without_mamba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_model = Mamba3CifarConfig()
    counts = {
        (160, 8, False): 8_999_999,
        (224, 18, False): 9_000_000,
        (288, 24, True): 11_000_001,
        (192, 12, True): 11_000_000,
    }

    def fake_evaluate_teacher_candidate(config: Mamba3CifarConfig) -> TeacherCandidateResult:
        count = counts[(config.d_model, config.depth, config.bidirectional)]
        accepted = config.target_min_params <= count <= config.target_max_params
        return TeacherCandidateResult(
            accepted=accepted,
            parameter_count=count,
            reason="accepted" if accepted else "outside target parameter range",
        )

    monkeypatch.setattr(teacher_hpo, "evaluate_teacher_candidate", fake_evaluate_teacher_candidate)

    rejected_small = evaluate_hpo_candidate(base_model, {"d_model": 160, "depth": 8})
    accepted_floor = evaluate_hpo_candidate(base_model, {"d_model": 224, "depth": 18})
    rejected_bidirectional = evaluate_hpo_candidate(
        base_model,
        {"d_model": 288, "depth": 24, "bidirectional": True},
    )
    accepted_bidirectional = evaluate_hpo_candidate(
        base_model,
        {"d_model": 192, "depth": 12, "bidirectional": True},
    )

    assert rejected_small[1].accepted is False
    assert rejected_small[1].parameter_count == 8_999_999
    assert accepted_floor[1].accepted is True
    assert accepted_floor[1].parameter_count == 9_000_000
    assert rejected_bidirectional[0].bidirectional is True
    assert rejected_bidirectional[1].accepted is False
    assert rejected_bidirectional[1].parameter_count == 11_000_001
    assert accepted_bidirectional[0].bidirectional is True
    assert accepted_bidirectional[1].accepted is True
    assert accepted_bidirectional[1].parameter_count == 11_000_000


def test_hpo_candidate_filter_rejects_unknown_override_keys() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate_hpo_candidate(load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True).model, {"typo": 1})


def test_scheduler_supports_cosine_and_wsd() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    cosine = build_lr_scheduler(
        optimizer,
        TeacherTrainConfig(epochs=2, warmup_epochs=1, schedule="cosine"),
        steps_per_epoch=2,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    optimizer.step()
    cosine.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    optimizer = torch.optim.SGD([parameter], lr=1.0)
    wsd = build_lr_scheduler(
        optimizer,
        TeacherTrainConfig(epochs=4, warmup_epochs=1, schedule="wsd", wsd_stable_fraction=0.75),
        steps_per_epoch=1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    optimizer.step()
    wsd.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_metadata_smoke_writes_resolved_config_without_touching_data(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    output_dir = tmp_path / "teacher"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cifar_mamba_fff.train_teacher",
            "--quick-smoke",
            "true",
            "--smoke-mode",
            "metadata",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_dir / "run_context.json").read_text(encoding="utf-8"))
    assert payload["quick_smoke"] is True
    assert payload["smoke_mode"] == "metadata"
    assert payload["resolved_config"]["data"]["use_test"] is False
    assert payload["resolved_config"]["model"]["target_min_params"] == 9_000_000


def test_run_context_and_training_metrics_output_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = TeacherRunConfig(
        seed=2026,
        dataset_name="cifar10",
        data=Cifar10DataConfig(
            data_dir=tmp_path / "data",
            batch_size=2,
            num_workers=0,
            quick_smoke=True,
            download=False,
            use_test=False,
        ),
        model=Mamba3CifarConfig(),
        train=TeacherTrainConfig(
            epochs=1,
            batch_size_per_gpu=2,
            num_workers=0,
            mixup=0.0,
            cutmix=0.0,
        ),
    )
    output_dir = tmp_path / "teacher"
    context = train_teacher.RunContext(output_dir, seed=run_config.seed, quick_smoke=True)
    context.prepare(configure_cuda=False)

    train_teacher.write_run_context(
        output_dir=output_dir,
        context=context,
        config_path=Path("configs/teacher_default.yaml"),
        raw_config={"dataset": {"name": "cifar10", "use_test": False}},
        run_config=run_config,
        smoke_mode="train",
    )

    monkeypatch.setattr(train_teacher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_teacher, "build_cifar10_loaders", lambda config: ([object()], [object()]))
    monkeypatch.setattr(
        train_teacher,
        "build_teacher_model",
        lambda model_config, *, device=None, enforce_target_params=True: (object(), 10_000_000),
    )
    monkeypatch.setattr(
        train_teacher,
        "build_muon_adamw_optimizer",
        lambda model, train_config, *, assignment_log_path=None: (
            object(),
            {
                "muon_tensors": 1,
                "adamw_tensors": 1,
                "muon_parameters": 9_000_000,
                "adamw_parameters": 1_000_000,
            },
        ),
    )
    monkeypatch.setattr(
        train_teacher,
        "build_lr_scheduler",
        lambda optimizer, train_config, *, steps_per_epoch: object(),
    )
    monkeypatch.setattr(
        train_teacher,
        "_train_one_step",
        lambda model, batch, optimizer, scheduler, run_config, device: {
            "train_loss": 2.3,
            "train_accuracy_hard_labels": 0.125,
            "lr_muon": 0.02,
            "lr_adamw": 0.001,
        },
    )
    monkeypatch.setattr(
        train_teacher,
        "_evaluate_steps",
        lambda model, loader, run_config, device, *, max_steps: {
            "val_loss": 2.2,
            "val_accuracy": 0.25,
            "val_steps": 1.0,
        },
    )

    summary = train_teacher.run_teacher_training(
        run_config,
        output_dir=output_dir,
        quick_smoke=True,
        save_checkpoint=False,
    )

    run_context = json.loads((output_dir / "run_context.json").read_text(encoding="utf-8"))
    assert {
        "output_dir",
        "seed",
        "quick_smoke",
        "git_commit",
        "cuda_visible_devices",
        "config_path",
        "smoke_mode",
        "config",
        "resolved_config",
    } <= set(run_context)
    assert run_context["resolved_config"]["data"]["use_test"] is False

    metrics_lines = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics_lines) == 1
    metrics = json.loads(metrics_lines[0])
    assert {
        "phase",
        "epoch",
        "parameter_count",
        "train_steps_total",
        "epoch_seconds",
        "train_loss",
        "train_accuracy_hard_labels",
        "lr_muon",
        "lr_adamw",
        "val_loss",
        "val_accuracy",
        "val_steps",
    } <= set(metrics)
    assert metrics["phase"] == "teacher_train"
    assert metrics["parameter_count"] == 10_000_000
    assert metrics["train_steps_total"] == 1
    assert metrics["val_accuracy"] == pytest.approx(0.25)

    assert summary["parameter_count"] == 10_000_000
    assert summary["best_val_accuracy"] == pytest.approx(0.25)
    assert summary["metrics_path"] == str(output_dir / "metrics.jsonl")
    summary_payload = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["quick_smoke"] is True

    with pytest.raises(FileExistsError, match="metrics"):
        train_teacher.run_teacher_training(
            run_config,
            output_dir=output_dir,
            quick_smoke=True,
            save_checkpoint=False,
        )
