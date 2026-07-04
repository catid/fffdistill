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
from cifar_mamba_fff.hpo.teacher_hpo import (
    CandidateFilterConfig,
    CudaKernelSmokeUnavailable,
    build_parameter_count_prefilter_pool,
    cuda_kernel_smoke_candidate,
    evaluate_hpo_candidate,
    parse_candidate_filter_config,
    resolve_hpo_run_config,
    run_teacher_hpo,
    sample_valid_hpo_candidates,
)
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


def _patch_cuda_training_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_teacher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_teacher.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(train_teacher.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(train_teacher.torch.cuda, "max_memory_allocated", lambda device=None: 123)
    monkeypatch.setattr(train_teacher.torch.cuda, "max_memory_reserved", lambda device=None: 456)


def test_default_teacher_config_parses_strict_sections() -> None:
    run_config = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)

    assert run_config.seed == 1337
    assert run_config.dataset_name == "cifar10"
    assert run_config.data.quick_smoke is True
    assert run_config.data.use_test is False
    assert run_config.data.batch_size == run_config.train.batch_size_per_gpu
    assert run_config.train.optimizer == "muon_adamw"
    assert run_config.train.precision == "bf16"
    assert run_config.model.d_model == 256
    assert run_config.model.depth == 20
    assert run_config.model.patch_size == 4
    assert run_config.model.d_state == 64
    assert run_config.model.headdim == 64
    assert run_config.model.mimo_rank == 2


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


def test_hpo_candidate_filter_config_parses_strictly() -> None:
    assert parse_candidate_filter_config({}) == CandidateFilterConfig()
    assert parse_candidate_filter_config({"candidate_filter": None}) == CandidateFilterConfig()
    assert parse_candidate_filter_config(
        {"candidate_filter": {"cuda_kernel_smoke": True, "kernel_smoke_batch_size": 2}}
    ) == CandidateFilterConfig(cuda_kernel_smoke=True, kernel_smoke_batch_size=2)
    assert parse_candidate_filter_config(
        {"candidate_filter": {"parameter_count_prefilter": True}}
    ) == CandidateFilterConfig(parameter_count_prefilter=True)

    with pytest.raises(ValueError, match="must be a mapping"):
        parse_candidate_filter_config({"candidate_filter": True})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_candidate_filter_config({"candidate_filter": {"typo": True}})
    with pytest.raises(ValueError, match="cuda_kernel_smoke must be a bool"):
        parse_candidate_filter_config({"candidate_filter": {"cuda_kernel_smoke": "false"}})
    with pytest.raises(ValueError, match="kernel_smoke_batch_size must be a positive integer"):
        parse_candidate_filter_config({"candidate_filter": {"kernel_smoke_batch_size": 1.5}})
    with pytest.raises(ValueError, match="parameter_count_prefilter must be a bool"):
        parse_candidate_filter_config({"candidate_filter": {"parameter_count_prefilter": "true"}})


def test_cuda_kernel_smoke_candidate_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    run_config = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    monkeypatch.setattr(teacher_hpo.torch.cuda, "is_available", lambda: False)

    with pytest.raises(CudaKernelSmokeUnavailable, match="requires CUDA"):
        cuda_kernel_smoke_candidate(run_config, batch_size=1)


def test_hpo_resolve_train_overrides_rebuilds_data_config() -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)

    run_config = resolve_hpo_run_config(
        base_run,
        {
            "batch_size_per_gpu": 1024,
            "num_workers": 0,
            "label_smoothing": 0.05,
            "mixup": 0.4,
            "cutmix": 0.5,
            "schedule": "wsd",
            "lr_muon": 0.03,
        },
        seed=9001,
        quick_smoke=True,
    )

    assert run_config.seed == 9001
    assert run_config.train.batch_size_per_gpu == 1024
    assert run_config.train.schedule == "wsd"
    assert run_config.train.lr_muon == pytest.approx(0.03)
    assert run_config.data.batch_size == 1024
    assert run_config.data.num_workers == 0
    assert run_config.data.seed == 9001
    assert run_config.data.split_seed == base_run.data.split_seed
    assert run_config.data.label_smoothing == pytest.approx(0.05)
    assert run_config.data.mixup == pytest.approx(0.4)
    assert run_config.data.cutmix == pytest.approx(0.5)
    assert run_config.data.use_test is False


def test_teacher_hpo_smoke_config_uses_known_kernel_safe_default() -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    hpo_config = teacher_hpo.load_yaml("configs/teacher_hpo_smoke.yaml")
    search_space = hpo_config["search_space"]
    overrides = {key: values[0] for key, values in search_space.items()}

    run_config = resolve_hpo_run_config(base_run, overrides, seed=2026, quick_smoke=True)

    assert run_config.model.d_model == 256
    assert run_config.model.depth == 20
    assert run_config.model.patch_size == 4
    assert run_config.model.d_state == 64
    assert run_config.model.headdim == 64
    assert run_config.model.mimo_rank == 2
    assert run_config.model.bidirectional is False
    assert run_config.train.batch_size_per_gpu == 512


def test_parameter_count_prefilter_pool_keeps_only_target_sized_model_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_model = Mamba3CifarConfig()

    def fake_evaluate(config: Mamba3CifarConfig) -> TeacherCandidateResult:
        accepted = config.d_model == 224 and config.depth == 20
        return TeacherCandidateResult(
            accepted=accepted,
            parameter_count=10_000_000 if accepted else 7_000_000,
            reason="accepted" if accepted else "outside target parameter range",
        )

    monkeypatch.setattr(teacher_hpo, "evaluate_teacher_candidate", fake_evaluate)

    pool = build_parameter_count_prefilter_pool(
        base_model,
        {
            "d_model": [160, 224],
            "depth": [8, 20],
            "batch_size_per_gpu": [512, 1024],
            "lr_muon_loguniform": [0.003, 0.05],
        },
    )

    assert pool == [{"d_model": 224, "depth": 20}]


def test_parameter_count_prefilter_overrides_invalid_sampled_model_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    sampled = iter(
        [
            {"d_model": 160, "depth": 8, "batch_size_per_gpu": 1024},
        ]
    )

    def fake_evaluate(config: Mamba3CifarConfig) -> TeacherCandidateResult:
        accepted = config.d_model == 224 and config.depth == 20
        return TeacherCandidateResult(
            accepted=accepted,
            parameter_count=10_000_000 if accepted else 7_000_000,
            reason="accepted" if accepted else "outside target parameter range",
        )

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: next(sampled))
    monkeypatch.setattr(teacher_hpo, "evaluate_teacher_candidate", fake_evaluate)

    candidates = sample_valid_hpo_candidates(
        base_run,
        {
            "d_model": [160, 224],
            "depth": [8, 20],
            "batch_size_per_gpu": [512, 1024],
        },
        quick_smoke=True,
        max_trials=1,
        max_attempts=1,
        rng=teacher_hpo.random.Random(123),
        event_log_path=tmp_path / "events.jsonl",
        candidate_filter=CandidateFilterConfig(parameter_count_prefilter=True),
    )

    assert len(candidates) == 1
    assert candidates[0].attempt_index == 0
    assert candidates[0].run_config.model.d_model == 224
    assert candidates[0].run_config.model.depth == 20
    assert candidates[0].run_config.data.batch_size == 1024
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == [{"accepted_model_shapes": 1, "event": "parameter_count_prefilter_pool"}]


def test_parameter_prefilter_still_rejects_known_cuda_kernel_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    sampled = iter(
        [
            {"d_model": 160, "depth": 8, "batch_size_per_gpu": 512},
            {"d_model": 160, "depth": 8, "batch_size_per_gpu": 512},
        ]
    )

    def fake_evaluate(config: Mamba3CifarConfig) -> TeacherCandidateResult:
        accepted = config.d_model == 224 and config.depth == 20
        return TeacherCandidateResult(
            accepted=accepted,
            parameter_count=10_000_000 if accepted else 7_000_000,
            reason="accepted" if accepted else "outside target parameter range",
        )

    smoke_calls = 0

    def fake_kernel_smoke(_run_config: TeacherRunConfig) -> TeacherCandidateResult:
        nonlocal smoke_calls
        smoke_calls += 1
        if smoke_calls == 1:
            return TeacherCandidateResult(
                False,
                10_000_000,
                "cuda_kernel_smoke_failed: InternalError: Failed to set the allowed dynamic shared memory size",
            )
        return TeacherCandidateResult(True, 10_000_000, "accepted")

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: next(sampled))
    monkeypatch.setattr(teacher_hpo, "evaluate_teacher_candidate", fake_evaluate)

    candidates = sample_valid_hpo_candidates(
        base_run,
        {
            "d_model": [160, 224],
            "depth": [8, 20],
            "batch_size_per_gpu": [512],
        },
        quick_smoke=True,
        max_trials=1,
        max_attempts=2,
        rng=teacher_hpo.random.Random(123),
        event_log_path=tmp_path / "events.jsonl",
        candidate_filter=CandidateFilterConfig(
            parameter_count_prefilter=True,
            cuda_kernel_smoke=True,
            kernel_smoke_batch_size=1,
        ),
        kernel_smoke_fn=fake_kernel_smoke,
    )

    assert len(candidates) == 1
    assert candidates[0].attempt_index == 1
    assert smoke_calls == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0] == {"accepted_model_shapes": 1, "event": "parameter_count_prefilter_pool"}
    assert events[1]["event"] == "rejected_pretrial"
    assert "dynamic shared memory" in events[1]["reason"]


def test_hpo_valid_resampling_does_not_count_rejected_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    sampled = iter(
        [
            {"d_model": 160},
            {"d_model": 224, "batch_size_per_gpu": 512},
            {"d_model": 288, "bidirectional": True},
            {"d_model": 224, "depth": 18, "batch_size_per_gpu": 1024},
        ]
    )

    def fake_sample(_search_space, *, rng):
        del rng
        return next(sampled)

    def fake_evaluate(config: Mamba3CifarConfig) -> TeacherCandidateResult:
        accepted = config.d_model == 224
        return TeacherCandidateResult(
            accepted=accepted,
            parameter_count=10_000_000 if accepted else 20_000_000,
            reason="accepted" if accepted else "outside target parameter range",
        )

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", fake_sample)
    monkeypatch.setattr(teacher_hpo, "evaluate_teacher_candidate", fake_evaluate)

    candidates = sample_valid_hpo_candidates(
        base_run,
        {},
        quick_smoke=True,
        max_trials=2,
        max_attempts=4,
        rng=teacher_hpo.random.Random(123),
        event_log_path=tmp_path / "events.jsonl",
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1]
    assert [candidate.attempt_index for candidate in candidates] == [1, 3]
    assert candidates[1].run_config.data.batch_size == 1024
    rejected = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in rejected] == ["rejected_pretrial", "rejected_pretrial"]


def test_hpo_kernel_smoke_rejection_does_not_consume_trial_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    sampled = iter(
        [
            {"d_model": 224, "batch_size_per_gpu": 512},
            {"d_model": 256, "batch_size_per_gpu": 1024},
        ]
    )

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: next(sampled))
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )

    def fake_kernel_smoke(run_config: TeacherRunConfig) -> TeacherCandidateResult:
        if run_config.model.d_model == 224:
            return TeacherCandidateResult(False, 10_000_000, "cuda_kernel_smoke_failed: synthetic")
        return TeacherCandidateResult(True, 10_000_000, "accepted")

    candidates = sample_valid_hpo_candidates(
        base_run,
        {},
        quick_smoke=True,
        max_trials=1,
        max_attempts=2,
        rng=teacher_hpo.random.Random(123),
        event_log_path=tmp_path / "events.jsonl",
        candidate_filter=CandidateFilterConfig(cuda_kernel_smoke=True, kernel_smoke_batch_size=1),
        kernel_smoke_fn=fake_kernel_smoke,
    )

    assert len(candidates) == 1
    assert candidates[0].trial_index == 0
    assert candidates[0].attempt_index == 1
    assert candidates[0].run_config.model.d_model == 256
    assert candidates[0].run_config.data.batch_size == 1024
    rejected = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rejected[0]["event"] == "rejected_pretrial"
    assert "cuda_kernel_smoke_failed" in rejected[0]["reason"]


def test_hpo_kernel_smoke_oom_is_pretrial_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run = load_teacher_run_config("configs/teacher_default.yaml", quick_smoke=True)
    sampled = iter([{"d_model": 224}, {"d_model": 224}])
    cleanup_calls: list[str] = []

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: next(sampled))
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )
    monkeypatch.setattr(teacher_hpo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(teacher_hpo.torch.cuda, "empty_cache", lambda: cleanup_calls.append("empty_cache"))
    monkeypatch.setattr(teacher_hpo.torch.cuda, "ipc_collect", lambda: cleanup_calls.append("ipc_collect"))
    smoke_calls = 0

    def fake_kernel_smoke(_run_config: TeacherRunConfig) -> TeacherCandidateResult:
        nonlocal smoke_calls
        smoke_calls += 1
        if smoke_calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. synthetic pretrial")
        return TeacherCandidateResult(True, 10_000_000, "accepted")

    candidates = sample_valid_hpo_candidates(
        base_run,
        {},
        quick_smoke=True,
        max_trials=1,
        max_attempts=2,
        rng=teacher_hpo.random.Random(123),
        event_log_path=tmp_path / "events.jsonl",
        candidate_filter=CandidateFilterConfig(cuda_kernel_smoke=True, kernel_smoke_batch_size=1),
        kernel_smoke_fn=fake_kernel_smoke,
    )

    assert len(candidates) == 1
    assert candidates[0].trial_index == 0
    assert candidates[0].attempt_index == 1
    assert cleanup_calls == ["empty_cache", "ipc_collect"]
    rejected = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rejected[0]["event"] == "rejected_pretrial"
    assert "OutOfMemoryError" in rejected[0]["reason"]


def test_hpo_runner_records_pruned_failed_and_successful_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "teacher.yaml"
    hpo_config = tmp_path / "hpo.yaml"
    base_config.write_text(Path("configs/teacher_default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    hpo_config.write_text(
        "prune_on: val_accuracy\nsearch_space:\n  d_model: [224]\n  depth: [18]\n",
        encoding="utf-8",
    )

    sampled = iter([{"d_model": 224}, {"d_model": 224}, {"d_model": 224}])

    def fake_sample(_search_space, *, rng):
        del rng
        return next(sampled)

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", fake_sample)
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )

    def fake_training_fn(
        run_config,
        *,
        output_dir,
        quick_smoke,
        max_train_steps,
        max_val_steps,
        save_checkpoint,
        epoch_callback,
    ):
        del run_config, quick_smoke, max_train_steps, max_val_steps, save_checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        trial_index = int(output_dir.name.rsplit("_", maxsplit=1)[1])
        if trial_index == 0:
            assert epoch_callback is not None
            epoch_callback({"epoch": 0, "val_accuracy": 0.1})
            raise AssertionError("prune callback should raise")
        if trial_index == 1:
            raise RuntimeError("synthetic failure")
        assert epoch_callback is not None
        epoch_callback({"epoch": 0, "val_accuracy": 0.9})
        return {"best_val_accuracy": 0.9, "train_steps_total": 1}

    summary = run_teacher_hpo(
        base_config_path=base_config,
        hpo_config_path=hpo_config,
        output_dir=tmp_path / "hpo",
        quick_smoke=True,
        max_trials=3,
        max_attempts=3,
        prune_min_value=0.5,
        training_fn=fake_training_fn,
    )

    assert summary["accepted_trials"] == 3
    assert summary["pruned"] == 1
    assert summary["failed_logic"] == 1
    assert summary["succeeded"] == 1
    assert summary["best_trial"] == 2
    statuses = [
        json.loads((tmp_path / "hpo" / "trials" / f"trial_{index:06d}" / "trial_summary.json").read_text(encoding="utf-8"))["status"]
        for index in range(3)
    ]
    assert statuses == ["pruned", "failed_logic", "succeeded"]
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "hpo" / "teacher_hpo_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "trial_pruned" in events


def test_hpo_runner_fails_when_no_valid_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "teacher.yaml"
    hpo_config = tmp_path / "hpo.yaml"
    base_config.write_text(Path("configs/teacher_default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    hpo_config.write_text("prune_on: val_accuracy\nsearch_space:\n  d_model: [160]\n", encoding="utf-8")
    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: {"d_model": 160})
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(False, 8_999_999, "outside target parameter range"),
    )

    with pytest.raises(RuntimeError, match="zero valid candidates"):
        run_teacher_hpo(
            base_config_path=base_config,
            hpo_config_path=hpo_config,
            output_dir=tmp_path / "hpo",
            quick_smoke=True,
            max_trials=1,
            max_attempts=1,
            training_fn=lambda *args, **kwargs: pytest.fail("training should not run"),
        )

    summary = json.loads((tmp_path / "hpo" / "teacher_hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed_zero_candidates"
    assert summary["accepted_trials"] == 0


def test_hpo_runner_fails_when_no_trials_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "teacher.yaml"
    hpo_config = tmp_path / "hpo.yaml"
    base_config.write_text(Path("configs/teacher_default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    hpo_config.write_text("prune_on: val_accuracy\nsearch_space:\n  d_model: [224]\n", encoding="utf-8")
    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: {"d_model": 224})
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )

    def failing_training_fn(*args, **kwargs):
        raise RuntimeError("synthetic logic failure")

    with pytest.raises(RuntimeError, match="zero successful trials"):
        run_teacher_hpo(
            base_config_path=base_config,
            hpo_config_path=hpo_config,
            output_dir=tmp_path / "hpo",
            quick_smoke=True,
            max_trials=1,
            max_attempts=1,
            training_fn=failing_training_fn,
        )

    summary = json.loads((tmp_path / "hpo" / "teacher_hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed_zero_successes"
    assert summary["failed_logic"] == 1
    assert summary["succeeded"] == 0


def test_hpo_runner_classifies_oom_and_cleans_cache_before_next_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "teacher.yaml"
    hpo_config = tmp_path / "hpo.yaml"
    base_config.write_text(Path("configs/teacher_default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    hpo_config.write_text("prune_on: val_accuracy\nsearch_space:\n  d_model: [224]\n", encoding="utf-8")
    sampled = iter([{"d_model": 224}, {"d_model": 224}])
    cleanup_calls: list[str] = []
    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", lambda _search_space, *, rng: next(sampled))
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )
    monkeypatch.setattr(teacher_hpo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(teacher_hpo.torch.cuda, "empty_cache", lambda: cleanup_calls.append("empty_cache"))
    monkeypatch.setattr(teacher_hpo.torch.cuda, "ipc_collect", lambda: cleanup_calls.append("ipc_collect"))

    def training_fn(run_config, *, output_dir, **kwargs):
        del run_config, kwargs
        trial_index = int(output_dir.name.rsplit("_", maxsplit=1)[1])
        if trial_index == 0:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. synthetic")
        return {"best_val_accuracy": 0.8, "train_steps_total": 1}

    summary = run_teacher_hpo(
        base_config_path=base_config,
        hpo_config_path=hpo_config,
        output_dir=tmp_path / "hpo",
        quick_smoke=True,
        max_trials=2,
        max_attempts=2,
        training_fn=training_fn,
    )

    assert summary["status"] == "completed"
    assert summary["failed_oom"] == 1
    assert summary["succeeded"] == 1
    assert cleanup_calls == ["empty_cache", "ipc_collect"]
    statuses = [
        json.loads((tmp_path / "hpo" / "trials" / f"trial_{index:06d}" / "trial_summary.json").read_text(encoding="utf-8"))["status"]
        for index in range(2)
    ]
    assert statuses == ["failed_oom", "succeeded"]


def test_hpo_seed_override_controls_sampling_and_trial_run_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = tmp_path / "teacher.yaml"
    hpo_config = tmp_path / "hpo.yaml"
    base_config.write_text(Path("configs/teacher_default.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    hpo_config.write_text("prune_on: val_accuracy\nsearch_space:\n  d_model: [224]\n", encoding="utf-8")
    rng_values: list[float] = []
    run_seeds: list[tuple[int, int, int]] = []

    def fake_sample(_search_space, *, rng):
        rng_values.append(rng.random())
        return {"d_model": 224}

    def training_fn(run_config, **kwargs):
        del kwargs
        run_seeds.append((run_config.seed, run_config.data.seed, run_config.data.split_seed))
        return {"best_val_accuracy": 0.7, "train_steps_total": 1}

    monkeypatch.setattr(teacher_hpo, "sample_teacher_overrides", fake_sample)
    monkeypatch.setattr(
        teacher_hpo,
        "evaluate_teacher_candidate",
        lambda config: TeacherCandidateResult(True, 10_000_000, "accepted"),
    )

    summary = run_teacher_hpo(
        base_config_path=base_config,
        hpo_config_path=hpo_config,
        output_dir=tmp_path / "hpo",
        quick_smoke=True,
        max_trials=2,
        max_attempts=2,
        seed=9001,
        training_fn=training_fn,
    )

    expected_rng = teacher_hpo.random.Random(9001)
    assert rng_values == [expected_rng.random(), expected_rng.random()]
    assert run_seeds == [(9001, 9001, 1337), (9002, 9002, 1337)]
    assert summary["seed"] == 9001


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
            "--seed",
            "9001",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_dir / "run_context.json").read_text(encoding="utf-8"))
    assert payload["quick_smoke"] is True
    assert payload["seed"] == 9001
    assert payload["smoke_mode"] == "metadata"
    assert payload["resolved_config"]["seed"] == 9001
    assert payload["resolved_config"]["data"]["seed"] == 9001
    assert payload["resolved_config"]["data"]["split_seed"] == 1337
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
    seed_calls: list[int] = []

    train_teacher.write_run_context(
        output_dir=output_dir,
        context=context,
        config_path=Path("configs/teacher_default.yaml"),
        raw_config={"dataset": {"name": "cifar10", "use_test": False}},
        run_config=run_config,
        smoke_mode="train",
    )

    _patch_cuda_training_runtime(monkeypatch)
    monkeypatch.setattr(train_teacher, "seed_everything", lambda seed: seed_calls.append(seed))
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
            "train_batch_size": 2.0,
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
        "train_images_seen",
        "epoch_train_images_seen",
        "epoch_train_elapsed_seconds",
        "epoch_train_images_per_second",
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
    assert metrics["train_images_seen"] == 2
    assert metrics["epoch_train_images_seen"] == 2
    assert metrics["epoch_train_elapsed_seconds"] > 0.0
    assert metrics["epoch_train_images_per_second"] > 0.0
    assert metrics["val_accuracy"] == pytest.approx(0.25)

    assert summary["parameter_count"] == 10_000_000
    assert summary["best_val_accuracy"] == pytest.approx(0.25)
    assert summary["metrics_path"] == str(output_dir / "metrics.jsonl")
    assert summary["batch_size_per_gpu"] == 2
    assert summary["train_images_seen"] == 2
    assert summary["train_images_per_second"] > 0.0
    assert summary["train_elapsed_seconds"] > 0.0
    assert summary["train_images_per_second_train_only"] > 0.0
    assert "peak_cuda_memory_allocated_bytes" in summary
    assert "peak_cuda_memory_reserved_bytes" in summary
    summary_payload = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["quick_smoke"] is True
    assert summary_payload["train_images_seen"] == 2
    assert seed_calls[0] == run_config.seed

    with pytest.raises(FileExistsError, match="metrics"):
        train_teacher.run_teacher_training(
            run_config,
            output_dir=output_dir,
            quick_smoke=True,
            save_checkpoint=False,
        )


def test_teacher_checkpoint_is_best_only_and_uses_atomic_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = TeacherRunConfig(
        seed=77,
        dataset_name="cifar10",
        data=Cifar10DataConfig(
            data_dir=tmp_path / "data",
            batch_size=2,
            num_workers=0,
            quick_smoke=False,
            download=False,
            use_test=False,
        ),
        model=Mamba3CifarConfig(),
        train=TeacherTrainConfig(
            epochs=2,
            batch_size_per_gpu=2,
            num_workers=0,
            mixup=0.0,
            cutmix=0.0,
        ),
    )

    class _TinyModel:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    val_accuracies = iter([0.5, 0.4])
    save_calls: list[tuple[Path, dict[str, object]]] = []

    _patch_cuda_training_runtime(monkeypatch)
    monkeypatch.setattr(train_teacher, "seed_everything", lambda seed: None)
    monkeypatch.setattr(train_teacher, "build_cifar10_loaders", lambda config: ([object()], [object()]))
    monkeypatch.setattr(
        train_teacher,
        "build_teacher_model",
        lambda model_config, *, device=None, enforce_target_params=True: (_TinyModel(), 10_000_000),
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
            "train_batch_size": 2.0,
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
            "val_accuracy": next(val_accuracies),
            "val_steps": 1.0,
        },
    )
    monkeypatch.setattr(
        train_teacher,
        "save_teacher_checkpoint_atomic",
        lambda path, payload: save_calls.append((path, dict(payload))),
    )

    summary = train_teacher.run_teacher_training(
        run_config,
        output_dir=tmp_path / "teacher",
        quick_smoke=False,
        save_checkpoint=True,
    )

    assert summary["best_val_accuracy"] == pytest.approx(0.5)
    assert len(save_calls) == 1
    assert save_calls[0][0] == tmp_path / "teacher" / "teacher_best.pt"
    assert save_calls[0][1]["metrics"]["val_accuracy"] == pytest.approx(0.5)


def test_save_teacher_checkpoint_atomic_uses_tmp_then_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "teacher_best.pt"
    calls: list[tuple[str, Path, Path | None]] = []

    def fake_torch_save(payload: object, path: Path) -> None:
        calls.append(("save", path, None))
        path.write_text("tmp", encoding="utf-8")

    def fake_replace(src: Path, dst: Path) -> None:
        calls.append(("replace", src, dst))
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        src.unlink()

    monkeypatch.setattr(train_teacher.torch, "save", fake_torch_save)
    monkeypatch.setattr(train_teacher.os, "replace", fake_replace)

    train_teacher.save_teacher_checkpoint_atomic(checkpoint_path, {"ok": True})

    tmp_path_expected = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    assert calls == [
        ("save", tmp_path_expected, None),
        ("replace", tmp_path_expected, checkpoint_path),
    ]
    assert checkpoint_path.read_text(encoding="utf-8") == "tmp"
    assert not tmp_path_expected.exists()
