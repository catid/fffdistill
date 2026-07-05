from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from cifar_mamba_fff.finetune_student import (
    _reached_train_step_limit,
    assemble_fff_student_from_artifacts,
    build_student_model,
    kd_loss,
    load_finetune_run_config,
    parse_finetune_run_config,
)
from cifar_mamba_fff.hpo.finetune_hpo import (
    run_finetune_hpo_trials,
    run_finetune_trial_command,
    write_finetune_hpo_trial_plan,
)
from cifar_mamba_fff.models.baseline_linears import (
    LowRankLinear,
    SharedOnlyLinear,
    SmallerDenseLinear,
)
from cifar_mamba_fff.models.replacement import make_fff_replacement
from cifar_mamba_fff.models.sparse_row_column import (
    CheckerboardSparseMoELinear,
    CoupledRowColumnLinear,
    SparseColumnLinear,
    SparseRowLinear,
)


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TinyConfigModel(nn.Module):
    def __init__(self, config: dict[str, int] | None = None) -> None:
        super().__init__()
        self.config = config or {"width": 64}
        width = int(self.config["width"])
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RectangularConfigModel(nn.Module):
    def __init__(self, config: dict[str, int] | None = None) -> None:
        super().__init__()
        self.config = config or {"in_features": 384, "out_features": 192}
        in_features = int(self.config["in_features"])
        out_features = int(self.config["out_features"])
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _minimal_config(tmp_path: Path) -> dict[str, object]:
    return {
        "seed": 7,
        "teacher_checkpoint": str(tmp_path / "teacher.pt"),
        "dataset": {
            "name": "cifar10",
            "data_dir": str(tmp_path / "cifar10"),
            "train_size": 128,
            "val_size": 32,
        },
        "student": {
            "source": "distill_artifacts",
            "distill_artifact_root": str(tmp_path / "artifacts"),
            "distill_config": str(tmp_path / "distill.yaml"),
            "min_in_features": 1,
            "min_out_features": 1,
        },
        "train": {
            "epochs": 1,
            "batch_size_per_gpu": 8,
            "num_workers": 0,
            "optimizer": "muon_adamw",
            "schedule": "wsd",
        },
        "losses": {
            "kd_temperature": 2.0,
            "lambda_kd": 0.7,
            "lambda_ce": 0.3,
            "lambda_hidden": 0.0,
            "lambda_balance": 0.0,
        },
    }


def test_finetune_default_config_parses_train_val_only() -> None:
    config = load_finetune_run_config("configs/finetune_default.yaml", quick_smoke=True)

    assert config.teacher_checkpoint is not None
    assert config.data.use_test is False
    assert config.student.source == "distill_artifacts"
    assert config.train.optimizer == "muon_adamw"
    assert config.losses.kd_temperature == pytest.approx(4.0)


def test_dense_copy_baseline_config_parses_validation_only_no_balance() -> None:
    config = load_finetune_run_config("configs/finetune_dense_copy_baseline.yaml", quick_smoke=False)

    assert config.teacher_checkpoint is not None
    assert config.data.use_test is False
    assert config.student.source == "dense_copy"
    assert config.student.allow_dense_copy is True
    assert config.student.require_full_replacement is False
    assert config.student.distill_artifact_root is None
    assert config.losses.lambda_balance == pytest.approx(0.0)
    assert config.losses.balance_recipe == "none"


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("configs/finetune_low_rank_baseline.yaml", "matched_low_rank"),
        ("configs/finetune_shared_only_baseline.yaml", "matched_shared_only"),
        ("configs/finetune_smaller_dense_baseline.yaml", "matched_smaller_dense"),
    ],
)
def test_matched_linear_baseline_yaml_parses_validation_only_no_balance(path: str, source: str) -> None:
    config = load_finetune_run_config(path, quick_smoke=False)

    assert config.teacher_checkpoint is not None
    assert config.data.use_test is False
    assert config.student.source == source
    assert config.student.require_full_replacement is True
    assert config.student.allow_matched_linear_baseline is True
    assert config.student.baseline_budget_source == "fff_config"
    assert config.student.distill_config == Path("configs/fff_distill_stage_f.yaml")
    assert config.student.baseline_parameter_budget_fraction == pytest.approx(0.50)
    assert config.losses.lambda_balance == pytest.approx(0.0)
    assert config.losses.balance_recipe == "none"


@pytest.mark.parametrize(
    ("base_path", "hpo_path", "case_prefix"),
    [
        (
            "configs/finetune_dense_copy_baseline.yaml",
            "configs/finetune_dense_copy_baseline_hpo.yaml",
            "dense_copy",
        ),
        (
            "configs/finetune_low_rank_baseline.yaml",
            "configs/finetune_low_rank_baseline_hpo.yaml",
            "low_rank",
        ),
        (
            "configs/finetune_shared_only_baseline.yaml",
            "configs/finetune_shared_only_baseline_hpo.yaml",
            "shared_only",
        ),
        (
            "configs/finetune_smaller_dense_baseline.yaml",
            "configs/finetune_smaller_dense_baseline_hpo.yaml",
            "smaller_dense",
        ),
    ],
)
def test_baseline_hpo_configs_plan_three_validation_seeds(
    tmp_path: Path,
    base_path: str,
    hpo_path: str,
    case_prefix: str,
) -> None:
    summary = write_finetune_hpo_trial_plan(
        base_config=yaml.safe_load(Path(base_path).read_text(encoding="utf-8")),
        hpo_config=yaml.safe_load(Path(hpo_path).read_text(encoding="utf-8")),
        output_dir=tmp_path / case_prefix,
        max_trials=3,
    )

    assert summary["test_accessed"] is False
    assert summary["accepted_trials"] == 3
    assert [trial["case"] for trial in summary["trials"]] == [
        f"{case_prefix}_seed21001",
        f"{case_prefix}_seed21002",
        f"{case_prefix}_seed21003",
    ]
    for seed, trial in zip([21001, 21002, 21003], summary["trials"], strict=True):
        trial_config = yaml.safe_load(Path(str(trial["config_path"])).read_text(encoding="utf-8"))
        assert trial_config["seed"] == seed
        assert trial_config["dataset"].get("use_test") is None
        assert trial_config["losses"]["lambda_balance"] == 0.0
        assert trial_config["losses"]["balance_recipe"] == "none"


def test_dense_copy_build_path_copies_teacher_without_replacements() -> None:
    config = load_finetune_run_config("configs/finetune_dense_copy_baseline.yaml", quick_smoke=True)
    teacher = TinyConfigModel({"width": 64})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "dense_copy"
    assert result.replacement_count == 0
    assert result.eligible_count == 1
    assert result.manifest == []
    assert result.model is not teacher
    for name, value in teacher.state_dict().items():
        assert torch.equal(result.model.state_dict()[name], value)


@pytest.mark.parametrize("source", ["matched_low_rank", "matched_shared_only", "matched_smaller_dense"])
def test_matched_linear_baseline_configs_parse(tmp_path: Path, source: str) -> None:
    raw = _minimal_config(tmp_path)
    raw["student"] = {
        "source": source,
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_parameter_budget_fraction": 0.5,
    }

    config = parse_finetune_run_config(raw, quick_smoke=True)

    assert config.student.source == source
    assert config.student.allow_matched_linear_baseline is True
    assert config.student.baseline_parameter_budget_fraction == pytest.approx(0.5)
    assert config.data.use_test is False


def test_matched_linear_baseline_requires_explicit_opt_in(tmp_path: Path) -> None:
    raw = _minimal_config(tmp_path)
    raw["student"] = {
        "source": "matched_low_rank",
        "min_in_features": 1,
        "min_out_features": 1,
    }

    with pytest.raises(ValueError, match="allow_matched_linear_baseline"):
        parse_finetune_run_config(raw, quick_smoke=True)


def test_generated_baseline_hpo_overrides_student_section(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {
        "cases": [
            {
                "name": "sparse_row",
                "student_source": "sparse_row",
                "allow_matched_linear_baseline": True,
                "baseline_budget_source": "dense_fraction",
                "baseline_parameter_budget_fraction": 0.5,
                "baseline_budget_tolerance_frac": 0.25,
                "sparse_row_banks": 16,
                "sparse_rows_per_token": 2,
                "sparse_activation": "gelu",
            },
            {
                "name": "checkerboard_mlp_shared",
                "student_source": "checkerboard_moe",
                "allow_matched_linear_baseline": True,
                "baseline_budget_source": "dense_fraction",
                "baseline_parameter_budget_fraction": 1.0,
                "baseline_budget_tolerance_frac": 4.0,
                "sparse_row_banks": 3,
                "sparse_rows_per_token": 2,
                "sparse_column_blocks": 4,
                "sparse_column_blocks_per_token": 2,
                "checkerboard_router_type": "mlp",
                "checkerboard_router_hidden_features": 5,
                "checkerboard_always_on_rows": 2,
            }
        ]
    }

    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "plan",
        max_trials=2,
    )

    trial_configs = [
        yaml.safe_load(Path(str(trial["config_path"])).read_text(encoding="utf-8"))
        for trial in plan["trials"]
    ]
    sparse_config, checkerboard_config = trial_configs
    assert sparse_config["student"]["source"] == "sparse_row"
    assert sparse_config["student"]["allow_matched_linear_baseline"] is True
    assert sparse_config["student"]["baseline_budget_source"] == "dense_fraction"
    assert sparse_config["student"]["baseline_budget_tolerance_frac"] == pytest.approx(0.25)
    assert sparse_config["student"]["sparse_row_banks"] == 16
    assert sparse_config["student"]["sparse_rows_per_token"] == 2
    assert sparse_config["student"]["sparse_activation"] == "gelu"
    assert checkerboard_config["student"]["source"] == "checkerboard_moe"
    assert checkerboard_config["student"]["allow_matched_linear_baseline"] is True
    assert checkerboard_config["student"]["baseline_budget_source"] == "dense_fraction"
    assert checkerboard_config["student"]["baseline_budget_tolerance_frac"] == pytest.approx(4.0)
    assert checkerboard_config["student"]["sparse_row_banks"] == 3
    assert checkerboard_config["student"]["sparse_rows_per_token"] == 2
    assert checkerboard_config["student"]["sparse_column_blocks"] == 4
    assert checkerboard_config["student"]["sparse_column_blocks_per_token"] == 2
    assert checkerboard_config["student"]["checkerboard_router_type"] == "mlp"
    assert checkerboard_config["student"]["checkerboard_router_hidden_features"] == 5
    assert checkerboard_config["student"]["checkerboard_always_on_rows"] == 2


@pytest.mark.parametrize(
    ("source", "expected_type"),
    [
        ("sparse_row", SparseRowLinear),
        ("sparse_column", SparseColumnLinear),
        ("sparse_row_column", CoupledRowColumnLinear),
        ("checkerboard_moe", CheckerboardSparseMoELinear),
    ],
)
def test_generated_sparse_baseline_build_paths_replace_eligible_linears(
    source: str,
    expected_type: type[nn.Module],
) -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": source,
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "dense_fraction",
        "baseline_parameter_budget_fraction": 1.0,
        # Tiny synthetic 8x8 layers are dominated by router/output-bank overhead.
        # The strict budget guard is tested separately below.
        "baseline_budget_tolerance_frac": 5.0,
        "sparse_row_banks": 4,
        "sparse_rows_per_token": 2,
        "sparse_column_blocks": 4,
        "sparse_column_blocks_per_token": 1,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == source
    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert isinstance(result.model.proj, expected_type)
    assert result.manifest[0].replacement_path.startswith(f"generated:{source}")
    x = torch.randn(3, 8)
    assert torch.isfinite(result.model(x)).all()


def test_generated_sparse_baseline_rejects_over_budget_replacement() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "sparse_row",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "dense_fraction",
        "baseline_parameter_budget_fraction": 1.0,
        "baseline_budget_tolerance_frac": 0.0,
        "sparse_row_banks": 4,
        "sparse_rows_per_token": 2,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    with pytest.raises(RuntimeError, match="exceeding budget"):
        build_student_model(
            loaded_teacher_model=teacher,
            config=config,
            device=torch.device("cpu"),
        )


def test_checkerboard_generated_baseline_launch_budget_matches_stage_f_shape() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "checkerboard_moe",
        "distill_config": "configs/fff_distill_stage_f.yaml",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "fff_config",
        "baseline_parameter_budget_fraction": 1.0,
        "sparse_row_banks": 2,
        "sparse_rows_per_token": 1,
        "sparse_column_blocks": 2,
        "sparse_column_blocks_per_token": 1,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = RectangularConfigModel({"in_features": 512, "out_features": 256})
    reference_budget = sum(
        parameter.numel()
        for parameter in make_fff_replacement(
            teacher.proj,
            config=yaml.safe_load(Path("configs/fff_distill_stage_f.yaml").read_text(encoding="utf-8"))["fff"],
        ).parameters()
    )

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "checkerboard_moe"
    assert result.replacement_count == 1
    assert isinstance(result.model.proj, CheckerboardSparseMoELinear)
    assert result.model.proj.expert_rank is not None
    assert result.manifest[0].parameters <= reference_budget
    assert result.manifest[0].parameters == sum(parameter.numel() for parameter in result.model.proj.parameters())
    assert "budget_tolerance_frac=0" in result.manifest[0].replacement_path
    assert "'factorized_experts': True" in result.manifest[0].replacement_path
    x = torch.randn(2, 512)
    assert result.model(x).shape == (2, 256)


def test_checkerboard_mlp_shared_generated_baseline_build_path_records_options() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "checkerboard_moe",
        "distill_config": "configs/fff_distill_stage_f.yaml",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "dense_fraction",
        "baseline_parameter_budget_fraction": 1.0,
        "baseline_budget_tolerance_frac": 4.0,
        "sparse_row_banks": 3,
        "sparse_rows_per_token": 2,
        "sparse_column_blocks": 4,
        "sparse_column_blocks_per_token": 2,
        "checkerboard_router_type": "mlp",
        "checkerboard_router_hidden_features": 5,
        "checkerboard_always_on_rows": 2,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert isinstance(result.model.proj, CheckerboardSparseMoELinear)
    assert result.model.proj.router_type == "mlp"
    assert result.model.proj.router_hidden_features == 5
    assert result.model.proj.always_on_rows == 2
    assert "checkerboard_router_type=mlp" in result.manifest[0].replacement_path
    assert "checkerboard_router_hidden_features=5" in result.manifest[0].replacement_path
    assert "checkerboard_always_on_rows=2" in result.manifest[0].replacement_path
    assert "'always_on_rows': 2" in result.manifest[0].replacement_path
    x = torch.randn(2, 8)
    assert result.model(x).shape == (2, 8)


def test_generated_official_fastfeedforward_build_path_replaces_eligible_linears() -> None:
    pytest.importorskip("fastfeedforward")
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "official_fastfeedforward",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "dense_fraction",
        "baseline_parameter_budget_fraction": 1.0,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "official_fastfeedforward"
    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert type(result.model.proj.module).__name__ == "FFF"
    assert result.manifest[0].replacement_path.startswith("generated:official_fastfeedforward")
    x = torch.randn(3, 8)
    assert torch.isfinite(result.model(x)).all()


def test_matched_low_rank_build_path_replaces_eligible_linears() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "matched_low_rank",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_parameter_budget_fraction": 0.5,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "matched_low_rank"
    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert isinstance(result.model.proj, LowRankLinear)
    assert result.manifest[0].parameters <= 8 * 8 + 8
    assert result.manifest[0].replacement_path.startswith("generated:matched_low_rank")
    x = torch.randn(3, 8)
    assert torch.isfinite(result.model(x)).all()


def test_matched_shared_only_build_path_replaces_eligible_linears() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "matched_shared_only",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_parameter_budget_fraction": 0.75,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "matched_shared_only"
    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert isinstance(result.model.proj, SharedOnlyLinear)
    assert result.manifest[0].parameters <= 8 * 8 + 8
    assert result.manifest[0].replacement_path.startswith("generated:matched_shared_only")
    x = torch.randn(3, 8)
    assert torch.isfinite(result.model(x)).all()


def test_matched_linear_fff_config_budget_uses_reference_fff_parameter_count(tmp_path: Path) -> None:
    distill_config = tmp_path / "fff_budget.yaml"
    distill_config.write_text(
        "\n".join(
            [
                "fff:",
                "  depth: 1",
                "  shared_rows: 0",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  route_rows_contribute: false",
                "  bias: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw = _minimal_config(tmp_path)
    raw["student"] = {
        "source": "matched_smaller_dense",
        "distill_config": str(distill_config),
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "fff_config",
        "baseline_parameter_budget_fraction": 1.0,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})
    expected_budget = sum(
        parameter.numel()
        for parameter in make_fff_replacement(
            teacher.proj,
            config={
                "depth": 1,
                "shared_rows": 0,
                "route_rows": 1,
                "leaf_rows": 1,
                "route_rows_contribute": False,
                "bias": True,
            },
        ).parameters()
    )

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "matched_smaller_dense"
    assert result.replacement_count == 1
    assert result.manifest[0].parameters <= expected_budget
    assert f"budget={expected_budget}" in result.manifest[0].replacement_path


def test_matched_shared_only_fff_config_budget_uses_reference_fff_parameter_count(
    tmp_path: Path,
) -> None:
    distill_config = tmp_path / "fff_budget.yaml"
    distill_config.write_text(
        "\n".join(
            [
                "fff:",
                "  depth: 1",
                "  shared_rows: 2",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  route_rows_contribute: false",
                "  activation: gelu",
                "  bias: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw = _minimal_config(tmp_path)
    raw["student"] = {
        "source": "matched_shared_only",
        "distill_config": str(distill_config),
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_budget_source": "fff_config",
        "baseline_parameter_budget_fraction": 1.0,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})
    expected_budget = sum(
        parameter.numel()
        for parameter in make_fff_replacement(
            teacher.proj,
            config={
                "depth": 1,
                "shared_rows": 2,
                "route_rows": 1,
                "leaf_rows": 1,
                "route_rows_contribute": False,
                "activation": "gelu",
                "bias": True,
            },
        ).parameters()
    )

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "matched_shared_only"
    assert result.replacement_count == 1
    assert isinstance(result.model.proj, SharedOnlyLinear)
    assert result.model.proj.activation == "gelu"
    assert result.manifest[0].parameters <= expected_budget
    assert f"budget={expected_budget}" in result.manifest[0].replacement_path


def test_matched_smaller_dense_full_budget_copies_teacher_prefix_exactly() -> None:
    raw = _minimal_config(Path("/tmp"))
    raw["student"] = {
        "source": "matched_smaller_dense",
        "min_in_features": 1,
        "min_out_features": 1,
        "allow_matched_linear_baseline": True,
        "baseline_parameter_budget_fraction": 1.0,
    }
    config = parse_finetune_run_config(raw, quick_smoke=True)
    teacher = TinyConfigModel({"width": 8})
    x = torch.randn(4, 8)
    expected = teacher(x)

    result = build_student_model(
        loaded_teacher_model=teacher,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.source == "matched_smaller_dense"
    assert result.replacement_count == 1
    assert isinstance(result.model.proj, SmallerDenseLinear)
    assert result.model.proj.active_out_features == 8
    torch.testing.assert_close(result.model(x), expected)


def test_finetune_config_rejects_unknown_keys_and_test_access(tmp_path: Path) -> None:
    raw = _minimal_config(tmp_path)
    with pytest.raises(ValueError, match="unknown keys"):
        parse_finetune_run_config(raw | {"surprise": 1}, quick_smoke=True)

    bad_dataset = dict(raw)
    bad_dataset["dataset"] = dict(raw["dataset"], use_test=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train-owned keys"):
        parse_finetune_run_config(bad_dataset, quick_smoke=True)

    missing_teacher = dict(raw)
    missing_teacher["teacher_checkpoint"] = None
    with pytest.raises(ValueError, match="teacher_checkpoint is required"):
        parse_finetune_run_config(missing_teacher, quick_smoke=False)


def test_kd_loss_is_finite_and_nonnegative() -> None:
    student = torch.tensor([[1.0, 0.0], [0.5, -0.25]])
    teacher = torch.tensor([[0.8, 0.2], [0.0, 1.0]])

    loss = kd_loss(student, teacher, temperature=2.0)

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_global_train_step_limit_boundary() -> None:
    assert _reached_train_step_limit(0, None) is False
    assert _reached_train_step_limit(0, 1) is False
    assert _reached_train_step_limit(1, 1) is True
    assert _reached_train_step_limit(2, 1) is True


def test_assemble_fff_student_from_artifacts_loads_strict_state(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    layer_dir = artifact_root / "work" / "0" / "trials" / "trial_000000" / "layers" / "proj"
    layer_dir.mkdir(parents=True)
    distill_config = tmp_path / "distill.yaml"
    distill_config.write_text(
        "\n".join(
            [
                "fff:",
                "  depth: 2",
                "  shared_rows: 0",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  activation: silu",
                "  hard_routing: true",
                "  route_rows_contribute: false",
                "  bias: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base = TinyLinearModel()
    replacement = make_fff_replacement(base.proj, config={"depth": 2, "leaf_rows": 1, "route_rows": 1})
    state_path = layer_dir / "fff_state.pt"
    torch.save(replacement.state_dict(), state_path)
    summary_path = artifact_root / "work" / "0" / "trials" / "trial_000000" / "layer_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "name": "proj",
                    "replacement_path": str(state_path),
                    "final_normalized_mse": 0.1,
                    "final_cosine_similarity": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = assemble_fff_student_from_artifacts(
        TinyLinearModel(),
        artifact_root=artifact_root,
        distill_config_path=distill_config,
        min_in_features=1,
        min_out_features=1,
        require_full_replacement=True,
        device=torch.device("cpu"),
    )

    assert result.replacement_count == 1
    assert result.eligible_count == 1
    assert result.manifest[0].name == "proj"


def test_assemble_fff_student_rejects_incomplete_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "layer_summary.json").write_text("[]", encoding="utf-8")
    distill_config = tmp_path / "distill.yaml"
    distill_config.write_text("fff:\n  depth: 2\n  route_rows: 1\n  leaf_rows: 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not match eligible"):
        assemble_fff_student_from_artifacts(
            TinyLinearModel(),
            artifact_root=artifact_root,
            distill_config_path=distill_config,
            min_in_features=1,
            min_out_features=1,
            require_full_replacement=True,
            device=torch.device("cpu"),
        )


def test_finetune_hpo_plan_and_execute_with_fake_runner(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {
        "study_name": "test_finetune",
        "cases": [
            {
                "name": "official_wsd",
                "fine_tune_epochs": 2,
                "lr_muon": 0.001,
                "schedule": "wsd",
                "kd_temperature": 4,
            }
        ],
    }
    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "plan",
        max_trials=1,
    )
    assert plan["test_accessed"] is False
    trial_config = Path(str(plan["trials"][0]["config_path"]))
    assert trial_config.exists()

    def fake_runner(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "succeeded",
            "summary": {
                "best_val_accuracy": 0.25,
                "train_steps_total": 1,
                "student_replacement_count": 64,
                "optimizer": "muon_adamw",
                "schedule": "wsd",
            },
            "test_accessed": False,
        }

    summary = run_finetune_hpo_trials(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "execute",
        max_trials=1,
        trial_runner=fake_runner,
    )
    assert summary["succeeded"] == 1
    assert summary["test_accessed"] is False


def test_finetune_hpo_seed_override_is_written_to_trial_config(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {"cases": [{"name": "seeded", "fine_tune_epochs": 1}]}

    seen_config_paths: list[Path] = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        seen_config_paths.append(Path(str(kwargs["config_path"])))
        return {
            "status": "succeeded",
            "summary": {"best_val_accuracy": 0.25, "train_steps_total": 1},
            "test_accessed": False,
        }

    summary = run_finetune_hpo_trials(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "execute",
        max_trials=1,
        seed=9001,
        trial_runner=fake_runner,
    )

    assert summary["seed"] == 9001
    assert seen_config_paths
    trial_config = yaml.safe_load(seen_config_paths[0].read_text(encoding="utf-8"))
    assert trial_config["seed"] == 9001


def test_gc5_optimizer_hpo_config_plans_matched_lr_budget(tmp_path: Path) -> None:
    base = yaml.safe_load(Path("configs/finetune_stage_h_train_eval.yaml").read_text())
    hpo = yaml.safe_load(Path("configs/finetune_optimizer_hpo_gc5.yaml").read_text())

    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "gc5_plan",
        max_trials=15,
    )

    assert plan["test_accessed"] is False
    assert len(plan["trials"]) == 15
    configs = [
        yaml.safe_load(Path(str(trial["config_path"])).read_text(encoding="utf-8"))
        for trial in plan["trials"]
    ]
    names = {str(trial["case"]) for trial in plan["trials"]}
    assert names == {
        f"{family}_lr_{tier}"
        for family in {
            "official_muon_cosine",
            "official_muon_wsd",
            "pace_muon_cosine",
            "normuon_cosine",
            "pace_normuon_cosine",
        }
        for tier in {"low", "base", "high"}
    }
    assert {config["dataset"].get("use_test", False) for config in configs} == {False}
    assert {config["train"]["epochs"] for config in configs} == {3}
    assert {config["train"]["batch_size_per_gpu"] for config in configs} == {32}
    assert {config["train"]["num_workers"] for config in configs} == {4}
    assert {config["losses"]["lambda_balance"] for config in configs} == {0.001}
    assert {config["losses"]["balance_recipe"] for config in configs} == {"split_minleaf"}
    assert {config["train"]["optimizer"] for config in configs} == {
        "muon_adamw",
        "pace_muon",
        "normuon_adamw",
        "pace_normuon",
    }


def test_optimizer_multiseed_hpo_config_uses_shared_seed_budget(tmp_path: Path) -> None:
    base = yaml.safe_load(Path("configs/finetune_stage_h_train_eval.yaml").read_text())
    hpo = yaml.safe_load(Path("configs/finetune_optimizer_hpo_multiseed.yaml").read_text())

    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "optimizer_multiseed_plan",
        max_trials=15,
    )

    assert plan["test_accessed"] is False
    assert len(plan["trials"]) == 15
    configs = [
        yaml.safe_load(Path(str(trial["config_path"])).read_text(encoding="utf-8"))
        for trial in plan["trials"]
    ]
    cases = [str(trial["case"]) for trial in plan["trials"]]
    families = {case.rsplit("_seed", maxsplit=1)[0] for case in cases}
    assert families == {
        "official_muon_cosine",
        "official_muon_wsd",
        "pace_muon_cosine",
        "normuon_cosine",
        "pace_normuon_cosine",
    }
    assert {
        int(case.rsplit("_seed", maxsplit=1)[1])
        for case in cases
        if case.startswith("official_muon_cosine")
    } == {31001, 31002, 31003}
    assert {config["dataset"].get("use_test", False) for config in configs} == {False}
    assert {config["train"]["epochs"] for config in configs} == {3}
    assert {config["train"]["optimizer"] for config in configs} == {
        "muon_adamw",
        "pace_muon",
        "normuon_adamw",
        "pace_normuon",
    }
    assert {config["train"]["schedule"] for config in configs} == {"cosine", "wsd"}


def test_bank_muon_hpo_config_writes_bank_policy_override(tmp_path: Path) -> None:
    base = yaml.safe_load(Path("configs/finetune_stage_h_train_eval.yaml").read_text())
    hpo = yaml.safe_load(Path("configs/finetune_bank_muon_hpo.yaml").read_text())

    plan = write_finetune_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo,
        output_dir=tmp_path / "bank_muon_plan",
        max_trials=6,
    )

    configs = [
        yaml.safe_load(Path(str(trial["config_path"])).read_text(encoding="utf-8"))
        for trial in plan["trials"]
    ]
    by_case = {
        str(trial["case"]): config["train"]["fff_bank_muon"]
        for trial, config in zip(plan["trials"], configs, strict=True)
    }
    assert {value for case, value in by_case.items() if case.startswith("adamw_fallback")} == {False}
    assert {value for case, value in by_case.items() if case.startswith("muon_banks")} == {True}
    assert {config["dataset"].get("use_test", False) for config in configs} == {False}


def test_finetune_hpo_rejects_test_accessed_trial(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {"cases": [{"name": "bad", "fine_tune_epochs": 1}]}

    def fake_runner(**_kwargs: object) -> dict[str, object]:
        return {"status": "succeeded", "test_accessed": True}

    with pytest.raises(RuntimeError, match="zero successful"):
        run_finetune_hpo_trials(
            base_config=base,
            hpo_config=hpo,
            output_dir=tmp_path / "execute",
            max_trials=1,
            trial_runner=fake_runner,
        )


def test_finetune_hpo_rejects_test_accessed_child_summary(tmp_path: Path) -> None:
    base = _minimal_config(tmp_path)
    hpo = {"cases": [{"name": "bad_summary", "fine_tune_epochs": 1}]}

    def fake_runner(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "succeeded",
            "test_accessed": False,
            "summary": {"best_val_accuracy": 0.1, "test_accessed": True},
        }

    with pytest.raises(RuntimeError, match="zero successful"):
        run_finetune_hpo_trials(
            base_config=base,
            hpo_config=hpo,
            output_dir=tmp_path / "execute_summary",
            max_trials=1,
            trial_runner=fake_runner,
        )

    trial_result = json.loads(
        (
            tmp_path
            / "execute_summary"
            / "trials"
            / "trial_000000_bad_summary"
            / "trial_result.json"
        ).read_text(encoding="utf-8")
    )
    assert trial_result["status"] == "failed_logic"
    assert trial_result["test_accessed"] is True


def test_finetune_hpo_command_runs_training_for_non_smoke_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_finetune_trial_command(
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "trial",
        quick_smoke=False,
        max_train_steps=1,
        max_val_steps=1,
    )
    run_finetune_trial_command(
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "smoke",
        quick_smoke=True,
        max_train_steps=1,
        max_val_steps=1,
    )

    assert commands[0][commands[0].index("--smoke-mode") + 1] == "train"
    assert commands[0][commands[0].index("--quick-smoke") + 1] == "false"
    assert commands[1][commands[1].index("--smoke-mode") + 1] == "metadata"
    assert commands[1][commands[1].index("--quick-smoke") + 1] == "true"
