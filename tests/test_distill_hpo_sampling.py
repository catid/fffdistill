from __future__ import annotations

import json
import random
from collections.abc import Mapping
from copy import deepcopy

import pytest

import cifar_mamba_fff.hpo.distill_hpo as distill_hpo
from cifar_mamba_fff.hpo.distill_hpo import (
    apply_distill_hpo_overrides,
    canonicalize_distill_route_overrides,
    sample_distill_overrides,
    sample_valid_distill_hpo_candidates,
    write_distill_hpo_trial_plan,
)
from cifar_mamba_fff.hpo.distill_hpo import (
    main as distill_hpo_main,
)
from cifar_mamba_fff.models.fff_linear import FFFLinear, FFFLinearConfig
from cifar_mamba_fff.utils import load_yaml


def _fff_kwargs(overrides: Mapping[str, object]) -> dict[str, object]:
    fields = set(FFFLinearConfig.__dataclass_fields__)
    return {key: value for key, value in overrides.items() if key in fields}


def _build_layer(overrides: Mapping[str, object]) -> FFFLinear:
    return FFFLinear(
        16,
        8,
        shared_rows=1,
        bias=False,
        **_fff_kwargs(overrides),
    )


def test_distill_hpo_config_samples_only_valid_route_role_combinations() -> None:
    config = load_yaml("configs/fff_distill_hpo.yaml")
    rng = random.Random(123)
    roles_seen: set[str] = set()

    for _ in range(250):
        overrides = sample_distill_overrides(config["search_space"], rng=rng)
        layer = _build_layer(overrides)
        diagnostics = layer.diagnostics()
        role = diagnostics["route_row_role"]
        roles_seen.add(str(role))

        if role == "routing_only":
            assert diagnostics["route_rows_contribute"] is False
            assert diagnostics["route_result_rows"] == 0
            assert diagnostics["route_output_rows_per_node"] == 0
        elif role == "shared_routing_and_output":
            assert diagnostics["route_rows_contribute"] is True
            assert diagnostics["route_result_rows"] == 0
            assert diagnostics["route_output_rows_per_node"] > 0
        elif role == "split_routing_output":
            assert diagnostics["route_rows_contribute"] is True
            assert diagnostics["route_result_rows"] > 0
            assert diagnostics["route_output_rows_per_node"] > 0
        else:
            raise AssertionError(f"unexpected role {role!r}")

    assert roles_seen == {
        "routing_only",
        "shared_routing_and_output",
        "split_routing_output",
    }


def test_routing_only_canonicalizes_to_no_output_rows() -> None:
    overrides = canonicalize_distill_route_overrides(
        {
            "route_row_role": "routing_only",
            "route_rows_contribute": False,
            "route_result_rows": 2,
            "route_rows_output_count": 1,
            "route_rows_output_fraction": 1.0,
        }
    )

    assert overrides["route_row_role"] == "routing_only"
    assert overrides["route_rows_contribute"] is False
    assert overrides["route_result_rows"] == 0
    assert overrides["route_rows_output_count"] == 0
    assert overrides["route_rows_output_fraction"] is None
    assert _build_layer(overrides).diagnostics()["route_output_rows_per_node"] == 0


def test_shared_route_output_canonicalizes_to_shared_role() -> None:
    overrides = canonicalize_distill_route_overrides(
        {
            "route_row_role": "shared_routing_and_output",
            "route_rows": 2,
            "route_result_rows": 2,
            "route_rows_output_count": 1,
            "route_rows_output_fraction": None,
        }
    )
    diagnostics = _build_layer(overrides).diagnostics()

    assert overrides["route_rows_contribute"] is True
    assert overrides["route_result_rows"] == 0
    assert diagnostics["route_row_role"] == "shared_routing_and_output"
    assert diagnostics["route_output_rows_per_node"] == 1


def test_split_route_output_requires_positive_result_rows() -> None:
    overrides = canonicalize_distill_route_overrides(
        {
            "route_row_role": "split_routing_output",
            "route_result_rows": 2,
            "route_rows_output_count": "all",
            "route_rows_output_fraction": 0.5,
        }
    )
    diagnostics = _build_layer(overrides).diagnostics()

    assert overrides["route_rows_contribute"] is True
    assert diagnostics["route_row_role"] == "split_routing_output"
    assert diagnostics["route_result_rows"] == 2
    assert diagnostics["route_output_rows_per_node"] == 1


def test_legacy_route_rows_contribute_alias_promotes_to_shared_output() -> None:
    overrides = canonicalize_distill_route_overrides(
        {
            "route_rows": 2,
            "route_rows_contribute": True,
        }
    )
    diagnostics = _build_layer(overrides).diagnostics()

    assert overrides["route_row_role"] == "shared_routing_and_output"
    assert overrides["route_rows_contribute"] is True
    assert diagnostics["route_output_rows_per_node"] == 2


def test_invalid_route_role_and_split_controls_raise_before_training() -> None:
    with pytest.raises(ValueError, match="unknown route_row_role"):
        canonicalize_distill_route_overrides({"route_row_role": "surprise"})

    with pytest.raises(ValueError, match="positive route_result_rows"):
        canonicalize_distill_route_overrides(
            {
                "route_row_role": "split_routing_output",
                "route_result_rows": 0,
                "route_rows_output_count": 1,
            }
        )

    with pytest.raises(ValueError, match="positive route_result_rows choice"):
        sample_distill_overrides(
            {
                "route_rows": [1],
                "route_row_role": ["split_routing_output"],
                "route_output_controls": {
                    "split_routing_output": {
                        "route_result_rows": [0],
                        "route_rows_output_count": [1],
                    },
                },
            },
            rng=random.Random(1),
        )


def test_valid_distill_hpo_candidates_do_not_count_rejected_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = iter(
        [
            {"route_row_role": "routing_only"},
            {"route_row_role": "shared_routing_and_output"},
            {"route_row_role": "split_routing_output"},
        ]
    )

    def fake_sample(_search_space: Mapping[str, object], *, rng: random.Random) -> dict[str, object]:
        del rng
        return next(sampled)

    monkeypatch.setattr(distill_hpo, "sample_distill_overrides", fake_sample)

    candidates = sample_valid_distill_hpo_candidates(
        {},
        max_trials=2,
        max_attempts=3,
        rng=random.Random(123),
        validate_fn=lambda overrides: overrides["route_row_role"] != "routing_only",
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1]
    assert [candidate.attempt_index for candidate in candidates] == [1, 2]


def test_distill_hpo_overrides_map_to_concrete_config_sections() -> None:
    base = load_yaml("configs/fff_distill_default.yaml")
    original_base = deepcopy(base)
    config = apply_distill_hpo_overrides(
        base,
        {
            "shared_unrouted_frac": 0.2,
            "route_rows": 2,
            "leaf_rows": 4,
            "depth": 6,
            "route_row_role": "split_routing_output",
            "route_rows_contribute": True,
            "route_result_rows": 2,
            "route_rows_output_count": 1,
            "route_rows_output_fraction": None,
            "activation": "gelu",
            "region_leak": 0.01,
            "master_leaf": True,
            "balance_recipe": "split_minleaf_margin",
            "balance_coeff": 0.003,
            "min_leaf_tokens": 64,
            "router_recipe": "hard_em_utility_ste",
            "utility_loss_coeff": 0.3,
            "utility_tau": 0.1,
            "locoprop_refit": "every_250",
            "ridge_lambda": 0.001,
            "locoprop_blend_alpha": 0.5,
        },
    )

    assert config["fff"]["shared_unrouted_frac"] == 0.2
    assert config["fff"]["route_rows"] == 2
    assert config["fff"]["leaf_rows"] == 4
    assert config["fff"]["depth"] == 6
    assert config["fff"]["route_row_role"] == "split_routing_output"
    assert config["fff"]["route_result_rows"] == 2
    assert config["fff"]["route_rows_output_count"] == 1
    assert config["fff"]["activation"] == "gelu"
    assert config["fff"]["region_leak"] == 0.01
    assert config["fff"]["master_leaf"] is True
    assert config["balance"]["recipe"] == "split_minleaf_margin"
    assert config["balance"]["coeff"] == 0.003
    assert config["balance"]["min_leaf_tokens"] == 64
    assert config["router"]["recipe"] == "hard_em_utility_ste"
    assert config["router"]["loss_coeff"] == 0.3
    assert config["router"]["utility_temperature"] == 0.1
    assert config["locoprop"]["enabled"] is True
    assert config["locoprop"]["interval_steps"] == 250
    assert config["locoprop"]["ridge_lambda"] == 0.001
    assert config["locoprop"]["blend_alpha"] == 0.5
    assert config["teacher_checkpoint"] is None
    assert config["hpo_overrides"]["router_recipe"] == "hard_em_utility_ste"
    assert base == original_base


def test_distill_hpo_rejects_invalid_locoprop_refit() -> None:
    for invalid in ("sometimes", "every_0", "every_x", True):
        with pytest.raises(ValueError, match=r"locoprop_refit|interval"):
            apply_distill_hpo_overrides({}, {"locoprop_refit": invalid})


def test_distill_hpo_override_mapper_rejects_unknown_keys_and_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="unknown distill HPO override keys"):
        apply_distill_hpo_overrides({}, {"typo_router_recipe": "vanilla_ste"})

    with pytest.raises(ValueError, match="duplicate router override aliases"):
        apply_distill_hpo_overrides(
            {},
            {"router_loss_coeff": 0.1, "utility_loss_coeff": 0.3},
        )

    with pytest.raises(ValueError, match="duplicate router override aliases"):
        apply_distill_hpo_overrides(
            {},
            {"temperature": 1.0, "tau": 0.5},
        )


def test_distill_hpo_override_mapper_rejects_non_mapping_base_sections() -> None:
    with pytest.raises(ValueError, match="fff must be a mapping"):
        apply_distill_hpo_overrides({"fff": []}, {"route_rows": 1})


def test_distill_hpo_override_mapper_canonicalizes_route_controls() -> None:
    config = apply_distill_hpo_overrides(
        {},
        {
            "route_rows": 2,
            "route_row_role": "routing_only",
            "route_rows_contribute": True,
            "route_rows_output_count": 1,
        },
    )

    assert config["fff"]["route_row_role"] == "shared_routing_and_output"
    assert config["fff"]["route_rows_contribute"] is True
    assert config["fff"]["route_result_rows"] == 0
    assert config["fff"]["route_rows_output_count"] == 1


def test_distill_hpo_trial_plan_writes_configs_without_test_access(tmp_path) -> None:
    base = load_yaml("configs/fff_distill_default.yaml")
    hpo_config = load_yaml("configs/fff_distill_hpo.yaml")

    summary = write_distill_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=3,
        max_attempts=16,
        seed=123,
        teacher_checkpoint="/tmp/teacher_best.pt",
    )

    assert summary["accepted_trials"] == 3
    assert summary["teacher_checkpoint"] == "/tmp/teacher_best.pt"
    assert summary["test_accessed"] is False
    for trial in summary["trials"]:
        config_path = tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "distill_config.yaml"
        record_path = tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "trial_config.json"
        config = load_yaml(config_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert config["teacher_checkpoint"] == "/tmp/teacher_best.pt"
        assert record["test_accessed"] is False
        assert record["execute_ready"] is True
    written_summary = json.loads((tmp_path / "distill_hpo_summary.json").read_text(encoding="utf-8"))
    assert written_summary["accepted_trials"] == 3
    assert written_summary["test_accessed"] is False


def test_distill_hpo_cli_materializes_dry_run_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    output_dir = tmp_path / "plan"
    monkeypatch.setattr(
        "sys.argv",
        [
            "distill_hpo",
            "--base-config",
            "configs/fff_distill_default.yaml",
            "--hpo-config",
            "configs/fff_distill_hpo.yaml",
            "--output-dir",
            str(output_dir),
            "--max-trials",
            "2",
            "--max-attempts",
            "16",
            "--seed",
            "321",
        ],
    )

    assert distill_hpo_main() == 0
    summary = json.loads((output_dir / "distill_hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_trials"] == 2
    assert summary["teacher_checkpoint"] is None
    assert summary["test_accessed"] is False
    assert (output_dir / "trials" / "trial_000000" / "distill_config.yaml").exists()
