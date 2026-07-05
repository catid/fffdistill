from __future__ import annotations

import json
import random
from collections.abc import Mapping
from copy import deepcopy

import pytest

import cifar_mamba_fff.hpo.distill_hpo as distill_hpo
from cifar_mamba_fff.distill_linears import (
    LinearDistillConfig,
    reject_unknown_distill_config_keys,
)
from cifar_mamba_fff.hpo.distill_hpo import (
    apply_distill_hpo_overrides,
    canonicalize_distill_route_overrides,
    run_distill_hpo_trials,
    sample_distill_hpo_cases,
    sample_distill_overrides,
    sample_valid_distill_hpo_candidates,
    write_distill_hpo_trial_plan,
)
from cifar_mamba_fff.hpo.distill_hpo import (
    main as distill_hpo_main,
)
from cifar_mamba_fff.models.fff_linear import FFFLinear, FFFLinearConfig
from cifar_mamba_fff.utils import load_yaml

STAGE_C_ROUTER_RECIPES = [
    "vanilla_ste",
    "clipped_ste",
    "sigmoid_surrogate_ste",
    "st_gumbel",
    "utility_targeted_ste",
    "hard_em_utility_ste",
    "expert_choice_imitation",
]


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


def _stage_c_router_search_space() -> dict[str, object]:
    return {
        "shared_unrouted_frac": [0.10],
        "route_rows": [1],
        "leaf_rows": [2],
        "depth": [5],
        "route_row_role": ["routing_only"],
        "activation": ["silu"],
        "region_leak": [0.0],
        "master_leaf": [False],
        "balance_recipe": ["split_minleaf"],
        "balance_coeff": [0.001],
        "min_leaf_tokens": [64],
        "router_recipe": STAGE_C_ROUTER_RECIPES,
        "locoprop_refit": ["off"],
    }


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


def test_default_distill_hpo_sampler_still_uses_random_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = iter(
        [
            {"route_row_role": "routing_only"},
            {"route_row_role": "shared_routing_and_output"},
        ]
    )
    calls = 0

    def fake_sample(_search_space: Mapping[str, object], *, rng: random.Random) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert isinstance(rng, random.Random)
        return next(sampled)

    monkeypatch.setattr(distill_hpo, "sample_distill_overrides", fake_sample)

    candidates = sample_valid_distill_hpo_candidates(
        {},
        max_trials=2,
        max_attempts=2,
        rng=random.Random(123),
    )

    assert calls == 2
    assert [candidate.overrides["route_row_role"] for candidate in candidates] == [
        "routing_only",
        "shared_routing_and_output",
    ]


def test_random_sampler_grid_offset_avoids_replaying_first_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample_with_offset(grid_offset: int) -> list[distill_hpo.DistillHpoCandidate]:
        sampled = iter(
            [
                {"route_row_role": "routing_only"},
                {"route_row_role": "shared_routing_and_output"},
            ]
        )

        def fake_sample(_search_space: Mapping[str, object], *, rng: random.Random) -> dict[str, object]:
            assert isinstance(rng, random.Random)
            return next(sampled)

        monkeypatch.setattr(distill_hpo, "sample_distill_overrides", fake_sample)
        return sample_valid_distill_hpo_candidates(
            {},
            max_trials=1,
            max_attempts=2,
            rng=random.Random(123),
            grid_offset=grid_offset,
        )

    first_slot = sample_with_offset(0)
    second_slot = sample_with_offset(1)

    assert [candidate.attempt_index for candidate in first_slot] == [0]
    assert [candidate.attempt_index for candidate in second_slot] == [1]
    assert first_slot[0].overrides != second_slot[0].overrides


def test_random_sampler_grid_offset_skips_only_valid_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = iter(
        [
            {"route_row_role": "invalid"},
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
        max_attempts=4,
        rng=random.Random(123),
        grid_offset=1,
        validate_fn=lambda overrides: overrides["route_row_role"] != "invalid",
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1]
    assert [candidate.attempt_index for candidate in candidates] == [2, 3]
    assert [candidate.overrides["route_row_role"] for candidate in candidates] == [
        "shared_routing_and_output",
        "split_routing_output",
    ]


def test_grid_sampler_writes_unique_stage_c_router_trials(tmp_path) -> None:
    summary = write_distill_hpo_trial_plan(
        base_config=load_yaml("configs/fff_distill_default.yaml"),
        hpo_config={"sampler": "grid", "search_space": _stage_c_router_search_space()},
        output_dir=tmp_path,
        max_trials=7,
        max_attempts=7,
        seed=123,
    )

    recipes = [trial["overrides"]["router_recipe"] for trial in summary["trials"]]
    assert summary["sampler"] == "grid"
    assert summary["accepted_trials"] == 7
    assert recipes == STAGE_C_ROUTER_RECIPES
    assert len(set(recipes)) == len(STAGE_C_ROUTER_RECIPES)
    for trial in summary["trials"]:
        assert trial["attempt_index"] == trial["trial_index"]
        assert trial["overrides"]["route_row_role"] == "routing_only"
        assert trial["overrides"]["route_rows_contribute"] is False
        assert trial["overrides"]["route_result_rows"] == 0


def test_grid_sampler_respects_max_trials_truncation() -> None:
    candidates = sample_valid_distill_hpo_candidates(
        _stage_c_router_search_space(),
        max_trials=3,
        max_attempts=7,
        rng=random.Random(123),
        sampler="grid",
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1, 2]
    assert [candidate.attempt_index for candidate in candidates] == [0, 1, 2]
    assert [candidate.overrides["router_recipe"] for candidate in candidates] == STAGE_C_ROUTER_RECIPES[:3]


def test_grid_sampler_respects_grid_offset() -> None:
    candidates = sample_valid_distill_hpo_candidates(
        _stage_c_router_search_space(),
        max_trials=2,
        max_attempts=7,
        rng=random.Random(123),
        sampler="grid",
        grid_offset=3,
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1]
    assert [candidate.attempt_index for candidate in candidates] == [3, 4]
    assert [candidate.overrides["router_recipe"] for candidate in candidates] == [
        "st_gumbel",
        "utility_targeted_ste",
    ]


def test_cases_sampler_respects_offsets_and_preserves_case_names() -> None:
    candidates = sample_distill_hpo_cases(
        [
            {
                "name": "none",
                "route_rows": 2,
                "route_row_role": "routing_only",
                "route_rows_output_count": 0,
            },
            {
                "name": "shared_one",
                "route_rows": 2,
                "route_row_role": "shared_routing_and_output",
                "route_rows_output_count": 1,
            },
            {
                "name": "split_half",
                "route_rows": 2,
                "route_row_role": "split_routing_output",
                "route_result_rows": 2,
                "route_rows_output_count": "all",
                "route_rows_output_fraction": 0.5,
            },
        ],
        max_trials=2,
        max_attempts=3,
        grid_offset=1,
    )

    assert [candidate.trial_index for candidate in candidates] == [0, 1]
    assert [candidate.attempt_index for candidate in candidates] == [1, 2]
    assert [candidate.overrides["case_name"] for candidate in candidates] == [
        "shared_one",
        "split_half",
    ]


def test_cases_sampler_derives_and_validates_launch_seed_and_router_family() -> None:
    candidates = sample_distill_hpo_cases(
        [
            {
                "name": "seed_from_metadata",
                "launch_seed": 21001,
                "router_family": "vanilla_ste",
                "overrides": {
                    "router_recipe": "vanilla_ste",
                    "route_rows": 1,
                    "route_row_role": "routing_only",
                },
            }
        ],
        max_trials=1,
        max_attempts=1,
    )
    assert candidates[0].overrides["seed"] == 21001

    with pytest.raises(ValueError, match="launch_seed"):
        sample_distill_hpo_cases(
            [
                {
                    "name": "mismatch",
                    "launch_seed": 21001,
                    "overrides": {
                        "seed": 21002,
                        "route_rows": 1,
                        "route_row_role": "routing_only",
                    },
                }
            ],
            max_trials=1,
            max_attempts=1,
        )

    with pytest.raises(ValueError, match="router_family"):
        sample_distill_hpo_cases(
            [
                {
                    "name": "router_mismatch",
                    "router_family": "utility_targeted_ste",
                    "overrides": {
                        "router_recipe": "vanilla_ste",
                        "route_rows": 1,
                        "route_row_role": "routing_only",
                    },
                }
            ],
            max_trials=1,
            max_attempts=1,
        )


def test_cases_sampler_trial_plan_writes_exact_named_case(tmp_path) -> None:
    summary = write_distill_hpo_trial_plan(
        base_config=load_yaml("configs/fff_distill_default.yaml"),
        hpo_config={
            "sampler": "cases",
            "cases": [
                {
                    "name": "shared_half",
                    "overrides": {
                        "include_indices": [32],
                        "router_recipe": "vanilla_ste",
                        "shared_unrouted_frac": 0.1,
                        "route_rows": 2,
                        "leaf_rows": 2,
                        "depth": 5,
                        "route_row_role": "shared_routing_and_output",
                        "route_rows_output_count": "all",
                        "route_rows_output_fraction": 0.5,
                    },
                }
            ],
        },
        output_dir=tmp_path,
        max_trials=1,
        max_attempts=1,
        seed=123,
        teacher_checkpoint="/tmp/teacher_best.pt",
    )

    trial = summary["trials"][0]
    config = load_yaml(tmp_path / "trials" / "trial_000000" / "distill_config.yaml")
    assert summary["sampler"] == "cases"
    assert trial["overrides"]["case_name"] == "shared_half"
    assert config["hpo_overrides"]["case_name"] == "shared_half"
    assert config["fff"]["route_row_role"] == "shared_routing_and_output"
    assert config["fff"]["route_rows_output_count"] == "all"
    assert config["fff"]["route_rows_output_fraction"] == 0.5


def test_grid_sampler_route_role_control_combinations_are_valid() -> None:
    candidates = sample_valid_distill_hpo_candidates(
        {
            "route_rows": [1, 2],
            "route_row_role": [
                "routing_only",
                "shared_routing_and_output",
                "split_routing_output",
            ],
            "route_output_controls": {
                "routing_only": {},
                "shared_routing_and_output": {
                    "route_rows_output_count": [0, 1, "all"],
                    "route_rows_output_fraction": [None],
                },
                "split_routing_output": {
                    "route_result_rows": [0, 1, 2],
                    "route_rows_output_count": [0, 1, "all"],
                    "route_rows_output_fraction": [None, 0.5],
                },
            },
        },
        max_trials=22,
        max_attempts=64,
        rng=random.Random(123),
        sampler="grid",
    )
    roles_seen: set[str] = set()

    for candidate in candidates:
        diagnostics = _build_layer(candidate.overrides).diagnostics()
        role = str(diagnostics["route_row_role"])
        roles_seen.add(role)
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
            "utility_hard": False,
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
    assert config["router"]["utility_hard"] is False
    assert config["locoprop"]["enabled"] is True
    assert config["locoprop"]["interval_steps"] == 250
    assert config["locoprop"]["ridge_lambda"] == 0.001
    assert config["locoprop"]["blend_alpha"] == 0.5
    assert config["teacher_checkpoint"] is None
    assert config["hpo_overrides"]["router_recipe"] == "hard_em_utility_ste"
    assert base == original_base


def test_distill_hpo_overrides_can_select_eligible_layer_indices() -> None:
    base = load_yaml("configs/fff_distill_default.yaml")

    config = apply_distill_hpo_overrides(
        base,
        {
            "include_indices": [0, 32, 60],
            "router_recipe": "vanilla_ste",
            "route_rows": 1,
            "route_row_role": "routing_only",
        },
    )

    assert config["eligible_linear"]["include_indices"] == [0, 32, 60]


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

    with pytest.raises(ValueError, match="eligible_linear include_indices and include_names"):
        apply_distill_hpo_overrides(
            {},
            {"include_indices": [0], "include_names": ["layer"]},
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
        grid_offset=2,
    )

    assert summary["accepted_trials"] == 3
    assert summary["grid_offset"] == 2
    assert summary["teacher_checkpoint"] == "/tmp/teacher_best.pt"
    assert summary["test_accessed"] is False
    assert [trial["attempt_index"] for trial in summary["trials"]] == [2, 3, 4]
    for trial in summary["trials"]:
        config_path = tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "distill_config.yaml"
        record_path = tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "trial_config.json"
        config = load_yaml(config_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert config["teacher_checkpoint"] == "/tmp/teacher_best.pt"
        assert "hpo_overrides" in config
        reject_unknown_distill_config_keys(config)
        assert record["test_accessed"] is False
        assert record["execute_ready"] is True
    written_summary = json.loads((tmp_path / "distill_hpo_summary.json").read_text(encoding="utf-8"))
    assert written_summary["accepted_trials"] == 3
    assert written_summary["grid_offset"] == 2
    assert written_summary["test_accessed"] is False


def test_stage_f_train_eval_shards_cover_all_eligible_layers_once() -> None:
    config = load_yaml("configs/fff_distill_stage_f_train_eval_layer_shards.yaml")
    search_space = config["search_space"]
    shards = search_space["include_indices"]
    flattened = [index for shard in shards for index in shard]

    assert config["max_trials"] == 12
    assert len(shards) == 12
    assert sorted(flattened) == list(range(64))
    assert len(flattened) == len(set(flattened))
    assert search_space["router_recipe"] == ["vanilla_ste"]
    assert search_space["route_row_role"] == ["split_routing_output"]
    assert search_space["locoprop_refit"] == ["every_500"]


@pytest.mark.parametrize(
    ("base_path", "hpo_path", "route_row_role", "route_rows_contribute", "route_result_rows"),
    [
        (
            "configs/fff_distill_route_output_routing_only.yaml",
            "configs/fff_distill_route_output_routing_only_shards.yaml",
            "routing_only",
            False,
            0,
        ),
        (
            "configs/fff_distill_route_output_shared_all.yaml",
            "configs/fff_distill_route_output_shared_all_shards.yaml",
            "shared_routing_and_output",
            True,
            0,
        ),
        (
            "configs/fff_distill_route_output_split_all.yaml",
            "configs/fff_distill_route_output_split_all_shards.yaml",
            "split_routing_output",
            True,
            2,
        ),
    ],
)
def test_full_student_route_output_shard_configs_plan_all_layers(
    tmp_path,
    base_path: str,
    hpo_path: str,
    route_row_role: str,
    route_rows_contribute: bool,
    route_result_rows: int,
) -> None:
    base = load_yaml(base_path)
    hpo_config = load_yaml(hpo_path)

    summary = write_distill_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=12,
        max_attempts=12,
        seed=123,
    )

    assert summary["test_accessed"] is False
    assert summary["execution_contract"]["sample_split"] == "train_eval"
    shards = [trial["overrides"]["include_indices"] for trial in summary["trials"]]
    assert sorted(index for shard in shards for index in shard) == list(range(64))
    for trial in summary["trials"]:
        config = load_yaml(tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "distill_config.yaml")
        reject_unknown_distill_config_keys(config)
        assert config["fff"]["route_row_role"] == route_row_role
        assert config["fff"]["route_rows_contribute"] is route_rows_contribute
        assert config["fff"]["route_result_rows"] == route_result_rows
        assert config["fff"]["route_rows"] == 2
        assert config["fff"]["leaf_rows"] == 2
        assert config["hpo_overrides"]["include_indices"] == trial["overrides"]["include_indices"]


@pytest.mark.parametrize(
    ("hpo_path", "router_recipe"),
    [
        ("configs/fff_distill_router_st_gumbel_full_shards.yaml", "st_gumbel"),
        ("configs/fff_distill_router_utility_targeted_full_shards.yaml", "utility_targeted_ste"),
        ("configs/fff_distill_router_hard_em_full_shards.yaml", "hard_em_utility_ste"),
        ("configs/fff_distill_router_expert_choice_full_shards.yaml", "expert_choice_imitation"),
    ],
)
def test_full_student_router_family_shard_configs_plan_all_layers(
    tmp_path,
    hpo_path: str,
    router_recipe: str,
) -> None:
    base = load_yaml("configs/fff_distill_stage_f.yaml")
    hpo_config = load_yaml(hpo_path)

    summary = write_distill_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=12,
        max_attempts=12,
        seed=123,
    )

    assert summary["test_accessed"] is False
    assert summary["execution_contract"]["sample_split"] == "train_eval"
    shards = [trial["overrides"]["include_indices"] for trial in summary["trials"]]
    assert sorted(index for shard in shards for index in shard) == list(range(64))
    for trial in summary["trials"]:
        config = load_yaml(tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "distill_config.yaml")
        reject_unknown_distill_config_keys(config)
        assert config["router"]["recipe"] == router_recipe
        assert config["balance"]["recipe"] == "split_minleaf"
        assert config["fff"]["route_row_role"] == "split_routing_output"
        assert config["fff"]["route_result_rows"] == 2
        assert config["hpo_overrides"]["include_indices"] == trial["overrides"]["include_indices"]


def test_l7k_equal_budget_router_config_plans_strict_hard_layer_cases(tmp_path) -> None:
    base = load_yaml("configs/fff_distill_default.yaml")
    hpo_config = load_yaml("configs/fff_distill_router_equal_budget_hard_layers.yaml")
    contract = hpo_config["execution_contract"]
    cases = hpo_config["cases"]
    router_families = [
        "vanilla_ste",
        "clipped_ste",
        "sigmoid_surrogate_ste",
        "st_gumbel",
        "expert_choice_imitation",
        "utility_targeted_ste",
        "hard_em_utility_ste",
    ]
    launch_seeds = [21001, 21002, 21003]

    assert hpo_config["sampler"] == "cases"
    assert hpo_config["max_trials"] == len(router_families) * len(launch_seeds)
    assert contract["sample_split"] == "train_eval"
    assert contract["metric_split"] == "holdout"
    assert contract["metric_holdout_fraction"] == pytest.approx(0.10)
    assert contract["test_accessed"] is False
    assert contract["launch_seeds"] == launch_seeds
    assert len(cases) == hpo_config["max_trials"]
    assert [
        (case["router_family"], case["launch_seed"])
        for case in cases
    ] == [(family, seed) for family in router_families for seed in launch_seeds]

    summary = write_distill_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=hpo_config["max_trials"],
        max_attempts=hpo_config["max_trials"],
        seed=123,
    )

    assert summary["sampler"] == "cases"
    assert summary["accepted_trials"] == hpo_config["max_trials"]
    assert summary["test_accessed"] is False
    assert summary["execution_contract"] == contract
    assert [trial["attempt_index"] for trial in summary["trials"]] == list(range(hpo_config["max_trials"]))
    for trial in summary["trials"]:
        overrides = trial["overrides"]
        expected_seed = cases[int(trial["trial_index"])]["launch_seed"]
        assert overrides["seed"] == expected_seed
        assert overrides["include_indices"] == [35, 39, 41, 43, 45, 47]
        assert overrides["shared_unrouted_frac"] == pytest.approx(0.20)
        assert overrides["route_rows"] == 1
        assert overrides["route_result_rows"] == 2
        assert overrides["leaf_rows"] == 4
        assert overrides["depth"] == 5
        assert overrides["route_rows_contribute"] is True
        assert overrides["route_row_role"] == "split_routing_output"
        assert overrides["route_rows_output_count"] == "all"
        assert overrides["route_rows_output_fraction"] == pytest.approx(0.5)
        assert overrides["hard_routing"] is True
        assert overrides["balance_recipe"] == "split_minleaf"

        config = load_yaml(tmp_path / "trials" / f"trial_{trial['trial_index']:06d}" / "distill_config.yaml")
        reject_unknown_distill_config_keys(config)
        assert config["seed"] == expected_seed
        assert config["eligible_linear"]["include_indices"] == [35, 39, 41, 43, 45, 47]
        assert config["fff"]["route_row_role"] == "split_routing_output"
        assert config["fff"]["route_result_rows"] == 2
        assert config["balance"]["recipe"] == "split_minleaf"
        distill_config = LinearDistillConfig.from_mapping(config.get("distill"))
        assert distill_config.metric_holdout_fraction == pytest.approx(0.10)


def test_l7k_execution_contract_rejects_wrong_sample_split(tmp_path) -> None:
    hpo_config = load_yaml("configs/fff_distill_router_equal_budget_hard_layers.yaml")

    with pytest.raises(ValueError, match="sample_split"):
        run_distill_hpo_trials(
            base_config=load_yaml("configs/fff_distill_default.yaml"),
            hpo_config=hpo_config,
            output_dir=tmp_path,
            max_trials=1,
            max_attempts=1,
            seed=123,
            teacher_checkpoint="/tmp/teacher_best.pt",
            sample_split="val",
            trial_runner=lambda **kwargs: pytest.fail("runner should not be called"),
        )


def test_l7k_execution_contract_rejects_metric_holdout_mismatch(tmp_path) -> None:
    hpo_config = deepcopy(load_yaml("configs/fff_distill_router_equal_budget_hard_layers.yaml"))
    hpo_config["execution_contract"]["metric_holdout_fraction"] = 0.25

    with pytest.raises(ValueError, match="metric_holdout_fraction"):
        write_distill_hpo_trial_plan(
            base_config=load_yaml("configs/fff_distill_default.yaml"),
            hpo_config=hpo_config,
            output_dir=tmp_path,
            max_trials=1,
            max_attempts=1,
            seed=123,
        )


def test_distill_hpo_generated_config_still_rejects_real_unknown_keys(tmp_path) -> None:
    base = load_yaml("configs/fff_distill_default.yaml")
    hpo_config = {
        "sampler": "cases",
        "cases": [
            {
                "name": "runtime_valid",
                "overrides": {
                    "include_indices": [35],
                    "router_recipe": "vanilla_ste",
                    "route_rows": 1,
                    "route_row_role": "routing_only",
                    "locoprop_refit": "off",
                },
            }
        ],
    }

    write_distill_hpo_trial_plan(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=1,
        max_attempts=1,
        seed=123,
        teacher_checkpoint="/tmp/teacher_best.pt",
    )

    config = load_yaml(tmp_path / "trials" / "trial_000000" / "distill_config.yaml")
    assert config["hpo_overrides"]["case_name"] == "runtime_valid"
    reject_unknown_distill_config_keys(config)

    config["stale_typo"] = True
    with pytest.raises(ValueError, match="unknown top-level keys: stale_typo"):
        reject_unknown_distill_config_keys(config)


def test_distill_hpo_execute_runs_trials_and_records_mixed_results(tmp_path) -> None:
    base = load_yaml("configs/fff_distill_default.yaml")
    hpo_config = load_yaml("configs/fff_distill_hpo.yaml")
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        output_dir = kwargs["output_dir"]
        assert hasattr(output_dir, "name")
        trial_index = int(output_dir.name.rsplit("_", maxsplit=1)[1])  # type: ignore[union-attr]
        if trial_index == 1:
            return {"status": "failed_logic", "reason": "synthetic", "test_accessed": False}
        return {
            "status": "succeeded",
            "returncode": 0,
            "distill_mse": 0.1,
            "test_accessed": False,
        }

    summary = run_distill_hpo_trials(
        base_config=base,
        hpo_config=hpo_config,
        output_dir=tmp_path,
        max_trials=2,
        max_attempts=16,
        seed=123,
        teacher_checkpoint="/tmp/teacher_best.pt",
        trial_runner=fake_runner,
    )

    assert summary["mode"] == "distill_hpo_execute"
    assert summary["status"] == "completed"
    assert summary["succeeded"] == 1
    assert summary["failed_logic"] == 1
    assert summary["test_accessed"] is False
    assert len(calls) == 2
    assert calls[0]["teacher_checkpoint"] == "/tmp/teacher_best.pt"
    assert calls[0]["quick_smoke"] is False
    first_result = json.loads((tmp_path / "trials" / "trial_000000" / "trial_result.json").read_text())
    second_result = json.loads((tmp_path / "trials" / "trial_000001" / "trial_result.json").read_text())
    assert first_result["status"] == "succeeded"
    assert second_result["status"] == "failed_logic"


def test_distill_hpo_execute_requires_teacher_checkpoint_for_non_smoke(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="teacher_checkpoint is required"):
        run_distill_hpo_trials(
            base_config=load_yaml("configs/fff_distill_default.yaml"),
            hpo_config=load_yaml("configs/fff_distill_hpo.yaml"),
            output_dir=tmp_path,
            max_trials=1,
            max_attempts=16,
            seed=123,
            teacher_checkpoint=None,
            quick_smoke=False,
            trial_runner=lambda **kwargs: pytest.fail("runner should not be called"),
        )


def test_distill_hpo_cli_execute_uses_runner_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    output_dir = tmp_path / "execute"
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "succeeded", "returncode": 0, "test_accessed": False}

    monkeypatch.setattr(distill_hpo, "run_distill_trial_command", fake_runner)
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
            "1",
            "--max-attempts",
            "16",
            "--execute-trials",
            "true",
            "--quick-smoke",
            "true",
        ],
    )

    assert distill_hpo_main() == 0
    summary = json.loads((output_dir / "distill_hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "distill_hpo_execute"
    assert summary["succeeded"] == 1
    assert summary["test_accessed"] is False
    assert calls[0]["quick_smoke"] is True
    assert calls[0]["teacher_checkpoint"] is None


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
