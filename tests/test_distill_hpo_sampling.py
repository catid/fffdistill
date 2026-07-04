from __future__ import annotations

import random
from collections.abc import Mapping

import pytest

import cifar_mamba_fff.hpo.distill_hpo as distill_hpo
from cifar_mamba_fff.hpo.distill_hpo import (
    canonicalize_distill_route_overrides,
    sample_distill_overrides,
    sample_valid_distill_hpo_candidates,
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
