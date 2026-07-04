from __future__ import annotations

import argparse
import copy
import itertools
import math
import random
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from cifar_mamba_fff.models.fff_linear import FFFLinearConfig
from cifar_mamba_fff.utils import bool_arg, load_yaml, write_json

RouteRowRole = Literal["routing_only", "shared_routing_and_output", "split_routing_output"]
RouteRowsOutputCount = int | Literal["all"] | None
DistillHpoSampler = Literal["random", "grid"]

ROUTE_ROW_ROLES: tuple[RouteRowRole, ...] = (
    "routing_only",
    "shared_routing_and_output",
    "split_routing_output",
)
ROUTE_CONTROL_KEYS = {
    "route_rows_contribute",
    "route_row_role",
    "route_result_rows",
    "route_rows_output_count",
    "route_rows_output_fraction",
}
CONDITIONAL_ROUTE_CONTROL_KEYS = {
    "route_output_controls",
    "route_row_controls",
    "route_control_space",
}
FFF_OVERRIDE_KEYS = {
    "shared_unrouted_frac",
    "shared_rows",
    "route_rows",
    "route_result_rows",
    "leaf_rows",
    "depth",
    "route_rows_contribute",
    "route_row_role",
    "route_rows_output_count",
    "route_rows_output_fraction",
    "activation",
    "hard_routing",
    "train_temperature",
    "region_leak",
    "master_leaf",
    "bias",
}
BALANCE_OVERRIDE_MAP = {
    "balance_recipe": "recipe",
    "balance_coeff": "coeff",
    "min_leaf_tokens": "min_leaf_tokens",
    "margin": "margin",
    "margin_coeff": "margin_coeff",
    "master_leaf_penalty": "master_leaf_penalty",
}
ROUTER_OVERRIDE_MAP = {
    "router_recipe": "recipe",
    "router_loss_coeff": "loss_coeff",
    "utility_loss_coeff": "loss_coeff",
    "imitation_coeff": "loss_coeff",
    "temperature": "temperature",
    "tau": "temperature",
    "clip_margin": "clip",
    "utility_tau": "utility_temperature",
    "capacity_factor": "expert_choice_capacity_factor",
}
LOCO_PROP_OVERRIDE_MAP = {
    "ridge_lambda": "ridge_lambda",
    "locoprop_blend_alpha": "blend_alpha",
    "damp_optimizer_state_after_refit": "damp_optimizer_state_after_refit",
}
KNOWN_OVERRIDE_KEYS = (
    FFF_OVERRIDE_KEYS
    | set(BALANCE_OVERRIDE_MAP)
    | set(ROUTER_OVERRIDE_MAP)
    | set(LOCO_PROP_OVERRIDE_MAP)
    | {"locoprop_refit"}
)


@dataclass(frozen=True)
class DistillHpoCandidate:
    trial_index: int
    attempt_index: int
    overrides: dict[str, object]

    def record(self) -> dict[str, object]:
        return {
            "trial_index": self.trial_index,
            "attempt_index": self.attempt_index,
            "overrides": self.overrides,
        }


DistillTrialRunner = Callable[..., dict[str, object]]


def _sample_loguniform(rng: random.Random, bounds: Sequence[object]) -> float:
    if len(bounds) != 2:
        raise ValueError("loguniform bounds must contain exactly two values")
    low, high = float(bounds[0]), float(bounds[1])
    if low <= 0.0 or high <= low:
        raise ValueError("loguniform bounds must be positive and increasing")
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _choice(values: object, *, key: str, rng: random.Random) -> object:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"search_space.{key} must be a non-empty sequence")
    return rng.choice(list(values))


def _choice_values(values: object, *, key: str) -> list[object]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"search_space.{key} must be a non-empty sequence")
    return list(values)


def _sample_mapping(search_space: Mapping[str, object], *, rng: random.Random) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key, values in search_space.items():
        if key.endswith("_loguniform"):
            overrides[key.removesuffix("_loguniform")] = _sample_loguniform(rng, values)  # type: ignore[arg-type]
        else:
            overrides[key] = _choice(values, key=key, rng=rng)
    return overrides


def _positive_ints(values: object, *, key: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"search_space.{key} must be a non-empty sequence")
    positive: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            positive.append(value)
    return positive


def _count_requested(count: object) -> bool:
    return count not in (None, 0)


def _fraction_requested(fraction: object) -> bool:
    return fraction is not None and float(fraction) > 0.0


def _selected_rows_for_pair(
    *,
    max_rows: int,
    count: RouteRowsOutputCount,
    fraction: object,
) -> int:
    candidates: list[int] = []
    if count is None:
        if fraction is None:
            candidates.append(max_rows)
    elif count == "all":
        candidates.append(max_rows)
    else:
        candidates.append(min(int(count), max_rows))

    if fraction is not None:
        fraction_value = float(fraction)
        if not 0.0 <= fraction_value <= 1.0:
            raise ValueError("route_rows_output_fraction must be in [0, 1]")
        candidates.append(min(math.ceil(fraction_value * max_rows), max_rows))
    return min(candidates) if candidates else 0


def _sample_output_controls(
    search_space: Mapping[str, object],
    *,
    role: RouteRowRole,
    max_rows: int,
    rng: random.Random,
) -> dict[str, object]:
    if role == "routing_only":
        return {
            "route_rows_output_count": 0,
            "route_rows_output_fraction": None,
        }

    counts = list(_sequence_or_default(search_space.get("route_rows_output_count"), [None]))
    fractions = list(_sequence_or_default(search_space.get("route_rows_output_fraction"), [None]))
    valid_pairs: list[tuple[RouteRowsOutputCount, object]] = []
    for count in counts:
        if count is not None and count != "all" and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError("route_rows_output_count choices must be non-negative ints, 'all', or null")
        for fraction in fractions:
            selected_rows = _selected_rows_for_pair(
                max_rows=max_rows,
                count=count,  # type: ignore[arg-type]
                fraction=fraction,
            )
            if selected_rows > 0:
                valid_pairs.append((count, fraction))  # type: ignore[arg-type]
    if not valid_pairs:
        raise ValueError(f"{role} requires at least one positive route output control choice")
    count, fraction = rng.choice(valid_pairs)
    return {
        "route_rows_output_count": count,
        "route_rows_output_fraction": fraction,
    }


def _grid_output_controls(
    search_space: Mapping[str, object],
    *,
    role: RouteRowRole,
    max_rows: int,
) -> list[dict[str, object]]:
    if role == "routing_only":
        return [
            {
                "route_rows_output_count": 0,
                "route_rows_output_fraction": None,
            }
        ]

    counts = list(_sequence_or_default(search_space.get("route_rows_output_count"), [None]))
    fractions = list(_sequence_or_default(search_space.get("route_rows_output_fraction"), [None]))
    controls: list[dict[str, object]] = []
    for count in counts:
        if count is not None and count != "all" and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError("route_rows_output_count choices must be non-negative ints, 'all', or null")
        for fraction in fractions:
            selected_rows = _selected_rows_for_pair(
                max_rows=max_rows,
                count=count,  # type: ignore[arg-type]
                fraction=fraction,
            )
            if selected_rows > 0:
                controls.append(
                    {
                        "route_rows_output_count": count,
                        "route_rows_output_fraction": fraction,
                    }
                )
    if not controls:
        raise ValueError(f"{role} requires at least one positive route output control choice")
    return controls


def _sequence_or_default(value: object, default: Sequence[object]) -> Sequence[object]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("route control choices must be non-empty sequences")
    return value


def _conditional_route_space(search_space: Mapping[str, object]) -> Mapping[str, object] | None:
    present = [key for key in CONDITIONAL_ROUTE_CONTROL_KEYS if key in search_space]
    if len(present) > 1:
        raise ValueError(
            "only one conditional route control section is allowed: "
            + ", ".join(sorted(CONDITIONAL_ROUTE_CONTROL_KEYS))
        )
    if not present:
        return None
    value = search_space[present[0]]
    if not isinstance(value, Mapping):
        raise ValueError(f"search_space.{present[0]} must be a mapping")
    return value


def _sample_role(search_space: Mapping[str, object], *, rng: random.Random) -> RouteRowRole:
    if "route_row_role" in search_space:
        role = _choice(search_space["route_row_role"], key="route_row_role", rng=rng)
    elif "route_rows_contribute" in search_space:
        contributes = _choice(search_space["route_rows_contribute"], key="route_rows_contribute", rng=rng)
        role = "shared_routing_and_output" if contributes else "routing_only"
    else:
        role = "routing_only"
    if role not in ROUTE_ROW_ROLES:
        raise ValueError(f"unknown route_row_role: {role!r}")
    return role


def _grid_roles(search_space: Mapping[str, object]) -> list[RouteRowRole]:
    if "route_row_role" in search_space:
        role_values = _choice_values(search_space["route_row_role"], key="route_row_role")
        roles = [str(role) for role in role_values]
    elif "route_rows_contribute" in search_space:
        contributes_values = _choice_values(
            search_space["route_rows_contribute"],
            key="route_rows_contribute",
        )
        roles = [
            "shared_routing_and_output" if contributes else "routing_only"
            for contributes in contributes_values
        ]
    else:
        roles = ["routing_only"]

    normalized: list[RouteRowRole] = []
    for role in roles:
        if role not in ROUTE_ROW_ROLES:
            raise ValueError(f"unknown route_row_role: {role!r}")
        normalized.append(role)  # type: ignore[arg-type]
    return normalized


def _route_rows_from_overrides(overrides: Mapping[str, object]) -> int:
    value = overrides.get("route_rows", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("route_rows must be 1 or 2 before route-output sampling")
    return value


def _sample_role_controls(
    search_space: Mapping[str, object],
    *,
    role: RouteRowRole,
    route_rows: int,
    rng: random.Random,
) -> dict[str, object]:
    route_result_rows = 0
    max_output_rows = route_rows
    if role == "split_routing_output":
        positive = _positive_ints(
            search_space.get("route_result_rows", [1]),
            key="route_result_rows",
        )
        if not positive:
            raise ValueError("split_routing_output requires a positive route_result_rows choice")
        route_result_rows = rng.choice(positive)
        max_output_rows = route_result_rows

    output_controls = _sample_output_controls(
        search_space,
        role=role,
        max_rows=max_output_rows,
        rng=rng,
    )
    return {
        "route_rows_contribute": role != "routing_only",
        "route_row_role": role,
        "route_result_rows": route_result_rows,
        **output_controls,
    }


def _grid_role_controls(
    search_space: Mapping[str, object],
    *,
    role: RouteRowRole,
    route_rows: int,
) -> list[dict[str, object]]:
    if role == "routing_only":
        return [
            {
                "route_rows_contribute": False,
                "route_row_role": role,
                "route_result_rows": 0,
                "route_rows_output_count": 0,
                "route_rows_output_fraction": None,
            }
        ]

    if role == "split_routing_output":
        route_result_rows_values = _positive_ints(
            search_space.get("route_result_rows", [1]),
            key="route_result_rows",
        )
        if not route_result_rows_values:
            raise ValueError("split_routing_output requires a positive route_result_rows choice")
    else:
        route_result_rows_values = [0]

    controls: list[dict[str, object]] = []
    for route_result_rows in route_result_rows_values:
        max_output_rows = route_result_rows if role == "split_routing_output" else route_rows
        for output_controls in _grid_output_controls(
            search_space,
            role=role,
            max_rows=max_output_rows,
        ):
            controls.append(
                {
                    "route_rows_contribute": True,
                    "route_row_role": role,
                    "route_result_rows": route_result_rows,
                    **output_controls,
                }
            )
    return controls


def sample_distill_overrides(
    search_space: Mapping[str, object],
    *,
    rng: random.Random,
) -> dict[str, object]:
    """Sample role-aware distillation HPO overrides.

    The route-output controls are conditional because many independent flat
    combinations are invalid or equivalent dead trials. This function accepts
    both the legacy flat search space and an optional role-keyed
    ``route_output_controls`` mapping.
    """

    conditional_route_space = _conditional_route_space(search_space)
    generic_space = {
        key: value
        for key, value in search_space.items()
        if key not in ROUTE_CONTROL_KEYS and key not in CONDITIONAL_ROUTE_CONTROL_KEYS
    }
    overrides = _sample_mapping(generic_space, rng=rng)
    role = _sample_role(search_space, rng=rng)
    route_rows = _route_rows_from_overrides(overrides)

    role_space: Mapping[str, object]
    if conditional_route_space is None:
        role_space = search_space
    else:
        value = conditional_route_space.get(role, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"route controls for {role} must be a mapping")
        role_space = value
    overrides.update(
        _sample_role_controls(
            role_space,
            role=role,
            route_rows=route_rows,
            rng=rng,
        )
    )
    return canonicalize_distill_route_overrides(overrides)


def _grid_mapping(search_space: Mapping[str, object]) -> Iterator[dict[str, object]]:
    if not search_space:
        yield {}
        return
    keys: list[str] = []
    values_by_key: list[list[object]] = []
    for key, values in search_space.items():
        if key.endswith("_loguniform"):
            raise ValueError(f"grid sampler does not support search_space.{key}")
        keys.append(key)
        values_by_key.append(_choice_values(values, key=key))
    for values in itertools.product(*values_by_key):
        yield dict(zip(keys, values, strict=True))


def iter_grid_distill_overrides(search_space: Mapping[str, object]) -> Iterator[dict[str, object]]:
    """Generate deterministic route-aware Cartesian distillation HPO overrides."""

    conditional_route_space = _conditional_route_space(search_space)
    generic_space = {
        key: value
        for key, value in search_space.items()
        if key not in ROUTE_CONTROL_KEYS and key not in CONDITIONAL_ROUTE_CONTROL_KEYS
    }

    for generic_overrides in _grid_mapping(generic_space):
        route_rows = _route_rows_from_overrides(generic_overrides)
        for role in _grid_roles(search_space):
            role_space: Mapping[str, object]
            if conditional_route_space is None:
                role_space = search_space
            else:
                value = conditional_route_space.get(role, {})
                if value is None:
                    value = {}
                if not isinstance(value, Mapping):
                    raise ValueError(f"route controls for {role} must be a mapping")
                role_space = value
            for role_controls in _grid_role_controls(
                role_space,
                role=role,
                route_rows=route_rows,
            ):
                overrides = dict(generic_overrides)
                overrides.update(role_controls)
                yield canonicalize_distill_route_overrides(overrides)


def canonicalize_distill_route_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    """Normalize legacy route knobs before constructing an ``FFFLinear``."""

    normalized = dict(overrides)
    role = normalized.get("route_row_role", "routing_only")
    contributes = bool(normalized.get("route_rows_contribute", False))
    if contributes and role == "routing_only":
        role = "shared_routing_and_output"
    if role not in ROUTE_ROW_ROLES:
        raise ValueError(f"unknown route_row_role: {role!r}")

    count = normalized.get("route_rows_output_count")
    fraction = normalized.get("route_rows_output_fraction")
    route_rows = _route_rows_from_overrides(normalized)
    result_rows = normalized.get("route_result_rows", 0)
    if role == "split_routing_output":
        if isinstance(result_rows, bool) or not isinstance(result_rows, int) or result_rows <= 0:
            raise ValueError("split_routing_output requires positive route_result_rows")
        max_output_rows = result_rows
    else:
        max_output_rows = route_rows
    selected_rows = _selected_rows_for_pair(
        max_rows=max_output_rows,
        count=count,  # type: ignore[arg-type]
        fraction=fraction,
    )
    if selected_rows == 0:
        role = "routing_only"

    if role == "routing_only":
        normalized["route_rows_contribute"] = False
        normalized["route_row_role"] = "routing_only"
        normalized["route_result_rows"] = 0
        normalized["route_rows_output_count"] = 0
        normalized["route_rows_output_fraction"] = None
    elif role == "shared_routing_and_output":
        normalized["route_rows_contribute"] = True
        normalized["route_row_role"] = role
        normalized["route_result_rows"] = 0
        _validate_output_controls_present(normalized, max_output_rows=max_output_rows)
    else:
        normalized["route_rows_contribute"] = True
        normalized["route_row_role"] = role
        _validate_output_controls_present(normalized, max_output_rows=max_output_rows)

    fff_fields = set(FFFLinearConfig.__dataclass_fields__)
    fff_kwargs = {
        "in_features": 16,
        "out_features": 8,
        "shared_rows": 1,
    } | {key: value for key, value in normalized.items() if key in fff_fields}
    FFFLinearConfig(**fff_kwargs).validate()
    return normalized


def _freeze_for_identity(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple((key, _freeze_for_identity(nested)) for key, nested in sorted(value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_for_identity(nested) for nested in value)
    return value


def _overrides_identity(overrides: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple((key, _freeze_for_identity(value)) for key, value in sorted(overrides.items()))


def distill_hpo_sampler(hpo_config: Mapping[str, object]) -> DistillHpoSampler:
    sampler = hpo_config.get("sampler", hpo_config.get("sampling", "random"))
    if sampler not in ("random", "grid"):
        raise ValueError("distill HPO sampler must be 'random' or 'grid'")
    return sampler  # type: ignore[return-value]


def _validate_output_controls_present(
    overrides: Mapping[str, object],
    *,
    max_output_rows: int,
) -> None:
    if (
        _selected_rows_for_pair(
            max_rows=max_output_rows,
            count=overrides.get("route_rows_output_count"),  # type: ignore[arg-type]
            fraction=overrides.get("route_rows_output_fraction"),
        )
        == 0
    ):
        raise ValueError(f"{overrides.get('route_row_role')} requires route output controls")


def sample_valid_distill_hpo_candidates(
    search_space: Mapping[str, object],
    *,
    max_trials: int,
    max_attempts: int,
    rng: random.Random,
    sampler: DistillHpoSampler = "random",
    validate_fn: Callable[[Mapping[str, object]], bool] | None = None,
) -> list[DistillHpoCandidate]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if max_attempts < max_trials:
        raise ValueError("max_attempts must be >= max_trials")
    candidates: list[DistillHpoCandidate] = []
    if sampler == "random":
        for attempt_index in range(max_attempts):
            if len(candidates) >= max_trials:
                break
            overrides = sample_distill_overrides(search_space, rng=rng)
            if validate_fn is not None and not validate_fn(overrides):
                continue
            candidates.append(
                DistillHpoCandidate(
                    trial_index=len(candidates),
                    attempt_index=attempt_index,
                    overrides=overrides,
                )
            )
        return candidates
    if sampler != "grid":
        raise ValueError("distill HPO sampler must be 'random' or 'grid'")

    seen: set[tuple[tuple[str, object], ...]] = set()
    for attempt_index, overrides in enumerate(iter_grid_distill_overrides(search_space)):
        if attempt_index >= max_attempts or len(candidates) >= max_trials:
            break
        identity = _overrides_identity(overrides)
        if identity in seen:
            continue
        seen.add(identity)
        if validate_fn is not None and not validate_fn(overrides):
            continue
        candidates.append(
            DistillHpoCandidate(
                trial_index=len(candidates),
                attempt_index=attempt_index,
                overrides=overrides,
            )
        )
    return candidates


def _as_mapping(value: object, *, section: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return dict(value)


def _locoprop_refit_config(value: object) -> dict[str, object]:
    if value in (None, "off", False):
        return {"enabled": False}
    if not isinstance(value, str) or not value.startswith("every_"):
        raise ValueError("locoprop_refit must be off or every_<positive-int>")
    try:
        interval = int(value.removeprefix("every_"))
    except ValueError as exc:
        raise ValueError("locoprop_refit must be off or every_<positive-int>") from exc
    if interval <= 0:
        raise ValueError("locoprop_refit interval must be positive")
    return {"enabled": True, "interval_steps": interval}


def _reject_unknown_override_keys(overrides: Mapping[str, object]) -> None:
    unknown = sorted(set(overrides) - KNOWN_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown distill HPO override keys: {', '.join(unknown)}")


def _apply_alias_overrides(
    section: dict[str, object],
    overrides: Mapping[str, object],
    alias_map: Mapping[str, str],
    *,
    section_name: str,
) -> None:
    assigned_by_target: dict[str, str] = {}
    for source, target in alias_map.items():
        if source not in overrides:
            continue
        if target in assigned_by_target:
            raise ValueError(
                f"duplicate {section_name} override aliases for {target}: "
                f"{assigned_by_target[target]} and {source}"
            )
        assigned_by_target[target] = source
        section[target] = overrides[source]


def _canonical_fff_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    fff_overrides = {key: overrides[key] for key in FFF_OVERRIDE_KEYS if key in overrides}
    if set(fff_overrides).intersection(ROUTE_CONTROL_KEYS | {"route_rows"}):
        route_normalized = canonicalize_distill_route_overrides(fff_overrides)
        fff_overrides.update(
            {
                key: route_normalized[key]
                for key in ROUTE_CONTROL_KEYS
                if key in route_normalized
            }
        )
    return fff_overrides


def apply_distill_hpo_overrides(
    base_config: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    _reject_unknown_override_keys(overrides)
    config: dict[str, object] = copy.deepcopy(dict(base_config))
    fff = _as_mapping(config.get("fff"), section="fff")
    balance = _as_mapping(config.get("balance"), section="balance")
    router = _as_mapping(config.get("router"), section="router")
    locoprop = _as_mapping(config.get("locoprop"), section="locoprop")

    fff.update(_canonical_fff_overrides(overrides))
    _apply_alias_overrides(
        balance,
        overrides,
        BALANCE_OVERRIDE_MAP,
        section_name="balance",
    )
    _apply_alias_overrides(
        router,
        overrides,
        ROUTER_OVERRIDE_MAP,
        section_name="router",
    )

    if "locoprop_refit" in overrides:
        locoprop.update(_locoprop_refit_config(overrides["locoprop_refit"]))
    _apply_alias_overrides(
        locoprop,
        overrides,
        LOCO_PROP_OVERRIDE_MAP,
        section_name="locoprop",
    )

    config["fff"] = fff
    config["balance"] = balance
    config["router"] = router
    config["locoprop"] = locoprop
    config["hpo_overrides"] = dict(overrides)
    return config


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=True), encoding="utf-8")


def write_distill_hpo_trial_plan(
    *,
    base_config: Mapping[str, object],
    hpo_config: Mapping[str, object],
    output_dir: Path,
    max_trials: int,
    max_attempts: int,
    seed: int,
    teacher_checkpoint: str | None = None,
) -> dict[str, object]:
    search_space = hpo_config.get("search_space")
    if not isinstance(search_space, Mapping):
        raise ValueError("distill HPO config must contain a search_space mapping")
    sampler = distill_hpo_sampler(hpo_config)
    candidates = sample_valid_distill_hpo_candidates(
        search_space,
        max_trials=max_trials,
        max_attempts=max_attempts,
        rng=random.Random(seed),
        sampler=sampler,
    )
    if not candidates:
        raise RuntimeError("distill HPO produced zero valid candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_records: list[dict[str, object]] = []
    for candidate in candidates:
        trial_dir = output_dir / "trials" / f"trial_{candidate.trial_index:06d}"
        trial_config = apply_distill_hpo_overrides(base_config, candidate.overrides)
        if teacher_checkpoint is not None:
            trial_config["teacher_checkpoint"] = teacher_checkpoint
        config_path = trial_dir / "distill_config.yaml"
        _write_yaml(config_path, trial_config)
        record = {
            **candidate.record(),
            "config_path": str(config_path),
            "output_dir": str(trial_dir),
            "execute_ready": teacher_checkpoint is not None,
            "test_accessed": False,
        }
        write_json(trial_dir / "trial_config.json", record | {"config": trial_config})
        trial_records.append(record)
    summary = {
        "mode": "distill_hpo_plan",
        "status": "planned",
        "study_name": hpo_config.get("study_name", "fff_distill_hpo"),
        "requested_trials": max_trials,
        "accepted_trials": len(candidates),
        "max_attempts": max_attempts,
        "sampler": sampler,
        "seed": seed,
        "teacher_checkpoint": teacher_checkpoint,
        "test_accessed": False,
        "trials": trial_records,
    }
    write_json(output_dir / "distill_hpo_summary.json", summary)
    return summary


def run_distill_trial_command(
    *,
    config_path: Path,
    output_dir: Path,
    teacher_checkpoint: str | None,
    quick_smoke: bool,
    sample_split: str = "train",
    max_sample_batches: int = 1,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "cifar_mamba_fff.distill_linears",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--quick-smoke",
        str(quick_smoke).lower(),
        "--sample-split",
        sample_split,
        "--max-sample-batches",
        str(max_sample_batches),
    ]
    if teacher_checkpoint is not None:
        command.extend(["--checkpoint", teacher_checkpoint])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    summary_path = output_dir / "distill_summary.json"
    payload: dict[str, object] = {
        "status": "succeeded" if completed.returncode == 0 else "failed_logic",
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout[-8192:],
        "stderr": completed.stderr[-8192:],
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "test_accessed": False,
    }
    if summary_path.exists():
        payload["summary"] = load_yaml(summary_path)
    return payload


def run_distill_hpo_trials(
    *,
    base_config: Mapping[str, object],
    hpo_config: Mapping[str, object],
    output_dir: Path,
    max_trials: int,
    max_attempts: int,
    seed: int,
    teacher_checkpoint: str | None,
    quick_smoke: bool = False,
    sample_split: str = "train",
    max_sample_batches: int = 1,
    trial_runner: DistillTrialRunner = run_distill_trial_command,
) -> dict[str, object]:
    if not quick_smoke and teacher_checkpoint is None:
        raise RuntimeError("teacher_checkpoint is required for executable distill HPO")
    plan = write_distill_hpo_trial_plan(
        base_config=base_config,
        hpo_config=hpo_config,
        output_dir=output_dir,
        max_trials=max_trials,
        max_attempts=max_attempts,
        seed=seed,
        teacher_checkpoint=teacher_checkpoint,
    )
    succeeded = 0
    failed_logic = 0
    trial_results: list[dict[str, object]] = []
    for trial in plan["trials"]:
        if not isinstance(trial, Mapping):
            raise ValueError("distill HPO trial records must be mappings")
        trial_index = int(trial["trial_index"])
        trial_dir = output_dir / "trials" / f"trial_{trial_index:06d}"
        config_path = trial_dir / "distill_config.yaml"
        try:
            runner_result = trial_runner(
                config_path=config_path,
                output_dir=trial_dir,
                teacher_checkpoint=teacher_checkpoint,
                quick_smoke=quick_smoke,
                sample_split=sample_split,
                max_sample_batches=max_sample_batches,
            )
        except Exception as exc:
            runner_result = {
                "status": "failed_logic",
                "returncode": None,
                "reason": f"{type(exc).__name__}: {exc}",
                "test_accessed": False,
            }
        status = str(runner_result.get("status", "failed_logic"))
        if status == "succeeded":
            succeeded += 1
        else:
            failed_logic += 1
        record = {
            **dict(trial),
            "status": status,
            "result": runner_result,
            "test_accessed": bool(runner_result.get("test_accessed", False)),
        }
        write_json(trial_dir / "trial_result.json", record)
        trial_results.append(record)

    summary = {
        **{key: value for key, value in plan.items() if key != "trials"},
        "mode": "distill_hpo_execute",
        "status": "completed" if succeeded > 0 else "failed_zero_successes",
        "succeeded": succeeded,
        "failed_logic": failed_logic,
        "quick_smoke": quick_smoke,
        "sample_split": sample_split,
        "max_sample_batches": max_sample_batches,
        "test_accessed": any(bool(result["test_accessed"]) for result in trial_results),
        "trials": trial_results,
    }
    write_json(output_dir / "distill_hpo_summary.json", summary)
    if succeeded == 0:
        raise RuntimeError("distill HPO completed with zero successful trials")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/fff_distill_default.yaml")
    parser.add_argument("--hpo-config", default="configs/fff_distill_hpo.yaml")
    parser.add_argument("--output-dir", default="outputs/distill_hpo")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--sample-split", choices=("train", "val"), default="train")
    parser.add_argument("--max-sample-batches", type=int, default=1)
    parser.add_argument(
        "--execute-trials",
        type=bool_arg,
        default=False,
        help="Execute planned trials instead of only materializing configs.",
    )
    args = parser.parse_args(argv)
    if args.execute_trials:
        summary = run_distill_hpo_trials(
            base_config=load_yaml(args.base_config),
            hpo_config=load_yaml(args.hpo_config),
            output_dir=Path(args.output_dir),
            max_trials=args.max_trials,
            max_attempts=args.max_attempts,
            seed=args.seed,
            teacher_checkpoint=args.teacher_checkpoint,
            quick_smoke=args.quick_smoke,
            sample_split=args.sample_split,
            max_sample_batches=args.max_sample_batches,
            trial_runner=run_distill_trial_command,
        )
        print(f"distill HPO executed: {summary['succeeded']} succeeded")
        return 0
    summary = write_distill_hpo_trial_plan(
        base_config=load_yaml(args.base_config),
        hpo_config=load_yaml(args.hpo_config),
        output_dir=Path(args.output_dir),
        max_trials=args.max_trials,
        max_attempts=args.max_attempts,
        seed=args.seed,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    print(f"distill HPO plan written: {summary['accepted_trials']} trials")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
