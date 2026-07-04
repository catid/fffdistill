from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from cifar_mamba_fff.distill_linears import main
from cifar_mamba_fff.models.fff_linear import FFFLinearConfig

RouteRowRole = Literal["routing_only", "shared_routing_and_output", "split_routing_output"]
RouteRowsOutputCount = int | Literal["all"] | None

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


@dataclass(frozen=True)
class DistillHpoCandidate:
    trial_index: int
    attempt_index: int
    overrides: dict[str, object]


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
    validate_fn: Callable[[Mapping[str, object]], bool] | None = None,
) -> list[DistillHpoCandidate]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if max_attempts < max_trials:
        raise ValueError("max_attempts must be >= max_trials")
    candidates: list[DistillHpoCandidate] = []
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

if __name__ == "__main__":
    raise SystemExit(main())
