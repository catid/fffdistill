from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from torch import nn

_ADAMW_GROUP = "adamw"
_MUON_GROUP = "muon"
_CLASSIFIER_NAME_COMPONENTS = frozenset({"classifier", "head"})


@dataclass(frozen=True)
class ParamAssignment:
    name: str
    shape: tuple[int, ...]
    group: Literal["muon", "adamw"]
    reason: str


def _name_components(name: str) -> tuple[str, ...]:
    return tuple(component for part in name.lower().split(".") for component in part.split("_") if component)


def _module_for_parameter(model: nn.Module, name: str) -> nn.Module | None:
    module_name, separator, _ = name.rpartition(".")
    if not separator:
        return model
    modules = dict(model.named_modules())
    return modules.get(module_name)


def _is_norm_module(module: nn.Module | None) -> bool:
    if module is None:
        return False
    norm_types: tuple[type[nn.Module], ...] = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.batchnorm._BatchNorm,
        nn.modules.instancenorm._InstanceNorm,
        nn.modules.normalization.RMSNorm,
    )
    return isinstance(module, norm_types)


def _is_embedding_module(module: nn.Module | None) -> bool:
    return isinstance(module, (nn.Embedding, nn.EmbeddingBag))


def _excluded_name_reason(name: str, adamw_name_fragments: tuple[str, ...]) -> str | None:
    components = _name_components(name)
    lowered = name.lower()

    if "bias" in components:
        return "bias parameter"
    if components and _CLASSIFIER_NAME_COMPONENTS.intersection(components):
        return "classifier/head parameter name"
    if "embedding" in components or "embed" in components:
        return "embedding parameter name"
    if "norm" in components or "bn" in components:
        return "normalization parameter name"

    handled_fragments = {
        "bias",
        "norm",
        "bn",
        "embedding",
        "embed",
        "head",
        "classifier",
    }
    for fragment in adamw_name_fragments:
        normalized_fragment = fragment.lower()
        if normalized_fragment in handled_fragments:
            continue
        if normalized_fragment in lowered:
            return f"name matched excluded fragment {fragment!r}"
    return None


def format_param_assignment(assignment: ParamAssignment) -> str:
    shape = "x".join(str(dimension) for dimension in assignment.shape) or "scalar"
    return (
        f"group={assignment.group} name={assignment.name} "
        f"shape={shape} reason={assignment.reason}"
    )


def format_param_assignments(assignments: Iterable[ParamAssignment]) -> list[str]:
    return [format_param_assignment(assignment) for assignment in assignments]


def _validate_split(
    model: nn.Module,
    muon_params: list[nn.Parameter],
    adamw_params: list[nn.Parameter],
) -> None:
    muon_ids = {id(param) for param in muon_params}
    adamw_ids = {id(param) for param in adamw_params}
    overlap = muon_ids & adamw_ids
    if overlap:
        raise ValueError(f"Muon/AdamW parameter split has {len(overlap)} overlapping parameters")

    assigned = muon_ids | adamw_ids
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    missing = trainable - assigned
    extra = assigned - trainable
    if missing or extra:
        raise ValueError(
            "Muon/AdamW parameter split is inconsistent: "
            f"missing={len(missing)} extra={len(extra)}"
        )


def split_muon_adamw_parameters(
    model: nn.Module,
    *,
    adamw_name_fragments: tuple[str, ...] = ("bias", "norm", "bn", "embedding", "embed", "head", "classifier"),
    assignment_logger: Callable[[str], None] | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[ParamAssignment]]:
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    assignments: list[ParamAssignment] = []
    seen: set[int] = set()

    for name, param in model.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        if id(param) in seen:
            raise ValueError(f"parameter {name} appears more than once")
        seen.add(id(param))

        shape = tuple(param.shape)
        module = _module_for_parameter(model, name)
        excluded_name_reason = _excluded_name_reason(name, adamw_name_fragments)
        if _is_embedding_module(module):
            adamw_params.append(param)
            assignments.append(ParamAssignment(name, shape, _ADAMW_GROUP, "embedding module"))
        elif _is_norm_module(module):
            adamw_params.append(param)
            assignments.append(ParamAssignment(name, shape, _ADAMW_GROUP, "normalization module"))
        elif excluded_name_reason is not None:
            adamw_params.append(param)
            assignments.append(ParamAssignment(name, shape, _ADAMW_GROUP, excluded_name_reason))
        elif param.ndim == 2:
            muon_params.append(param)
            assignments.append(ParamAssignment(name, shape, _MUON_GROUP, "hidden matrix parameter"))
        else:
            adamw_params.append(param)
            assignments.append(
                ParamAssignment(name, shape, _ADAMW_GROUP, f"{param.ndim}D parameter")
            )

    _validate_split(model, muon_params, adamw_params)
    if assignment_logger is not None:
        for line in format_param_assignments(assignments):
            assignment_logger(line)
    return muon_params, adamw_params, assignments
