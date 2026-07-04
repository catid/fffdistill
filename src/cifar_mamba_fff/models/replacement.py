from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .fff_linear import FFFLinear


@dataclass(frozen=True)
class LinearReport:
    name: str
    in_features: int
    out_features: int
    weight_shape: tuple[int, int]
    bias_shape: tuple[int] | None
    parameters: int
    included: bool
    reason: str

    def log_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplacementRecord:
    name: str
    original_type: str
    replacement_type: str
    in_features: int
    out_features: int
    original_parameters: int
    replacement_parameters: int

    def log_record(self) -> dict[str, object]:
        return asdict(self)


def linear_parameter_count(module: nn.Linear) -> int:
    return module.in_features * module.out_features + (
        module.out_features if module.bias is not None else 0
    )


def _linear_reason(
    module: nn.Linear,
    *,
    min_in_features: int,
    min_out_features: int,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if module.in_features < min_in_features:
        reasons.append(f"in_features {module.in_features} < {min_in_features}")
    if module.out_features < min_out_features:
        reasons.append(f"out_features {module.out_features} < {min_out_features}")
    if reasons:
        return False, "skipped: " + "; ".join(reasons)
    return True, "eligible"


def discover_linear_layers(
    model: nn.Module, *, min_in_features: int = 64, min_out_features: int = 64
) -> list[LinearReport]:
    reports: list[LinearReport] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        included, reason = _linear_reason(
            module,
            min_in_features=min_in_features,
            min_out_features=min_out_features,
        )
        reports.append(
            LinearReport(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                weight_shape=tuple(module.weight.shape),
                bias_shape=tuple(module.bias.shape) if module.bias is not None else None,
                parameters=linear_parameter_count(module),
                included=included,
                reason=reason,
            )
        )
    return reports


def linear_reports_as_log_records(reports: Iterable[LinearReport]) -> list[dict[str, object]]:
    return [report.log_record() for report in reports]


def _flatten_feature_tensor(tensor: torch.Tensor, features: int) -> torch.Tensor:
    if tensor.shape[-1] != features:
        raise RuntimeError(f"expected last dimension {features}, got {tensor.shape[-1]}")
    return tensor.detach().reshape(-1, features).cpu()


class LinearCapture:
    def __init__(self, module: nn.Linear) -> None:
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self._handle = module.register_forward_hook(self._hook)
        self._closed = False

    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = _flatten_feature_tensor(inputs[0], self.in_features)
        y = _flatten_feature_tensor(output, self.out_features)
        self.inputs.append(x)
        self.outputs.append(y)

    def close(self) -> None:
        if self._closed:
            return
        self._handle.remove()
        self._closed = True

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.inputs:
            raise RuntimeError("no tensors captured")
        return torch.cat(self.inputs, dim=0), torch.cat(self.outputs, dim=0)


class LinearCaptureSet:
    def __init__(self, model: nn.Module, reports: Iterable[LinearReport]) -> None:
        modules = dict(model.named_modules())
        self.captures: dict[str, LinearCapture] = {}
        try:
            for report in reports:
                if not report.included:
                    continue
                module = modules.get(report.name)
                if not isinstance(module, nn.Linear):
                    raise KeyError(f"{report.name!r} is not an nn.Linear in the model")
                self.captures[report.name] = LinearCapture(module)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> LinearCaptureSet:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        for capture in self.captures.values():
            capture.close()

    def tensors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.captures[name].tensors()

    def all_tensors(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {name: capture.tensors() for name, capture in self.captures.items()}


def replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parent, child_name = _module_parent(root, dotted_name)
    if child_name not in parent._modules:
        raise KeyError(f"{dotted_name!r} is not a registered child module")
    parent._modules[child_name] = replacement


def _module_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError("dotted_name must name a child module")
    parent = root
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            raise KeyError(f"{dotted_name!r} does not resolve at {part!r}")
        parent = child
    return parent, parts[-1]


def get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    parent, child_name = _module_parent(root, dotted_name)
    child = parent._modules.get(child_name)
    if child is None:
        raise KeyError(f"{dotted_name!r} is not a registered child module")
    return child


def select_progressive_reports(
    reports: Iterable[LinearReport],
    *,
    step: int | None = None,
    step_size: int = 1,
    max_replacements: int | None = None,
) -> list[LinearReport]:
    if step is not None and step < 0:
        raise ValueError("step must be non-negative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if max_replacements is not None and max_replacements < 0:
        raise ValueError("max_replacements must be non-negative")

    eligible = [report for report in reports if report.included]
    if step is not None:
        eligible = eligible[: step * step_size]
    if max_replacements is not None:
        eligible = eligible[:max_replacements]
    return eligible


def _fff_kwargs_from_config(linear: nn.Linear, config: Mapping[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(config or {})
    shared_unrouted_frac = kwargs.pop("shared_unrouted_frac", None)
    if shared_unrouted_frac is not None and "shared_rows" not in kwargs:
        if not 0.0 <= float(shared_unrouted_frac) <= 1.0:
            raise ValueError("shared_unrouted_frac must be in [0, 1]")
        kwargs["shared_rows"] = round(linear.out_features * float(shared_unrouted_frac))
    kwargs.setdefault("bias", linear.bias is not None)
    return kwargs


def make_fff_replacement(
    linear: nn.Linear,
    *,
    config: Mapping[str, Any] | None = None,
) -> FFFLinear:
    kwargs = _fff_kwargs_from_config(linear, config)
    replacement = FFFLinear(
        linear.in_features,
        linear.out_features,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
        **kwargs,
    )
    replacement.train(linear.training)
    return replacement


ReplacementFactory = Callable[[str, nn.Linear, LinearReport], nn.Module]


def _module_matches_report_features(module: nn.Module, report: LinearReport) -> bool:
    return (
        getattr(module, "in_features", None) == report.in_features
        and getattr(module, "out_features", None) == report.out_features
    )


def replace_linear_layers(
    model: nn.Module,
    reports: Iterable[LinearReport],
    *,
    fff_config: Mapping[str, Any] | None = None,
    replacement_factory: ReplacementFactory | None = None,
    step: int | None = None,
    step_size: int = 1,
    max_replacements: int | None = None,
    skip_existing_replacements: bool = True,
) -> list[ReplacementRecord]:
    selected = select_progressive_reports(
        reports,
        step=step,
        step_size=step_size,
        max_replacements=max_replacements,
    )
    records: list[ReplacementRecord] = []
    for report in selected:
        original = get_module(model, report.name)
        if not isinstance(original, nn.Linear):
            if skip_existing_replacements and _module_matches_report_features(original, report):
                continue
            raise TypeError(f"{report.name!r} is {type(original).__name__}, not nn.Linear")
        replacement = (
            replacement_factory(report.name, original, report)
            if replacement_factory is not None
            else make_fff_replacement(original, config=fff_config)
        )
        replacement.train(original.training)
        replace_module(model, report.name, replacement)
        records.append(
            ReplacementRecord(
                name=report.name,
                original_type=type(original).__name__,
                replacement_type=type(replacement).__name__,
                in_features=report.in_features,
                out_features=report.out_features,
                original_parameters=report.parameters,
                replacement_parameters=sum(parameter.numel() for parameter in replacement.parameters()),
            )
        )
    return records
