from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LinearReport:
    name: str
    in_features: int
    out_features: int
    parameters: int
    included: bool
    reason: str


def discover_linear_layers(
    model: nn.Module, *, min_in_features: int = 64, min_out_features: int = 64
) -> list[LinearReport]:
    reports: list[LinearReport] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        params = module.in_features * module.out_features + (
            module.out_features if module.bias is not None else 0
        )
        included = module.in_features >= min_in_features and module.out_features >= min_out_features
        reason = "eligible" if included else "below minimum feature threshold"
        reports.append(
            LinearReport(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                parameters=params,
                included=included,
                reason=reason,
            )
        )
    return reports


class LinearCapture:
    def __init__(self, module: nn.Linear) -> None:
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = inputs[0].detach().flatten(0, -2).cpu()
        y = output.detach().flatten(0, -2).cpu()
        self.inputs.append(x)
        self.outputs.append(y)

    def close(self) -> None:
        self._handle.remove()

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.inputs:
            raise RuntimeError("no tensors captured")
        return torch.cat(self.inputs, dim=0), torch.cat(self.outputs, dim=0)


def replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)
