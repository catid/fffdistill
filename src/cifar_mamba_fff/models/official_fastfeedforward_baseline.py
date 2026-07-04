from __future__ import annotations

import importlib

import torch
from torch import nn


def make_official_fff(in_features: int, out_features: int, **kwargs: object) -> nn.Module:
    module = importlib.import_module("fastfeedforward")
    cls = getattr(module, "FFF", None)
    if cls is None:
        raise RuntimeError("fastfeedforward imported, but FFF symbol was not found")
    try:
        return cls(in_features, out_features, **kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Installed fastfeedforward.FFF is not shape-compatible with "
            f"({in_features}, {out_features}) and kwargs {kwargs}"
        ) from exc


def forward_smoke(module: nn.Module, in_features: int, device: str = "cuda") -> tuple[int, ...]:
    x = torch.randn(2, in_features, device=device)
    y = module.to(device)(x)
    return tuple(y.shape)
