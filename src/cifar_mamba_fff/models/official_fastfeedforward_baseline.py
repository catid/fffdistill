from __future__ import annotations

import importlib

import torch
from torch import nn


def make_official_fff(
    input_width: int,
    leaf_width: int,
    output_width: int,
    depth: int,
    **kwargs: object,
) -> nn.Module:
    module = importlib.import_module("fastfeedforward")
    cls = getattr(module, "FFF", None)
    if cls is None:
        raise RuntimeError("fastfeedforward imported, but FFF symbol was not found")
    try:
        return cls(
            input_width=input_width,
            leaf_width=leaf_width,
            output_width=output_width,
            depth=depth,
            **kwargs,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Installed fastfeedforward.FFF is not shape-compatible with "
            f"(input_width={input_width}, leaf_width={leaf_width}, "
            f"output_width={output_width}, depth={depth}) and kwargs {kwargs}"
        ) from exc


def forward_smoke(module: nn.Module, input_width: int, device: str = "cuda") -> tuple[int, ...]:
    x = torch.randn(2, input_width, device=device)
    y = module.to(device)(x)
    return tuple(y.shape)
