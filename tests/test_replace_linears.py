from __future__ import annotations

import torch
from torch import nn

from cifar_mamba_fff.models.replacement import LinearCapture, discover_linear_layers, replace_module


def test_discover_and_replace_linear() -> None:
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    reports = discover_linear_layers(model, min_in_features=8, min_out_features=8)
    assert reports[0].included
    assert not reports[1].included
    replace_module(model, "2", nn.Linear(16, 3))
    assert model(torch.randn(4, 8)).shape == (4, 3)


def test_linear_capture_flattens_leading_dims() -> None:
    layer = nn.Linear(5, 7)
    capture = LinearCapture(layer)
    y = layer(torch.randn(2, 3, 5))
    assert y.shape == (2, 3, 7)
    capture.close()
    x_cap, y_cap = capture.tensors()
    assert x_cap.shape == (6, 5)
    assert y_cap.shape == (6, 7)
