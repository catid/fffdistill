from __future__ import annotations

import torch
from torch.nn import functional as F

from cifar_mamba_fff.models.fff_linear import FFFLinear


def test_overparameterized_fff_overfits_tiny_linear_regression_batch() -> None:
    torch.manual_seed(20240704)
    x = torch.randn(10, 3)
    true_weight = torch.randn(3, 2)
    true_bias = torch.randn(2)
    target = x @ true_weight + true_bias
    layer = FFFLinear(
        3,
        2,
        depth=2,
        shared_rows=0,
        route_rows=2,
        leaf_rows=4,
        hard_routing=False,
        activation="gelu",
        master_leaf=True,
        bias=True,
    )
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.035, weight_decay=0.0)

    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()

    final_mse = F.mse_loss(layer(x), target)

    assert final_mse.item() < 2e-4
