from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cifar_mamba_fff.losses.balance import (
    master_fallback_usage_cap_penalty,
    min_leaf_occupancy_loss,
    route_entropy_loss,
    route_margin_loss,
    split_balance_loss,
    uniform_leaf_balance_loss,
    usage_cap_penalty,
)


def test_balance_losses_are_finite_and_differentiable() -> None:
    logits = torch.randn(16, 3, 2, requires_grad=True)
    probs = logits.softmax(dim=-1)
    leaf_probs = torch.rand(16, 8, requires_grad=True).softmax(dim=-1)
    loss = (
        split_balance_loss(probs)
        + min_leaf_occupancy_loss(leaf_probs, min_occupancy=0.05)
        + uniform_leaf_balance_loss(leaf_probs)
        + route_margin_loss(logits, target_margin=0.1)
        - 0.01 * route_entropy_loss(logits)
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_split_balance_pair_and_scalar_inputs() -> None:
    balanced_pairs = torch.full((4, 3, 2), 0.5)
    assert torch.allclose(split_balance_loss(balanced_pairs), torch.tensor(0.0))

    scalar_right = torch.tensor(
        [
            [0.9, 0.1, 0.5],
            [0.9, 0.1, 0.5],
        ],
        dtype=torch.float32,
    )
    per_node = split_balance_loss(scalar_right, pair_probs=False, reduction="none")
    torch.testing.assert_close(per_node, torch.tensor([0.16, 0.16, 0.0]))
    assert split_balance_loss(scalar_right, pair_probs=False) > 0.0


def test_split_balance_has_finite_gradients() -> None:
    logits = torch.tensor(
        [[[2.0, -0.5], [0.2, 1.0]], [[1.5, -0.2], [0.0, 0.7]]],
        requires_grad=True,
    )
    probs = logits.softmax(dim=-1)
    loss = split_balance_loss(probs)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_leaf_occupancy_and_uniform_balance() -> None:
    healthy = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    assert torch.allclose(
        min_leaf_occupancy_loss(healthy, min_occupancy=0.2),
        torch.tensor(0.0),
    )
    assert torch.allclose(uniform_leaf_balance_loss(healthy), torch.tensor(0.0))

    starved = torch.tensor(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
    )
    assert min_leaf_occupancy_loss(starved, min_occupancy=0.1) > 0.0
    assert uniform_leaf_balance_loss(starved) > 0.0


def test_route_margin_penalty_and_gradients() -> None:
    weak = torch.tensor([[0.1, 0.2], [0.0, -0.1]], requires_grad=True)
    strong = torch.tensor([[-3.0, 3.0], [4.0, -4.0]])

    weak_loss = route_margin_loss(weak, target_margin=1.0)
    strong_loss = route_margin_loss(strong, target_margin=1.0)

    assert weak_loss > strong_loss
    assert torch.allclose(strong_loss, torch.tensor(0.0))

    weak_loss.backward()
    assert weak.grad is not None
    assert torch.isfinite(weak.grad).all()


def test_route_entropy_from_logits_and_probs() -> None:
    logits = torch.tensor([[0.0, 0.0], [6.0, -6.0]], requires_grad=True)
    entropy = route_entropy_loss(logits, from_logits=True, reduction="none")

    assert entropy[0] > entropy[1]
    entropy.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    probs = torch.tensor([[0.5, 0.5], [0.99, 0.01]])
    entropy_from_probs = route_entropy_loss(probs, from_logits=False, reduction="none")
    assert entropy_from_probs[0] > entropy_from_probs[1]
    assert torch.isfinite(entropy_from_probs).all()


def test_usage_cap_penalties() -> None:
    under_cap = torch.tensor([0.1, 0.2, 0.3])
    over_cap = torch.tensor([0.1, 0.7, 0.9])

    assert torch.allclose(usage_cap_penalty(under_cap, cap=0.5), torch.tensor(0.0))
    assert usage_cap_penalty(over_cap, cap=0.5) > 0.0

    master = torch.tensor([0.2, 0.8])
    fallback = torch.tensor([0.6, 0.1])
    penalty = master_fallback_usage_cap_penalty(
        master,
        fallback,
        master_cap=0.5,
        fallback_cap=0.5,
        reduction="none",
    )
    torch.testing.assert_close(penalty, torch.tensor([0.01, 0.09]))


def test_balance_losses_fail_fast_on_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="pair route_probs"):
        split_balance_loss(torch.randn(3), pair_probs=True)
    with pytest.raises(ValueError, match="route_logits"):
        route_margin_loss(torch.randn(2, 3))
    with pytest.raises(ValueError, match="dim"):
        route_entropy_loss(torch.randn(2, 2), dim=3)
    with pytest.raises(ValueError, match="non-negative"):
        route_entropy_loss(torch.tensor([[0.5, -0.1]]), from_logits=False)
    with pytest.raises(ValueError, match="same shape"):
        master_fallback_usage_cap_penalty(
            torch.ones(2),
            torch.ones(3),
            master_cap=0.5,
            fallback_cap=0.5,
        )
