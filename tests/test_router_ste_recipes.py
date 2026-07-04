from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cifar_mamba_fff.losses.router_ste import (
    clipped_ste,
    expert_choice_assignment,
    expert_choice_imitation,
    hard_concrete_expected_l0,
    hard_concrete_gate,
    hard_concrete_row_gates,
    hard_concrete_sample,
    hard_em_targets,
    hard_em_utility_ste,
    no_ste_soft_router,
    router_recipe_diagnostics,
    sigmoid_surrogate_ste,
    st_gumbel,
    utility_targeted_ce,
    utility_targeted_ste,
    validate_finite_tensor,
    vanilla_ste,
)


def _assert_one_hot(values: torch.Tensor) -> None:
    assert torch.all((values == 0.0) | (values == 1.0))
    torch.testing.assert_close(values.sum(dim=-1), torch.ones(values.shape[:-1]))


def _assert_finite_diagnostics(diagnostics: dict[str, torch.Tensor]) -> None:
    assert diagnostics
    for value in diagnostics.values():
        assert torch.isfinite(value).all()


def test_vanilla_ste_accepts_tau_alias_for_existing_smoke() -> None:
    logits = torch.randn(8, requires_grad=True)

    gates = vanilla_ste(logits, tau=1.0)
    gates.sum().backward()

    assert gates.shape == logits.shape
    assert logits.grad is not None


def test_no_ste_soft_router_matches_softmax() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0], [0.3, -0.2, 0.5]])

    probs = no_ste_soft_router(logits, temperature=0.7)

    torch.testing.assert_close(probs, torch.softmax(logits / 0.7, dim=-1))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))

    diagnostics = router_recipe_diagnostics(probs, logits=logits)
    _assert_finite_diagnostics(diagnostics)
    assert diagnostics["hard_fraction"] == 0.0


def test_vanilla_ste_hard_forward_and_soft_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.2, -0.4]], requires_grad=True)
    weights = torch.tensor([[0.0, 1.0, 3.0], [2.0, -1.0, 0.5]])

    routed = vanilla_ste(logits)
    _assert_one_hot(routed)
    torch.testing.assert_close(routed.argmax(dim=-1), logits.argmax(dim=-1))

    (routed * weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0

    diagnostics = router_recipe_diagnostics(routed.detach(), logits=logits.detach())
    _assert_finite_diagnostics(diagnostics)
    assert diagnostics["hard_fraction"] == 1.0


def test_sigmoid_surrogate_ste_hard_forward_and_gradient() -> None:
    logits = torch.tensor([[0.5, -0.3], [-0.1, 0.4]], requires_grad=True)
    weights = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

    routed = sigmoid_surrogate_ste(logits, temperature=0.5)
    _assert_one_hot(routed)
    torch.testing.assert_close(routed.argmax(dim=-1), logits.argmax(dim=-1))

    (routed * weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_clipped_ste_clamps_backward_gradient() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    weights = torch.tensor([[10.0, -4.0, 0.1]])

    routed = clipped_ste(logits, clip=0.25)
    _assert_one_hot(routed)
    (routed * weights).sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().max() <= 0.250001
    torch.testing.assert_close(logits.grad, weights.clamp(min=-0.25, max=0.25))


def test_st_gumbel_eval_is_deterministic_and_training_can_be_seeded() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.2, -0.4]])

    eval_first = st_gumbel(logits, training=False, hard=True)
    eval_second = st_gumbel(logits, training=False, hard=True)
    torch.testing.assert_close(eval_first, eval_second)
    _assert_one_hot(eval_first)
    torch.testing.assert_close(eval_first.argmax(dim=-1), logits.argmax(dim=-1))

    gen_a = torch.Generator().manual_seed(1234)
    gen_b = torch.Generator().manual_seed(1234)
    train_first = st_gumbel(logits, training=True, hard=True, generator=gen_a)
    train_second = st_gumbel(logits, training=True, hard=True, generator=gen_b)
    torch.testing.assert_close(train_first, train_second)
    _assert_one_hot(train_first)


def test_utility_targeted_ce_and_hard_em_targets() -> None:
    router_logits = torch.tensor(
        [[0.0, 1.0, -0.5], [2.0, -1.0, 0.0]],
        requires_grad=True,
    )
    utility = torch.tensor([[0.1, 0.8, 0.2], [0.0, 0.2, 0.9]])

    targets = hard_em_targets(utility)
    torch.testing.assert_close(targets, torch.tensor([1, 2]))

    loss = utility_targeted_ce(router_logits, utility)
    expected = F.cross_entropy(router_logits, targets)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()


def test_utility_targeted_ste_routes_to_utility_and_keeps_router_gradients() -> None:
    router_logits = torch.tensor(
        [[0.0, 1.0, -0.5], [2.0, -1.0, 0.0], [-0.2, 0.1, 0.3]],
        requires_grad=True,
    )
    utility = torch.tensor([[0.1, 0.8, 0.2], [0.3, 0.2, 0.9], [1.2, 0.0, -0.1]])
    weights = torch.tensor([[1.0, -2.0, 0.5], [0.3, 2.0, -1.0], [-0.5, 1.5, 0.2]])

    routed, diagnostics = utility_targeted_ste(
        router_logits,
        utility,
        return_diagnostics=True,
    )

    _assert_one_hot(routed.detach())
    torch.testing.assert_close(routed.detach().argmax(dim=-1), utility.argmax(dim=-1))
    _assert_finite_diagnostics(diagnostics)
    assert all(not value.requires_grad for value in diagnostics.values())
    assert torch.count_nonzero(diagnostics["expert_usage"]) == 3

    (routed * weights).sum().backward()
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()
    assert router_logits.grad.abs().sum() > 0.0


def test_utility_targeted_ste_can_use_soft_utility_targets() -> None:
    router_logits = torch.tensor([[0.1, 0.0, -0.2]], requires_grad=True)
    utility = torch.tensor([[0.0, 2.0, 1.0]])

    routed = utility_targeted_ste(router_logits, utility, hard=False, utility_temperature=0.5)

    expected = torch.softmax(utility / 0.5, dim=-1)
    torch.testing.assert_close(routed.detach(), expected)
    torch.testing.assert_close(routed.sum(dim=-1), torch.ones(1))

    routed[:, 1].sum().backward()
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()
    assert router_logits.grad.abs().sum() > 0.0


def test_hard_em_utility_ste_returns_non_degenerate_assignments_and_gradients() -> None:
    router_logits = torch.tensor(
        [[0.2, 1.0, -0.5], [1.2, 0.0, -0.1], [-0.3, 0.4, 0.6]],
        requires_grad=True,
    )
    utility = torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.1, -0.2], [0.4, 0.2, 1.5]])
    weights = torch.tensor([[0.5, -1.0, 0.1], [1.0, 0.2, -0.4], [-0.3, 0.6, 2.0]])

    routed, targets, diagnostics = hard_em_utility_ste(
        router_logits,
        utility,
        return_targets=True,
        return_diagnostics=True,
    )

    _assert_one_hot(routed.detach())
    torch.testing.assert_close(targets, torch.tensor([1, 0, 2]))
    assert targets.unique().numel() == 3
    _assert_finite_diagnostics(diagnostics)
    assert all(not value.requires_grad for value in diagnostics.values())

    (routed * weights).sum().backward()
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()
    assert router_logits.grad.abs().sum() > 0.0


def test_expert_choice_assignment_selects_top_capacity_per_expert() -> None:
    utility = torch.tensor(
        [
            [0.1, 3.0, 0.0],
            [0.5, 2.0, 4.0],
            [0.9, 1.0, 2.0],
            [0.8, 0.5, 3.0],
        ]
    )

    assignment = expert_choice_assignment(utility, capacity=2)

    assert assignment.dtype == torch.bool
    assert assignment.shape == utility.shape
    torch.testing.assert_close(assignment.sum(dim=0), torch.tensor([2, 2, 2]))
    assert assignment[2, 0]
    assert assignment[3, 0]
    assert assignment[0, 1]
    assert assignment[1, 1]
    assert assignment[1, 2]
    assert assignment[3, 2]


def test_expert_choice_imitation_loss_uses_assignment_and_updates_logits() -> None:
    router_logits = torch.tensor(
        [
            [0.1, 0.0, 0.2],
            [0.4, -0.2, 0.3],
            [-0.1, 0.5, 0.6],
            [0.8, 0.1, -0.4],
        ],
        requires_grad=True,
    )
    utility = torch.tensor(
        [
            [0.1, 3.0, 0.0],
            [0.5, 2.0, 4.0],
            [0.9, 1.0, 2.0],
            [0.8, 0.5, 3.0],
        ]
    )

    loss, assignment = expert_choice_imitation(
        router_logits,
        utility,
        capacity=2,
        return_assignment=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    torch.testing.assert_close(assignment.sum(dim=0), torch.tensor([2, 2, 2]))
    assert assignment.any(dim=0).all()

    loss.backward()
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()
    assert router_logits.grad.abs().sum() > 0.0


def test_hard_concrete_sample_expected_l0_and_gate_helper() -> None:
    log_alpha = torch.tensor([-4.0, 0.0, 4.0], requires_grad=True)
    generator = torch.Generator().manual_seed(99)

    sample = hard_concrete_sample(log_alpha, generator=generator)
    expected = hard_concrete_expected_l0(log_alpha)
    gate_sample, gate_expected = hard_concrete_gate(log_alpha, training=False)

    assert torch.all(sample >= 0.0)
    assert torch.all(sample <= 1.0)
    assert torch.all(expected > 0.0)
    assert torch.all(expected < 1.0)
    assert expected[0] < expected[1] < expected[2]
    torch.testing.assert_close(gate_expected, expected)
    assert torch.all(gate_sample >= 0.0)
    assert torch.all(gate_sample <= 1.0)

    expected.sum().backward()
    assert log_alpha.grad is not None
    assert torch.isfinite(log_alpha.grad).all()


def test_hard_concrete_row_gates_are_deterministic_in_eval_with_finite_diagnostics() -> None:
    log_alpha = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)

    gates_a, expected_a, diagnostics = hard_concrete_row_gates(
        log_alpha,
        training=False,
        l0_weight=0.25,
        return_diagnostics=True,
    )
    gates_b, expected_b = hard_concrete_row_gates(log_alpha, training=False)

    torch.testing.assert_close(gates_a, gates_b)
    torch.testing.assert_close(expected_a, expected_b)
    assert torch.all(gates_a >= 0.0)
    assert torch.all(gates_a <= 1.0)
    assert expected_a[0] < expected_a[1] < expected_a[2] < expected_a[3]
    _assert_finite_diagnostics(diagnostics)
    for name, value in diagnostics.items():
        if name != "l0_penalty":
            assert not value.requires_grad
    assert diagnostics["l0_penalty"].requires_grad

    diagnostics["l0_penalty"].backward()
    assert log_alpha.grad is not None
    assert torch.isfinite(log_alpha.grad).all()
    assert log_alpha.grad.abs().sum() > 0.0


def test_router_helpers_fail_fast_on_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        no_ste_soft_router(torch.tensor(1.0))
    with pytest.raises(ValueError, match="same shape"):
        utility_targeted_ce(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError, match="at least two experts"):
        utility_targeted_ce(torch.randn(3, 1), torch.randn(3, 1))
    with pytest.raises(ValueError, match="tokens, experts"):
        expert_choice_assignment(torch.randn(2, 3, 4), capacity=1)
    with pytest.raises(ValueError, match="capacity"):
        expert_choice_assignment(torch.randn(2, 3), capacity=0)
    with pytest.raises(ValueError, match="gamma"):
        hard_concrete_expected_l0(torch.randn(3), gamma=0.1)
    with pytest.raises(ValueError, match="temperature"):
        vanilla_ste(torch.randn(2, 3), temperature=0.0)
    with pytest.raises(ValueError, match="same shape"):
        utility_targeted_ste(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError, match="same shape"):
        expert_choice_imitation(torch.randn(2, 3), torch.randn(2, 4), capacity=1)
    with pytest.raises(ValueError, match="l0_weight"):
        hard_concrete_row_gates(torch.randn(3), l0_weight=-1.0)


def test_router_debug_finite_validation_is_explicit() -> None:
    bad = torch.tensor([[0.5, float("nan")]])

    with pytest.raises(ValueError, match="finite"):
        validate_finite_tensor("bad", bad)
    with pytest.raises(ValueError, match="finite"):
        router_recipe_diagnostics(bad, validate_finite=True)
