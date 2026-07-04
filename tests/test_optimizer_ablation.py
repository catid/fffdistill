from __future__ import annotations

import torch
from torch import nn

import cifar_mamba_fff.train_teacher as train_teacher
from cifar_mamba_fff.optim.normuon import MuonNorMuon
from cifar_mamba_fff.optim.pace import PaceOptimizer
from cifar_mamba_fff.train_teacher import TeacherTrainConfig


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(torch.relu(self.hidden(x))))


class FakeOfficialMuon(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.01)


def _one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()


def test_pace_muon_wraps_official_muon_baseline(monkeypatch) -> None:
    monkeypatch.setattr(train_teacher, "_import_official_muon_class", lambda: FakeOfficialMuon)
    model = TinyNet()
    config = TeacherTrainConfig(optimizer="pace_muon", pace_pullback_c=1e-3)

    optimizer, summary = train_teacher.build_training_optimizer(model, config)

    assert isinstance(optimizer, PaceOptimizer)
    assert isinstance(optimizer.base_optimizer, FakeOfficialMuon)
    assert summary["optimizer_family"] == "pace_muon"
    assert summary["base_optimizer_family"] == "muon_adamw"
    assert summary["uses_ema_eval"] is True
    assert summary["optimizer_experiments_commit"] == "689568d71ebe92093e5f5bf433127a5184ef0c35"
    assert train_teacher._scheduler_optimizer(optimizer) is optimizer.base_optimizer

    live_before = [parameter.detach().clone() for parameter in model.parameters()]
    _one_step(model, optimizer)
    with optimizer.use_ema_weights():
        ema_weights = [parameter.detach().clone() for parameter in model.parameters()]
        assert any(
            not torch.allclose(live, ema)
            for live, ema in zip(live_before, ema_weights, strict=True)
        )
    assert optimizer._swapped is False


def test_normuon_and_pace_normuon_are_explicit_non_official_ablations() -> None:
    model = TinyNet()
    config = TeacherTrainConfig(optimizer="normuon_adamw")

    optimizer, summary = train_teacher.build_training_optimizer(model, config)

    assert isinstance(optimizer, MuonNorMuon)
    assert summary["optimizer_family"] == "normuon_adamw"
    assert "optimizer_experiments" in str(summary["optimizer_source"])
    _one_step(model, optimizer)
    hidden_state = optimizer.state[model.hidden.weight]
    assert hidden_state["normuon_second_momentum"].shape == (4, 1)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    pace_optimizer, pace_summary = train_teacher.build_training_optimizer(
        TinyNet(),
        TeacherTrainConfig(optimizer="pace_normuon", pace_pullback_c=0.0),
    )
    assert isinstance(pace_optimizer, PaceOptimizer)
    assert isinstance(pace_optimizer.base_optimizer, MuonNorMuon)
    assert pace_summary["optimizer_family"] == "pace_normuon"
    assert pace_summary["base_optimizer_family"] == "normuon_adamw"
    assert pace_summary["uses_ema_eval"] is True
