from __future__ import annotations

import pytest
import torch
from torch import nn

from cifar_mamba_fff.models.fff_linear import FFFLinear
from cifar_mamba_fff.optim.muon_groups import (
    format_param_assignments,
    split_muon_adamw_parameters,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 5)
        self.pos_embedding = nn.Parameter(torch.zeros(8, 5))
        self.scalar = nn.Parameter(torch.zeros(()))
        self.vector = nn.Parameter(torch.zeros(5))
        self.hidden = nn.Linear(5, 6)
        self.attention_heads = nn.Linear(6, 6, bias=False)
        self.norm = nn.LayerNorm(6)
        self.bn = nn.BatchNorm1d(6)
        self.classifier = nn.Linear(6, 3)
        self.head = nn.Linear(6, 3)
        self.frozen = nn.Linear(6, 6)
        for param in self.frozen.parameters():
            param.requires_grad_(False)


def test_muon_param_groups_assign_expected_categories() -> None:
    model = TinyModel()
    muon, adamw, assignments = split_muon_adamw_parameters(model)

    names_by_group = {
        group: {assignment.name for assignment in assignments if assignment.group == group}
        for group in ("muon", "adamw")
    }
    assert names_by_group["muon"] == {"hidden.weight", "attention_heads.weight"}
    assert names_by_group["adamw"] == {
        "embed.weight",
        "pos_embedding",
        "scalar",
        "vector",
        "hidden.bias",
        "norm.weight",
        "norm.bias",
        "bn.weight",
        "bn.bias",
        "classifier.weight",
        "classifier.bias",
        "head.weight",
        "head.bias",
    }

    assigned_ids = [id(p) for p in muon + adamw]
    assert len(assigned_ids) == len(set(assigned_ids))
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(assigned_ids) == trainable_ids
    assert len(assignments) == len(trainable_ids)


def test_muon_param_groups_log_exact_assignments() -> None:
    model = TinyModel()
    logged: list[str] = []
    _, _, assignments = split_muon_adamw_parameters(model, assignment_logger=logged.append)

    assert logged == format_param_assignments(assignments)
    assert any(
        line == "group=muon name=hidden.weight shape=6x5 reason=hidden matrix parameter"
        for line in logged
    )
    assert any(
        line == "group=adamw name=classifier.weight shape=3x6 reason=classifier/head parameter name"
        for line in logged
    )


def test_muon_param_groups_keep_fff_bias_banks_in_adamw() -> None:
    model = nn.Sequential(
        nn.Linear(8, 8),
        FFFLinear(
            8,
            8,
            depth=2,
            shared_rows=2,
            route_rows=2,
            route_result_rows=1,
            leaf_rows=1,
            route_row_role="split_routing_output",
            route_rows_output_count="all",
        ),
    )

    _, _, assignments = split_muon_adamw_parameters(model)
    by_name = {assignment.name: assignment for assignment in assignments}

    assert by_name["0.weight"].group == "muon"
    assert by_name["1.shared_weight"].group == "muon"
    assert by_name["1.shared_output"].group == "muon"
    for name in (
        "1.shared_bias",
        "1.route_bias",
        "1.route_result_bias",
        "1.leaf_bias",
        "1.bias",
    ):
        assert by_name[name].group == "adamw"
        assert by_name[name].reason == "bias parameter"
    for name in (
        "1.route_weight",
        "1.route_result_weight",
        "1.route_result_output",
        "1.leaf_weight",
        "1.leaf_output",
    ):
        assert by_name[name].group == "adamw"
        assert by_name[name].reason == "3D parameter"


def test_muon_param_groups_reject_duplicate_parameter_aliases() -> None:
    model = nn.Module()
    shared = nn.Parameter(torch.ones(2, 2))
    model.first = shared
    model.second = shared

    with pytest.raises(ValueError, match="appears more than once"):
        split_muon_adamw_parameters(model)
