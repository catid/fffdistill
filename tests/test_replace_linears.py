from __future__ import annotations

import pytest
import torch
from torch import nn

from cifar_mamba_fff.distill_linears import linear_replacement_plan
from cifar_mamba_fff.distill_linears import main as distill_linears_main
from cifar_mamba_fff.models.fff_linear import FFFLinear
from cifar_mamba_fff.models.replacement import (
    LinearCapture,
    LinearCaptureSet,
    LinearReport,
    discover_linear_layers,
    linear_reports_as_log_records,
    make_fff_replacement,
    replace_linear_layers,
    replace_module,
    select_progressive_reports,
)


def test_discover_and_replace_linear() -> None:
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    reports = discover_linear_layers(model, min_in_features=8, min_out_features=8)
    assert reports[0].included
    assert not reports[1].included
    assert reports[0].weight_shape == (16, 8)
    assert reports[0].bias_shape == (16,)
    assert reports[0].parameters == 8 * 16 + 16
    assert reports[0].reason == "eligible"
    assert reports[1].reason == "skipped: out_features 2 < 8"
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


def test_discovery_log_records_include_included_and_skipped_reasons() -> None:
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 8))
    reports = discover_linear_layers(model, min_in_features=5, min_out_features=6)
    records = linear_reports_as_log_records(reports)

    assert records == [
        {
            "name": "0",
            "in_features": 4,
            "out_features": 4,
            "weight_shape": (4, 4),
            "bias_shape": None,
            "parameters": 16,
            "included": False,
            "reason": "skipped: in_features 4 < 5; out_features 4 < 6",
        },
        {
            "name": "1",
            "in_features": 4,
            "out_features": 8,
            "weight_shape": (8, 4),
            "bias_shape": (8,),
            "parameters": 40,
            "included": False,
            "reason": "skipped: in_features 4 < 5",
        },
    ]


def test_linear_capture_set_collects_only_included_layers() -> None:
    model = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3))
    reports = discover_linear_layers(model, min_in_features=5, min_out_features=5)

    with LinearCaptureSet(model, reports) as captures:
        y = model(torch.randn(2, 4, 5))

    assert y.shape == (2, 4, 3)
    assert set(captures.captures) == {"0"}
    x_cap, y_cap = captures.tensors("0")
    assert x_cap.shape == (8, 5)
    assert y_cap.shape == (8, 7)


def test_make_fff_replacement_preserves_linear_interface_shape_dtype_and_bias() -> None:
    linear = nn.Linear(6, 4, bias=False, dtype=torch.float64)
    replacement = make_fff_replacement(
        linear,
        config={
            "shared_unrouted_frac": 0.5,
            "depth": 2,
            "route_rows": 1,
            "leaf_rows": 1,
        },
    )

    assert isinstance(replacement, FFFLinear)
    assert replacement.in_features == 6
    assert replacement.out_features == 4
    assert replacement.shared_rows == 2
    assert replacement.bias is None
    assert replacement(torch.randn(3, 6, dtype=torch.float64)).shape == (3, 4)


def test_replace_linear_layers_supports_exact_factory_and_nested_names() -> None:
    model = nn.Sequential(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        nn.Linear(4, 2),
    )
    x = torch.randn(3, 4)
    expected = model(x)
    reports = discover_linear_layers(model, min_in_features=4, min_out_features=2)

    def copy_linear(_name: str, linear: nn.Linear, _report) -> nn.Linear:
        copied = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        copied.load_state_dict(linear.state_dict())
        return copied

    records = replace_linear_layers(model, reports, replacement_factory=copy_linear)

    assert [record.name for record in records] == ["0.0", "1"]
    assert all(record.replacement_type == "Linear" for record in records)
    assert torch.equal(model(torch.randn(0, 4)), torch.empty(0, 2))
    torch.testing.assert_close(model(x), expected, rtol=0.0, atol=0.0)


def test_make_fff_replacement_preserves_eval_mode() -> None:
    linear = nn.Linear(6, 4)
    linear.eval()

    replacement = make_fff_replacement(linear, config={"depth": 1, "route_rows": 1, "leaf_rows": 1})

    assert not replacement.training


def test_replace_linear_layers_preserves_factory_replacement_eval_mode() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model.eval()
    reports = discover_linear_layers(model, min_in_features=4, min_out_features=4)

    def fresh_linear(_name: str, linear: nn.Linear, _report) -> nn.Linear:
        return nn.Linear(linear.in_features, linear.out_features)

    replace_linear_layers(model, reports, replacement_factory=fresh_linear)

    assert not model[0].training


def test_progressive_replacement_advances_existing_model_idempotently() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 2))
    reports = discover_linear_layers(model, min_in_features=8, min_out_features=8)

    first_records = replace_linear_layers(
        model,
        reports,
        step=1,
        step_size=1,
        fff_config={"depth": 1, "route_rows": 1, "leaf_rows": 1},
    )
    second_records = replace_linear_layers(
        model,
        reports,
        step=2,
        step_size=1,
        fff_config={"depth": 1, "route_rows": 1, "leaf_rows": 1},
    )

    assert [record.name for record in first_records] == ["0"]
    assert [record.name for record in second_records] == ["1"]
    assert isinstance(model[0], FFFLinear)
    assert isinstance(model[1], FFFLinear)


def test_linear_capture_set_cleans_up_hooks_when_later_report_is_stale() -> None:
    model = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 7))
    stale_report = LinearReport(
        name="missing",
        in_features=7,
        out_features=7,
        weight_shape=(7, 7),
        bias_shape=(7,),
        parameters=56,
        included=True,
        reason="eligible",
    )
    reports = [*discover_linear_layers(model, min_in_features=5, min_out_features=7)[:1], stale_report]

    with pytest.raises(KeyError):
        LinearCaptureSet(model, reports)

    assert len(model[0]._forward_hooks) == 0


def test_progressive_report_selection_and_plan_are_deterministic() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 2))
    config = {"eligible_linear": {"min_in_features": 8, "min_out_features": 8}}
    reports = discover_linear_layers(model, min_in_features=8, min_out_features=8)

    selected = select_progressive_reports(reports, step=1, step_size=1)
    plan = linear_replacement_plan(model, config, progressive_step=1, progressive_step_size=1)

    assert [report.name for report in selected] == ["0"]
    assert plan["selected_replacements"] == ["0"]
    assert plan["progressive"] == {
        "step": 1,
        "step_size": 1,
        "selected_count": 1,
        "eligible_count": 2,
    }


def test_distill_quick_smoke_rejects_invalid_progressive_step_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "distill_linears",
            "--quick-smoke",
            "true",
            "--output-dir",
            str(tmp_path),
            "--progressive-step-size",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="step_size must be positive"):
        distill_linears_main()
