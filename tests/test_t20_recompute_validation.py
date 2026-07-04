from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import cifar_mamba_fff.t20_recompute_validation as t20
from cifar_mamba_fff.models.replacement import make_fff_replacement


class _TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(image.reshape(image.shape[0], -1))


def test_t20_recompute_validation_replaces_one_layer_without_test_access(
    monkeypatch,
    tmp_path: Path,
) -> None:
    teacher = _TinyTeacher()
    replacement = make_fff_replacement(
        teacher.linear,
        config={
            "shared_rows": 4,
            "depth": 1,
            "route_rows": 1,
            "leaf_rows": 1,
            "route_row_role": "routing_only",
            "route_rows_output_count": 0,
        },
    )
    state_path = tmp_path / "fff_state.pt"
    torch.save(replacement.state_dict(), state_path)
    config_path = tmp_path / "distill.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 1337",
                "teacher_checkpoint: unused.pt",
                "eligible_linear:",
                "  min_in_features: 6",
                "  min_out_features: 4",
                "fff:",
                "  shared_rows: 4",
                "  depth: 1",
                "  route_rows: 1",
                "  leaf_rows: 1",
                "  route_row_role: routing_only",
                "  route_rows_output_count: 0",
                "distill:",
                "  steps: 1",
                "  batch_size: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seen_loader_data = []

    def fake_load_teacher_for_distillation(**kwargs):
        assert kwargs["checkpoint_path"] == tmp_path / "teacher.pt"
        assert kwargs["device"] == torch.device("cpu")
        return SimpleNamespace(
            model=teacher,
            selected_val_accuracy=0.9418,
            parameter_count=1234,
            run_config=SimpleNamespace(
                seed=2026,
                data=SimpleNamespace(use_test=False),
            ),
        )

    def fake_build_cifar10_loaders(data_config):
        seen_loader_data.append(data_config)
        images = torch.randn(3, 1, 2, 3)
        labels = torch.tensor([0, 1, 2])
        return [], [(images, labels)]

    monkeypatch.setattr(t20, "load_teacher_for_distillation", fake_load_teacher_for_distillation)
    monkeypatch.setattr(t20, "build_cifar10_loaders", fake_build_cifar10_loaders)

    result = t20.recompute_single_layer_validation(
        teacher_checkpoint=tmp_path / "teacher.pt",
        distill_config_path=config_path,
        replacement_state=state_path,
        output_dir=tmp_path / "out",
        layer_name="linear",
        quick_smoke=True,
        device=torch.device("cpu"),
        max_val_steps=1,
        precision="fp32",
    )

    assert result["test_accessed"] is False
    assert result["layer"] == "linear"
    assert result["eligible_index"] == 0
    assert result["validation_steps_after_replacement"] == 1.0
    assert seen_loader_data and seen_loader_data[0].use_test is False
    saved = json.loads((tmp_path / "out" / "t20_validation_recompute.json").read_text())
    assert saved["test_accessed"] is False
    context = json.loads((tmp_path / "out" / "run_context.json").read_text())
    assert context["test_accessed"] is False
    assert context["sample_split"] == "val"
