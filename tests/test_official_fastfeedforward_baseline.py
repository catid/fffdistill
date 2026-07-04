from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.models.official_fastfeedforward_baseline import (
    forward_smoke,
    make_official_fff,
)


def test_make_official_fff_matches_installed_signature_cpu() -> None:
    pytest.importorskip("fastfeedforward")

    input_width = 7
    leaf_width = 3
    output_width = 5
    depth = 2
    model = make_official_fff(
        input_width=input_width,
        leaf_width=leaf_width,
        output_width=output_width,
        depth=depth,
    )

    assert model.input_width == input_width
    assert model.leaf_width == leaf_width
    assert model.output_width == output_width
    assert int(model.depth.item()) == depth
    assert forward_smoke(model, input_width=input_width, device="cpu") == (2, output_width)

    y = model(torch.randn(4, input_width))
    assert y.shape == (4, output_width)
