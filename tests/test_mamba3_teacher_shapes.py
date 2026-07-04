from __future__ import annotations

import pytest
import torch

from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig, Mamba3CifarTeacher


def test_teacher_config_validates_patch_size() -> None:
    with pytest.raises(ValueError):
        Mamba3CifarConfig(patch_size=3).validate()


def test_teacher_config_validates_mamba_shape_constraints() -> None:
    with pytest.raises(ValueError, match="d_model \\* expand"):
        Mamba3CifarConfig(d_model=130, headdim=64).validate()
    with pytest.raises(ValueError, match="sequence length"):
        Mamba3CifarConfig(patch_size=4, chunk_size=24).validate()


def test_default_teacher_parameter_count_is_in_target_range() -> None:
    pytest.importorskip("mamba_ssm")
    model = Mamba3CifarTeacher(Mamba3CifarConfig())
    total = model.assert_target_parameter_count()
    assert 9_000_000 <= total <= 11_000_000


def test_teacher_shape_with_official_mamba3_if_installed() -> None:
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("official Mamba3 teacher smoke requires CUDA Triton/TileLang kernels")
    model = Mamba3CifarTeacher(
        Mamba3CifarConfig(
            d_model=128,
            depth=1,
            patch_size=4,
            d_state=64,
            expand=2,
            headdim=64,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=16,
            target_min_params=1,
            target_max_params=10_000_000,
        )
    ).cuda()
    assert {param.dtype for param in model.parameters()} == {torch.float32}
    x = torch.randn(2, 3, 32, 32, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
        loss = y.float().square().mean()
    loss.backward()
    assert y.shape == (2, 10)
    assert {param.dtype for param in model.parameters()} == {torch.float32}
