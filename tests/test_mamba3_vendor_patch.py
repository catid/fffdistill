from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_patch_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "apply_vendor_patches.py"
    spec = importlib.util.spec_from_file_location("apply_vendor_patches", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_mamba3_mimo_tilelang_kernel_available_if_installed() -> None:
    pytest.importorskip("mamba_ssm")
    import mamba_ssm.modules.mamba3 as mamba3_module

    assert getattr(mamba3_module, "Mamba3", None) is not None
    assert getattr(mamba3_module, "mamba3_mimo_combined", None) is not None


def test_vendor_patch_markers_present_if_official_mamba3_installed() -> None:
    pytest.importorskip("mamba_ssm")
    patch_script = _load_patch_script()

    for patch in patch_script._patches():
        text = patch.path.read_text(encoding="utf-8")
        assert patch.marker in text, f"{patch.name} marker missing from {patch.path}"
