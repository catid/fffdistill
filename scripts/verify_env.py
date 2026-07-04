#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import inspect
import json
import platform
import sys
from collections.abc import Iterable

MAMBA_COMMIT = "ed6ce09e4d802e274b1ecc7205757b892e180a93"
MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def _find_mamba3() -> type:
    candidates: Iterable[tuple[str, str]] = (
        ("mamba_ssm", "Mamba3"),
        ("mamba_ssm.modules.mamba3", "Mamba3"),
        ("mamba_ssm.modules.mamba3_simple", "Mamba3"),
        ("mamba_ssm.modules.mamba3_mimo", "Mamba3"),
    )
    errors: list[str] = []
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on installed extension
            errors.append(f"{module_name}: import failed: {exc}")
            continue
        cls = getattr(module, attr, None)
        if cls is not None:
            print(f"official Mamba3 symbol: {module_name}.{attr}")
            return cls
        errors.append(f"{module_name}: no {attr}")
    joined = "\n  ".join(errors)
    raise RuntimeError(f"official Mamba3 class was not found in mamba_ssm:\n  {joined}")


def _normalize_repo_url(url: str) -> str:
    normalized = url.lower().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _verify_direct_url(package: str, expected_repo: str, expected_commit: str) -> None:
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required official package {package!r} is not installed") from exc
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{package} is missing direct_url.json; cannot verify official source commit")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{package} has invalid direct_url.json: {exc}") from exc
    url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        raise RuntimeError(f"{package} direct_url.json does not contain vcs_info")
    commit_id = str(vcs_info.get("commit_id", ""))
    vcs = str(vcs_info.get("vcs", ""))
    if vcs != "git":
        raise RuntimeError(f"{package} must be installed from official git source, got vcs={vcs!r}")
    if _normalize_repo_url(url) != _normalize_repo_url(expected_repo):
        raise RuntimeError(f"{package} source URL mismatch: got {url!r}, expected {expected_repo!r}")
    if commit_id != expected_commit:
        raise RuntimeError(f"{package} commit mismatch: got {commit_id!r}, expected {expected_commit!r}")
    print(f"{package} source commit verified: {commit_id}")


def _instantiate_mamba3(cls: type, *, d_model: int, device: str):
    kwargs = {
        "d_model": d_model,
        "d_state": 64,
        "headdim": 64,
        "is_mimo": True,
        "mimo_rank": 2,
        "chunk_size": 16,
    }
    try:
        module = cls(**kwargs).to(device=device)
    except TypeError as exc:
        signature = inspect.signature(cls)
        raise RuntimeError(
            "Could not instantiate official Mamba3 with required MIMO kwargs. "
            f"Signature: {signature}. Kwargs: {kwargs}. Error: {exc}"
        ) from exc
    if getattr(module, "is_mimo", None) is not True:
        raise RuntimeError("Official Mamba3 constructor did not preserve requested is_mimo=True")
    if getattr(module, "mimo_rank", None) != 2:
        raise RuntimeError("Official Mamba3 constructor did not preserve requested mimo_rank=2")
    if getattr(module, "chunk_size", None) != 16:
        raise RuntimeError("Official Mamba3 constructor did not preserve requested chunk_size=16")
    print(f"Mamba3 constructor kwargs: {kwargs}")
    return module


def _verify_muon_step(device: str) -> None:
    import torch

    module_names = ("muon", "Muon", "muon_optimizer")
    imported = None
    import_errors: list[str] = []
    for module_name in module_names:
        try:
            imported = importlib.import_module(module_name)
            print(f"official Muon import: {module_name}")
            break
        except Exception as exc:  # pragma: no cover - depends on package
            import_errors.append(f"{module_name}: {exc}")
    if imported is None:
        raise RuntimeError("Could not import official Muon package: " + "; ".join(import_errors))
    _verify_direct_url("muon-optimizer", "https://github.com/KellerJordan/Muon", MUON_COMMIT)

    optimizer_cls = (
        getattr(imported, "SingleDeviceMuonWithAuxAdam", None)
        or getattr(imported, "MuonWithAuxAdam", None)
        or getattr(imported, "Muon", None)
    )
    if optimizer_cls is None:
        available = ", ".join(name for name in dir(imported) if "Muon" in name or "muon" in name)
        raise RuntimeError(f"Muon package imported, but no Muon optimizer class found. Available: {available}")

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)).to(device)
    x = torch.randn(4, 8, device=device)
    loss = model(x).square().mean()
    loss.backward()

    hidden_matrix_params = [model[0].weight]
    adamw_fallback_params = [p for name, p in model.named_parameters() if name != "0.weight"]
    try:
        if optimizer_cls.__name__.endswith("WithAuxAdam"):
            optimizer = optimizer_cls(
                [
                    {
                        "params": hidden_matrix_params,
                        "lr": 1e-3,
                        "momentum": 0.95,
                        "weight_decay": 0.0,
                        "use_muon": True,
                    },
                    {
                        "params": adamw_fallback_params,
                        "lr": 1e-4,
                        "betas": (0.9, 0.95),
                        "eps": 1e-10,
                        "weight_decay": 0.0,
                        "use_muon": False,
                    },
                ]
            )
        else:
            optimizer = optimizer_cls(list(hidden_matrix_params), lr=1e-3)
    except (AssertionError, TypeError) as exc:
        raise RuntimeError(
            f"Official Muon class {optimizer_cls} imported, but simple construction failed: {exc}. "
            "Inspect T05 and adapt grouping to the installed API before training."
        ) from exc
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"official Muon one-step smoke: {optimizer_cls.__name__}")


def _verify_fastfeedforward(device: str, quick_smoke: bool) -> None:
    import torch

    module = importlib.import_module("fastfeedforward")
    fff_cls = getattr(module, "FFF", None)
    if fff_cls is None:
        raise RuntimeError("fastfeedforward imported, but FFF symbol was not found")
    print("fastfeedforward.FFF import: ok")
    try:
        model = fff_cls(8, 4, 4, 2)
        x = torch.randn(2, 8, device=device)
        model = model.to(device)
        y = model(x)
        print(f"fastfeedforward.FFF forward shape: {tuple(y.shape)}")
    except Exception as exc:
        raise RuntimeError(f"fastfeedforward.FFF forward smoke failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-smoke", type=_bool_arg, default=False)
    parser.add_argument("--allow-restricted-visible-gpus", type=_bool_arg, default=False)
    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA: {torch.version.cuda}")
    if "dev" not in torch.__version__ or "+cu130" not in torch.__version__:
        raise RuntimeError(
            "Expected PyTorch nightly cu130, e.g. '*dev*+cu130'. "
            f"Installed torch is {torch.__version__}. "
            "Do not continue until setup preserves the nightly cu130 wheel."
        )
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected torch CUDA 13.0, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; CPU-only fallback is forbidden")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device_count = torch.cuda.device_count()
    print(f"visible CUDA devices: {device_count}")
    if device_count < 1:
        raise RuntimeError("No CUDA devices are visible")
    if device_count < 2 and not args.allow_restricted_visible_gpus:
        raise RuntimeError(
            "Full local verification on work expects 2 visible GPUs. "
            "Pass --allow-restricted-visible-gpus true only for intentional CUDA_VISIBLE_DEVICES restriction."
        )
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / 1024**3
        print(f"gpu[{idx}]: {props.name}, {total_gib:.1f} GiB")

    device = "cuda:0"
    mamba3_cls = _find_mamba3()
    _verify_direct_url("mamba-ssm", "https://github.com/state-spaces/mamba.git", MAMBA_COMMIT)
    module = _instantiate_mamba3(mamba3_cls, d_model=128, device=device)
    try:
        import mamba_ssm.modules.mamba3 as mamba3_module

        if getattr(mamba3_module, "mamba3_mimo_combined", None) is None:
            raise RuntimeError("official Mamba3 MIMO TileLang kernel is unavailable")
        print("Mamba3 TileLang MIMO kernel: available")
    except Exception as exc:
        raise RuntimeError("Could not verify official Mamba3 MIMO TileLang kernel") from exc
    x = torch.randn(1, 64, 128, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = module(x)
    if isinstance(y, tuple):
        y = y[0]
    y.float().square().mean().backward()
    print(f"Mamba3 BF16 forward/backward: output_shape={tuple(y.shape)}")

    _verify_muon_step(device)
    _verify_fastfeedforward(device, args.quick_smoke)
    print("environment verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
