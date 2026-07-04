# Setup Failure Report

Generated during Stage A environment verification.

## Summary

Initial `scripts/setup.sh` completed, but `python scripts/verify_env.py --quick-smoke true` failed at the official Mamba-3 import gate. No fallback architecture or optimizer was substituted.

This blocker is now resolved locally by `scripts/apply_vendor_patches.py`, which applies minimal upstream-compatible Python 3.12 patches while preserving the official TileLang MIMO Mamba-3 training kernels.

## Environment Observed

- Python: 3.12.11 inside `.venv`.
- Local GPUs visible: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- `mamba-ssm`: 2.3.2.post1 from official `state-spaces/mamba` commit `ed6ce09e4d802e274b1ecc7205757b892e180a93`.
- `muon-optimizer`: 0.1.0 from official `KellerJordan/Muon` commit `f98f1cacc0263b04290753e32be8d498c1efc806`.
- `fastfeedforward`: 0.2.1.
- Torch after first setup run: `2.12.1+cu130`. This is CUDA 13.0 but not the requested nightly wheel. `scripts/setup.sh` and `scripts/verify_env.py` were updated so future runs install Mamba with `--no-deps` and fail if torch is not a `*dev*+cu130` nightly.
- Verified torch after repair: `2.14.0.dev20260702+cu130`.

## Failure

The official package contains `mamba_ssm.modules.mamba3.Mamba3`, but importing `mamba_ssm` fails while importing the official Mamba-3 TileLang MIMO path:

```text
AttributeError: attribute '__dict__' of 'type' objects is not writable
```

The traceback enters:

```text
mamba_ssm/__init__.py
mamba_ssm/modules/mamba3.py
mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py
mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py
tilelang/__init__.py
tilelang/3rdparty/tvm/python/tvm/ir/attrs.py
tvm_ffi/registry.py
```

`mamba3.py` treats the TileLang MIMO import as optional only for `ImportError`, but this environment raises `AttributeError`, so the import aborts before `Mamba3` can be instantiated.

## Consequence

The initial official dependency install failed the required install/import/run gate. Teacher training, HPO, and distillation were correctly blocked until the dependency path was repaired.

## Vendor Patch

`scripts/apply_vendor_patches.py` applies these pinned patches:

- `apache-tvm-ffi==0.1.12`: skip reflected fields named `__dict__` and `__weakref__` in `tvm_ffi.registry._add_class_attrs`. CPython forbids assigning those descriptors to a type.
- `mamba-ssm==2.3.2.post1`: catch non-`ImportError` failures from the optional Cute inference-step import. This does not disable the TileLang MIMO training path; `mamba3_mimo_combined` remains required and verified.
- `tilelang==0.1.8`: add `_inst` and `__weakref__` slots to TVM-derived Python wrapper objects and route `__setattr__` through `object.__getattribute__`. This allows decorated Python visitors to initialize state during TileLang lowering on Python 3.12.

These patches are documented in `scripts/patches/mamba3_python312_tilelang_tvm.patch`.

## Verification

After reinstalling nightly torch cu130, rebuilding official Mamba from pinned upstream commit
`ed6ce09e4d802e274b1ecc7205757b892e180a93` with `--no-deps`, and applying vendor patches:

```text
python scripts/verify_env.py --quick-smoke true
```

passed with:

```text
torch: 2.14.0.dev20260702+cu130
official Mamba3 symbol: mamba_ssm.Mamba3
Mamba3 constructor kwargs: {'d_model': 128, 'd_state': 64, 'headdim': 64, 'is_mimo': True, 'mimo_rank': 2, 'chunk_size': 16}
Mamba3 TileLang MIMO kernel: available
Mamba3 BF16 forward/backward: output_shape=(1, 64, 128)
official Muon one-step smoke: SingleDeviceMuonWithAuxAdam
fastfeedforward.FFF import: ok
environment verification passed
```

The Mamba smoke keeps parameters in FP32 and uses BF16 CUDA autocast. Converting the whole official Mamba3 module to BF16 converts `B_bias`/`C_bias`, but the upstream TileLang MIMO kernel expects those bias tensors in FP32.

`scripts/verify_env.py` now fails if the official constructor does not preserve
`is_mimo=True`, `mimo_rank=2`, and `chunk_size=16`. The teacher shape test also uses a
CUDA BF16 autocast MIMO-compatible smoke shape instead of implying CPU execution is
supported by upstream Triton/TileLang kernels. The default quick environment gate also
runs a shape-compatible `fastfeedforward.FFF` CUDA forward smoke.

Full repository quality gate after this repair:

```text
bash scripts/run_tests.sh
All checks passed!
124 passed, 11 warnings
```
