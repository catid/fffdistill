#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

CONSTRAINTS="scripts/constraints-py312-cu130.txt"
MAMBA_COMMIT="ed6ce09e4d802e274b1ecc7205757b892e180a93"
MUON_COMMIT="f98f1cacc0263b04290753e32be8d498c1efc806"

uv pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --constraint "${CONSTRAINTS}"

uv pip install \
    numpy pandas tqdm rich matplotlib scikit-learn pyyaml \
    optuna wandb pytest ruff einops tabulate \
    packaging ninja psutil fabric paramiko filelock \
    --constraint "${CONSTRAINTS}"

uv pip install \
    tilelang apache-tvm-ffi quack-kernels transformers \
    --constraint "${CONSTRAINTS}"

MAMBA_FORCE_BUILD=TRUE uv pip install --no-cache-dir --force-reinstall --no-deps \
    "git+https://github.com/state-spaces/mamba.git@${MAMBA_COMMIT}" --no-build-isolation

uv pip install --no-deps "git+https://github.com/KellerJordan/Muon@${MUON_COMMIT}"
uv pip install fastfeedforward --constraint "${CONSTRAINTS}"
uv pip install -e ".[dev]" --constraint "${CONSTRAINTS}"
python scripts/apply_vendor_patches.py
