#!/usr/bin/env bash
set -euo pipefail

python -m cifar_mamba_fff.fairness "$@"
