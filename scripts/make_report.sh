#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-.venv/bin/python}" -m cifar_mamba_fff.make_report "$@"
