#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi
"${PYTHON_BIN}" scripts/verify_env.py --quick-smoke true
"${PYTHON_BIN}" -m ruff check .
"${PYTHON_BIN}" -m pytest -q "$@"
