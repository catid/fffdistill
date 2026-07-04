#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
git fetch "$REMOTE"
git status --short --branch
