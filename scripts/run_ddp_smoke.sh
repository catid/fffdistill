#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

OUT_DIR="${1:-outputs/ddp_smoke/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="$("${PYTHON_BIN}" - <<'PY'
import torch

cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if cuda_count == 1:
    print(1)
elif cuda_count > 1:
    print(min(2, cuda_count))
else:
    print(2)
PY
)"
fi

COMMON_ARGS=(
    --quick-smoke "${QUICK_SMOKE:-true}"
    --device "${DEVICE:-auto}"
    --batch-size "${BATCH_SIZE:-128}"
    --hidden-size "${HIDDEN_SIZE:-512}"
    --iterations "${ITERATIONS:-50}"
    --warmup "${WARMUP:-5}"
)

# BENCH_ARGS is intentionally split by the shell so callers can pass normal CLI flags.
# Example: BENCH_ARGS="--quick-smoke false --iterations 200 --batch-size 512".
"${PYTHON_BIN}" -m cifar_mamba_fff.distributed \
    --strategy single \
    "${COMMON_ARGS[@]}" \
    ${BENCH_ARGS:-} \
    --output "${OUT_DIR}/single_gpu.jsonl" \
    2>&1 | tee "${OUT_DIR}/single_gpu.stdout"

TORCHRUN=("${PYTHON_BIN}" -m torch.distributed.run)

"${TORCHRUN[@]}" \
    --standalone \
    --nproc-per-node "${NPROC_PER_NODE}" \
    -m cifar_mamba_fff.distributed \
    --strategy ddp \
    "${COMMON_ARGS[@]}" \
    ${BENCH_ARGS:-} \
    --output "${OUT_DIR}/ddp.jsonl" \
    2>&1 | tee "${OUT_DIR}/ddp.stdout"

printf 'Recorded DDP smoke outputs in %s\n' "${OUT_DIR}"
