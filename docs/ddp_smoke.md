# DDP Smoke and Benchmark Harness

Beads task: `fff-dil`

This harness records a synthetic CIFAR-shaped forward/backward benchmark for local
single-process training and local `torchrun` DDP. It does not use CIFAR-10 test data
or launch remote cluster jobs.

## Quick Smoke

```bash
PYTHONPATH=src bash scripts/run_ddp_smoke.sh outputs/ddp_smoke/local_quick
```

The script writes:

- `outputs/ddp_smoke/local_quick/single_gpu.jsonl`
- `outputs/ddp_smoke/local_quick/ddp.jsonl`
- matching `*.stdout` logs

## Local Benchmark

Run this on the main GPU host when GPUs are available:

```bash
PYTHONPATH=src \
NPROC_PER_NODE=2 \
BENCH_ARGS="--quick-smoke false --batch-size 512 --hidden-size 512 --iterations 200 --warmup 20" \
bash scripts/run_ddp_smoke.sh outputs/ddp_benchmark/local_ddp_vs_single
```

Compare `samples_per_second` in `single_gpu.jsonl` and `ddp.jsonl`. Record the chosen
strategy in the task notes and scheduler documentation after the real GPU run. For
CIFAR-10-sized HPO, keep concurrent one-GPU trials as the default until this benchmark
proves DDP is faster and stable.

## Current Status

Prepared only. No real GPU DDP result is claimed by this document.
