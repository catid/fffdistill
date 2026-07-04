# DDP Smoke and Benchmark Harness

Beads task: `fff-dil`

This harness records a synthetic CIFAR-shaped forward/backward benchmark for local
single-process training and local `torchrun` DDP. It does not use CIFAR-10 test data
or launch remote cluster jobs. The DDP leg is launched as
`.venv/bin/python -m torch.distributed.run` by default so the single-process and
DDP legs use the same Python, PyTorch, CUDA, and package environment.

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

Compare `samples_per_second` in `single_gpu.jsonl` and `ddp.jsonl`. For
CIFAR-10-sized HPO, keep concurrent one-GPU trials as the default until this benchmark
proves DDP is faster and stable.

## Local GPU Result

Run:

```bash
PYTHONPATH=src \
NPROC_PER_NODE=2 \
BENCH_ARGS="--quick-smoke false --batch-size 512 --hidden-size 512 --iterations 200 --warmup 20 --dtype bfloat16" \
bash scripts/run_ddp_smoke.sh outputs/ddp_benchmark/local_ddp_vs_single_20260704_6bcb9d6
```

Result on `work` at git `6bcb9d6d00b51467a118ddaade4a07ec1f787842`:

| strategy | GPUs | global batch | samples/s | note |
| --- | ---: | ---: | ---: | --- |
| single process | 1 | 512 | 2,567,217.93 | one GPU baseline |
| DDP/NCCL | 2 | 1024 | 3,670,635.52 | valid, but only 1.43x one GPU |
| two independent one-GPU trials | 2 | 2 x 512 | about 5,134,435.85 | inferred from one-GPU baseline |

Decision: keep parallel one-GPU trials as the default for CIFAR-10 HPO and
distillation sweeps. Use DDP only for explicitly larger-batch final training
experiments after a fresh benchmark shows a time-to-accuracy benefit.

Artifacts kept in ignored output storage:

- `outputs/ddp_benchmark/local_ddp_vs_single_20260704_6bcb9d6/single_gpu.jsonl`
- `outputs/ddp_benchmark/local_ddp_vs_single_20260704_6bcb9d6/ddp.jsonl`
- `outputs/ddp_benchmark/local_ddp_vs_single_20260704_6bcb9d6/nvidia_dmon.log`
