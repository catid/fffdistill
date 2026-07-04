# T18 Grouped FFF Throughput Profiling

Status: report generator implemented and representative CUDA BF16 profiling run
on 2026-07-04.

Beads task: `fff-evq`.

## Reproducible Report Command

The benchmark now has a multi-shape report mode for teacher-linear-sized FFF
replacements:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m cifar_mamba_fff.benchmark_fff \
  --profile-report teacher-linear \
  --device cuda:0 \
  --dtype bfloat16 \
  --iterations 20 \
  --warmup 5 \
  --measure-backward true \
  --include-compiled true \
  --report-include-naive true \
  --report-markdown-out docs/t18_grouped_fff_profile_gpu0.md \
  --json true | tee outputs/t18_grouped_fff_profile_gpu0.jsonl
```

For a bounded launch-only check without CUDA:

```bash
.venv/bin/python -m cifar_mamba_fff.benchmark_fff \
  --profile-report teacher-linear \
  --quick-smoke true \
  --device cpu \
  --iterations 1 \
  --warmup 0 \
  --report-include-naive false \
  --report-markdown-out outputs/fff-evq_profile_quick.md \
  --json false
```

GPU utilization is intentionally not inferred from PyTorch elapsed time. Capture it
with a host sampler around the CUDA command, for example:

```bash
mkdir -p outputs/t18_grouped_fff_profile
nvidia-smi dmon -s pucm -d 1 -o DT \
  > outputs/t18_grouped_fff_profile/gpu0_dmon.log &
sampler_pid=$!
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m cifar_mamba_fff.benchmark_fff \
  --profile-report teacher-linear \
  --device cuda:0 \
  --dtype bfloat16 \
  --iterations 20 \
  --warmup 5 \
  --measure-backward true \
  --include-compiled true \
  --report-include-naive true \
  --report-markdown-out docs/t18_grouped_fff_profile_gpu0.md \
  --json true | tee outputs/t18_grouped_fff_profile/gpu0.jsonl
kill "$sampler_pid"
```

## Matrix

`--profile-report teacher-linear` runs these representative shapes:

| case | batch | in | out | FFF recipe |
| --- | ---: | ---: | ---: | --- |
| `mamba3_d_model_square_256` | 1024 | 256 | 256 | depth 5, split route-output, leaf rows 4 |
| `mamba3_expand_in_256x512` | 1024 | 256 | 512 | depth 5, split route-output, leaf rows 4 |
| `mamba3_expand_out_512x256` | 1024 | 512 | 256 | depth 5, split route-output, leaf rows 4 |

Every timed row includes:

- `tokens_per_second`
- `dense_tokens_per_second`
- `tokens_per_second_fraction_of_dense`
- `dense_slowdown`
- `grouped_vs_naive_speedup`, when the naive path is included
- `python_per_token_hot_path`
- `sort_bucket_strategy`
- `sort_bucket_seconds_per_iteration`
- `sort_bucket_overhead_fraction`
- `grouped_leaf_path`, `effective_region_leak`, and active/stored row metadata

Component rows are emitted for `fff_route_setup` and either
`fff_selected_leaf_kernel` or `fff_all_leaf_kernel`.

## Current Hot-Path Evidence

The deployable grouped path is `FFFLinear.forward_grouped`. It computes routing for
the whole flattened batch, then uses tensor gather plus `torch.bmm` for selected
hard leaves, or all-leaf `einsum` for soft/leaky training paths. It does not sort
or bucket tokens by leaf. Therefore grouped rows report:

- `python_per_token_hot_path=false`
- `sort_bucket_strategy=not_used_selected_leaf_gather_bmm` for hard selected-leaf eval
- `sort_bucket_seconds_per_iteration=0.0`
- `sort_bucket_overhead_fraction=0.0`

The intentionally slow `FFFLinear.forward_naive` path remains the correctness
reference. It loops over tokens and leaves in Python, so benchmark rows report
`python_per_token_hot_path=true` for `fff_naive`.

## CUDA BF16 Result

Run on `work` GPU 0 with git
`14bc2e27f435fc3c7d86e4878d99f1a63e95d441`. The later safety commit
`6bcb9d6d00b51467a118ddaade4a07ec1f787842` changed scheduler/DDP launch guards
but did not change `benchmark_fff.py` or `FFFLinear`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m cifar_mamba_fff.benchmark_fff \
  --profile-report teacher-linear \
  --device cuda:0 \
  --dtype bfloat16 \
  --iterations 20 \
  --warmup 5 \
  --measure-backward true \
  --include-compiled true \
  --report-include-naive true \
  --report-markdown-out outputs/t18_grouped_fff_profile/gpu0_20260704_14bc2e2/profile.md \
  --json true | tee outputs/t18_grouped_fff_profile/gpu0_20260704_14bc2e2/profile.jsonl
```

Forward throughput:

| case | dense tokens/s | grouped tokens/s | compiled grouped tokens/s | naive tokens/s |
| --- | ---: | ---: | ---: | ---: |
| `mamba3_d_model_square_256` | 155,187,960.82 | 2,279,169.05 | 10,931,773.57 | 2,270.98 |
| `mamba3_expand_in_256x512` | 179,049,151.10 | 2,458,003.90 | 10,604,613.11 | 2,277.45 |
| `mamba3_expand_out_512x256` | 180,009,140.97 | 2,447,006.13 | 10,380,515.85 | 2,274.08 |

Forward+backward throughput:

| case | dense tokens/s | grouped tokens/s | compiled grouped tokens/s | naive tokens/s |
| --- | ---: | ---: | ---: | ---: |
| `mamba3_d_model_square_256` | 7,826,983.91 | 799,265.36 | 2,302,369.46 | 938.75 |
| `mamba3_expand_in_256x512` | 7,974,751.81 | 770,303.48 | 2,447,105.83 | 936.65 |
| `mamba3_expand_out_512x256` | 8,541,632.95 | 769,254.49 | 2,468,382.59 | 934.86 |

Profiler interpretation:

- Grouped eager is 68x-74x slower than dense forward and 9.8x-11.1x slower than
  dense forward+backward on these small teacher-linear shapes.
- `torch.compile` improves grouped FFF substantially: compiled grouped is
  14x-17x slower than dense forward and 3.3x-3.5x slower than dense
  forward+backward.
- Naive remains unusable for throughput, but it is useful as a correctness
  reference: grouped-vs-naive max absolute BF16 diff was `0.0078125`.
- The selected hard eval path has no Python per-token hot path and no sort/bucket
  overhead; route setup and selected-leaf tensor kernels are the measured
  components.
- `nvidia-smi dmon` showed GPU 0 activity up to 97%-98% SM during compile/profile
  phases. GPU 1 stayed idle because this was a single-GPU profiler gate.

Artifacts kept in ignored output storage:

- `outputs/t18_grouped_fff_profile/gpu0_20260704_14bc2e2/profile.jsonl`
- `outputs/t18_grouped_fff_profile/gpu0_20260704_14bc2e2/profile.md`
- `outputs/t18_grouped_fff_profile/gpu0_20260704_14bc2e2/nvidia_dmon.log`

Decision: compiled grouped FFF is the fastest valid current path for these shapes,
but it remains materially slower than dense Linear. Long full-student sweeps may
proceed only with this limitation reported; any final speed claim must use measured
throughput and cannot imply dense-linear speed parity.
