# T18 Grouped FFF Throughput Profiling

Status: report generator implemented on 2026-07-04; representative CUDA profiling
still needs to be run before long full-student sweeps.

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

## Still Needed

Before long full-student sweeps, run the CUDA BF16 command above on at least one
representative GPU, keep the JSONL and utilization log, and compare dense,
grouped, naive, and compiled grouped rows. If compiled grouped is unavailable or
slower on the active PyTorch/CUDA stack, record that limitation in the generated
Markdown report rather than treating it as an optimization result.
