# T16 Smoke and Profiling Summary

Run ID: `20260704T024027Z`

Git commit under test: `926a05a6d68fc0ce540d7e6cf698d51da24e67b8`

This is Stage A/T16 smoke evidence only. It is not CIFAR-10 validation, teacher HPO, distillation HPO, or final test evidence.

## Gates Run

- `bash scripts/run_tests.sh`: passed on `work` with environment verification, ruff, and 155 pytest tests.
- Strict `scripts/verify_cluster.py --allow-incomplete false`: passed for `work`, `ripper`, `foureyes`, and `ai`.
- Scheduler dry-run all slots: 12 queued slots materialized.
- Scheduler dry-run usable slots: 10 queued slots materialized after excluding occupied `foureyes:2` and `foureyes:3`.
- Remote sync: used `rsync` from `work` because remote GitHub SSH auth previously failed with `publickey`.
- Concurrent quick GPU smoke: passed on all 10 currently usable GPUs: `work` 0-1, `ripper` 0-3, `foureyes` 0-1, `ai` 0-1.

Raw logs are intentionally ignored under `outputs/t16_smoke/20260704T024027Z/`.

## Occupancy

`foureyes` GPUs 2 and 3 were occupied before and after smoke:

| machine | gpu | free memory | utilization |
| --- | ---: | ---: | ---: |
| foureyes | 2 | 15029 MiB | 100% |
| foureyes | 3 | 15029 MiB | 100% |

Those GPUs were excluded with `--unavailable-slot foureyes:2 --unavailable-slot foureyes:3`.

Local `nvidia-smi dmon` during the representative hard-routing run sampled GPU 0 at up to 40% SM utilization and 132 W. This confirms the profiling command exercised CUDA; the benchmark is still a microbenchmark, not a full training workload.

## Correctness Smoke

Quick BF16 grouped-vs-naive smoke on `work` GPU 0:

| tokens | grouped path | max abs diff |
| ---: | --- | ---: |
| 16 | selected leaf | 0.00390625 |

Small BF16 grouped-vs-naive timing on `work` GPU 0:

| tokens | dense tok/s | grouped tok/s | naive tok/s | max abs diff |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 46.25M | 1.19M | 4.40K | 0.01171875 |

Naive is intentionally opt-in for nontrivial profiling because it is a Python-loop correctness path and is too slow for routine representative sweeps.

## Representative Work GPU 0/1 Profiling

BF16, batch 1024, in/out 192, depth 4, 20 iterations unless noted.

| regime | route role | grouped path | active rows | stored rows | dense fwd tok/s | grouped fwd tok/s | dense fwd+bwd tok/s | grouped fwd+bwd tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard routing | routing only | selected leaf | 4 | 64 | 140.48M | 2.73M | 7.34M | 998K |
| soft routing | routing only | all leaves | 34 | 64 | 142.58M | 1.75M | 7.57M | 410K |
| split route output | split_routing_output | selected leaf | 12 | 94 | 141.94M | 2.19M | 7.60M | 752K |

Additional local utilization-sampled hard-routing run, 3000 iterations:

| regime | dense fwd tok/s | grouped fwd tok/s | dense fwd+bwd tok/s | grouped fwd+bwd tok/s |
| --- | ---: | ---: | ---: | ---: |
| hard routing | 159.53M | 2.90M | 11.41M | 1.33M |

## Cross-Machine Quick Smoke

Concurrent quick smoke used BF16, 16 tokens, hard routing, grouped path only.

| machine | GPUs used | grouped tok/s range |
| --- | --- | ---: |
| work | 0-1 | 1.59K-1.67K |
| ripper | 0-3 | 1.59K-1.68K |
| foureyes | 0-1 | 1.70K-1.74K |
| ai | 0-1 | 2.61K |

The quick-smoke token rates are dominated by setup and tiny-kernel overhead; they are only pass/fail GPU launch evidence.

## T16 Conclusions

- Scheduler dry-run now records command, machine, GPU id, output dir, seed, git commit, `PYTHONPATH=src`, and `CUDA_VISIBLE_DEVICES`.
- The scheduler supports explicit `--unavailable-slot machine:gpu` exclusions so occupied GPUs are not treated as runnable capacity.
- Representative grouped FFF is much faster than naive correctness mode but still much slower than dense `nn.Linear` at these small row counts.
- Soft routing is slower than hard selected-leaf routing because grouped execution must evaluate all leaves.
- Split route-output contribution increases active rows and reduces grouped throughput versus routing-only hard routing.
- Long HPO should still wait for the open P0 review fixes. The current FFF-bank
  optimizer policy is documented as AdamW fallback for 3D replacement banks; any
  optimizer conclusion for assembled FFF students must label that policy or use a
  tested bank-specific Muon grouping.
