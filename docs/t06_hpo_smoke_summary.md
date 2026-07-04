# T06 HPO Smoke Summary

Date: 2026-07-04

Commit: `167658379711cccf67c460c6a9d2c6635bb33c24`

Command:

```bash
python -m cifar_mamba_fff.gpu_scheduler \
  --dry-run false \
  --quick-smoke true \
  --smoke-mode metadata \
  --job-kind teacher_hpo \
  --teacher-hpo-config configs/teacher_hpo_smoke.yaml \
  --run-id t06_hpo_smoke_1676583 \
  --hpo-trials-per-job 1 \
  --hpo-max-attempts-per-job 8 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --unavailable-slot foureyes:2 \
  --unavailable-slot foureyes:3 \
  --wait true \
  --wait-timeout-s 600 \
  --poll-interval-s 2 \
  --queue-out outputs/t06_hpo_smoke_queue.jsonl \
  --launch-results-out outputs/t06_hpo_smoke_launch.jsonl \
  --collect-root outputs/t06_hpo_smoke_collected
```

Scope:

- 10 one-GPU teacher HPO smoke jobs launched and succeeded.
- Each job ran one bounded HPO trial with `--max-train-steps 1` and `--max-val-steps 1`.
- `configs/teacher_hpo_smoke.yaml` fixes the search space to the known CUDA-valid 9,527,370 parameter official Mamba3 default.
- `foureyes:2` and `foureyes:3` were excluded because unrelated jobs occupied those GPUs.
- All collected `trial_config.json` files recorded `run_config.data.use_test=false`.

| Machine | GPU | HPO seed | Trial status | Params | Train imgs/s | Peak reserved GiB |
|---|---:|---:|---|---:|---:|---:|
| work | 0 | 1337 | succeeded | 9,527,370 | 101.373 | 5.145 |
| work | 1 | 1338 | succeeded | 9,527,370 | 101.482 | 5.145 |
| ripper | 0 | 1339 | succeeded | 9,527,370 | 100.643 | 5.141 |
| ripper | 1 | 1340 | succeeded | 9,527,370 | 100.419 | 5.141 |
| ripper | 2 | 1341 | succeeded | 9,527,370 | 100.123 | 5.141 |
| ripper | 3 | 1342 | succeeded | 9,527,370 | 100.237 | 5.141 |
| foureyes | 0 | 1343 | succeeded | 9,527,370 | 99.761 | 5.141 |
| foureyes | 1 | 1344 | succeeded | 9,527,370 | 100.069 | 5.145 |
| ai | 0 | 1347 | succeeded | 9,527,370 | 104.656 | 5.141 |
| ai | 1 | 1348 | succeeded | 9,527,370 | 103.653 | 5.145 |

Next gate:

- Before broad teacher HPO, add a CUDA kernel-smoke prefilter or narrow the first broad search config. The broad `configs/teacher_hpo.yaml` still includes shapes that previously hit an official Mamba3 TileLang dynamic shared-memory failure.
