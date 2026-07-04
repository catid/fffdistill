# T06b Teacher Smoke Summary

Date: 2026-07-04

Commit: `bce660ae50a1d33e9cc499864e2957362253ac2f`

Command:

```bash
python -m cifar_mamba_fff.gpu_scheduler \
  --dry-run false \
  --quick-smoke true \
  --smoke-mode train \
  --unavailable-slot foureyes:2 \
  --unavailable-slot foureyes:3 \
  --wait true \
  --wait-timeout-s 600 \
  --poll-interval-s 2 \
  --queue-out outputs/t06b_train_queue.jsonl \
  --launch-results-out outputs/t06b_scheduler_train_launch.jsonl \
  --collect-root outputs/t06b_scheduler_train_collected
```

Scope:

- 10 one-GPU BF16 train smokes launched and succeeded.
- `foureyes:2` and `foureyes:3` were excluded because unrelated `/home/catid/sps_transformer/.venv/bin/python` jobs occupied about 82 GiB each.
- All collected `run_context.json` files recorded `resolved_config.data.use_test=false`.
- All runs used the official Mamba3 CIFAR teacher default with 9,527,370 trainable parameters.
- CIFAR train readiness preflight passed on `work`, `ripper`, `foureyes`, and `ai` with `test_accessed=false`.

| Machine | GPU | Seed | GPU class | Train imgs/s | End-to-end imgs/s | Peak reserved GiB | Status |
|---|---:|---:|---|---:|---:|---:|---|
| work | 0 | 1337 | RTX PRO 6000 Blackwell Workstation | 101.686 | 94.629 | 5.145 | succeeded |
| work | 1 | 1338 | RTX PRO 6000 Blackwell Workstation | 100.621 | 93.670 | 5.145 | succeeded |
| ripper | 0 | 1339 | RTX PRO 6000 Blackwell Max-Q | 100.156 | 92.694 | 5.141 | succeeded |
| ripper | 1 | 1340 | RTX PRO 6000 Blackwell Max-Q | 100.030 | 92.576 | 5.141 | succeeded |
| ripper | 2 | 1341 | RTX PRO 6000 Blackwell Max-Q | 99.502 | 91.957 | 5.141 | succeeded |
| ripper | 3 | 1342 | RTX PRO 6000 Blackwell Max-Q | 100.420 | 92.513 | 5.141 | succeeded |
| foureyes | 0 | 1343 | RTX PRO 6000 Blackwell Max-Q | 100.122 | 92.920 | 5.145 | succeeded |
| foureyes | 1 | 1344 | RTX PRO 6000 Blackwell Max-Q | 99.494 | 92.658 | 5.145 | succeeded |
| ai | 0 | 1347 | RTX 5090 | 103.743 | 98.465 | 5.141 | succeeded |
| ai | 1 | 1348 | RTX 5090 | 103.636 | 98.460 | 5.145 | succeeded |

Notes:

- The first 10-slot train-smoke attempt exposed a scheduler detach bug: remote `nohup` jobs could start successfully while the SSH launch command timed out and was classified as `failed_infra`. Commit `bce660a` fixes this by closing stdin for the detached launch and probing `status.json` on timeout.
- These are smoke-test throughput numbers, not final teacher training throughput. They include one train step and one validation step; `Train imgs/s` is train-loop-only timing and `End-to-end imgs/s` includes validation and run overhead.
