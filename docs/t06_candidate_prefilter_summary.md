# T06 Teacher HPO Candidate Prefilter Summary

Date: 2026-07-04

## Purpose

The broad teacher HPO search includes official Mamba-3 shapes that pass the 9M-11M parameter-count gate but fail the optimized TileLang CUDA kernel path on the target GPUs. A CUDA BF16 kernel-smoke prefilter was added so those candidates are rejected before they consume HPO trial slots.

## Implementation

- Added strict `candidate_filter` parsing to `configs/teacher_hpo.yaml` and `configs/teacher_hpo_smoke.yaml`.
- Added an opt-in CUDA kernel smoke in `src/cifar_mamba_fff/hpo/teacher_hpo.py`.
- The smoke instantiates the official `Mamba3CifarTeacher`, runs one synthetic BF16 CUDA forward and backward pass, checks output shape and finite loss, synchronizes CUDA, and rejects failures as `rejected_pretrial`.
- The smoke uses synthetic tensors only. It does not build CIFAR loaders and does not access the CIFAR-10 test split.
- Kernel-smoke rejection happens after parameter-count acceptance and before assigning a trial slot.

## Smoke Evidence

Broad HPO candidate-filter smoke:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m cifar_mamba_fff.hpo.teacher_hpo \
  --quick-smoke true \
  --max-candidates 4 \
  --output-dir outputs/teacher_hpo_prefilter_smoke_20260704_060027
```

Result:

- accepted: 0
- rejected: 4
- one parameter-valid candidate was rejected by CUDA smoke:
  `cuda_kernel_smoke_failed: InternalError: Failed to set the allowed dynamic shared memory size to 146080`
- other rejections were outside the 9M-11M parameter-count range.

Pinned known-safe HPO execute smoke:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m cifar_mamba_fff.hpo.teacher_hpo \
  --quick-smoke true \
  --execute-trials true \
  --hpo-config configs/teacher_hpo_smoke.yaml \
  --max-trials 1 \
  --max-attempts 1 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --output-dir outputs/teacher_hpo_prefilter_execute_smoke_20260704_060131
```

Result:

- accepted_trials: 1
- succeeded: 1
- failed_logic: 0
- failed_oom: 0
- best_val_accuracy: 0.18359375
- quick_smoke: true

## Test Evidence

```bash
source .venv/bin/activate
ruff check src/cifar_mamba_fff/hpo/teacher_hpo.py tests/test_teacher_training.py
PYTHONPATH=src pytest -q tests/test_teacher_training.py
bash scripts/run_tests.sh
```

Results:

- focused ruff: passed
- focused teacher HPO tests: 24 passed
- full gate: environment verification passed, ruff passed, 222 pytest tests passed

Independent read-only subworker review found no integration issues and confirmed the smoke uses synthetic tensors rather than CIFAR data.

## Multi-Machine Broad HPO Smoke

After committing and syncing the prefilter to remotes, a broad-config scheduler smoke was launched on the 10 currently usable GPUs. `foureyes:2` and `foureyes:3` were excluded because they were still occupied by unrelated high-memory jobs.

```bash
source .venv/bin/activate
PYTHONPATH=src python -m cifar_mamba_fff.gpu_scheduler \
  --dry-run false \
  --quick-smoke true \
  --smoke-mode metadata \
  --job-kind teacher_hpo \
  --teacher-hpo-config configs/teacher_hpo.yaml \
  --run-id t06_prefilter_broad_smoke_20260704_0604 \
  --hpo-trials-per-job 1 \
  --hpo-max-attempts-per-job 64 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --unavailable-slot foureyes:2 \
  --unavailable-slot foureyes:3 \
  --wait true \
  --wait-timeout-s 1200 \
  --poll-interval-s 5
```

Results:

| Slot | Status | Rejected | Kernel Rejects | Accepted Config | Smoke Val Acc |
| --- | --- | ---: | ---: | --- | ---: |
| work:0 | succeeded | 8 | 1 | d=160 depth=24 p=4 state=64 head=64 rank=2 bi=True bs=512 cosine params=9,772,554 | 0.18359375 |
| work:1 | succeeded | 30 | 3 | d=288 depth=16 p=2 state=64 head=64 rank=2 bi=False bs=256 wsd params=9,525,066 | 0.171875 |
| ai:0 | succeeded | 18 | 0 | d=224 depth=24 p=2 state=64 head=32 rank=2 bi=False bs=512 cosine params=9,141,386 | 0.1171875 |
| ai:1 | succeeded | 27 | 0 | d=288 depth=8 p=4 state=64 head=64 rank=2 bi=True bs=256 wsd params=9,480,138 | 0.13671875 |
| foureyes:0 | succeeded | 8 | 0 | d=288 depth=16 p=4 state=64 head=32 rank=2 bi=False bs=256 wsd params=9,641,706 | 0.12890625 |
| foureyes:1 | succeeded | 7 | 0 | d=256 depth=20 p=2 state=64 head=64 rank=2 bi=False bs=1024 cosine params=9,567,306 | 0.17578125 |
| ripper:0 | succeeded | 0 | 0 | d=192 depth=16 p=2 state=64 head=64 rank=2 bi=True bs=512 cosine params=9,053,258 | 0.13671875 |
| ripper:1 | failed_logic | 64 | 5 | none | n/a |
| ripper:2 | succeeded | 49 | 4 | d=224 depth=24 p=2 state=64 head=32 rank=2 bi=False bs=512 wsd params=9,141,386 | 0.15625 |
| ripper:3 | succeeded | 17 | 3 | d=288 depth=16 p=2 state=64 head=64 rank=2 bi=False bs=256 cosine params=9,525,066 | 0.078125 |

Interpretation:

- The broad config is viable but inefficient: 9 of 10 jobs found a parameter-valid and kernel-safe candidate within 64 attempts.
- One job exhausted all 64 attempts and failed with zero valid candidates. This is expected HPO accounting, not a training success.
- Real HPO launches should use a higher `--hpo-max-attempts-per-job` or a documented kernel-safe narrowed config if zero-candidate jobs waste too much GPU time.
- Scheduler artifact collection was patched after this run so default collection paths include `--run-id`, preventing later HPO waves from overwriting collected summaries.
