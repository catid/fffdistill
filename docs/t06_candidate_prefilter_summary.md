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
