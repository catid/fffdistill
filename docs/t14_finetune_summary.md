# T14 End-to-End KD Fine-Tuning Summary

Status: implementation complete, real FFF end-to-end KD fine-tuning blocked by an official Mamba-3 TileLang backward failure. No CIFAR-10 test data was accessed.

## Implemented

- Replaced the previous `finetune_student.py` metadata stub with a strict BF16 CUDA fine-tune trainer.
- Added strict config parsing for `teacher_checkpoint`, CIFAR-10 train/validation data, student assembly, optimizer/schedule settings, and KD losses.
- Added FFF student assembly from Stage F distillation artifacts:
  - reads all `layer_summary.json` files under the Stage F artifact root;
  - requires an exact match to the eligible Linear set when `require_full_replacement: true`;
  - reconstructs every `FFFLinear` from `configs/fff_distill_stage_f.yaml`;
  - loads every `fff_state.pt` with `strict=True`;
  - writes `student_assembly_manifest.json/.csv`.
- Collected remote Stage F `fff_state.pt` artifacts locally under the ignored `outputs/scheduler_distill_hpo/distill_stage_f_shards_20260704_091746` tree; all 64 eligible linears have state files available locally.
- Added KD fine-tune losses:
  - cross entropy on CIFAR-10 train labels;
  - KL distillation from teacher logits;
  - pooled `forward_features` normalized MSE when enabled;
  - FFF route balance regularization hooks when enabled.
- Added a fine-tune HPO plan/execute wrapper with per-trial configs/results, zero-success failure, and `test_accessed=false` safeguards.
- Added tests for config validation, artifact assembly, KD loss, HPO no-test-access behavior, and CLI metadata smoke.

## Verification Gates

- `bash scripts/run_tests.sh` passed after the initial T14 implementation: environment verification, ruff, and 324 pytest tests.
- Focused FFF equivalence after the grouped leak-path fix: `73 passed`.
- Focused T14 tests: `8 passed`.
- CIFAR-10 test access: false for every T14 smoke/HPO artifact.

## FFF Runtime Bug Fixed

The first CUDA smoke at batch size 512 failed before backward with CUDA OOM inside `FFFLinear._regular_leaf_output_grouped`.

Cause:
- Stage F uses `region_leak=0.01` with hard routing and no fallback leaf.
- The previous grouped path materialized `[tokens, leaves, out_features]` for the leak-to-all-leaves contribution.
- At batch 512 this attempted a 16.53 GiB allocation inside one FFF linear and exhausted a 95 GiB GPU.

Fix:
- `FFFLinear` now computes hard-routed leak output as `(1 - leak) * selected_leaf + leak * mean(all_leaf_outputs)`.
- The uniform all-leaf term is computed in leaf chunks and accumulates directly to `[tokens, out_features]`.
- Grouped-vs-naive tests pass after the fix.
- Fine-tune default batch size was reduced to 32, and fine-tune HPO now searches `[8, 16, 32, 64]`.

## Blocking Official Mamba-3 Issue

After the FFF memory fix, batch-32 end-to-end KD smoke reaches backward and fails inside the official Mamba-3 TileLang MIMO backward:

```text
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 143808
```

Key evidence:

- Dense teacher backward with the exact selected teacher checkpoint config succeeds at batch 32.
- The selected teacher checkpoint is an official Mamba-3 model with:
  - `d_model=192`
  - `depth=16`
  - `patch_size=2`
  - `d_state=64`
  - `headdim=64`
  - `is_mimo=true`
  - `mimo_rank=2`
  - `bidirectional=true`
  - `parameter_count=9,053,258`
  - validation accuracy `0.9418`
- The fully assembled FFF student has all 64 replacements loaded and reaches the KD loss.
- The same Mamba backward failure occurs with `lambda_balance=0.0`, so balance hooks are not the cause.
- Freezing non-FFF Mamba parameters does not avoid the failure; gradients still need to propagate through Mamba to earlier FFF modules.
- Returning contiguous tensors from `FFFLinear` does not avoid the failure.
- Installed-package diagnostics lowering official `mamba_mimo_bwd_combined(..., bb_threads=256)` to 128, 64, and 32 lowered the requested shared memory slightly but did not make the kernel launch valid.
- Overriding Mamba `chunk_size=8` is not viable for this shape; the official TileLang forward compile fails its GEMM warp partition check.

No substitute architecture, optimizer, CPU path, or fake Mamba path was used.

## Smoke Artifacts

| Run | Purpose | Result | Test Accessed |
| --- | --- | --- | --- |
| `outputs/t14_finetune_smoke_20260704` | initial full FFF KD smoke, batch 512 | failed CUDA OOM in grouped FFF leak path | false |
| `outputs/t14_finetune_smoke_20260704_b32` | batch 32 after FFF memory fix | failed official Mamba TileLang backward shared-memory setting | false |
| `outputs/t14_finetune_smoke_20260704_b32_nobalance` | same, balance hooks disabled | same Mamba TileLang backward failure | false |
| `outputs/t14_finetune_smoke_20260704_b32_nobalance_contig` | same, FFF outputs forced contiguous | same Mamba TileLang backward failure | false |

## Limitation

T14 cannot honestly report validation or final-test fine-tuned FFF student accuracy until the official Mamba-3 TileLang backward issue is fixed for FFF-replaced `in_proj` modules. The implemented trainer is therefore a reproducible blocked path, not a successful Stage G result.
