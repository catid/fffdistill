# T14 End-to-End KD Fine-Tuning Summary

Status: implementation complete; the assembled-FFF BF16 CUDA KD path is unblocked by
the `FFFLinear` autocast fix. No CIFAR-10 test data was accessed by T14 smoke or HPO
artifacts.

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
- Focused FFF/T14 regression gate after the autocast and HPO-command fixes:
  `pytest -q tests/test_finetune_student.py tests/test_fff_grouped_matches_naive.py`
  passed with `44 passed`.
- Student final-evaluation gate added:
  `pytest -q tests/test_student_final_eval.py tests/test_finetune_student.py`
  passed with `13 passed`; the evaluator CLI help imports successfully.
- Focused ruff gate passed for the changed T14/FFF files.
- CIFAR-10 test access: false for every T14 smoke/HPO artifact.

## FFF Runtime Bug Fixed

The first CUDA smoke at batch size 512 failed before backward with CUDA OOM inside `FFFLinear._regular_leaf_output_grouped`.

Cause:
- Stage F configured `region_leak=0.01` with hard routing and no fallback leaf.
- After T18 review, `region_leak` is train-only: eval/inference uses
  `effective_region_leak=0.0` and keeps selected-leaf grouped execution. See
  `docs/t18_region_leak_policy.md` for the bounded CUDA BF16 smoke.
- The previous grouped path materialized `[tokens, leaves, out_features]` for the leak-to-all-leaves contribution.
- At batch 512 this attempted a 16.53 GiB allocation inside one FFF linear and exhausted a 95 GiB GPU.

Fix:
- `FFFLinear` now computes hard-routed leak output as `(1 - leak) * selected_leaf + leak * mean(all_leaf_outputs)`.
- The uniform all-leaf term is computed in leaf chunks and accumulates directly to `[tokens, out_features]`.
- Grouped-vs-naive tests pass after the fix.
- Fine-tune default batch size was reduced to 32, and fine-tune HPO now searches `[8, 16, 32, 64]`.

## Resolved Official Mamba-3 TileLang Smoke Failure

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

Root cause:
- Assembled FFF students keep FP32 trainable parameters and rely on CUDA BF16 autocast,
  matching the teacher precision contract.
- Dense `nn.Linear` returns BF16 under CUDA BF16 autocast, but `FFFLinear` could return
  FP32 because its output accumulation and final bias addition used FP32 tensors.
- Feeding FP32 activations into official Mamba-3 selected the FP32 TileLang backward
  specialization, which requested too much dynamic shared memory for the target GPUs.

Fix:
- `FFFLinear` now keeps parameters unchanged but casts only the public forward output to
  the active autocast dtype when device autocast is enabled.
- Without autocast, FP32 inputs and FP32 parameters still produce FP32 outputs.
- Regression coverage verifies grouped and naive FFF outputs return BF16 under CUDA BF16
  autocast while parameters remain FP32.

No substitute architecture, optimizer, CPU path, or fake Mamba path was used.

## Smoke Artifacts

| Run | Purpose | Result | Test Accessed |
| --- | --- | --- | --- |
| `outputs/t14_finetune_smoke_20260704` | initial full FFF KD smoke, batch 512 | failed CUDA OOM in grouped FFF leak path | false |
| `outputs/t14_finetune_smoke_20260704_b32` | batch 32 after FFF memory fix | failed official Mamba TileLang backward shared-memory setting | false |
| `outputs/t14_finetune_smoke_20260704_b32_nobalance` | same, balance hooks disabled | same Mamba TileLang backward failure | false |
| `outputs/t14_finetune_smoke_20260704_b32_nobalance_contig` | same, FFF outputs forced contiguous | same Mamba TileLang backward failure | false |
| `outputs/t14_finetune_smoke_autocast_fix_train_nobalance_real` | batch-32 assembled FFF KD, one train step + one val step, no balance | succeeded; 64/64 linears replaced, official Muon+AdamW, no test access | false |
| `outputs/t14_finetune_hpo_autocast_fix_train_nobalance_real` | fine-tune HPO wrapper, one train-mode trial, one train step + one val step, no balance | succeeded; wrapper launched `--smoke-mode train`, metrics summary written | false |
| `outputs/t14_finetune_smoke_autocast_fix_train_balance_globalcap` | batch-32 assembled FFF KD, one train step + one val step, default balance enabled | succeeded; `train_steps_total=1`, 64/64 linears replaced, official Muon+AdamW | false |
| `outputs/t14_student_final_partial_autocast_fix_nobalance_qfalse_v2` | selected one-step HPO checkpoint, partial CIFAR-10 test evaluation with `max_test_steps=1` | succeeded; `partial_test_evaluation=true`, `test_accuracy_partial=0.3125`, 64/64 linears replaced | true |

## Additional Safety Fix

- The fine-tune HPO wrapper previously launched metadata smoke for every trial, even when
  `quick_smoke=false`. Real HPO now launches `--smoke-mode train`; metadata mode remains
  only for quick-smoke entrypoint checks.
- `--max-train-steps` previously capped steps per epoch. It now caps total train steps
  globally, preventing a bounded smoke from accidentally running one step for every epoch.
- Added `cifar_mamba_fff.evaluate_student` and `scripts/evaluate_student_final.sh` for
  strict selected-checkpoint student test evaluation. The evaluator requires CUDA,
  requires checkpoint validation metrics, rebuilds the assembled FFF student, loads the
  selected checkpoint, marks `test_accessed=true`, and separates partial from full test
  metrics. T18 later hardened this path to require a validation-selection record with
  `selected_for_final_eval=true` by default; untracked or below-target evaluations must
  be explicitly labeled as failure analysis.

## Limitation

T14 now has a validation-selected partial final-test artifact for the one-step HPO
checkpoint, but this is not a final quality claim. Full CIFAR-10 test accuracy and
multi-seed recipe claims remain final-report work and must stay separated from smoke and
partial-test evidence.
