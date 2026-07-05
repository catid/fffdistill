# Project Plan

This file preserves the original research plan in repository form and records approved improvements. Beads is the source of truth for task state; this file is the durable human-readable plan.

## Original Objective

Build, test, commit, push, and run a reproducible CIFAR-10 research project for BF16 Fast Feedforward Network distillation from an official Mamba-3 teacher. The final repository remote is `git@github.com:catid/fffdistill.git`.

The teacher is a roughly 10M-parameter official Mamba-3 CIFAR-10 model trained from scratch to at least 90% final test accuracy. The student replaces eligible `nn.Linear` layers with FFFLinear modules. FFF distillation is studied first in BF16 on GPU, not BitNet or CPU.

## Non-Negotiable Constraints

- Use official Mamba-3 from `https://github.com/state-spaces/mamba`.
- Use official Muon from `https://github.com/KellerJordan/Muon`.
- Use `fastfeedforward` from `https://github.com/pbelcak/fastfeedforward` as canonical FFF reference/baseline where shape-compatible.
- Do not substitute fake Mamba blocks, Mamba-2, Transformer, ResNet, AdamW-only, CPU-only, smaller models, another dataset, or fake dependencies.
- If official Mamba-3 or official Muon fails to install/import/run, stop, diagnose, and report failure.
- CIFAR-10 test set is used only for final selected checkpoints after validation/HPO.
  Allowed test accesses must be backed by selection manifests and summarized in
  `docs/final_report.md` and the fairness tables. Current open optimizer,
  router-family, route-output, official-FFF, and row/column sparse baseline
  tasks remain validation-only until their own selection records exist.
- Do not claim success without saved logs, configs, metrics, tests, profiler outputs, Beads task records, git commits, and reproducible commands.

## Hardware Assumption

- `work`: 2x NVIDIA RTX Pro 6000 GPUs.
- `ripper`: 4x Pro6000 MaxQ GPUs.
- `foureyes`: 4x Pro6000 MaxQ GPUs.
- `ai`: 2x RTX 5090 GPUs.
- Use concurrent one-GPU trials by default for CIFAR-10-sized HPO unless DDP benchmarking proves faster.
- Do not run long experiments until correctness tests, profiler sanity, SSH/job launch sanity, and GPU utilization checks pass.
- Use parallel workers/subagents for independent implementation, review, and test slices. Main Codex remains responsible for integration review, deterministic checks, Beads updates, coherent commits, and pushes.

## Beads Task Graph

- T00 repo/bootstrap/gitignore/setup.
- T01 environment verification for PyTorch nightly cu130, Mamba-3, Muon, fastfeedforward.
- T02 cluster SSH/GPU inventory and remote job launcher.
- T03 CIFAR-10 data pipeline.
- T04 official Mamba-3 vision teacher wrapper.
- T05 official Muon parameter grouping.
- T06 teacher training/HPO.
- T07 custom FFFLinear correctness path.
- T08 grouped/batched FFF GPU path.
- T09 balance regularizers.
- T10 STE/router recipes.
- T11 LocoProp-S ridge refit.
- T12 Linear hook/capture/replacement pipeline.
- T13 distillation HPO.
- T14 end-to-end KD fine-tuning.
- T15 baselines and fair comparison checks.
- T16 profiling/throughput analysis.
- T17 final report.
- T18 adversarial bugfix/review pass.
- T19 optimizer ablations and WSD schedule tuning.
- T20 route-row output contribution ablation.
- T22 official Mamba-3 TileLang/Python 3.12 vendor patch and strict BF16 CUDA verification.
- T23 FFF route-output empty-batch bugfix.
- T24 LocoProp-S ridge refit no-grad/default non-differentiable solve bugfix.
- T25 CUDA BF16 grouped-vs-naive FFF regression coverage.
- T26 remote setup and `verify_env.py` on all GPU machines.
- T27 row-plus-column sparse activation and alternative sublinear baselines.

## Current Original-Plan Gap Tasks

The 2026-07-04 audit found that the repository has strong smoke, validation, and one selected Stage H final-test result, but several original-plan requirements remain open as follow-up work:

- `fff-i6e`: run validation-selected full-student final-test router-family comparisons.
- `fff-dza`: run full-student route-output ablations with active-FLOP accounting.
- `fff-5sa`: run the official `fastfeedforward` matched-budget baseline where shape-compatible.
- `fff-ytn`: run multi-seed matched-budget optimizer/WSD selection and final evaluation.
- `fff-v9q`: evaluate a bank-specific Muon grouping or explicit optimizer policy for 3D FFF replacement banks.
- `fff-rcs`: generate full-student Pareto source data for accuracy, active rows/FLOPs, and throughput after the comparison tasks complete.
- `fff-6rw`: evaluate row-plus-column sparse activation and checkerboard MoE/sublinear baselines.

## Optimizer Plan Amendment

Start with simple official Muon plus AdamW fallback until the teacher, distillation, and fine-tuning paths are stable. Insert optimizer ablations after the simple-Muon teacher and layerwise distillation baseline are working, before final KD/fairness/reporting.

Current optimizer policy: the audited split sends hidden 2D matrix parameters to
Muon and sends biases, excluded names, and non-2D tensors to AdamW fallback. FFF
replacement banks such as `route_weight`, `route_output`, `route_result_weight`,
`route_result_output`, `leaf_weight`, and `leaf_output` are 3D tensors, so they use
AdamW fallback in assembled FFF student fine-tuning unless a future tested
bank-specific Muon grouping is implemented. Optimizer conclusions for assembled
FFF students must state whether replacement banks used AdamW fallback or Muon.

Optimizer/schedule ablations and evidence:

- Official `muon_adamw` / `official_muon`: unchanged KellerJordan/Muon
  `SingleDeviceMuonWithAuxAdam` baseline with this repo's audited Muon/AdamW
  parameter split.
- `pace_muon`: optimizer-experiments PACE wrapper around the official Muon
  baseline, not the standalone hand-rolled `PaceMuon`; validation/checkpoint
  selection must use EMA weights.
- `normuon_adamw` / `muon_normuon`: vendored optimizer-experiments Muon+NorMuon
  ablation using the same audited parameter split. This is not the official Muon
  package and must be reported separately.
- `pace_normuon`: PACE wrapper around the vendored Muon+NorMuon ablation.
- WSD learning-rate schedule as a trainer-side schedule option, reported
  separately from optimizer family.
- Other optimizer-experiments techniques only after the baseline and first two ablations are reproducible.

Optimizer comparisons must use the same teacher checkpoint, split, seeds, training tokens/epochs, HPO budget, and validation-selection protocol as comparable baselines. Report optimizer family and LR schedule separately.
Vendored optimizer-experiments code is sourced from commit
`689568d71ebe92093e5f5bf433127a5184ef0c35`.

Current GC5 optimizer/WSD validation status:

- GC5 matched-budget full-student optimizer/WSD validation completed 15 one-seed
  validation-only cells with seed `1337`, `4218` train steps per cell, the same
  Stage H selected student inputs, and `test_accessed=false` throughout.
- WSD cells in GC5 are limited to the official Muon family; PACE+Muon, NorMuon,
  and PACE+NorMuon GC5 rows are cosine-only LR-tier ablations.
- The best validation family was `official_muon_cosine_lr_base` at `0.914800`.
  This does not establish a final-test optimizer ranking; no GC5-selected
  optimizer checkpoint has been evaluated on the CIFAR-10 final test set.
- Committed evidence is in `docs/fff_gc5_optimizer_wsd_validation_trials.csv`,
  `docs/fff_gc5_optimizer_wsd_validation_families.csv`, and
  `docs/fff_gc5_optimizer_wsd_validation_summary.md`.
- A 15-cell, 3-seed follow-up validation config is staged in
  `configs/finetune_optimizer_hpo_multiseed.yaml`; the first 12 jobs were
  launched on all 12 GPUs at commit `88c624b` and remain validation-only until
  summarized and selected.

## Core Research Axes

FFF variants:

- shared always-on/unrouted rows;
- 1 or 2 route rows per routing node;
- 1 to 4 rows per leaf;
- optional route-row output contribution;
- partial route-row output accumulation, where some rows introduced for routing are also accumulated into the output instead of being used only for decisions;
- optional master/fallback leaf;
- balance regularization;
- multiple STE/router-training recipes;
- batched/grouped GPU execution;
- LocoProp-S-style output-vector ridge refits.

Router recipes:

- no_ste_soft_router;
- vanilla_ste;
- clipped_ste;
- sigmoid_surrogate_ste;
- st_gumbel;
- utility_targeted_ste;
- hard_em_utility_ste;
- expert_choice_imitation;
- hard_concrete_row_gates;
- optional dselect_k_leaf_router if core tree experiments already work.

Forward-looking router-family comparisons must treat `utility_targeted_ste` and
`hard_em_utility_ste` as distinct algorithms. A Fable review found older
distillation paths always called `utility_targeted_ste(..., hard=True)`, making
it effectively hard-EM for the main route target. The staged fix adds
`router.utility_hard` and sets utility-targeted full-student configs to
`utility_hard: false`; older L7K utility-vs-hard-EM rows are caveated and should
not be used as decisive evidence that two independent algorithms agree.

LocoProp-S means fixed routes, fixed input-side row activations, teacher Linear output as local target, local ridge solve for output vectors, and optional blend back into the trainable model.

Route-row contribution ablation:

- none: route rows are used only for routing decisions;
- shared-role: the same introduced route rows are used for both routing and output contribution;
- split-role: some introduced rows are used only for routing and separate introduced rows are used only for output contribution along the visited path;
- one: exactly one visited route-result row per routing node contributes to output;
- all: all route-result rows at visited routing nodes contribute to output;
- partial/budget-matched: a reasonable count or fraction contributes when multiple route/path rows are available, chosen so active rows/FLOPs can be compared fairly against leaf/shared-row alternatives.

Reports must include the number of active contributing route rows, active-row/FLOP budget, MSE, accuracy, throughput, and route diagnostics.

Sparse row/column activation ablation:

- Evaluate FFF row activation against methods that sparsify both input rows and output columns, with matched active-FLOP, active-row, parameter, and training-token budgets where possible.
- Compare custom FFF against alternative sublinear baselines such as token-choice MoE, expert-choice MoE, and checkerboard-style sparse row-and-column activation where a token activates a sparse set of rows and a sparse set of output columns/blocks.
- Include variants where the router selects row experts only, column experts only, and coupled row/column blocks; include a checkerboard MoE baseline that activates sparse row banks and sparse column banks jointly.
- Include a checkerboard variant with an MLP router that selects sparse tiles plus a small number of always-on/shared rows that contribute every token. Compare it as a distinct family against the linear-router checkerboard variant, with router parameters, always-on rows, active FLOPs, and throughput reported separately.
- Keep the same official Mamba-3 teacher, CIFAR-10 split policy, distillation token budget, optimizer/schedule budget, validation-selection protocol, and test-access rules as the FFF experiments.
- Report local distillation MSE/cosine, validation accuracy after replacement or assembled-student fine-tune, selected final-test accuracy only after validation selection, active rows, active columns, active FLOPs, stored rows/columns, throughput, GPU utilization, router/load balance, dead experts/leaves, and fairness budget notes.
- Generated validation baselines are staged through the normal fine-tune HPO path
  using `student.source` values `official_fastfeedforward`, `sparse_row`,
  `sparse_column`, `sparse_row_column`, and `checkerboard_moe`. The first
  launchable validation grid is `configs/finetune_generated_sublinear_baselines_hpo.yaml`,
  including the appended `checkerboard_mlp_shared` family.
  These sparse baselines have explicit budget notes and are not a substitute for
  distilled FFF replacements.

Current T20 status:

- Route-output named-case config `configs/fff_distill_t20_route_output_cases.yaml` covers
  `none_routing_only`, shared one/all/half, and split one/all/half route-output cases.
- T20 run `t20_route_output_cases_20260704_092832` completed local offsets 0-1. The first
  mixed local/remote launch exposed stale remote commits and failed remote preflight
  honestly; after `scripts/sync_repo_remote.sh all`, rerun
  `t20_route_output_cases_remote_20260704_092948` completed offsets 2-6 on synced remotes.
- Final T20 evidence covers all seven named cases, all on CIFAR-10 validation split only,
  all with `test_accessed=false`, at commit `9265ec0bf80db186315ce3bc3301ffe82b8932ee`.
  Single-layer validation accuracy was measured by loading each ignored FFF state into the
  selected teacher and evaluating the full 5k validation split with test disabled.
- In the eligible-index-32 one-layer ablation, route-output contribution improved local
  MSE over `none_routing_only` for every contributing case. `split_all` had the best local
  NMSE (`0.284449`), while `split_half_fraction` was fastest (`37399.5` tokens/s). See
  `docs/t20_route_row_output_ablation.md` and
  `docs/t20_route_row_output_ablation_results.csv`.

## Experiment Stages

Stage A: environment and bugfix gates.

Current Stage A status:

- Official Mamba-3 import/run is repaired locally with pinned upstream dependencies and
  documented vendor patches for `apache-tvm-ffi==0.1.12`, `mamba-ssm==2.3.2.post1`,
  and `tilelang==0.1.8`.
- `scripts/verify_env.py` requires nightly torch cu130, CUDA, official `Mamba3`,
  `is_mimo=True`, TileLang MIMO availability, and BF16 CUDA autocast forward/backward.
- `scripts/run_tests.sh` includes the local CUDA environment gate before ruff/pytest.
- Cluster inventory sees all 12 expected GPUs, and remote project environments have passed
  the bounded verification/scheduler gates needed for one-GPU smoke jobs. At the last T06
  smoke checkpoint, `foureyes:2` and `foureyes:3` were intentionally excluded because
  unrelated high-memory jobs were already occupying them.
- Setup now uses pinned constraints/commits for the observed Python 3.12/cu130 Stage A
  environment; update the constraints file deliberately when changing dependency versions.
- Completed Stage A bugfixes include FFF route-output empty-batch handling and no-grad
  LocoProp-S ridge refits.
- Teacher HPO now has opt-in candidate prefilters for both 9M-11M parameter-count
  compatibility and CUDA BF16 kernel smoke. Broad HPO and pinned HPO smoke configs
  enable the parameter-count pool first, then reject official Mamba-3 candidates that
  fail the optimized TileLang CUDA path before they consume HPO trial slots. The CUDA
  smoke uses a synthetic batch size of 1 and never touches CIFAR-10 test data.
- `configs/teacher_hpo_safe.yaml` is the refill config for keeping idle GPUs busy
  while broad teacher HPO runs elsewhere: it pins the known CUDA-valid 9.5M
  official Mamba-3 shape and searches only training hyperparameters, WSD/cosine,
  augmentation, drop-path, and batch size.
- The parameter-count prefilter scans only fields that change the teacher parameter
  count; non-count knobs such as `drop_path` remain sampled per trial without
  multiplying the official model-construction pool.
- Long teacher HPO may start only after the current prefilter code is committed/pushed,
  remotes are synced to that commit, current GPU occupancy is rechecked, and unavailable
  slots are explicitly excluded.

Stage B: teacher HPO with all usable GPUs, validation pruning, and final CIFAR-10 test only after selection.

- T06 selected `ripper:0` from `teacher_hpo_wave1_20260704_0618` as the teacher
  checkpoint after validation accuracy reached `0.9418`. The gated final CIFAR-10
  test evaluation was run once for that selected checkpoint and reached `0.9399`
  test accuracy. See `docs/t06_teacher_hpo_final_summary.md`.
- Downstream distillation should use the ignored local checkpoint copy
  `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt` and pass it via
  `--teacher-checkpoint` or `--checkpoint`; do not commit the checkpoint.

Stage C: STE/router smoke tests on one representative middle Linear layer.

- Layerwise distillation now has CPU-tested executable HPO orchestration: dry-run planning
  remains available without a teacher checkpoint, while `--execute-trials true` records
  per-trial configs/results and requires a selected teacher checkpoint for non-smoke runs.
- Balance HPO knobs are active in layerwise distillation. `balance.recipe/coeff`,
  min-leaf occupancy, uniform leaf balance, and margin regularization are parsed, added to
  the training loss, and logged in layer metrics.
- LocoProp-S is wired as an optional output-vector refit in layerwise distillation. It
  builds a fixed-route/fixed-activation local basis, solves a ridge problem in FP32,
  blends solved output vectors back into the FFF layer, and logs solve telemetry plus
  before/after MSE.
- Stage C router recipe smoke was rerun after fixing distributed grid offsets. Run
  `distill_stage_c_router_offsets_20260704_090349` covered each of the seven configured
  router recipes exactly once, used train-split activation samples only, and reported
  `test_accessed=false`. The earlier `distill_stage_c_router_20260704_085938` wave is
  invalid for router comparison because every slot ran the first grid candidate.

Stage D: single-layer architecture sweep.

- Stage D architecture sweep run `distill_stage_d_arch_20260704_090730` completed 10/10
  validation-split trials without CIFAR-10 test access. The best short-run local recipe
  was `vanilla_ste` with `split_routing_output`, `shared_unrouted_frac=0.2`,
  `route_rows=1`, `leaf_rows=4`, `depth=5`, `route_rows_output_count=all`,
  `route_rows_output_fraction=0.5`, and LocoProp-S enabled. See
  `docs/t13_stage_d_arch_summary.md`.

Stage E: representative-layer sweep.

- Stage E representative-layer run `distill_stage_e_layers_20260704_091514` applied the
  Stage D recipe to eligible indices `[0]`, `[32]`, and `[60]` on the validation split.
  All records report `test_accessed=false`. The recipe transferred well to the early and
  late `in_proj` examples, while the middle layer remained harder. See
  `docs/t13_stage_e_layer_summary.md`.

Stage F: full layerwise distillation HPO.

- Stage F run `distill_stage_f_shards_20260704_091746` sharded all 64 eligible Linear
  layers exactly once across 10 one-GPU jobs on `work`, `ripper`, `foureyes`, and `ai`.
  Jobs succeeded 10/10, covered indices `0..63` with no missing/duplicate layers, and
  every artifact reports `test_accessed=false`. `ripper:1` and `foureyes:0` were
  intentionally excluded because pre-launch checks showed persistent anomalous 100%
  utilization with negligible memory and no visible compute PID.
- Stage F is validation-split layerwise evidence, not final student accuracy. Mean final
  normalized MSE across eligible Linear layers was `0.288707`, mean cosine similarity was
  `0.807726`, and mean measured distillation throughput was `42752.8` tokens/s. LocoProp-S
  succeeded and decreased or preserved local MSE for every layer. Late and middle
  `out_proj` layers remain the hardest and need additional recipe/budget tuning before
  final student claims. See `docs/t13_stage_f_layerwise_summary.md`.
- T18 review found that the completed Stage F layerwise pass trained FFF replacements on
  CIFAR-10 validation-split images and then reported layerwise metrics from that capture
  stream. This is not CIFAR-10 test leakage, but it is train-on-validation leakage for
  layerwise distillation metrics. The corrected distillation default is now
  `--sample-split train_eval`, which captures CIFAR-10 train-split images through
  eval/no-augmentation transforms, and `LinearDistillConfig` records deterministic held-out
  token metrics. The validation-capture Stage F artifacts remain legacy/provenance
  evidence and are superseded for corrected layerwise claims.
- Corrected full Stage F run `distill_stage_f_train_eval_shards_20260704_abd09d5`
  sharded all 64 eligible Linear layers exactly once across all 12 configured GPUs
  (`work`, `ripper`, `foureyes`, and `ai`) at commit
  `abd09d5537753f870e247aefaed01fe8091b483d`. All 12 one-GPU shard jobs succeeded with
  `sample_split=train_eval`, `metric_split=holdout`, 29,491 fit tokens and 3,277 held-out
  metric tokens per layer, and `test_accessed=false` throughout. Mean held-out normalized
  MSE was `0.292891`, median NMSE `0.186459`, mean cosine `0.809442`, mean throughput
  `42562.0` tokens/s, mean dead leaves `8.44`, and worst held-out NMSE `0.702147` on
  `blocks.11.forward_block.mixer.mixer.out_proj`. LocoProp-S succeeded and decreased or
  preserved local MSE for every corrected layer record. See
  `docs/t13_stage_f_train_eval_layerwise_summary.md` and
  `docs/t13_stage_f_train_eval_layerwise_summary.csv`.
- Hard `out_proj` train-eval recipe sweep
  `hard_outproj_router_balance_train_eval_20260704_819c2c6` ran 12 one-GPU cases across
  all 12 GPUs at commit `819c2c6f05c199821aedf8003617eb55ef4935db`, with
  `sample_split=train_eval`, held-out token metrics, and `test_accessed=false` throughout.
  The run covered eligible indices `[35, 39, 41, 43, 45, 47]` for vanilla, clipped,
  sigmoid, ST-Gumbel, utility, hard-EM, expert-choice, balance, route-row, and depth
  variants. All 12 jobs succeeded after the `hpo_overrides` strict-config provenance
  bugfix. Best mean NMSE was `vanilla_depth_6` at `0.681258`; lowest mean dead leaves was
  `st_gumbel_split_minleaf` at `0.333333`; the fastest case was the unbalanced vanilla
  baseline at `44108.7` tokens/s. This is hard-layer recipe-selection evidence, not a
  corrected full Stage F pass or final student result. See
  `docs/hard_outproj_router_balance_train_eval_summary.md` and
  `docs/hard_outproj_router_balance_train_eval_results.csv`.
- Equal-budget router-family validation is now summarized in
  `docs/l7k_router_equal_budget_summary.md`, `docs/l7k_router_equal_budget_results.csv`,
  `docs/l7k_router_equal_budget_family_summary.md`, and
  `docs/l7k_router_equal_budget_families.csv`. The collected waves cover seven router
  families, three comparison seeds per family, and six hard `out_proj` layer records per
  seed. All 126 layer records are validation-only train-eval activation-capture results
  with `test_accessed=false`. They support router recipe selection but do not replace
  full-student final-test comparisons.
- Fairness limitation to handle in T15/T18: Stage F trial configs retain the base
  distillation seed even though scheduler status metadata records per-slot launch seeds.
  Future multi-seed comparisons must explicitly vary and report training seeds.

Stage G: end-to-end KD fine-tuning.

- T14 now has an implemented validation-only BF16 CUDA KD fine-tuning path for
  assembled Stage F FFF students. It strictly loads all 64 FFF replacement states,
  keeps CIFAR-10 test disabled, writes assembly manifests/metrics, and supports the
  optimizer/schedule families added in T19.
- The assembled-FFF KD smoke blocker is fixed. Root cause: `FFFLinear` returned FP32
  activations under CUDA BF16 autocast with FP32 parameters, while dense `nn.Linear`
  returns BF16. Those FP32 activations selected the official Mamba-3 FP32 TileLang
  backward specialization, which requested excessive dynamic shared memory. `FFFLinear`
  now casts only its public forward output to the active device autocast dtype, leaving
  FP32 parameters and no-autocast FP32 behavior intact.
- Verified batch-32 official Mamba-3 BF16 KD smoke with all 64 eligible linears
  replaced, official Muon+AdamW, forward/backward/optimizer step, and
  `test_accessed=false`:
  `outputs/t14_finetune_smoke_autocast_fix_train_nobalance_real`,
  `outputs/t14_finetune_hpo_autocast_fix_train_nobalance_real`, and
  `outputs/t14_finetune_smoke_autocast_fix_train_balance_globalcap`.
- T14 safety fixes: real fine-tune HPO now launches train mode instead of metadata mode
  when `quick_smoke=false`, and `--max-train-steps` is a global cap rather than a
  per-epoch cap.
- Student final-evaluation plumbing now exists in `cifar_mamba_fff.evaluate_student` and
  `scripts/evaluate_student_final.sh`. It requires checkpoint validation metrics, rebuilds
  the assembled FFF student, loads the selected checkpoint, marks `test_accessed=true`,
  and separates partial from full CIFAR-10 test metrics. T18 hardened this gate so student
  final evaluation requires a validation-selection record by default; the record must
  mark `selected_for_final_eval=true`, match checkpoint path, validation accuracy, and
  checkpoint hash when available, and it must not have accessed CIFAR-10 test.
  Below-target or untracked final evaluations now require explicit failure-analysis
  overrides recorded in output metadata. A validation-selected one-step HPO checkpoint
  was evaluated with `max_test_steps=1` at
  `outputs/t14_student_final_partial_autocast_fix_nobalance_qfalse_v2`, yielding
  `test_accuracy_partial=0.3125` with all 64 FFF replacements loaded.
- No substitute architecture, optimizer, CPU path, smaller model, or fake Mamba path was
  used. Full final CIFAR-10 test accuracy remains reserved for validation-selected
  checkpoints only and must not be inferred from smoke or partial-test artifacts.

Stage H: final repeated runs for validation-selected recipes.

- The validation-selected `no_balance_cosine` Stage H FFF student family has a
  full CIFAR-10 final-test evaluation after validation selection. Its three-seed
  validation mean is `0.914800`, final-test mean accuracy is `0.915133`, and
  final-test standard deviation is `0.003421`.
- Other router, architecture, optimizer, and baseline rows remain validation-only
  unless their evidence split is explicitly marked `final_test` or
  `partial_final_test`.

Fairness and baselines:

- T15 fairness reporting is generated by `cifar_mamba_fff.fairness` into
  `docs/t15_fairness_summary.md` and `docs/t15_fairness_summary.csv`.
- The table aggregates committed evidence only: selected dense teacher final test, Stage F
  corrected train-eval layerwise FFF held-out token metrics, legacy validation-capture
  Stage F provenance, T20 route-output single-layer ablations, T19 optimizer/schedule
  smoke ablations, GC5 matched-budget optimizer/WSD validation rows, T14 KD/final-eval
  artifacts, official `fastfeedforward.FFF` shape
  smoke, three-seed final-test dense-copy/low-rank/shared-only/smaller-dense baseline
  evidence, and explicit limitation rows when required baselines lack metrics.
- Fairness validation enforces that CIFAR-10 test access is reported only on
  `final_test` or `partial_final_test` rows. The shared-only rows full-student
  baseline now has validation-selected three-seed final-test evidence with mean
  validation accuracy `0.910000` and mean final-test accuracy `0.906267`.
- T18/GC5 hardened fairness/report provenance checks: required source CSVs and teacher
  summary files must exist with expected row counts, the base fairness table must
  contain exactly 33 rows, the optional GC5 family summary must contain exactly
  15 rows when present, the committed fairness table must contain 48 rows with
  GC5 included, exactly 10 rows may report test access, test-access booleans
  must parse canonically, and final-report validation rejects stale committed
  fairness CSVs by regenerating rows from the source evidence.

## Required Quality Gates

- Static/code sanity: ruff, imports, config validation, no fallback path, no TODO in critical path.
- Unit tests: shapes, route IDs, grouped-vs-naive equivalence, STE behavior, optimizer grouping, LocoProp MSE decrease, replacement correctness.
- Numerical sanity: overfit tiny teacher-linear distillation batch and shared-only baseline.
- STE sanity: gradients, finite entropy, no NaNs, utility/EM assignments non-degenerate.
- Distributed sanity: local DDP smoke and one-GPU scheduler smoke across available machines.
- Profiling sanity: no Python per-token hot path; naive/grouped/dense throughput measured.
- Result sanity: equal budgets, no test leakage, no impossible metrics or overwritten checkpoints.
- Adversarial review: record bugs, fixes, residual risk, and rerun checks.

## Reporting

`docs/final_report.md` must report environment, Beads/git, teacher, FFF validation, STE comparison, architecture sweep, layerwise distillation, KD fine-tuning, fairness, Pareto plots, best recipes, and honest limitations.
