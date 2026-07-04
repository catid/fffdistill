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

## Optimizer Plan Amendment

Start with simple official Muon plus AdamW fallback until the teacher, distillation, and fine-tuning paths are stable. Insert optimizer ablations after the simple-Muon teacher and layerwise distillation baseline are working, before final KD/fairness/reporting.

Planned optimizer/schedule ablations:

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

LocoProp-S means fixed routes, fixed input-side row activations, teacher Linear output as local target, local ridge solve for output vectors, and optional blend back into the trainable model.

Route-row contribution ablation:

- none: route rows are used only for routing decisions;
- shared-role: the same introduced route rows are used for both routing and output contribution;
- split-role: some introduced rows are used only for routing and separate introduced rows are used only for output contribution along the visited path;
- one: exactly one visited route-result row per routing node contributes to output;
- all: all route-result rows at visited routing nodes contribute to output;
- partial/budget-matched: a reasonable count or fraction contributes when multiple route/path rows are available, chosen so active rows/FLOPs can be compared fairly against leaf/shared-row alternatives.

Reports must include the number of active contributing route rows, active-row/FLOP budget, MSE, accuracy, throughput, and route diagnostics.

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
- Fairness limitation to handle in T15/T18: Stage F trial configs retain the base
  distillation seed even though scheduler status metadata records per-slot launch seeds.
  Future multi-seed comparisons must explicitly vary and report training seeds.

Stage G: end-to-end KD fine-tuning.

Stage H: final repeated runs with at least three seeds for top validation-selected recipes.

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
