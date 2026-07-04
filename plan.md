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

- PACE+Muon from `https://github.com/catid/optimizer_experiments`.
- PACE+NorMuon from `https://github.com/catid/optimizer_experiments`.
- WSD learning-rate schedule as a trainer-side schedule option.
- Other optimizer-experiments techniques only after the baseline and first two ablations are reproducible.

Optimizer comparisons must use the same teacher checkpoint, split, seeds, training tokens/epochs, HPO budget, and validation-selection protocol as comparable baselines. Report optimizer family and LR schedule separately.

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
- Long teacher HPO may start only after the current prefilter code is committed/pushed,
  remotes are synced to that commit, current GPU occupancy is rechecked, and unavailable
  slots are explicitly excluded.

Stage B: teacher HPO with all usable GPUs, validation pruning, and final CIFAR-10 test only after selection.

Stage C: STE/router smoke tests on one representative middle Linear layer.

Stage D: single-layer architecture sweep.

Stage E: representative-layer sweep.

Stage F: full layerwise distillation HPO.

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
