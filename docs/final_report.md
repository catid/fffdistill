# Final Report

Report date: 2026-07-04

This repository implements a BF16 GPU research pipeline for distilling eligible
`nn.Linear` layers in an official Mamba-3 CIFAR-10 teacher into custom Fast
Feedforward Network replacements. Results below are separated as smoke,
validation, partial final-test, and full final-test evidence. Missing experiments
are listed as limitations instead of filled with invented metrics.

## References

- Mamba-3 paper: https://arxiv.org/pdf/2603.15569
- Official Mamba repository: https://github.com/state-spaces/mamba
- Muon repository: https://github.com/KellerJordan/Muon
- Muon writeup: https://kellerjordan.github.io/posts/muon/
- Fast Feedforward Networks paper: https://arxiv.org/pdf/2308.14711
- Official fastfeedforward repository: https://github.com/pbelcak/fastfeedforward
- LocoProp paper: https://arxiv.org/pdf/2106.06199
- LocoProp Google Research code: https://github.com/google-research/google-research/tree/master/locoprop
- Proximal Backpropagation: https://arxiv.org/pdf/1706.04638
- Simple Linear Neuron Boosting: https://arxiv.org/pdf/2502.01131
- FOOF / Gradient Descent on Neurons: https://arxiv.org/pdf/2201.12250
- Layerwise preconditioning / feature learning: https://proceedings.mlr.press/v267/zhang25bh/zhang25bh.pdf
- Straight-through estimator theory: https://arxiv.org/pdf/1903.05662
- Gumbel-Softmax: https://arxiv.org/pdf/1611.01144
- Concrete distribution: https://arxiv.org/pdf/1611.00712
- Hard Concrete / L0 gates: https://arxiv.org/pdf/1712.01312
- Expert Choice routing: https://arxiv.org/pdf/2202.09368
- DSelect-k: https://arxiv.org/pdf/2106.03760
- Optimizer experiments: https://github.com/catid/optimizer_experiments

## Environment

- Python: 3.12.11.
- PyTorch: `2.14.0.dev20260702+cu130`.
- CUDA runtime reported by torch: 13.0.
- Official Mamba-3: `state-spaces/mamba` commit `ed6ce09e4d802e274b1ecc7205757b892e180a93`, imported as `mamba_ssm.Mamba3`.
- Official Muon: `KellerJordan/Muon` commit `f98f1cacc0263b04290753e32be8d498c1efc806`, using `SingleDeviceMuonWithAuxAdam`.
- `fastfeedforward.FFF`: installed and shape-smoke tested through `src/cifar_mamba_fff/models/official_fastfeedforward_baseline.py`.
- Local machine `work`: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, about 95 GiB each.
- Remote inventory: `ripper` 4x RTX PRO 6000 Blackwell Max-Q, `foureyes` 4x RTX PRO 6000 Blackwell Max-Q, `ai` 2x RTX 5090. Strict verification saw all 12 configured GPUs; `foureyes:2-3` were occupied during some runs and were excluded when appropriate.

Detailed inventory: `docs/cluster_inventory.md`.

## Beads And Git

- Repository remote: `git@github.com:catid/fffdistill.git`.
- Report commit at T17 start: `fae2047178eed2ba846374d44c88b7d39cfcf902`.
- Key closed task IDs include T00-T16, T19, T20, T22, and blocker `fff-qb3.34`.
- T14 and T15 were closed with explicit limitations after deterministic gates passed.
- Final-report task `fff-qb3.18` is closed. Adversarial review task `fff-qb3.19`
  records the T18 closure gate and residual limitations in `docs/bugfix_log.md`.

## Teacher

Selected dense teacher:

- Architecture: official Mamba-3 CIFAR wrapper, no fake block or alternate architecture.
- Parameters: `9,053,258`.
- Config: `d_model=192`, `depth=16`, `patch_size=2`, `d_state=64`, `headdim=64`, `is_mimo=true`, `mimo_rank=2`, `bidirectional=true`, `drop_path=0.05`.
- Training: BF16 autocast, official Muon plus AdamW fallback groups, cosine LR.
- Selected validation accuracy: `0.9418`.
- Final CIFAR-10 test accuracy: `0.9399`.
- Final test access: true, only after validation selection.
- Training elapsed: `8,367.94` s; train-only throughput: `1,097.22` images/s.

Detailed teacher summary: `docs/t06_teacher_hpo_final_summary.md`.

## FFF Validation

Grouped-vs-naive correctness and profiling:

- BF16 grouped-vs-naive smoke: max absolute difference `0.00390625` on 16 tokens.
- Small BF16 timing on `work:0`: dense `46.25M` tok/s forward, grouped `1.19M` tok/s forward, naive `4.40K` tok/s.
- Representative hard-routing forward/backward: dense `7.34M` tok/s, grouped `998K` tok/s.
- Region-leak inference policy after T18 review: configured `region_leak` is train-only. Eval/inference uses `effective_region_leak=0.0`, reports `region_leak_policy=train_only`, and keeps the selected-leaf grouped path. A bounded CUDA BF16 eval smoke with configured `region_leak=0.01` measured grouped `2.05M` tok/s, naive `1.96K` tok/s, dense `135.66M` tok/s, and grouped-vs-naive max difference `0.0078125`; see `docs/t18_region_leak_policy.md`.
- Naive remains a correctness path, not a training path.

LocoProp-S:

- Stage F LocoProp-S succeeded for every eligible layer.
- Mean local MSE before refit: `4.549086`; after refit: `1.289520`.
- Every recorded refit decreased or preserved local MSE.

Detailed profiling: `docs/t16_smoke_profile_summary.md`.

## STE And Routing

Implemented router recipes include vanilla STE, clipped/sigmoid surrogate families,
ST-Gumbel, utility-targeted STE, hard-EM utility, expert-choice imitation, and hard
concrete row gates. Stage F selected the working short-budget recipe:

- Router recipe: `vanilla_ste`.
- Route role: `split_routing_output`.
- `route_rows=1`, `route_result_rows=2`, `route_rows_output_count=all`,
  `route_rows_output_fraction=0.5`.
- Mean route entropy in Stage F CSV is `0.0` for the hard routed selected recipe.
- Mean dead leaves: `16.48`.

This is not a final claim that vanilla STE is best. Utility/EM/expert-choice methods
are implemented and smoke-tested, but the full fair multi-seed router comparison
remains incomplete.

Stage C compared seven router recipes on one representative layer with a fixed
architecture and no test access:

| Router | Final NMSE | Cosine | Tokens/s | Dead leaves |
| --- | ---: | ---: | ---: | ---: |
| `vanilla_ste` | 0.072189 | 0.966050 | 29978.8 | 18 |
| `clipped_ste` | 0.075208 | 0.964121 | 28997.3 | 21 |
| `sigmoid_surrogate_ste` | 0.074365 | 0.964792 | 28217.2 | 16 |
| `st_gumbel` | 0.077582 | 0.962831 | 29926.5 | 25 |
| `utility_targeted_ste` | 0.095430 | 0.954309 | 25461.2 | 30 |
| `hard_em_utility_ste` | 0.095430 | 0.954308 | 25308.2 | 30 |
| `expert_choice_imitation` | 0.282055 | 0.887177 | 25572.1 | 23 |

Stage C details: `docs/t13_stage_c_router_summary.md`.

## Architecture Sweep

Stage D swept 10 one-layer architecture/recipe trials using the Stage C leading
router families (`vanilla_ste`, `clipped_ste`, `sigmoid_surrogate_ste`,
`st_gumbel`) on CIFAR-10 validation samples only. The best local MSE recipe was:

- `vanilla_ste`, `split_routing_output`, depth `5`, shared fraction `0.2`,
  route rows `1`, leaf rows `4`, LocoProp `every_500`.
- Final NMSE: `0.004615`.
- Cosine: `0.997880`.
- Throughput: `44221.9` tokens/s.
- Active rows/token: `345.0`; stored rows: `433`.

Stage D details: `docs/t13_stage_d_arch_summary.md`.

Route-output contribution ablation on one representative layer
`blocks.8.forward_block.mixer.mixer.in_proj`:

| Case | Role | Output rows/node | Active rows/token | Effective stored rows | Final NMSE | Cosine | Tokens/s | Val accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none_routing_only` | `routing_only` | 0 | 108 | 232 | 0.300582 | 0.853248 | 36521.3 | 0.9404 |
| `shared_one_per_node` | `shared_routing_and_output` | 1 | 113 | 232 | 0.293318 | 0.857536 | 29078.7 | 0.9406 |
| `shared_all` | `shared_routing_and_output` | 2 | 118 | 232 | 0.287133 | 0.860755 | 28094.1 | 0.9404 |
| `shared_half_fraction` | `shared_routing_and_output` | 1 | 113 | 232 | 0.293337 | 0.857533 | 28063.3 | 0.9404 |
| `split_one_per_node` | `split_routing_output` | 1 | 113 | 263 | 0.292199 | 0.857408 | 34295.6 | 0.9396 |
| `split_all` | `split_routing_output` | 2 | 118 | 294 | 0.284449 | 0.861225 | 28240.1 | 0.9398 |
| `split_half_fraction` | `split_routing_output` | 1 | 113 | 263 | 0.292116 | 0.857402 | 37399.5 | 0.9396 |

Best local MSE in this one-layer sweep was `split_all`; fastest contributing case was
`split_half_fraction`. This is validation-only single-layer evidence.

Detailed route-output report: `docs/t20_route_row_output_ablation.md`.

## Layerwise Distillation

Stage F distilled all 64 eligible Linear layers once each across 10 one-GPU jobs on
`work`, `ripper`, `foureyes`, and `ai`.

- Data split: CIFAR-10 validation split sampling, 2 batches per shard.
- T18 provenance caveat: these Stage F artifacts fit layer replacements on validation-split images and report metrics from that capture stream. This is not CIFAR-10 test leakage, but it is train-on-validation leakage for layerwise distillation metrics. The corrected code defaults distillation/HPO sampling to `train_eval`, which uses train-split images with eval/no-augmentation transforms, and computes deterministic held-out token metrics. Stage F must be rerun in that mode before final full-student FFF quality claims.
- Test access: false for all records.
- Mean final normalized MSE: `0.288707`.
- Median final normalized MSE: `0.188059`.
- Mean cosine similarity: `0.807726`.
- Mean throughput: `42,752.8` tokens/s.
- Hardest layers are middle/late `out_proj` modules; worst final NMSE was `0.683897`.

Detailed layer table: `docs/t13_stage_f_layerwise_summary.md` and
`docs/t13_stage_f_layerwise_summary.csv`.

## End-To-End KD

T14 implemented the assembled FFF student fine-tuning path:

- All 64 eligible linears can be replaced from Stage F artifacts.
- BF16 CUDA official Mamba-3 KD forward/backward/optimizer step passes.
- Optimizer: official Muon plus AdamW fallback. Under the current audited split,
  hidden 2D matrix parameters use Muon, while biases, excluded names, and non-2D
  tensors use AdamW fallback. The assembled FFF student's 3D replacement banks
  (`route_weight`, `route_output`, `route_result_weight`, `route_result_output`,
  `leaf_weight`, and `leaf_output`) therefore used AdamW fallback; no
  Muon-specific conclusion is claimed for those banks.
- Losses: CE, KD KL, optional hidden MSE, balance regularization.
- `FFFLinear` now respects CUDA autocast output dtype, preventing accidental FP32
  activations from selecting the official Mamba-3 FP32 TileLang backward kernel.
- HPO wrapper now launches real train mode when `quick_smoke=false`.
- `--max-train-steps` is a global cap.

Validation/smoke evidence:

- No-balance one-step assembled KD: succeeded, `train_steps_total=1`, 64/64 replacements.
- Balance-enabled one-step assembled KD: succeeded, `train_steps_total=1`, 64/64 replacements.
- One-step HPO selected checkpoint: validation accuracy `0.40625`.
- Partial selected-checkpoint CIFAR-10 test evaluation: `max_test_steps=1`,
  `test_accuracy_partial=0.3125`, `test_accessed=true`.
- Student final evaluation is now hardened to require a validation-selection record
  with `selected_for_final_eval=true` by default; below-target or untracked
  evaluations require explicit failure-analysis overrides that are recorded in
  output metadata.

This partial test result is final-evaluation plumbing evidence, not a final quality
claim. No full multi-seed FFF student final accuracy is claimed.

Detailed T14 report: `docs/t14_finetune_summary.md`.

## Fairness

T15 generated fairness tables from committed evidence only:

- `docs/t15_fairness_summary.md`
- `docs/t15_fairness_summary.csv`

The table contains 28 rows and enforces that `test_accessed=true` appears only on
`final_test` or `partial_final_test` rows. Exactly 2 fairness rows have
`test_accessed=true`. It includes explicit `not_run` rows for
dense-copy student, shared-only rows baseline, matched low-rank Linear, and matched
smaller dense Linear. These are limitations, not hidden successes.
The final-report validator also checks the committed fairness CSV row count,
test-access row count, and key prose metrics against their source CSV/Markdown
artifacts, then regenerates fairness rows from source evidence and rejects a stale CSV.

Optimizer ablation smoke:

- Official Muon + cosine and Official Muon + WSD are both present.
- PACE+Muon, NorMuon, and PACE+NorMuon are present as optimizer-experiments ablations.
- Equal budget: 2 train steps, 512 images, train/validation only.
- These are correctness/provenance smoke runs, not optimizer quality rankings. T18 fixed the future T19 protocol so every optimizer/schedule case iterates the same explicit seed list, and NorMuon/PACE+NorMuon rows require an update-RMS calibration note or optimizer-specific LR sweep before quality claims.
- Current FFF-bank optimizer policy: assembled-student 3D replacement banks use
  AdamW fallback under the audited split. Future FFF-student optimizer comparisons
  must state whether replacement banks used AdamW fallback or a tested Muon bank
  grouping. The T19 rows are dense-teacher smoke rows, not FFF-bank evidence.

Detailed optimizer report: `docs/t19_optimizer_ablation_summary.md`.

## Pareto Inputs

Plot-ready CSV inputs are committed:

- Accuracy vs active rows: `docs/t20_route_row_output_ablation_results.csv`.
- Accuracy vs throughput: `docs/t20_route_row_output_ablation_results.csv`.
- MSE vs active rows: `docs/t20_route_row_output_ablation_results.csv` and `docs/t13_stage_f_layerwise_summary.csv`.
- Dead leaves vs balance/recipe: `docs/t13_stage_d_arch_summary.csv`.
- Route entropy vs accuracy: `docs/t20_route_row_output_ablation_results.csv`.
- STE/route method vs MSE/throughput: `docs/t13_stage_c_router_summary.csv`, `docs/t13_stage_d_arch_summary.csv`, and `docs/t13_stage_f_layerwise_summary.csv`.

Rendered plots are committed under `docs/pareto_plots/`:

- [Accuracy vs active rows](pareto_plots/accuracy_vs_active_rows.svg).
- [Accuracy vs throughput](pareto_plots/accuracy_vs_throughput.svg).
- [MSE vs active rows](pareto_plots/mse_vs_active_rows.svg).
- [Dead leaves vs balance](pareto_plots/dead_leaves_vs_balance.svg).
- [Route entropy vs accuracy](pareto_plots/route_entropy_vs_accuracy.svg).
- [STE method vs MSE/throughput](pareto_plots/ste_method_vs_mse_throughput.svg).

Regenerate and validate with:

```bash
.venv/bin/python docs/render_pareto_plots.py
.venv/bin/python docs/render_pareto_plots.py --check
```

The plot provenance and missing/limited-source notes are in
[`docs/pareto_plots/missing_sources.md`](pareto_plots/missing_sources.md).
In particular, the accuracy plots use T20 one-layer replacement validation
accuracy; no full FFF-student accuracy-vs-active-rows or accuracy-vs-throughput
CSV is committed.

## Best Recipes

Within the evidence that exists:

- Best teacher quality: selected official Mamba-3 dense teacher, `0.9399` final test accuracy.
- Best single-layer route-output MSE: `split_all`, final NMSE `0.284449`.
- Best single-layer route-output speed among contributing cases: `split_half_fraction`, `37399.5` tokens/s.
- Best Stage F layerwise recipe used for assembly: `vanilla_ste + split_routing_output + LocoProp-S`, because it completed all 64 eligible layers.
- Best overall Pareto FFF student: not established. Full multi-seed end-to-end FFF accuracy is not available.

## Limitations

- Full FFF-replaced student final CIFAR-10 accuracy is not established; only a one-batch partial selected-checkpoint test artifact exists.
- Existing Stage F layerwise FFF artifacts used validation-split activation capture and are leakage-limited for layerwise validation metrics. Rerun Stage F with `train_eval` capture and held-out token metrics before any Stage H full-student FFF claim.
- Several required baselines are not run: dense teacher-copied student, shared-only rows baseline, matched low-rank Linear, matched smaller dense Linear.
- Utility-targeted, hard-EM, expert-choice, and ST-Gumbel router recipes are implemented but not fully compared under equal final budgets.
- Grouped FFF is much faster than naive but still far slower than dense Linear in current PyTorch implementation.
- FFF-bank optimizer policy is currently AdamW fallback for 3D replacement banks;
  optimizer conclusions for assembled FFF students must remain labeled accordingly
  until a bank-specific Muon grouping is implemented and validated.
- Remote GitHub SSH auth failed earlier on remote hosts; rsync from `work` was used for remote sync.
- Remote artifact collection was hardened after several compact summaries were generated; older summaries may depend on pre-hardening tail-only artifact collection. See `docs/t18_artifact_integrity.md`.
- Some GPUs were intentionally excluded during runs due occupied/anomalous utilization.
- The official Mamba-3 Python 3.12 path required documented vendor patches in dependencies, while preserving the official Mamba-3 TileLang MIMO kernels.
- Test metrics must not be reused for HPO or recipe selection. The one-step student partial test result is recorded only after selecting that checkpoint by validation metrics.
