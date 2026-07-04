# T15 Fair Baseline And Comparison Checks

This table aggregates committed evidence only. Any unavailable comparison is
reported explicitly instead of being filled with invented metrics. Smoke,
validation, partial final-test, and full final-test rows are deliberately separated.

| Method | Split | Status | Test accessed | Val acc | Final/partial test acc | NMSE | Tokens/s | Budget note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| dense_mamba3_teacher | final_test | selected_full_test | true | 0.941800 | 0.939900 |  |  | Full teacher HPO selection followed by one full CIFAR-10 test evaluation. |
| assembled_fff_stage_f_train_eval_layerwise | train_eval_layerwise_holdout | completed | false |  |  | 0.292891 | 42561.999688 | Corrected train_eval activation capture, 2 sample batches per layer shard, 10 percent held-out token metric split. |
| assembled_fff_stage_f_validation_capture_legacy | legacy_validation_capture_layerwise | superseded_leakage_limited | false |  |  | 0.288707 | 42752.755801 | Legacy validation-split layerwise distillation, 2 sample batches per layer shard; leakage-limited for layerwise validation metrics. |
| route_output_none_routing_only | validation_single_layer | completed | false | 0.940400 |  | 0.300582 | 36521.263064 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_shared_one_per_node | validation_single_layer | completed | false | 0.940600 |  | 0.293318 | 29078.716514 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_shared_all | validation_single_layer | completed | false | 0.940400 |  | 0.287133 | 28094.089306 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_shared_half_fraction | validation_single_layer | completed | false | 0.940400 |  | 0.293337 | 28063.255503 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_split_one_per_node | validation_single_layer | completed | false | 0.939600 |  | 0.292199 | 34295.624071 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_split_all | validation_single_layer | completed | false | 0.939800 |  | 0.284449 | 28240.111486 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| route_output_split_half_fraction | validation_single_layer | completed | false | 0.939600 |  | 0.292116 | 37399.486771 | Matched representative layer and token budget; active/stored rows are reported per route-output setting. |
| optimizer_official_muon_cosine | validation_smoke | completed | false | 0.169922 |  |  | 100.826328 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| optimizer_official_muon_wsd | validation_smoke | completed | false | 0.109375 |  |  | 100.563436 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| optimizer_pace_muon_ema_control | validation_smoke | completed | false | 0.138672 |  |  | 13.114041 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| optimizer_pace_muon_c1e3 | validation_smoke | completed | false | 0.099609 |  |  | 13.220083 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| optimizer_normuon_wsd | validation_smoke | completed | false | 0.130859 |  |  | 13.864579 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| optimizer_pace_normuon_c1e3 | validation_smoke | completed | false | 0.179688 |  |  | 96.410718 | Equal two-step teacher smoke budget; no FFF replacement banks; not a quality ranking. |
| t14_t14_finetune_smoke_20260704 | validation_smoke | failed | false |  |  |  |  | initial full FFF KD smoke batch 512 |
| t14_t14_finetune_smoke_20260704_b32 | validation_smoke | failed | false |  |  |  |  | batch 32 after FFF memory fix |
| t14_t14_finetune_smoke_20260704_b32_nobalance | validation_smoke | failed | false |  |  |  |  | batch 32 balance disabled |
| t14_t14_finetune_smoke_20260704_b32_nobalance_contig | validation_smoke | failed | false |  |  |  |  | batch 32 balance disabled FFF contiguous output |
| t14_t14_finetune_smoke_autocast_fix_train_nobalance_real | validation_smoke | succeeded | false |  |  |  |  | batch 32 assembled FFF KD one train/val step no balance |
| t14_t14_finetune_hpo_autocast_fix_train_nobalance_real | validation_smoke | succeeded | false |  |  |  |  | HPO wrapper train-mode one trial one train/val step no balance |
| t14_t14_finetune_smoke_autocast_fix_train_balance_globalcap | validation_smoke | succeeded | false |  |  |  |  | batch 32 assembled FFF KD one global train step with balance |
| t14_t14_student_final_partial_autocast_fix_nobalance_qfalse_v2 | partial_final_test | succeeded partial_test_accuracy=0.3125 | true |  | 0.312500 |  |  | selected one-step HPO checkpoint partial CIFAR-10 test max_test_steps=1 |
| fff_student_stage_h_no_balance_cosine | final_test | selected_full_test | true | 0.914800 | 0.914933 |  |  | Validation-selected no_balance_cosine family, three seeds, full CIFAR-10 test evaluation after selection. |
| official_fastfeedforward_fff | shape_smoke | shape_compatible_forward_tested | false |  |  |  |  | CPU API/shape smoke only. |
| matched_smaller_dense_linear | validation | completed | false | 0.931933 |  |  |  | Three seeds, three fine-tune epochs, 4218 train steps each, same teacher checkpoint and validation-only selection protocol. |
| dense_teacher_copied_student | validation | completed | false | 0.931667 |  |  |  | Three seeds, three fine-tune epochs, 4218 train steps each, same teacher checkpoint and validation-only selection protocol. |
| matched_low_rank_linear | validation | completed | false | 0.926400 |  |  |  | Three seeds, three fine-tune epochs, 4218 train steps each, same teacher checkpoint and validation-only selection protocol. |
| shared_only_rows_baseline | validation | completed | false | 0.910000 |  |  |  | Three seeds, three fine-tune epochs, 4218 train steps each, same teacher checkpoint and validation-only selection protocol. |
| gc5_official_muon_cosine_lr_base | validation | completed | false | 0.914800 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_muon_cosine_lr_base | validation | completed | false | 0.913600 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_normuon_cosine_lr_low | validation | completed | false | 0.912200 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_normuon_cosine_lr_low | validation | completed | false | 0.912200 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_official_muon_wsd_lr_base | validation | completed | false | 0.907600 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_muon_cosine_lr_high | validation | completed | false | 0.906600 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_normuon_cosine_lr_base | validation | completed | false | 0.898200 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_official_muon_wsd_lr_low | validation | completed | false | 0.897200 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_official_muon_cosine_lr_low | validation | completed | false | 0.893800 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_normuon_cosine_lr_base | validation | completed | false | 0.892000 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_muon_cosine_lr_low | validation | completed | false | 0.892000 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_official_muon_cosine_lr_high | validation | completed | false | 0.887800 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_official_muon_wsd_lr_high | validation | completed | false | 0.876000 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_pace_normuon_cosine_lr_high | validation | completed | false | 0.802800 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |
| gc5_normuon_cosine_lr_high | validation | completed | false | 0.747000 |  |  |  | GC5 matched-budget full-student fine-tune validation cell; same teacher, split, distillation artifact source, three epochs, and no CIFAR-10 test access. |

## Fairness Checks

- CIFAR-10 test access appears only in `final_test` or `partial_final_test` rows.
- Optimizer ablations use equal two-step smoke budgets and are not ranked as final quality results.
- Current Muon grouping sends only hidden 2D matrix parameters to Muon; assembled FFF replacement banks such as route_weight, route_output, route_result_weight, route_result_output, leaf_weight, and leaf_output are 3D tensors and use AdamW fallback unless a future tested bank-specific Muon grouping is implemented. Any assembled-student optimizer conclusion must state whether replacement banks used AdamW fallback or a tested Muon bank grouping.
- Route-output ablations use one representative layer with reported active/stored row budgets.
- Stage F layerwise rows are legacy validation-capture MSE/cosine/throughput evidence, not clean held-out validation metrics and not final accuracy.
- Stage F train-eval rows use CIFAR-10 train images with eval/no-augmentation transforms and held-out token metrics; they are clean layerwise distillation evidence, not final accuracy.
- Required baselines without committed metrics are explicitly marked `not_run`; completed baseline rows are validation-only unless separately marked final-test.
