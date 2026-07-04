# T15 Missing Baseline Validation Summary

- Collected roots: `outputs/scheduler_collected/dense_copy_baseline_validation_20260704_2f20bb1_wave0, outputs/scheduler_collected/dense_copy_baseline_validation_20260704_02e0197_wave1, outputs/scheduler_collected/low_rank_baseline_validation_20260704_02e0197_wave0, outputs/scheduler_collected/low_rank_baseline_validation_20260704_a1e311f_wave1, outputs/scheduler_collected/smaller_dense_baseline_validation_20260704_a1e311f_wave0`
- Trial rows: `9`
- Families: `3`
- CIFAR-10 test accessed: `false`
- Accuracy statistic: family standard deviation is sample standard deviation; one-trial families report `0.000000`.

These are validation summaries for Stage H fine-tune HPO runs. Rows with `test_accessed=true` are rejected rather than summarized.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Mean train steps | Best case | Best seed | Best val acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smaller_dense | 3 | 21001,21002,21003 | 0.931933 | 0.002444 | 4218.000000 | smaller_dense_seed21003 | 21003 | 0.934600 |
| dense_copy | 3 | 21001,21002,21003 | 0.931667 | 0.002023 | 4218.000000 | dense_copy_seed21002 | 21002 | 0.934000 |
| low_rank | 3 | 21001,21002,21003 | 0.926400 | 0.002227 | 4218.000000 | low_rank_seed21001 | 21001 | 0.928800 |

## Trial Rows

| Run | Machine | GPU | Case | Family | Seed | Best val acc | Train steps | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_copy_baseline_validation_20260704_02e0197_wave1 | ai | 0 | dense_copy_seed21003 | dense_copy | 21003 | 0.930600 | 4218 | outputs/scheduler_finetune_hpo/dense_copy_baseline_validation_20260704_02e0197_wave1/ai/0/trials/trial_000002_dense_copy_seed21003/student_best.pt |
| dense_copy_baseline_validation_20260704_2f20bb1_wave0 | ai | 0 | dense_copy_seed21001 | dense_copy | 21001 | 0.930400 | 4218 | outputs/scheduler_finetune_hpo/dense_copy_baseline_validation_20260704_2f20bb1_wave0/ai/0/trials/trial_000000_dense_copy_seed21001/student_best.pt |
| dense_copy_baseline_validation_20260704_2f20bb1_wave0 | ai | 1 | dense_copy_seed21002 | dense_copy | 21002 | 0.934000 | 4218 | outputs/scheduler_finetune_hpo/dense_copy_baseline_validation_20260704_2f20bb1_wave0/ai/1/trials/trial_000001_dense_copy_seed21002/student_best.pt |
| low_rank_baseline_validation_20260704_02e0197_wave0 | ai | 1 | low_rank_seed21001 | low_rank | 21001 | 0.928800 | 4218 | outputs/scheduler_finetune_hpo/low_rank_baseline_validation_20260704_02e0197_wave0/ai/1/trials/trial_000000_low_rank_seed21001/student_best.pt |
| low_rank_baseline_validation_20260704_a1e311f_wave1 | work | 0 | low_rank_seed21002 | low_rank | 21002 | 0.926000 | 4218 | outputs/scheduler_finetune_hpo/low_rank_baseline_validation_20260704_a1e311f_wave1/work/0/trials/trial_000001_low_rank_seed21002/student_best.pt |
| low_rank_baseline_validation_20260704_a1e311f_wave1 | work | 1 | low_rank_seed21003 | low_rank | 21003 | 0.924400 | 4218 | outputs/scheduler_finetune_hpo/low_rank_baseline_validation_20260704_a1e311f_wave1/work/1/trials/trial_000002_low_rank_seed21003/student_best.pt |
| smaller_dense_baseline_validation_20260704_a1e311f_wave0 | ai | 0 | smaller_dense_seed21002 | smaller_dense | 21002 | 0.931400 | 4218 | outputs/scheduler_finetune_hpo/smaller_dense_baseline_validation_20260704_a1e311f_wave0/ai/0/trials/trial_000001_smaller_dense_seed21002/student_best.pt |
| smaller_dense_baseline_validation_20260704_a1e311f_wave0 | ai | 1 | smaller_dense_seed21003 | smaller_dense | 21003 | 0.934600 | 4218 | outputs/scheduler_finetune_hpo/smaller_dense_baseline_validation_20260704_a1e311f_wave0/ai/1/trials/trial_000002_smaller_dense_seed21003/student_best.pt |
| smaller_dense_baseline_validation_20260704_a1e311f_wave0 | foureyes | 0 | smaller_dense_seed21001 | smaller_dense | 21001 | 0.929800 | 4218 | outputs/scheduler_finetune_hpo/smaller_dense_baseline_validation_20260704_a1e311f_wave0/foureyes/0/trials/trial_000000_smaller_dense_seed21001/student_best.pt |
