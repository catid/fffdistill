# Fine-Tune HPO Validation Summary

- Collected roots: `outputs/scheduler_collected/stage_h_validation_full3ep_20260704_5b5e4e1`
- Trial rows: `12`
- Families: `4`
- CIFAR-10 test accessed: `false`
- Accuracy statistic: family standard deviation is sample standard deviation; one-trial families report `0.000000`.

These are validation summaries for Stage H fine-tune HPO runs. Rows with `test_accessed=true` are rejected rather than summarized.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Mean train steps | Best case | Best seed | Best val acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_balance_cosine | 3 | 21001,21002,21003 | 0.914800 | 0.004060 | 4218.000000 | no_balance_cosine_seed21002 | 21002 | 0.918400 |
| baseline_cosine | 3 | 21001,21002,21003 | 0.908000 | 0.001709 | 4218.000000 | baseline_cosine_seed21003 | 21003 | 0.909800 |
| wsd | 3 | 21001,21002,21003 | 0.903067 | 0.001301 | 4218.000000 | wsd_seed21002 | 21002 | 0.904400 |
| low_lr_cosine | 3 | 21001,21002,21003 | 0.889533 | 0.001724 | 4218.000000 | low_lr_cosine_seed21002 | 21002 | 0.891400 |

## Trial Rows

| Run | Machine | GPU | Case | Family | Seed | Best val acc | Train steps | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ai | 0 | no_balance_cosine_seed21002 | no_balance_cosine | 21002 | 0.918400 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ai/0/trials/trial_000010_no_balance_cosine_seed21002/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ai | 1 | no_balance_cosine_seed21003 | no_balance_cosine | 21003 | 0.915600 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ai/1/trials/trial_000011_no_balance_cosine_seed21003/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | foureyes | 0 | low_lr_cosine_seed21001 | low_lr_cosine | 21001 | 0.889200 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/foureyes/0/trials/trial_000006_low_lr_cosine_seed21001/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | foureyes | 1 | low_lr_cosine_seed21002 | low_lr_cosine | 21002 | 0.891400 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/foureyes/1/trials/trial_000007_low_lr_cosine_seed21002/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | foureyes | 2 | low_lr_cosine_seed21003 | low_lr_cosine | 21003 | 0.888000 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/foureyes/2/trials/trial_000008_low_lr_cosine_seed21003/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | foureyes | 3 | no_balance_cosine_seed21001 | no_balance_cosine | 21001 | 0.910400 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/foureyes/3/trials/trial_000009_no_balance_cosine_seed21001/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ripper | 0 | baseline_cosine_seed21003 | baseline_cosine | 21003 | 0.909800 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ripper/0/trials/trial_000002_baseline_cosine_seed21003/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ripper | 1 | wsd_seed21001 | wsd | 21001 | 0.903000 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ripper/1/trials/trial_000003_wsd_seed21001/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ripper | 2 | wsd_seed21002 | wsd | 21002 | 0.904400 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ripper/2/trials/trial_000004_wsd_seed21002/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | ripper | 3 | wsd_seed21003 | wsd | 21003 | 0.901800 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/ripper/3/trials/trial_000005_wsd_seed21003/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | work | 0 | baseline_cosine_seed21001 | baseline_cosine | 21001 | 0.906400 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/work/0/trials/trial_000000_baseline_cosine_seed21001/student_best.pt |
| stage_h_validation_full3ep_20260704_5b5e4e1 | work | 1 | baseline_cosine_seed21002 | baseline_cosine | 21002 | 0.907800 | 4218 | outputs/scheduler_finetune_hpo/stage_h_validation_full3ep_20260704_5b5e4e1/work/1/trials/trial_000001_baseline_cosine_seed21002/student_best.pt |
