# Optimizer/WSD Multiseed Validation Summary

- Collected roots: `outputs/scheduler_collected/optimizer_wsd_multiseed_validation_20260705_000419_88c624b, outputs/scheduler_collected/optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14`
- Trial rows: `15`
- Families: `5`
- CIFAR-10 test accessed: `false`
- Accuracy statistic: family standard deviation is sample standard deviation; one-trial families report `0.000000`.

These are validation summaries for Stage H fine-tune HPO runs. Rows with `test_accessed=true` are rejected rather than summarized.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Mean train steps | Best case | Best seed | Best val acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pace_normuon_cosine | 3 | 31001,31002,31003 | 0.913867 | 0.003301 | 4218.000000 | pace_normuon_cosine_seed31003 | 31003 | 0.916600 |
| normuon_cosine | 3 | 31001,31002,31003 | 0.911933 | 0.000231 | 4218.000000 | normuon_cosine_seed31002 | 31002 | 0.912200 |
| pace_muon_cosine | 3 | 31001,31002,31003 | 0.911133 | 0.001890 | 4218.000000 | pace_muon_cosine_seed31003 | 31003 | 0.912600 |
| official_muon_cosine | 3 | 31001,31002,31003 | 0.909867 | 0.002003 | 4218.000000 | official_muon_cosine_seed31003 | 31003 | 0.911400 |
| official_muon_wsd | 3 | 31001,31002,31003 | 0.907333 | 0.004636 | 4218.000000 | official_muon_wsd_seed31001 | 31001 | 0.910400 |

## Trial Rows

| Run | Machine | GPU | Case | Family | Seed | Best val acc | Train steps | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14 | ai | 0 | pace_normuon_cosine_seed31003 | pace_normuon_cosine | 31003 | 0.916600 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14/ai/0/trials/trial_000014_pace_normuon_cosine_seed31003/student_best.pt |
| optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14 | work | 0 | pace_normuon_cosine_seed31001 | pace_normuon_cosine | 31001 | 0.914800 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14/work/0/trials/trial_000012_pace_normuon_cosine_seed31001/student_best.pt |
| optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14 | work | 1 | pace_normuon_cosine_seed31002 | pace_normuon_cosine | 31002 | 0.910200 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_1411104_wave1_offsets12_14/work/1/trials/trial_000013_pace_normuon_cosine_seed31002/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ai | 0 | normuon_cosine_seed31002 | normuon_cosine | 31002 | 0.912200 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ai/0/trials/trial_000010_normuon_cosine_seed31002/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ai | 1 | normuon_cosine_seed31003 | normuon_cosine | 31003 | 0.911800 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ai/1/trials/trial_000011_normuon_cosine_seed31003/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | foureyes | 0 | pace_muon_cosine_seed31001 | pace_muon_cosine | 31001 | 0.911800 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/foureyes/0/trials/trial_000006_pace_muon_cosine_seed31001/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | foureyes | 1 | pace_muon_cosine_seed31002 | pace_muon_cosine | 31002 | 0.909000 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/foureyes/1/trials/trial_000007_pace_muon_cosine_seed31002/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | foureyes | 2 | pace_muon_cosine_seed31003 | pace_muon_cosine | 31003 | 0.912600 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/foureyes/2/trials/trial_000008_pace_muon_cosine_seed31003/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | foureyes | 3 | normuon_cosine_seed31001 | normuon_cosine | 31001 | 0.911800 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/foureyes/3/trials/trial_000009_normuon_cosine_seed31001/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ripper | 0 | official_muon_cosine_seed31003 | official_muon_cosine | 31003 | 0.911400 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ripper/0/trials/trial_000002_official_muon_cosine_seed31003/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ripper | 1 | official_muon_wsd_seed31001 | official_muon_wsd | 31001 | 0.910400 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ripper/1/trials/trial_000003_official_muon_wsd_seed31001/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ripper | 2 | official_muon_wsd_seed31002 | official_muon_wsd | 31002 | 0.902000 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ripper/2/trials/trial_000004_official_muon_wsd_seed31002/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | ripper | 3 | official_muon_wsd_seed31003 | official_muon_wsd | 31003 | 0.909600 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/ripper/3/trials/trial_000005_official_muon_wsd_seed31003/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | work | 0 | official_muon_cosine_seed31001 | official_muon_cosine | 31001 | 0.907600 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/work/0/trials/trial_000000_official_muon_cosine_seed31001/student_best.pt |
| optimizer_wsd_multiseed_validation_20260705_000419_88c624b | work | 1 | official_muon_cosine_seed31002 | official_muon_cosine | 31002 | 0.910600 | 4218 | outputs/scheduler_finetune_hpo/optimizer_wsd_multiseed_validation_20260705_000419_88c624b/work/1/trials/trial_000001_official_muon_cosine_seed31002/student_best.pt |
