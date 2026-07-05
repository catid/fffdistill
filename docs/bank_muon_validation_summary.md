# FFF Bank Muon Validation Summary

- Collected roots: `outputs/scheduler_collected/bank_muon_validation_1411104_wave0_part0, outputs/scheduler_collected/bank_muon_validation_cd134e8_wave0_part1_offsets1_4, outputs/scheduler_collected/bank_muon_validation_8d55431_wave0_part2_offset5_relaunch`
- Trial rows: `6`
- Families: `2`
- CIFAR-10 test accessed: `false`
- Accuracy statistic: family standard deviation is sample standard deviation; one-trial families report `0.000000`.

These are validation summaries for Stage H fine-tune HPO runs. Rows with `test_accessed=true` are rejected rather than summarized.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Mean train steps | Best case | Best seed | Best val acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamw_fallback_banks | 3 | 32001,32002,32003 | 0.906467 | 0.005201 | 4218.000000 | adamw_fallback_banks_seed32003 | 32003 | 0.911600 |
| muon_banks | 3 | 32001,32002,32003 | 0.897200 | 0.000872 | 4218.000000 | muon_banks_seed32003 | 32003 | 0.898200 |

## Trial Rows

| Run | Machine | GPU | Case | Family | Seed | Best val acc | Train steps | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bank_muon_validation_1411104_wave0_part0 | ai | 1 | adamw_fallback_banks_seed32001 | adamw_fallback_banks | 32001 | 0.901200 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_1411104_wave0_part0/ai/1/trials/trial_000000_adamw_fallback_banks_seed32001/student_best.pt |
| bank_muon_validation_8d55431_wave0_part2_offset5_relaunch | foureyes_scratch | 0 | muon_banks_seed32003 | muon_banks | 32003 | 0.898200 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_8d55431_wave0_part2_offset5_relaunch/foureyes_scratch/0/trials/trial_000005_muon_banks_seed32003/student_best.pt |
| bank_muon_validation_cd134e8_wave0_part1_offsets1_4 | ripper | 0 | adamw_fallback_banks_seed32002 | adamw_fallback_banks | 32002 | 0.906600 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_cd134e8_wave0_part1_offsets1_4/ripper/0/trials/trial_000001_adamw_fallback_banks_seed32002/student_best.pt |
| bank_muon_validation_cd134e8_wave0_part1_offsets1_4 | ripper | 1 | adamw_fallback_banks_seed32003 | adamw_fallback_banks | 32003 | 0.911600 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_cd134e8_wave0_part1_offsets1_4/ripper/1/trials/trial_000002_adamw_fallback_banks_seed32003/student_best.pt |
| bank_muon_validation_cd134e8_wave0_part1_offsets1_4 | ripper | 2 | muon_banks_seed32001 | muon_banks | 32001 | 0.896600 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_cd134e8_wave0_part1_offsets1_4/ripper/2/trials/trial_000003_muon_banks_seed32001/student_best.pt |
| bank_muon_validation_cd134e8_wave0_part1_offsets1_4 | ripper | 3 | muon_banks_seed32002 | muon_banks | 32002 | 0.896800 | 4218 | outputs/scheduler_finetune_hpo/bank_muon_validation_cd134e8_wave0_part1_offsets1_4/ripper/3/trials/trial_000004_muon_banks_seed32002/student_best.pt |
