# Optimizer/WSD Multiseed Final-Test Summary

- Collected roots: `outputs/scheduler_collected/optimizer_wsd_multiseed_final_test_20260705_7515c46_work01, outputs/scheduler_collected/optimizer_wsd_multiseed_final_test_20260705_7515c46_ai0`
- Trial rows: `3`
- Families: `1`
- CIFAR-10 test accessed: `true`
- Partial test evaluation: `false`
- Selection rule: Top validation family selected from docs/optimizer_wsd_multiseed_validation_families.csv; selected family pace_normuon_cosine across seeds 31001/31002/31003.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Below-target trials | Allow below target | Mean test acc | Std test acc | Best case | Best seed | Best test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pace_normuon_cosine | 3 | 31001,31002,31003 | 0.913867 | 0.003301 | 0 | false | 0.910767 | 0.000586 | pace_normuon_cosine_seed31001 | 31001 | 0.911200 |

## Trial Rows

| Case | Machine | GPU | Seed | Val acc | Min val gate | Allow below target | Test acc | Test steps | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pace_normuon_cosine_seed31001 | work | 0 | 31001 | 0.914800 | 0.900000 | false | 0.911200 | 20 | c7083a30f0b76e072f3036d1bacebd2a0372836a60245989ab94afe3758c2216 |
| pace_normuon_cosine_seed31002 | work | 1 | 31002 | 0.910200 | 0.900000 | false | 0.911000 | 20 | c6df04b7fa9071a7ae421c51b91d315fb1c36234afc1285dc1f97a18b2e93e47 |
| pace_normuon_cosine_seed31003 | ai | 0 | 31003 | 0.916600 | 0.900000 | false | 0.910100 | 20 | 5e51037939cbd2ea88ddc2d978e26ebf441e3860fef1e2ed5140fab450dc569c |
