# Stage H Final Student Test Summary

- Collected roots: `outputs/scheduler_collected/student_final_stage_h_validation_full3ep_20260704_a32f93f`
- Trial rows: `3`
- Families: `1`
- CIFAR-10 test accessed: `true`
- Partial test evaluation: `false`
- Selection rule: validation-selected `no_balance_cosine` family from Stage H full 3-epoch validation.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Mean test acc | Std test acc | Best case | Best seed | Best test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_balance_cosine | 3 | 21001,21002,21003 | 0.914800 | 0.004060 | 0.914933 | 0.004022 | no_balance_cosine_seed21002 | 21002 | 0.919400 |

## Trial Rows

| Case | Machine | GPU | Seed | Val acc | Test acc | Test steps | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_balance_cosine_seed21001 | foureyes | 3 | 21001 | 0.910400 | 0.913800 | 313 | e6a449a743985d6b7ea0dbdfc92af069f8c05feda78ee739a11ad967222c637d |
| no_balance_cosine_seed21002 | foureyes | 1 | 21002 | 0.918400 | 0.919400 | 313 | 0b154b1a81c0a387b38207bebbb11bb08566dbe14d185887523942c579b25f56 |
| no_balance_cosine_seed21003 | foureyes | 2 | 21003 | 0.915600 | 0.911600 | 313 | c878c6684bd743b4d8c992b3512f6f1349d619066426ec142b92e0cec0318088 |
