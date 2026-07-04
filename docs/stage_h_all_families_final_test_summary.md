# Stage H All-Families Final Student Test Summary

- Collected roots: `outputs/scheduler_collected/student_final_stage_h_all_families_20260704_10a7691, outputs/scheduler_collected/student_final_stage_h_all_families_work_retry_20260704_10a7691, outputs/scheduler_collected/student_final_stage_h_all_families_low_lr_failure_analysis_20260704_10a7691`
- Trial rows: `12`
- Families: `4`
- CIFAR-10 test accessed: `true`
- Partial test evaluation: `false`
- Selection rule: validation-selected Stage H full-student family checkpoints from docs/final_eval_selection/stage_h_all_families_20260704; low_lr_cosine rows were below the 0.90 validation gate and were run only as explicitly labeled failure analysis with allow_below_target=true.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Below-target trials | Allow below target | Mean test acc | Std test acc | Best case | Best seed | Best test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_balance_cosine | 3 | 21001,21002,21003 | 0.914800 | 0.004060 | 0 | false | 0.915133 | 0.003421 | no_balance_cosine_seed21002 | 21002 | 0.919000 |
| baseline_cosine | 3 | 21001,21002,21003 | 0.908000 | 0.001709 | 0 | false | 0.909133 | 0.004020 | baseline_cosine_seed21003 | 21003 | 0.911700 |
| wsd | 3 | 21001,21002,21003 | 0.903067 | 0.001301 | 0 | false | 0.905700 | 0.001539 | wsd_seed21003 | 21003 | 0.907400 |
| low_lr_cosine | 3 | 21001,21002,21003 | 0.889533 | 0.001724 | 3 | true | 0.889133 | 0.001002 | low_lr_cosine_seed21002 | 21002 | 0.889900 |

## Trial Rows

| Case | Machine | GPU | Seed | Val acc | Min val gate | Allow below target | Test acc | Test steps | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_cosine_seed21001 | work | 0 | 21001 | 0.906400 | 0.900000 | false | 0.904500 | 20 | de1fe78e379c5b1e7e575f2f689f319993f54f1e2b98cf37c57062d5e335e5fd |
| baseline_cosine_seed21002 | work | 1 | 21002 | 0.907800 | 0.900000 | false | 0.911200 | 20 | 46bad4424aa671dac49852dffcc080e89a8581dcbf7950ada8654f0749506528 |
| baseline_cosine_seed21003 | ripper | 0 | 21003 | 0.909800 | 0.900000 | false | 0.911700 | 20 | 1b39356ef3c5596506fe70c77ef3a9c7cd50939970224a2274d8472f4bbfef66 |
| low_lr_cosine_seed21001 | foureyes | 0 | 21001 | 0.889200 | 0.900000 | true | 0.888000 | 20 | d8b2f9b17e10ba065ca35529d8716762e6343e7f0967f15343dceda4525b214b |
| low_lr_cosine_seed21002 | foureyes | 1 | 21002 | 0.891400 | 0.900000 | true | 0.889900 | 20 | 5562c8a36325949b4c98357942bc568f84f4fba60f5f5e3be518e02243ae46bd |
| low_lr_cosine_seed21003 | foureyes | 2 | 21003 | 0.888000 | 0.900000 | true | 0.889500 | 20 | 347f262bd79abd5f77a396df818956d957ee240a9139dab9a8fb13159ae426be |
| no_balance_cosine_seed21001 | foureyes | 3 | 21001 | 0.910400 | 0.900000 | false | 0.913900 | 20 | e6a449a743985d6b7ea0dbdfc92af069f8c05feda78ee739a11ad967222c637d |
| no_balance_cosine_seed21002 | ai | 0 | 21002 | 0.918400 | 0.900000 | false | 0.919000 | 20 | 0b154b1a81c0a387b38207bebbb11bb08566dbe14d185887523942c579b25f56 |
| no_balance_cosine_seed21003 | ai | 1 | 21003 | 0.915600 | 0.900000 | false | 0.912500 | 20 | c878c6684bd743b4d8c992b3512f6f1349d619066426ec142b92e0cec0318088 |
| wsd_seed21001 | ripper | 1 | 21001 | 0.903000 | 0.900000 | false | 0.905300 | 20 | 6d28611479b22a4588b64ed437a5b453c9f6f42c86c6c627dc2ae94c89d5c172 |
| wsd_seed21002 | ripper | 2 | 21002 | 0.904400 | 0.900000 | false | 0.904400 | 20 | 90ec1979928fe35a05a08cdb286040242a2ee4ed74ed93e39a1417db40f25876 |
| wsd_seed21003 | ripper | 3 | 21003 | 0.901800 | 0.900000 | false | 0.907400 | 20 | b14513d843c75a2b3050916036877ef55c687a35b13ceb14792d71fc6b4100ac |
