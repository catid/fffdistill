# T15 Matched Baseline Final Student Test Summary

- Collected roots: `outputs/scheduler_collected/student_final_t15_baselines_20260704_10a7691`
- Trial rows: `12`
- Families: `4`
- CIFAR-10 test accessed: `true`
- Partial test evaluation: `false`
- Selection rule: validation-selected matched baseline checkpoints from docs/final_eval_selection/t15_baselines_all_families_20260704; all rows satisfied the 0.90 selected-validation gate without allow_below_target override.

## Family Aggregate

| Family | Trials | Seeds | Mean val acc | Std val acc | Below-target trials | Allow below target | Mean test acc | Std test acc | Best case | Best seed | Best test acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_copy | 3 | 21001,21002,21003 | 0.931667 | 0.002023 | 0 | false | 0.927867 | 0.003408 | dense_copy_seed21002 | 21002 | 0.931800 |
| smaller_dense | 3 | 21001,21002,21003 | 0.931933 | 0.002444 | 0 | false | 0.927500 | 0.000200 | smaller_dense_seed21003 | 21003 | 0.927700 |
| low_rank | 3 | 21001,21002,21003 | 0.926400 | 0.002227 | 0 | false | 0.921200 | 0.001229 | low_rank_seed21002 | 21002 | 0.922100 |
| shared_only | 3 | 21001,21002,21003 | 0.910000 | 0.003079 | 0 | false | 0.906267 | 0.001429 | shared_only_seed21002 | 21002 | 0.907500 |

## Trial Rows

| Case | Machine | GPU | Seed | Val acc | Min val gate | Allow below target | Test acc | Test steps | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_copy_seed21001 | ai | 0 | 21001 | 0.930400 | 0.900000 | false | 0.925800 | 20 | 9cf777a2b6e9f4f038ef2e20807dcdba96a421874d37b21a2bc117fa155aa747 |
| dense_copy_seed21002 | ai | 1 | 21002 | 0.934000 | 0.900000 | false | 0.931800 | 20 | b71622da9e23fdbf1115a7582577193691e5c5569eb1dc7680e3bacd9b40b92e |
| dense_copy_seed21003 | ripper | 0 | 21003 | 0.930600 | 0.900000 | false | 0.926000 | 20 | 2356c1d2b9fe66a1480d7dcf8833cea7541bab59175675a6c7d8ab6dbb90c49d |
| low_rank_seed21001 | ripper | 1 | 21001 | 0.928800 | 0.900000 | false | 0.921700 | 20 | 429c89e1e0d7207d65d55f298ee86d980e9ce90e827e4031c5c4c24bfa5cb206 |
| low_rank_seed21002 | work | 0 | 21002 | 0.926000 | 0.900000 | false | 0.922100 | 20 | 59ceaabf9f742aaa051839988f3154577a10e792e7f45bcc8e78fd4655187dde |
| low_rank_seed21003 | work | 1 | 21003 | 0.924400 | 0.900000 | false | 0.919800 | 20 | 315be53e65fb610d258857a3dca33ec9ddc5e6e3f508e715e96bdbf38877acf0 |
| shared_only_seed21001 | foureyes | 1 | 21001 | 0.912600 | 0.900000 | false | 0.906600 | 20 | 406e28715b4671f7324622cffc2a7f456a028ca85e70d577cbbe128ff8063169 |
| shared_only_seed21002 | foureyes | 2 | 21002 | 0.906600 | 0.900000 | false | 0.907500 | 20 | e1f0eda0813c6ce967011b741aabb79c3a439524457a7a61e8a2f6de9cec13c6 |
| shared_only_seed21003 | foureyes | 3 | 21003 | 0.910800 | 0.900000 | false | 0.904700 | 20 | fbfd2aaa8b93d56f5a4fbc45d5acc088a882563a39966fa4ccd64f834f1edd1e |
| smaller_dense_seed21001 | foureyes | 0 | 21001 | 0.929800 | 0.900000 | false | 0.927500 | 20 | 8dbe42214ea9bf2b7fd2ae6dc308e3cb89e6b39691ce9aed842f140e73ef8c26 |
| smaller_dense_seed21002 | ripper | 2 | 21002 | 0.931400 | 0.900000 | false | 0.927300 | 20 | 5b8e0cb83c478f815ccd1c2f66e387779afe08210090e6e0c5599f9a9306947d |
| smaller_dense_seed21003 | ripper | 3 | 21003 | 0.934600 | 0.900000 | false | 0.927700 | 20 | e85ebcbf0ec412b0d9482a4500b59ec72212085cbb34c4f5878922e59b2536ea |
