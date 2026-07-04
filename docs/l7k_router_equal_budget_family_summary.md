# L7K Router Equal-Budget Family Summary

- Source rows: `docs/l7k_router_equal_budget_results.csv`
- Scope: validation-only train_eval activation-capture records; CIFAR-10 test was not accessed.
- Each router family uses three comparison seeds from the case names and six representative hard-layer records per seed.
- The CSV also preserves scheduler/HPO slot seeds separately because they differ from the comparison seeds.
- Commit provenance is mixed because later collection happened after a docs-only commit; run contexts in the artifacts retain the launch-time commit for each job.

| Router | Comparison seeds | Cases | Layers | Mean NMSE | Std NMSE | Mean cosine | Dead leaves | Entropy | p50 leaf tokens | Tokens/s | Churn | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expert_choice_imitation | 21001,21002,21003 | 3 | 18 | 0.684392 | 0.017040 | 0.525758 | 0.277778 | 0.689301 | 6.777778 | 25953.993893 | 0.756688 | false |
| utility_targeted_ste | 21001,21002,21003 | 3 | 18 | 0.688554 | 0.017387 | 0.525697 | 1.333333 | 0.000000 | 5.722222 | 23492.167564 | 0.772302 | false |
| hard_em_utility_ste | 21001,21002,21003 | 3 | 18 | 0.688585 | 0.017301 | 0.525663 | 1.722222 | 0.000000 | 5.694444 | 24142.829768 | 0.772760 | false |
| vanilla_ste | 21001,21002,21003 | 3 | 18 | 0.688923 | 0.017678 | 0.524498 | 1.166667 | 0.000000 | 4.638889 | 29612.779735 | 0.573424 | false |
| sigmoid_surrogate_ste | 21001,21002,21003 | 3 | 18 | 0.689130 | 0.017889 | 0.524236 | 1.500000 | 0.000000 | 4.833333 | 28769.942864 | 0.578747 | false |
| clipped_ste | 21001,21002,21003 | 3 | 18 | 0.689432 | 0.017624 | 0.524100 | 1.277778 | 0.000000 | 4.666667 | 29186.269206 | 0.579748 | false |
| st_gumbel | 21001,21002,21003 | 3 | 18 | 0.696783 | 0.017227 | 0.520019 | 0.500000 | 0.000000 | 6.972222 | 30330.800526 | 0.432103 | false |

## Selection Notes

- Best mean held-out NMSE: `expert_choice_imitation` at `0.684392`.
- Fastest mean throughput: `st_gumbel` at `30330.800526` tokens/s.
- Lowest mean dead leaves: `expert_choice_imitation` at `0.277778`.
