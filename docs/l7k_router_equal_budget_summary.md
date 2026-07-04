# L7K Router Equal-Budget Sweep Summary

- Run id: `l7k_router_equal_budget_20260704_a1e311f_waves0-3`
- Collected root: `outputs/scheduler_collected/l7k_router_equal_budget_20260704_a1e311f_waves0-3`
- Git commit(s): `a1e311fcf924d5a18b8b503605d3ae12509cc2c8, fbe1b3dd28945ade456d36e865d0f80e0998ea22`
- Machine/GPU slots: `foureyes:0, foureyes:1, foureyes:2, foureyes:3, ripper:0, ripper:1, ripper:2, ripper:3`
- Rows: `126` layer records from `21` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

The per-router family aggregate is in `docs/l7k_router_equal_budget_family_summary.md` and `docs/l7k_router_equal_budget_families.csv`. The commit list is mixed because some artifacts were collected after a docs-only commit; run contexts preserve each launch-time commit.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expert_choice_imitation_seed21002 | expert_choice_imitation | split_minleaf | split_routing_output | 5 | 6 | 0.683196 | 0.526824 | 0.500000 | 6.833333 | 25044.362773 | false |
| expert_choice_imitation_seed21001 | expert_choice_imitation | split_minleaf | split_routing_output | 5 | 6 | 0.683320 | 0.526229 | 0.166667 | 6.750000 | 26610.685337 | false |
| utility_targeted_ste_seed21002 | utility_targeted_ste | split_minleaf | split_routing_output | 5 | 6 | 0.686205 | 0.527445 | 1.166667 | 5.916667 | 23178.320999 | false |
| hard_em_utility_ste_seed21002 | hard_em_utility_ste | split_minleaf | split_routing_output | 5 | 6 | 0.686251 | 0.527400 | 1.833333 | 5.750000 | 24675.620287 | false |
| expert_choice_imitation_seed21003 | expert_choice_imitation | split_minleaf | split_routing_output | 5 | 6 | 0.686660 | 0.524222 | 0.166667 | 6.750000 | 26206.933568 | false |
| vanilla_ste_seed21001 | vanilla_ste | split_minleaf | split_routing_output | 5 | 6 | 0.686957 | 0.525264 | 0.833333 | 4.416667 | 30209.346904 | false |
| sigmoid_surrogate_ste_seed21001 | sigmoid_surrogate_ste | split_minleaf | split_routing_output | 5 | 6 | 0.687289 | 0.524924 | 1.333333 | 4.583333 | 28356.498254 | false |
| clipped_ste_seed21001 | clipped_ste | split_minleaf | split_routing_output | 5 | 6 | 0.687526 | 0.524700 | 1.000000 | 4.666667 | 29944.214920 | false |
| utility_targeted_ste_seed21001 | utility_targeted_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688420 | 0.526203 | 1.333333 | 5.833333 | 23920.273595 | false |
| hard_em_utility_ste_seed21001 | hard_em_utility_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688468 | 0.526146 | 1.833333 | 5.916667 | 24022.955428 | false |
| vanilla_ste_seed21002 | vanilla_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688546 | 0.525296 | 2.000000 | 4.750000 | 29594.615527 | false |
| sigmoid_surrogate_ste_seed21002 | sigmoid_surrogate_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688614 | 0.525029 | 2.166667 | 4.583333 | 29114.498072 | false |
| clipped_ste_seed21002 | clipped_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688964 | 0.524936 | 1.833333 | 4.500000 | 28870.200094 | false |
| utility_targeted_ste_seed21003 | utility_targeted_ste | split_minleaf | split_routing_output | 5 | 6 | 0.691037 | 0.523443 | 1.500000 | 5.416667 | 23377.908098 | false |
| hard_em_utility_ste_seed21003 | hard_em_utility_ste | split_minleaf | split_routing_output | 5 | 6 | 0.691037 | 0.523443 | 1.500000 | 5.416667 | 23729.913590 | false |
| vanilla_ste_seed21003 | vanilla_ste | split_minleaf | split_routing_output | 5 | 6 | 0.691267 | 0.522936 | 0.666667 | 4.750000 | 29034.376775 | false |
| sigmoid_surrogate_ste_seed21003 | sigmoid_surrogate_ste | split_minleaf | split_routing_output | 5 | 6 | 0.691486 | 0.522754 | 1.000000 | 5.333333 | 28838.832267 | false |
| clipped_ste_seed21003 | clipped_ste | split_minleaf | split_routing_output | 5 | 6 | 0.691806 | 0.522664 | 1.000000 | 4.833333 | 28744.392606 | false |
| st_gumbel_seed21003 | st_gumbel | split_minleaf | split_routing_output | 5 | 6 | 0.695468 | 0.520506 | 0.666667 | 7.083333 | 31221.783974 | false |
| st_gumbel_seed21002 | st_gumbel | split_minleaf | split_routing_output | 5 | 6 | 0.696952 | 0.520259 | 0.500000 | 6.833333 | 31624.796475 | false |
| st_gumbel_seed21001 | st_gumbel | split_minleaf | split_routing_output | 5 | 6 | 0.697929 | 0.519293 | 0.333333 | 7.000000 | 28145.821131 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| expert_choice_imitation_seed21001 | expert_choice_imitation | split_minleaf | 0.683320 | 0.166667 | 6.750000 | 26610.685337 |
| expert_choice_imitation_seed21003 | expert_choice_imitation | split_minleaf | 0.686660 | 0.166667 | 6.750000 | 26206.933568 |
| st_gumbel_seed21001 | st_gumbel | split_minleaf | 0.697929 | 0.333333 | 7.000000 | 28145.821131 |
| expert_choice_imitation_seed21002 | expert_choice_imitation | split_minleaf | 0.683196 | 0.500000 | 6.833333 | 25044.362773 |
| st_gumbel_seed21002 | st_gumbel | split_minleaf | 0.696952 | 0.500000 | 6.833333 | 31624.796475 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | expert_choice_imitation_seed21002 | 0.683196 | 0.500000 | 25044.362773 | lowest mean held-out NMSE in this run |
| balanced candidate | expert_choice_imitation_seed21001 | 0.683320 | 0.166667 | 26610.685337 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | st_gumbel_seed21002 | 0.696952 | 0.500000 | 31624.796475 | highest measured layer-distillation tokens/s |
| lowest dead leaves | expert_choice_imitation_seed21001 | 0.683320 | 0.166667 | 26610.685337 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| st_gumbel_seed21001 | blocks.11.forward_block.mixer.mixer.out_proj | 0.717901 | 0.480855 | 0 | 7.500000 | 29075.181796 |
| st_gumbel_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.716298 | 0.484826 | 0 | 7.000000 | 32852.574575 |
| st_gumbel_seed21002 | blocks.11.forward_block.mixer.mixer.out_proj | 0.713743 | 0.486247 | 1 | 7.000000 | 34863.123564 |
| sigmoid_surrogate_ste_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.712671 | 0.487909 | 0 | 6.000000 | 30092.425924 |
| clipped_ste_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.712495 | 0.488147 | 1 | 5.000000 | 28757.791643 |
| vanilla_ste_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.711556 | 0.488242 | 1 | 5.000000 | 29861.472930 |
| utility_targeted_ste_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.708587 | 0.491853 | 3 | 5.000000 | 23421.209694 |
| hard_em_utility_ste_seed21003 | blocks.11.forward_block.mixer.mixer.out_proj | 0.708587 | 0.491853 | 3 | 5.000000 | 26041.545763 |
| hard_em_utility_ste_seed21001 | blocks.11.forward_block.mixer.mixer.out_proj | 0.708088 | 0.491433 | 2 | 5.000000 | 26331.057806 |
| vanilla_ste_seed21001 | blocks.11.forward_block.mixer.mixer.out_proj | 0.707812 | 0.490202 | 0 | 4.500000 | 30964.387469 |
| clipped_ste_seed21001 | blocks.11.forward_block.mixer.mixer.out_proj | 0.707670 | 0.489416 | 0 | 6.000000 | 30662.255908 |
| utility_targeted_ste_seed21001 | blocks.11.forward_block.mixer.mixer.out_proj | 0.707644 | 0.491581 | 1 | 4.500000 | 23799.770995 |

## Interpretation

- The previous strict-config failure is resolved: every relaunched case reached `succeeded` and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
