# Hard Out-Projection Router/Balance Train-Eval Sweep

- Run id: `hard_outproj_router_balance_train_eval_20260704_819c2c6`
- Collected root: `outputs/scheduler_collected/hard_outproj_router_balance_train_eval_20260704_819c2c6`
- Git commit(s): `819c2c6f05c199821aedf8003617eb55ef4935db`
- Machine/GPU slots: `ai:0, ai:1, foureyes:0, foureyes:1, foureyes:2, foureyes:3, ripper:0, ripper:1, ripper:2, ripper:3, work:0, work:1`
- Rows: `72` layer records from `12` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla_depth_6 | vanilla_ste | split_minleaf | split_routing_output | 6 | 6 | 0.681258 | 0.528598 | 8.500000 | 2.666667 | 22819.160558 | false |
| expert_choice_split_minleaf | expert_choice_imitation | split_minleaf | split_routing_output | 5 | 6 | 0.685069 | 0.526112 | 0.500000 | 6.916667 | 26775.353756 | false |
| vanilla_route_rows_2 | vanilla_ste | split_minleaf | split_routing_output | 5 | 6 | 0.685918 | 0.525488 | 2.500000 | 4.166667 | 36981.380406 | false |
| utility_targeted_split_minleaf | utility_targeted_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688407 | 0.525566 | 1.166667 | 5.000000 | 24147.978313 | false |
| hard_em_split_minleaf | hard_em_utility_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688407 | 0.525566 | 1.166667 | 5.000000 | 22821.725415 | false |
| baseline_vanilla_none | vanilla_ste | none | split_routing_output | 5 | 6 | 0.688740 | 0.525026 | 1.166667 | 5.416667 | 44108.737110 | false |
| vanilla_split_minleaf | vanilla_ste | split_minleaf | split_routing_output | 5 | 6 | 0.688825 | 0.524994 | 1.000000 | 5.500000 | 30249.463032 | false |
| vanilla_split_minleaf_uniform | vanilla_ste | split_minleaf_uniform | split_routing_output | 5 | 6 | 0.689114 | 0.524859 | 1.166667 | 5.500000 | 29270.188332 | false |
| vanilla_split_minleaf_margin | vanilla_ste | split_minleaf_margin | split_routing_output | 5 | 6 | 0.689319 | 0.524787 | 1.166667 | 5.500000 | 30875.172186 | false |
| sigmoid_split_minleaf | sigmoid_surrogate_ste | split_minleaf | split_routing_output | 5 | 6 | 0.689542 | 0.524516 | 1.000000 | 5.333333 | 31783.634796 | false |
| clipped_split_minleaf | clipped_ste | split_minleaf | split_routing_output | 5 | 6 | 0.689695 | 0.524373 | 0.833333 | 5.166667 | 30340.137067 | false |
| st_gumbel_split_minleaf | st_gumbel | split_minleaf | split_routing_output | 5 | 6 | 0.697482 | 0.518969 | 0.333333 | 6.250000 | 28119.902613 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| st_gumbel_split_minleaf | st_gumbel | split_minleaf | 0.697482 | 0.333333 | 6.250000 | 28119.902613 |
| expert_choice_split_minleaf | expert_choice_imitation | split_minleaf | 0.685069 | 0.500000 | 6.916667 | 26775.353756 |
| clipped_split_minleaf | clipped_ste | split_minleaf | 0.689695 | 0.833333 | 5.166667 | 30340.137067 |
| vanilla_split_minleaf | vanilla_ste | split_minleaf | 0.688825 | 1.000000 | 5.500000 | 30249.463032 |
| sigmoid_split_minleaf | sigmoid_surrogate_ste | split_minleaf | 0.689542 | 1.000000 | 5.333333 | 31783.634796 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | vanilla_depth_6 | 0.681258 | 8.500000 | 22819.160558 | lowest mean held-out NMSE in this run |
| balanced candidate | expert_choice_split_minleaf | 0.685069 | 0.500000 | 26775.353756 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | baseline_vanilla_none | 0.688740 | 1.166667 | 44108.737110 | highest measured layer-distillation tokens/s |
| lowest dead leaves | st_gumbel_split_minleaf | 0.697482 | 0.333333 | 28119.902613 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| st_gumbel_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.719336 | 0.480661 | 0 | 6.000000 | 28983.838575 |
| clipped_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.710871 | 0.490729 | 0 | 6.000000 | 32770.192174 |
| vanilla_split_minleaf_margin | blocks.11.forward_block.mixer.mixer.out_proj | 0.710416 | 0.491380 | 0 | 7.000000 | 31460.596511 |
| sigmoid_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.709850 | 0.490813 | 0 | 6.500000 | 32459.307066 |
| vanilla_split_minleaf_uniform | blocks.11.forward_block.mixer.mixer.out_proj | 0.709414 | 0.491613 | 0 | 6.500000 | 28952.610367 |
| vanilla_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.707475 | 0.492576 | 1 | 6.000000 | 30283.457611 |
| baseline_vanilla_none | blocks.11.forward_block.mixer.mixer.out_proj | 0.707472 | 0.492553 | 1 | 6.000000 | 50429.238113 |
| expert_choice_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.705905 | 0.494894 | 0 | 6.500000 | 25657.061304 |
| utility_targeted_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.704842 | 0.495472 | 1 | 5.000000 | 26603.855795 |
| hard_em_split_minleaf | blocks.11.forward_block.mixer.mixer.out_proj | 0.704842 | 0.495472 | 1 | 5.000000 | 23947.137890 |
| st_gumbel_split_minleaf | blocks.9.reverse_block.mixer.mixer.out_proj | 0.703096 | 0.525505 | 2 | 5.000000 | 28168.522747 |
| vanilla_route_rows_2 | blocks.11.forward_block.mixer.mixer.out_proj | 0.702815 | 0.493951 | 3 | 4.500000 | 39006.219057 |

## Interpretation

- The previous strict-config failure is resolved: every relaunched case reached `succeeded` and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
