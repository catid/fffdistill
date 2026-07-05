# FFF-DZA shared_all_full_student_552dc50_wave0_offsets0_1

- Run id: `route_output_shared_all_full_student_552dc50_wave0_offsets0_1`
- Collected root: `outputs/scheduler_collected/route_output_shared_all_full_student_552dc50_wave0_offsets0_1`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ripper_idle:2, ripper_idle:3`
- Rows: `12` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards |  |  | shared_routing_and_output | 5 | 12 | 0.157506 | 0.918128 | 16.000000 | 1.041667 | 42423.452456 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards |  |  | 0.157506 | 16.000000 | 1.041667 | 42423.452456 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_shared_all_full_student_shards | 0.157506 | 16.000000 | 42423.452456 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_shared_all_full_student_shards | 0.157506 | 16.000000 | 42423.452456 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_shared_all_full_student_shards | 0.157506 | 16.000000 | 42423.452456 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_shared_all_full_student_shards | 0.157506 | 16.000000 | 42423.452456 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards | blocks.2.forward_block.mixer.mixer.out_proj | 0.407494 | 0.765341 | 11 | 2.500000 | 42696.854053 |
| fff_route_output_shared_all_full_student_shards | blocks.2.reverse_block.mixer.mixer.out_proj | 0.321950 | 0.820980 | 9 | 3.000000 | 48524.526843 |
| fff_route_output_shared_all_full_student_shards | blocks.1.reverse_block.mixer.mixer.out_proj | 0.310575 | 0.840852 | 14 | 1.500000 | 42005.913369 |
| fff_route_output_shared_all_full_student_shards | blocks.1.forward_block.mixer.mixer.out_proj | 0.288753 | 0.848295 | 12 | 3.500000 | 48619.076547 |
| fff_route_output_shared_all_full_student_shards | blocks.0.reverse_block.mixer.mixer.out_proj | 0.181342 | 0.917645 | 18 | 0.000000 | 45741.560917 |
| fff_route_output_shared_all_full_student_shards | blocks.0.forward_block.mixer.mixer.out_proj | 0.134688 | 0.937619 | 18 | 0.000000 | 35127.263855 |
| fff_route_output_shared_all_full_student_shards | blocks.2.reverse_block.mixer.mixer.in_proj | 0.061119 | 0.973710 | 17 | 0.000000 | 45781.451682 |
| fff_route_output_shared_all_full_student_shards | blocks.2.forward_block.mixer.mixer.in_proj | 0.057603 | 0.971913 | 14 | 2.000000 | 44503.025724 |
| fff_route_output_shared_all_full_student_shards | blocks.1.reverse_block.mixer.mixer.in_proj | 0.041796 | 0.980564 | 21 | 0.000000 | 29402.016167 |
| fff_route_output_shared_all_full_student_shards | blocks.1.forward_block.mixer.mixer.in_proj | 0.037217 | 0.982333 | 19 | 0.000000 | 46610.253985 |
| fff_route_output_shared_all_full_student_shards | blocks.0.forward_block.mixer.mixer.in_proj | 0.028160 | 0.987494 | 19 | 0.000000 | 29981.824315 |
| fff_route_output_shared_all_full_student_shards | blocks.0.reverse_block.mixer.mixer.in_proj | 0.019371 | 0.990792 | 20 | 0.000000 | 50087.662018 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
