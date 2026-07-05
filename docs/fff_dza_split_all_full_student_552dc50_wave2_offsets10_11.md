# FFF-DZA split_all_full_student_552dc50_wave2_offsets10_11

- Run id: `route_output_split_all_full_student_552dc50_wave2_offsets10_11`
- Collected root: `outputs/scheduler_collected/route_output_split_all_full_student_552dc50_wave2_offsets10_11`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:0, ai:1`
- Rows: `10` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards |  |  | split_routing_output | 5 | 10 | 0.135849 | 0.877994 | 12.500000 | 1.100000 | 49495.204991 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards |  |  | 0.135849 | 12.500000 | 1.100000 | 49495.204991 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_split_all_full_student_shards | 0.135849 | 12.500000 | 49495.204991 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_split_all_full_student_shards | 0.135849 | 12.500000 | 49495.204991 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_split_all_full_student_shards | 0.135849 | 12.500000 | 49495.204991 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_split_all_full_student_shards | 0.135849 | 12.500000 | 49495.204991 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards | blocks.13.reverse_block.mixer.mixer.out_proj | 0.280810 | 0.631812 | 4 | 3.500000 | 46145.126609 |
| fff_route_output_split_all_full_student_shards | blocks.14.forward_block.mixer.mixer.in_proj | 0.232970 | 0.931611 | 15 | 1.000000 | 45823.791979 |
| fff_route_output_split_all_full_student_shards | blocks.14.reverse_block.mixer.mixer.in_proj | 0.191598 | 0.932383 | 13 | 1.000000 | 47892.356183 |
| fff_route_output_split_all_full_student_shards | blocks.13.reverse_block.mixer.mixer.in_proj | 0.181915 | 0.920465 | 13 | 1.000000 | 36393.217535 |
| fff_route_output_split_all_full_student_shards | blocks.14.reverse_block.mixer.mixer.out_proj | 0.092763 | 0.784669 | 11 | 1.500000 | 39950.255314 |
| fff_route_output_split_all_full_student_shards | blocks.15.forward_block.mixer.mixer.in_proj | 0.083826 | 0.967623 | 21 | 0.000000 | 57926.581026 |
| fff_route_output_split_all_full_student_shards | blocks.14.forward_block.mixer.mixer.out_proj | 0.081427 | 0.809099 | 7 | 1.000000 | 46357.650430 |
| fff_route_output_split_all_full_student_shards | blocks.15.reverse_block.mixer.mixer.in_proj | 0.074834 | 0.965715 | 18 | 0.000000 | 57993.283338 |
| fff_route_output_split_all_full_student_shards | blocks.15.forward_block.mixer.mixer.out_proj | 0.071485 | 0.919205 | 12 | 1.000000 | 58800.770861 |
| fff_route_output_split_all_full_student_shards | blocks.15.reverse_block.mixer.mixer.out_proj | 0.066859 | 0.917353 | 11 | 1.000000 | 57669.016639 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
