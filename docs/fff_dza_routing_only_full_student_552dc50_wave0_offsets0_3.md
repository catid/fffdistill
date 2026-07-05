# FFF-DZA routing_only_full_student_552dc50_wave0_offsets0_3

- Run id: `route_output_routing_only_full_student_552dc50_wave0_offsets0_3`
- Collected root: `outputs/scheduler_collected/route_output_routing_only_full_student_552dc50_wave0_offsets0_3`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:1, ripper_idle:1, ripper_idle:2, ripper_idle:3`
- Rows: `24` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | routing_only | 5 | 24 | 0.243694 | 0.864217 | 17.375000 | 0.687500 | 50478.011376 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | 0.243694 | 17.375000 | 0.687500 | 50478.011376 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_routing_only_full_student_shards | 0.243694 | 17.375000 | 50478.011376 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_routing_only_full_student_shards | 0.243694 | 17.375000 | 50478.011376 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_routing_only_full_student_shards | 0.243694 | 17.375000 | 50478.011376 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_routing_only_full_student_shards | 0.243694 | 17.375000 | 50478.011376 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards | blocks.5.reverse_block.mixer.mixer.out_proj | 0.603728 | 0.611164 | 8 | 3.000000 | 51283.337270 |
| fff_route_output_routing_only_full_student_shards | blocks.5.forward_block.mixer.mixer.out_proj | 0.573198 | 0.650005 | 11 | 1.000000 | 51817.370391 |
| fff_route_output_routing_only_full_student_shards | blocks.4.reverse_block.mixer.mixer.out_proj | 0.538084 | 0.685761 | 20 | 0.000000 | 47902.880025 |
| fff_route_output_routing_only_full_student_shards | blocks.4.forward_block.mixer.mixer.out_proj | 0.533789 | 0.665403 | 17 | 0.000000 | 46200.435239 |
| fff_route_output_routing_only_full_student_shards | blocks.2.forward_block.mixer.mixer.out_proj | 0.472523 | 0.743623 | 25 | 0.000000 | 49842.006873 |
| fff_route_output_routing_only_full_student_shards | blocks.3.reverse_block.mixer.mixer.out_proj | 0.451683 | 0.743568 | 20 | 0.000000 | 45848.596722 |
| fff_route_output_routing_only_full_student_shards | blocks.3.forward_block.mixer.mixer.out_proj | 0.419485 | 0.731880 | 18 | 0.000000 | 57334.776349 |
| fff_route_output_routing_only_full_student_shards | blocks.1.forward_block.mixer.mixer.out_proj | 0.369523 | 0.834659 | 22 | 0.000000 | 56673.317940 |
| fff_route_output_routing_only_full_student_shards | blocks.2.reverse_block.mixer.mixer.out_proj | 0.355925 | 0.802852 | 20 | 0.000000 | 45385.285671 |
| fff_route_output_routing_only_full_student_shards | blocks.1.reverse_block.mixer.mixer.out_proj | 0.348910 | 0.819614 | 24 | 0.000000 | 52310.219118 |
| fff_route_output_routing_only_full_student_shards | blocks.0.reverse_block.mixer.mixer.out_proj | 0.185377 | 0.916021 | 18 | 0.000000 | 56679.840285 |
| fff_route_output_routing_only_full_student_shards | blocks.0.forward_block.mixer.mixer.out_proj | 0.156819 | 0.930075 | 24 | 0.000000 | 56612.685159 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
