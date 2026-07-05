# FFF-DZA shared_all_full_student_552dc50_wave1_offsets2_6

- Run id: `route_output_shared_all_full_student_552dc50_wave1_offsets2_6`
- Collected root: `outputs/scheduler_collected/route_output_shared_all_full_student_552dc50_wave1_offsets2_6`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:0, ai:1, ripper_idle:1, ripper_idle:2, ripper_idle:3`
- Rows: `27` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards |  |  | shared_routing_and_output | 5 | 27 | 0.337418 | 0.796690 | 8.148148 | 3.185185 | 49600.145484 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards |  |  | 0.337418 | 8.148148 | 3.185185 | 49600.145484 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_shared_all_full_student_shards | 0.337418 | 8.148148 | 49600.145484 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_shared_all_full_student_shards | 0.337418 | 8.148148 | 49600.145484 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_shared_all_full_student_shards | 0.337418 | 8.148148 | 49600.145484 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_shared_all_full_student_shards | 0.337418 | 8.148148 | 49600.145484 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards | blocks.9.forward_block.mixer.mixer.out_proj | 0.670063 | 0.545995 | 2 | 5.500000 | 44176.780094 |
| fff_route_output_shared_all_full_student_shards | blocks.8.reverse_block.mixer.mixer.out_proj | 0.644137 | 0.569188 | 5 | 3.500000 | 46061.404127 |
| fff_route_output_shared_all_full_student_shards | blocks.7.reverse_block.mixer.mixer.out_proj | 0.635839 | 0.587810 | 4 | 3.500000 | 49799.247174 |
| fff_route_output_shared_all_full_student_shards | blocks.8.forward_block.mixer.mixer.out_proj | 0.635143 | 0.567890 | 3 | 4.000000 | 53509.013681 |
| fff_route_output_shared_all_full_student_shards | blocks.6.forward_block.mixer.mixer.out_proj | 0.610299 | 0.625715 | 8 | 3.000000 | 54033.880570 |
| fff_route_output_shared_all_full_student_shards | blocks.6.reverse_block.mixer.mixer.out_proj | 0.608537 | 0.611054 | 6 | 3.500000 | 51410.238468 |
| fff_route_output_shared_all_full_student_shards | blocks.7.forward_block.mixer.mixer.out_proj | 0.582992 | 0.618042 | 4 | 4.500000 | 33742.851251 |
| fff_route_output_shared_all_full_student_shards | blocks.5.reverse_block.mixer.mixer.out_proj | 0.571922 | 0.643342 | 5 | 2.000000 | 56522.972484 |
| fff_route_output_shared_all_full_student_shards | blocks.5.forward_block.mixer.mixer.out_proj | 0.541158 | 0.672449 | 4 | 4.500000 | 60262.870120 |
| fff_route_output_shared_all_full_student_shards | blocks.4.forward_block.mixer.mixer.out_proj | 0.497948 | 0.693581 | 8 | 1.000000 | 56390.560745 |
| fff_route_output_shared_all_full_student_shards | blocks.4.reverse_block.mixer.mixer.out_proj | 0.486430 | 0.710063 | 6 | 5.000000 | 56878.242297 |
| fff_route_output_shared_all_full_student_shards | blocks.3.reverse_block.mixer.mixer.out_proj | 0.415040 | 0.764817 | 9 | 2.500000 | 57672.459990 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
