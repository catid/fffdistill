# FFF-DZA routing_only_full_student_552dc50_wave1_offsets4_8

- Run id: `route_output_routing_only_full_student_552dc50_wave1_offsets4_8`
- Collected root: `outputs/scheduler_collected/route_output_routing_only_full_student_552dc50_wave1_offsets4_8`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:0, ai:1, ripper_idle:1, ripper_idle:2, ripper_idle:3`
- Rows: `25` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | routing_only | 5 | 25 | 0.416981 | 0.736643 | 8.640000 | 2.300000 | 50431.488210 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | 0.416981 | 8.640000 | 2.300000 | 50431.488210 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_routing_only_full_student_shards | 0.416981 | 8.640000 | 50431.488210 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_routing_only_full_student_shards | 0.416981 | 8.640000 | 50431.488210 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_routing_only_full_student_shards | 0.416981 | 8.640000 | 50431.488210 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_routing_only_full_student_shards | 0.416981 | 8.640000 | 50431.488210 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards | blocks.11.reverse_block.mixer.mixer.out_proj | 0.719108 | 0.483276 | 1 | 5.000000 | 48960.113581 |
| fff_route_output_routing_only_full_student_shards | blocks.11.forward_block.mixer.mixer.out_proj | 0.715054 | 0.479503 | 4 | 4.000000 | 50684.861370 |
| fff_route_output_routing_only_full_student_shards | blocks.10.forward_block.mixer.mixer.out_proj | 0.711034 | 0.517226 | 0 | 4.000000 | 46523.495794 |
| fff_route_output_routing_only_full_student_shards | blocks.9.reverse_block.mixer.mixer.out_proj | 0.707092 | 0.512343 | 2 | 4.000000 | 39290.560673 |
| fff_route_output_routing_only_full_student_shards | blocks.10.reverse_block.mixer.mixer.out_proj | 0.697918 | 0.526470 | 5 | 4.000000 | 48303.996858 |
| fff_route_output_routing_only_full_student_shards | blocks.9.forward_block.mixer.mixer.out_proj | 0.697538 | 0.518572 | 3 | 4.000000 | 52365.778053 |
| fff_route_output_routing_only_full_student_shards | blocks.8.forward_block.mixer.mixer.out_proj | 0.672884 | 0.541204 | 1 | 3.000000 | 52684.225475 |
| fff_route_output_routing_only_full_student_shards | blocks.8.reverse_block.mixer.mixer.out_proj | 0.669367 | 0.547904 | 6 | 2.000000 | 47381.150002 |
| fff_route_output_routing_only_full_student_shards | blocks.7.reverse_block.mixer.mixer.out_proj | 0.667610 | 0.558190 | 10 | 1.500000 | 52522.384436 |
| fff_route_output_routing_only_full_student_shards | blocks.6.forward_block.mixer.mixer.out_proj | 0.646020 | 0.594459 | 11 | 2.000000 | 63789.963889 |
| fff_route_output_routing_only_full_student_shards | blocks.6.reverse_block.mixer.mixer.out_proj | 0.642358 | 0.579548 | 11 | 1.500000 | 59213.324802 |
| fff_route_output_routing_only_full_student_shards | blocks.7.forward_block.mixer.mixer.out_proj | 0.613286 | 0.591146 | 8 | 2.500000 | 43871.278379 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
