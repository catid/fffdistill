# FFF-DZA routing_only_full_student_552dc50_wave2_offsets9_11

- Run id: `route_output_routing_only_full_student_552dc50_wave2_offsets9_11`
- Collected root: `outputs/scheduler_collected/route_output_routing_only_full_student_552dc50_wave2_offsets9_11`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:0, ai:1, ripper_idle:1`
- Rows: `15` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | routing_only | 5 | 15 | 0.222217 | 0.811527 | 12.200000 | 1.566667 | 52151.717778 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards |  |  | 0.222217 | 12.200000 | 1.566667 | 52151.717778 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_routing_only_full_student_shards | 0.222217 | 12.200000 | 52151.717778 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_routing_only_full_student_shards | 0.222217 | 12.200000 | 52151.717778 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_routing_only_full_student_shards | 0.222217 | 12.200000 | 52151.717778 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_routing_only_full_student_shards | 0.222217 | 12.200000 | 52151.717778 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_routing_only_full_student_shards | blocks.12.reverse_block.mixer.mixer.out_proj | 0.599715 | 0.506541 | 2 | 5.000000 | 52123.215992 |
| fff_route_output_routing_only_full_student_shards | blocks.12.forward_block.mixer.mixer.out_proj | 0.563177 | 0.522462 | 1 | 3.000000 | 45424.335044 |
| fff_route_output_routing_only_full_student_shards | blocks.13.reverse_block.mixer.mixer.out_proj | 0.301499 | 0.615596 | 6 | 3.000000 | 64417.847012 |
| fff_route_output_routing_only_full_student_shards | blocks.13.forward_block.mixer.mixer.out_proj | 0.272326 | 0.613427 | 6 | 3.000000 | 63509.487646 |
| fff_route_output_routing_only_full_student_shards | blocks.14.forward_block.mixer.mixer.in_proj | 0.234328 | 0.927975 | 17 | 0.000000 | 64478.347122 |
| fff_route_output_routing_only_full_student_shards | blocks.12.reverse_block.mixer.mixer.in_proj | 0.202137 | 0.912842 | 13 | 1.000000 | 51818.815109 |
| fff_route_output_routing_only_full_student_shards | blocks.13.reverse_block.mixer.mixer.in_proj | 0.190594 | 0.917709 | 14 | 1.000000 | 46511.846437 |
| fff_route_output_routing_only_full_student_shards | blocks.13.forward_block.mixer.mixer.in_proj | 0.189598 | 0.918514 | 12 | 2.000000 | 51667.626114 |
| fff_route_output_routing_only_full_student_shards | blocks.14.reverse_block.mixer.mixer.in_proj | 0.177759 | 0.931002 | 19 | 0.000000 | 53446.491809 |
| fff_route_output_routing_only_full_student_shards | blocks.14.reverse_block.mixer.mixer.out_proj | 0.122436 | 0.767129 | 10 | 2.000000 | 37074.813011 |
| fff_route_output_routing_only_full_student_shards | blocks.14.forward_block.mixer.mixer.out_proj | 0.104416 | 0.796921 | 10 | 1.500000 | 53800.238480 |
| fff_route_output_routing_only_full_student_shards | blocks.15.reverse_block.mixer.mixer.out_proj | 0.103055 | 0.904373 | 15 | 1.000000 | 52515.627163 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
