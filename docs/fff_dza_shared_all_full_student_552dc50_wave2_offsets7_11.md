# FFF-DZA shared_all_full_student_552dc50_wave2_offsets7_11

- Run id: `route_output_shared_all_full_student_552dc50_wave2_offsets7_11`
- Collected root: `outputs/scheduler_collected/route_output_shared_all_full_student_552dc50_wave2_offsets7_11`
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
| fff_route_output_shared_all_full_student_shards |  |  | shared_routing_and_output | 5 | 25 | 0.303593 | 0.782280 | 7.360000 | 3.280000 | 45979.555287 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards |  |  | 0.303593 | 7.360000 | 3.280000 | 45979.555287 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_shared_all_full_student_shards | 0.303593 | 7.360000 | 45979.555287 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_shared_all_full_student_shards | 0.303593 | 7.360000 | 45979.555287 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_shared_all_full_student_shards | 0.303593 | 7.360000 | 45979.555287 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_shared_all_full_student_shards | 0.303593 | 7.360000 | 45979.555287 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_shared_all_full_student_shards | blocks.11.forward_block.mixer.mixer.out_proj | 0.693866 | 0.504385 | 0 | 6.000000 | 50863.070814 |
| fff_route_output_shared_all_full_student_shards | blocks.10.forward_block.mixer.mixer.out_proj | 0.684910 | 0.547205 | 2 | 6.000000 | 47819.690355 |
| fff_route_output_shared_all_full_student_shards | blocks.11.reverse_block.mixer.mixer.out_proj | 0.683843 | 0.513750 | 1 | 4.000000 | 49662.849774 |
| fff_route_output_shared_all_full_student_shards | blocks.10.reverse_block.mixer.mixer.out_proj | 0.675811 | 0.547163 | 5 | 5.500000 | 50012.850564 |
| fff_route_output_shared_all_full_student_shards | blocks.9.reverse_block.mixer.mixer.out_proj | 0.674844 | 0.545888 | 5 | 5.500000 | 37579.603768 |
| fff_route_output_shared_all_full_student_shards | blocks.12.reverse_block.mixer.mixer.out_proj | 0.564601 | 0.533500 | 2 | 4.500000 | 46791.273778 |
| fff_route_output_shared_all_full_student_shards | blocks.12.forward_block.mixer.mixer.out_proj | 0.536467 | 0.552263 | 0 | 6.000000 | 32296.606626 |
| fff_route_output_shared_all_full_student_shards | blocks.14.forward_block.mixer.mixer.in_proj | 0.298911 | 0.932260 | 12 | 3.500000 | 42689.186777 |
| fff_route_output_shared_all_full_student_shards | blocks.13.reverse_block.mixer.mixer.out_proj | 0.292311 | 0.625398 | 1 | 5.000000 | 43816.047771 |
| fff_route_output_shared_all_full_student_shards | blocks.12.reverse_block.mixer.mixer.in_proj | 0.256908 | 0.915546 | 9 | 3.000000 | 48676.112974 |
| fff_route_output_shared_all_full_student_shards | blocks.13.forward_block.mixer.mixer.out_proj | 0.252542 | 0.629221 | 3 | 5.000000 | 43408.971510 |
| fff_route_output_shared_all_full_student_shards | blocks.12.forward_block.mixer.mixer.in_proj | 0.204902 | 0.918530 | 7 | 2.000000 | 50165.906097 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
