# FFF-DZA split_all_full_student_552dc50_wave0_offsets0_4

- Run id: `route_output_split_all_full_student_552dc50_wave0_offsets0_4`
- Collected root: `outputs/scheduler_collected/route_output_split_all_full_student_552dc50_wave0_offsets0_4`
- Git commit(s): `552dc50163e3d2dbaf3700ebad7b12cd813ee3e3`
- Machine/GPU slots: `ai:0, ai:1, ripper_idle:1, ripper_idle:2, ripper_idle:3`
- Rows: `29` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are validation evidence for selecting FFF recipes, not CIFAR-10 final-test student results.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards |  |  | split_routing_output | 5 | 29 | 0.243473 | 0.865176 | 13.827586 | 1.431034 | 44914.243203 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards |  |  | 0.243473 | 13.827586 | 1.431034 | 44914.243203 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_split_all_full_student_shards | 0.243473 | 13.827586 | 44914.243203 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_split_all_full_student_shards | 0.243473 | 13.827586 | 44914.243203 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_split_all_full_student_shards | 0.243473 | 13.827586 | 44914.243203 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_split_all_full_student_shards | 0.243473 | 13.827586 | 44914.243203 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards | blocks.6.forward_block.mixer.mixer.out_proj | 0.608132 | 0.625986 | 8 | 1.000000 | 45556.409976 |
| fff_route_output_split_all_full_student_shards | blocks.6.reverse_block.mixer.mixer.out_proj | 0.603347 | 0.618017 | 4 | 2.500000 | 44768.304108 |
| fff_route_output_split_all_full_student_shards | blocks.5.reverse_block.mixer.mixer.out_proj | 0.561359 | 0.647597 | 8 | 1.000000 | 50035.927163 |
| fff_route_output_split_all_full_student_shards | blocks.5.forward_block.mixer.mixer.out_proj | 0.545806 | 0.679150 | 18 | 0.000000 | 47724.027163 |
| fff_route_output_split_all_full_student_shards | blocks.4.reverse_block.mixer.mixer.out_proj | 0.515548 | 0.710951 | 15 | 1.000000 | 45317.203269 |
| fff_route_output_split_all_full_student_shards | blocks.4.forward_block.mixer.mixer.out_proj | 0.500818 | 0.693295 | 17 | 0.000000 | 48869.081267 |
| fff_route_output_split_all_full_student_shards | blocks.2.forward_block.mixer.mixer.out_proj | 0.436108 | 0.767179 | 19 | 0.000000 | 49941.593307 |
| fff_route_output_split_all_full_student_shards | blocks.3.reverse_block.mixer.mixer.out_proj | 0.428336 | 0.769200 | 18 | 0.000000 | 45562.381385 |
| fff_route_output_split_all_full_student_shards | blocks.3.forward_block.mixer.mixer.out_proj | 0.419573 | 0.759761 | 18 | 0.000000 | 41649.505904 |
| fff_route_output_split_all_full_student_shards | blocks.2.reverse_block.mixer.mixer.out_proj | 0.326928 | 0.825167 | 18 | 0.000000 | 49891.411149 |
| fff_route_output_split_all_full_student_shards | blocks.1.reverse_block.mixer.mixer.out_proj | 0.311108 | 0.840022 | 25 | 0.000000 | 49359.124594 |
| fff_route_output_split_all_full_student_shards | blocks.1.forward_block.mixer.mixer.out_proj | 0.276298 | 0.857261 | 16 | 0.500000 | 48434.046305 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
