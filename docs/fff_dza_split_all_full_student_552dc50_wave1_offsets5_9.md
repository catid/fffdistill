# FFF-DZA split_all_full_student_552dc50_wave1_offsets5_9

- Run id: `route_output_split_all_full_student_552dc50_wave1_offsets5_9`
- Collected root: `outputs/scheduler_collected/route_output_split_all_full_student_552dc50_wave1_offsets5_9`
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
| fff_route_output_split_all_full_student_shards |  |  | split_routing_output | 5 | 25 | 0.399918 | 0.734865 | 6.360000 | 3.020000 | 39381.919761 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards |  |  | 0.399918 | 6.360000 | 3.020000 | 39381.919761 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_route_output_split_all_full_student_shards | 0.399918 | 6.360000 | 39381.919761 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_route_output_split_all_full_student_shards | 0.399918 | 6.360000 | 39381.919761 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_route_output_split_all_full_student_shards | 0.399918 | 6.360000 | 39381.919761 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_route_output_split_all_full_student_shards | 0.399918 | 6.360000 | 39381.919761 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_route_output_split_all_full_student_shards | blocks.11.forward_block.mixer.mixer.out_proj | 0.686633 | 0.515208 | 2 | 5.000000 | 45117.471797 |
| fff_route_output_split_all_full_student_shards | blocks.11.reverse_block.mixer.mixer.out_proj | 0.676954 | 0.517324 | 0 | 5.500000 | 47407.909229 |
| fff_route_output_split_all_full_student_shards | blocks.10.forward_block.mixer.mixer.out_proj | 0.674644 | 0.548991 | 2 | 5.000000 | 20311.236149 |
| fff_route_output_split_all_full_student_shards | blocks.9.reverse_block.mixer.mixer.out_proj | 0.671693 | 0.548730 | 2 | 4.000000 | 17030.243585 |
| fff_route_output_split_all_full_student_shards | blocks.10.reverse_block.mixer.mixer.out_proj | 0.665708 | 0.553818 | 4 | 3.000000 | 20109.589976 |
| fff_route_output_split_all_full_student_shards | blocks.9.forward_block.mixer.mixer.out_proj | 0.654441 | 0.553485 | 4 | 4.000000 | 48553.547710 |
| fff_route_output_split_all_full_student_shards | blocks.8.reverse_block.mixer.mixer.out_proj | 0.641421 | 0.575535 | 5 | 3.000000 | 49383.971985 |
| fff_route_output_split_all_full_student_shards | blocks.7.reverse_block.mixer.mixer.out_proj | 0.631983 | 0.587946 | 5 | 2.000000 | 46570.503379 |
| fff_route_output_split_all_full_student_shards | blocks.8.forward_block.mixer.mixer.out_proj | 0.631632 | 0.573057 | 3 | 2.000000 | 48017.282738 |
| fff_route_output_split_all_full_student_shards | blocks.7.forward_block.mixer.mixer.out_proj | 0.577209 | 0.623051 | 1 | 2.000000 | 35596.678814 |
| fff_route_output_split_all_full_student_shards | blocks.12.reverse_block.mixer.mixer.out_proj | 0.559399 | 0.544197 | 4 | 3.000000 | 45858.602244 |
| fff_route_output_split_all_full_student_shards | blocks.12.forward_block.mixer.mixer.out_proj | 0.533276 | 0.558620 | 4 | 4.000000 | 30470.284247 |

## Interpretation

- All summarized records passed strict status checks: scheduler and trial status are `succeeded`, and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
