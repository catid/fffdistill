# Stage F Train-Eval Full Layerwise Distillation

- Run id: `distill_stage_f_train_eval_shards_20260704_abd09d5`
- Collected root: `outputs/scheduler_collected/distill_stage_f_train_eval_shards_20260704_abd09d5`
- Git commit(s): `abd09d5537753f870e247aefaed01fe8091b483d`
- Machine/GPU slots: `ai:0, ai:1, foureyes:0, foureyes:1, foureyes:2, foureyes:3, ripper:0, ripper:1, ripper:2, ripper:3, work:0, work:1`
- Rows: `64` layer records from `1` HPO cases
- Sample split(s): `train_eval` with held-out token metrics
- Trial statuses: `succeeded`
- CIFAR-10 test accessed: `false`

These are train-split activation-capture results using eval/no-augmentation transforms and held-out token metrics. They are train-split held-out layerwise recipe-selection evidence, not CIFAR-10 final-test student results.
Legacy validation-capture Stage F artifacts are separate and superseded for corrected Stage F layerwise evidence.

## Case Aggregate

| Case | Router | Balance | Role | Depth | Rows | Mean NMSE | Mean cosine | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fff_distill_stage_f_train_eval_full_layerwise_shards | vanilla_ste | none | split_routing_output | 5 | 64 | 0.292891 | 0.809442 | 8.437500 | 3.156250 | 42561.999688 | false |

## Lowest Dead-Leaf Cases

| Case | Router | Balance | Mean NMSE | Mean dead leaves | Mean p50 leaf tokens | Mean tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_distill_stage_f_train_eval_full_layerwise_shards | vanilla_ste | none | 0.292891 | 8.437500 | 3.156250 | 42561.999688 |

## Selection Guidance

| Use | Case | Mean NMSE | Mean dead leaves | Mean tokens/s | Rationale |
| --- | --- | --- | --- | --- | --- |
| best quality | fff_distill_stage_f_train_eval_full_layerwise_shards | 0.292891 | 8.437500 | 42561.999688 | lowest mean held-out NMSE in this run |
| balanced candidate | fff_distill_stage_f_train_eval_full_layerwise_shards | 0.292891 | 8.437500 | 42561.999688 | lowest dead leaves within +0.01 mean NMSE of the quality pick |
| fastest | fff_distill_stage_f_train_eval_full_layerwise_shards | 0.292891 | 8.437500 | 42561.999688 | highest measured layer-distillation tokens/s |
| lowest dead leaves | fff_distill_stage_f_train_eval_full_layerwise_shards | 0.292891 | 8.437500 | 42561.999688 | lowest mean dead leaves regardless of quality drop |

## Hardest Layer Records

| Case | Layer | NMSE | Cosine | Dead leaves | p50 leaf tokens | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.11.forward_block.mixer.mixer.out_proj | 0.702147 | 0.498227 | 0 | 5.000000 | 48078.398259 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.11.reverse_block.mixer.mixer.out_proj | 0.694521 | 0.502934 | 1 | 4.500000 | 47159.586401 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.10.forward_block.mixer.mixer.out_proj | 0.694419 | 0.533967 | 0 | 5.000000 | 43471.226623 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.10.reverse_block.mixer.mixer.out_proj | 0.688097 | 0.534483 | 2 | 3.000000 | 43709.860475 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.9.reverse_block.mixer.mixer.out_proj | 0.687188 | 0.531699 | 0 | 5.000000 | 28901.785919 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.9.forward_block.mixer.mixer.out_proj | 0.670619 | 0.538932 | 0 | 5.000000 | 39299.133723 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.8.reverse_block.mixer.mixer.out_proj | 0.655267 | 0.559206 | 5 | 4.000000 | 44429.976162 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.8.forward_block.mixer.mixer.out_proj | 0.651982 | 0.556483 | 4 | 4.000000 | 40855.906863 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.7.reverse_block.mixer.mixer.out_proj | 0.650980 | 0.568855 | 4 | 5.500000 | 39544.523949 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.6.forward_block.mixer.mixer.out_proj | 0.627133 | 0.610071 | 4 | 3.500000 | 41265.675790 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.6.reverse_block.mixer.mixer.out_proj | 0.624975 | 0.599094 | 9 | 2.000000 | 44742.603718 |
| fff_distill_stage_f_train_eval_full_layerwise_shards | blocks.7.forward_block.mixer.mixer.out_proj | 0.601250 | 0.605726 | 2 | 4.000000 | 27572.901664 |

## Interpretation

- The previous strict-config failure is resolved: every relaunched case reached `succeeded` and all records preserve `test_accessed=false`.
- The summary covers the layer and recipe records collected in this distillation-HPO run.
- Dead-leaf and occupancy metrics should be used alongside NMSE before selecting a full-student recipe.
- These results feed corrected Stage F train_eval selection and equal-budget router comparison. They do not close final Stage H because no full-student final CIFAR-10 test evaluation is included here.
