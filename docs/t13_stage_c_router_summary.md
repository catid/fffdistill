# T13 Stage C Router Smoke Summary

- Run id: `distill_stage_c_router_offsets_20260704_090349`
- Git commit: `e200050983ee64756647b8c69856b3fc24f5cd21`
- Teacher checkpoint: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Data access: CIFAR-10 train split only; every summary reports `test_accessed=false`.
- Scope: fixed FFF architecture, one representative Linear layer, two train sample batches, 32,768 captured tokens, 100 distillation steps, LocoProp-S off.
- Invalid prior wave `distill_stage_c_router_20260704_085938` ran all jobs at grid index 0 (`vanilla_ste`) and is bug evidence only, not comparison evidence.

## Result

- Successful trials: `7/7`
- Router recipes covered: `vanilla_ste`, `clipped_ste`, `sigmoid_surrogate_ste`, `st_gumbel`, `utility_targeted_ste`, `hard_em_utility_ste`, `expert_choice_imitation`
- Best smoke MSE: `vanilla_ste` with final normalized MSE `0.072189` and cosine `0.966050`.

| Offset | Slot | Router | Final NMSE | Cosine | Tokens/s | Dead leaves | p50 leaf tokens | p90 leaf tokens |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `work:0` | `vanilla_ste` | 0.072189 | 0.966050 | 29978.8 | 18 | 0.00 | 23.30 |
| 1 | `work:1` | `clipped_ste` | 0.075208 | 0.964121 | 28997.3 | 21 | 0.00 | 7.70 |
| 2 | `ripper:0` | `sigmoid_surrogate_ste` | 0.074365 | 0.964792 | 28217.2 | 16 | 0.50 | 18.10 |
| 3 | `ripper:2` | `st_gumbel` | 0.077582 | 0.962831 | 29926.5 | 25 | 0.00 | 16.00 |
| 4 | `ripper:3` | `utility_targeted_ste` | 0.095430 | 0.954309 | 25461.2 | 30 | 0.00 | 0.00 |
| 5 | `foureyes:1` | `hard_em_utility_ste` | 0.095430 | 0.954308 | 25308.2 | 30 | 0.00 | 0.00 |
| 6 | `foureyes:2` | `expert_choice_imitation` | 0.282055 | 0.887177 | 25572.1 | 23 | 0.00 | 20.40 |

## Notes

- This is Stage C smoke evidence, not validation selection or final CIFAR-10 testing.
- `vanilla_ste` was strongest in this short fixed-architecture run; several methods are very close and need Stage D/E sweeps before selection.
- Dead leaves remain common at this short budget, so balance/depth/shared-row sweeps remain necessary.
