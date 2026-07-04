# T13 Stage D Architecture Sweep Summary

- Run id: `distill_stage_d_arch_20260704_090730`
- Git commit: `9139930965dc15ea9f05f81c9ff70f7aba3e3dcf`
- Teacher checkpoint: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Data access: CIFAR-10 validation split sampling only; every summary reports `test_accessed=false`.
- Scope: one representative Linear layer, 10 one-GPU random architecture trials, 65,536 captured-token cap, two validation sample batches, 150 distillation steps.
- Router family restricted to Stage C leaders: `vanilla_ste`, `clipped_ste`, `sigmoid_surrogate_ste`, `st_gumbel`.

## Result

- Successful trials: `10/10`
- Best validation-split local MSE: `0.004615` from `vanilla_ste` with role `split_routing_output`, depth `5`, shared fraction `0.2`, route rows `1`, leaf rows `4`, LocoProp `every_500`.
- Best cosine similarity: `0.997880`; throughput `44221.9` tokens/s; active rows/token `345.0`; stored rows `433`.

| Rank | Slot | Router | Arch | Balance | LocoProp | Final NMSE | Cosine | Tokens/s | Dead leaves |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `foureyes:2` | `vanilla_ste` | `d=5, shared=0.2, route=1, role=split_routing_output, route_out=all, leaf=4` | `none@0.001` | `every_500 (succeeded)` | 0.004615 | 0.997880 | 44221.9 | 14 |
| 2 | `work:1` | `sigmoid_surrogate_ste` | `d=7, shared=0.1, route=2, role=routing_only, route_out=0, leaf=2` | `none@0.001` | `every_500 (succeeded)` | 0.011141 | 0.994962 | 20161.3 | 118 |
| 3 | `ripper:2` | `st_gumbel` | `d=3, shared=0.1, route=2, role=shared_routing_and_output, route_out=all, leaf=2` | `none@0.01` | `every_500 (succeeded)` | 0.012248 | 0.994460 | 51741.5 | 4 |
| 4 | `ripper:0` | `clipped_ste` | `d=5, shared=0.1, route=1, role=shared_routing_and_output, route_out=all, leaf=1` | `none@0.01` | `every_500 (succeeded)` | 0.012347 | 0.994373 | 37200.8 | 29 |
| 5 | `foureyes:1` | `st_gumbel` | `d=7, shared=0.2, route=2, role=shared_routing_and_output, route_out=1, leaf=1` | `none@0.001` | `False (skipped)` | 0.019197 | 0.990924 | 20878.2 | 115 |
| 6 | `work:0` | `sigmoid_surrogate_ste` | `d=5, shared=0.05, route=2, role=split_routing_output, route_out=1, leaf=2` | `split_minleaf_margin@0.003` | `every_500 (succeeded)` | 0.020584 | 0.990688 | 30585.6 | 28 |
| 7 | `foureyes:3` | `sigmoid_surrogate_ste` | `d=5, shared=0.05, route=1, role=split_routing_output, route_out=all, leaf=1` | `split_minleaf_margin@0.001` | `every_500 (succeeded)` | 0.021530 | 0.990224 | 26896.0 | 26 |
| 8 | `ai:0` | `st_gumbel` | `d=4, shared=0.05, route=1, role=split_routing_output, route_out=1, leaf=2` | `split_minleaf_margin@0.0` | `every_500 (succeeded)` | 0.025693 | 0.988231 | 54333.8 | 13 |
| 9 | `ai:1` | `st_gumbel` | `d=4, shared=0.05, route=1, role=routing_only, route_out=0, leaf=2` | `split_minleaf_margin@0.003` | `every_500 (succeeded)` | 0.026493 | 0.987816 | 45113.8 | 14 |
| 10 | `ripper:3` | `vanilla_ste` | `d=4, shared=0.1, route=2, role=shared_routing_and_output, route_out=1, leaf=1` | `split_minleaf@0.001` | `False (skipped)` | 0.038516 | 0.981926 | 36428.3 | 8 |

## Notes

- This is a validation-split local Linear-distillation sweep, not end-to-end student validation and not CIFAR-10 test evaluation.
- Route-row output contribution variants appeared in the top result, supporting further Stage E/F sweeps with split route output and LocoProp enabled.
- Several low-MSE recipes still have dead leaves; occupancy remains a selection constraint, not just MSE.
