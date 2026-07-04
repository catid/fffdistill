# T13 Distillation HPO Smoke Summary

- Run id: `distill_hpo_train_smoke_20260704_084927`
- Git commit: `468dfb35ff7b690f22f703ed7fdbb3ff4949cd8e`
- Teacher checkpoint: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Teacher selected validation accuracy: `0.9418`; final teacher test accuracy was evaluated only at T06 selection and not reused here.
- Scheduler command: `python -m cifar_mamba_fff.gpu_scheduler --job-kind distill_hpo --quick-smoke false --dry-run false --allow-long-jobs true --distill-sample-split train --distill-max-sample-batches 1 --hpo-trials-per-job 1`
- Slots launched: 10 (`work:0,1`, `ripper:0,2,3`, `foureyes:1,2,3`, `ai:0,1`). Excluded anomaly slots: `ripper:1`, `foureyes:0`.
- Data access: CIFAR-10 train split only; every `distill_hpo_summary.json` reports `test_accessed=false`.
- Scope: one sampled HPO trial per slot, one eligible layer, one sample batch, 8,192 captured tokens, 20 distillation steps from `configs/fff_distill_smoke.yaml`.

## Aggregate Results

- Successful trials: `10/10`
- Final normalized MSE: min `0.087255`, median `0.124794`, max `0.312870`
- Initial normalized MSE range: `1.001518` to `1.037146`
- LocoProp-S actual MSE before/after: median `0.099567` -> `0.025336`
- Route-row role flags sampled: `routing_only`, `shared_routing_and_output`, `split_routing_output`.
- Raw artifacts remain under ignored `outputs/`; compact CSV is committed as `docs/t13_distill_hpo_smoke_summary.csv`.

## Per-Slot Results

| Slot | Seed | Router | FFF rows | Balance | LocoProp | Initial NMSE | Final NMSE | Dead leaves | Test accessed |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `ai:0` | 1347 | `sigmoid_surrogate_ste` | `d=5, shared=0.2, route=1, route_out=1, role=split_routing_output, leaf=1` | `split@0.01` | `every_250 (succeeded)` | 1.035708 | 0.087534 | 27 | `false` |
| `ai:1` | 1348 | `sigmoid_surrogate_ste` | `d=5, shared=0.2, route=1, route_out=0, role=routing_only, leaf=1` | `split@0.003` | `every_500 (succeeded)` | 1.036095 | 0.089175 | 29 | `false` |
| `foureyes:1` | 1344 | `clipped_ste` | `d=3, shared=0.2, route=2, route_out=0, role=shared_routing_and_output, leaf=2` | `split@0.01` | `every_100 (succeeded)` | 1.036560 | 0.087930 | 3 | `false` |
| `foureyes:2` | 1345 | `vanilla_ste` | `d=6, shared=0.0, route=2, route_out=2, role=split_routing_output, leaf=4` | `none@0.0003` | `every_100 (succeeded)` | 1.001518 | 0.233199 | 57 | `false` |
| `foureyes:3` | 1346 | `hard_em_utility_ste` | `d=8, shared=0.1, route=1, route_out=2, role=split_routing_output, leaf=4` | `split_minleaf_uniform@0.0003` | `every_100 (succeeded)` | 1.035656 | 0.124633 | 246 | `false` |
| `ripper:0` | 1339 | `sigmoid_surrogate_ste` | `d=2, shared=0.05, route=2, route_out=0, role=shared_routing_and_output, leaf=1` | `split_minleaf_uniform@0.0001` | `every_500 (succeeded)` | 1.026764 | 0.180896 | 1 | `false` |
| `ripper:2` | 1341 | `vanilla_ste` | `d=4, shared=0.2, route=2, route_out=0, role=routing_only, leaf=4` | `split_minleaf@0.01` | `every_500 (succeeded)` | 1.037146 | 0.087255 | 11 | `false` |
| `ripper:3` | 1342 | `utility_targeted_ste` | `d=8, shared=0.0, route=2, route_out=0, role=shared_routing_and_output, leaf=4` | `split_minleaf_margin@0.0` | `every_100 (succeeded)` | 1.032493 | 0.312870 | 246 | `false` |
| `work:0` | 1337 | `vanilla_ste` | `d=5, shared=0.1, route=1, route_out=2, role=split_routing_output, leaf=2` | `split@0.01` | `every_500 (succeeded)` | 1.027358 | 0.127377 | 26 | `false` |
| `work:1` | 1338 | `expert_choice_imitation` | `d=5, shared=0.1, route=2, route_out=0, role=routing_only, leaf=4` | `split_minleaf_margin@0.003` | `every_250 (succeeded)` | 1.020125 | 0.124955 | 28 | `false` |

## Notes

- This is smoke evidence only, not a validation or final result. CIFAR-10 test was not used.
- The HPO sampler exercised route-output contribution knobs, including `split_routing_output`, `shared_routing_and_output`, and `routing_only` roles.
- Several smoke configs still produced many dead leaves after 20 steps; that is expected at this smoke budget and remains a router/balance target for Stage C/D.
