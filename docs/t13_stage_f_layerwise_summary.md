# T13 Stage F Layerwise Distillation Summary

- Run id: `distill_stage_f_shards_20260704_091746`
- Git commit used for jobs: `f0053c0d904cb7c552f79e4d8c8fb534a3333aa1`
- Teacher checkpoint: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Teacher parameter count: `9,053,258`
- Selected teacher validation accuracy recorded in checkpoint summary: `0.9418`
- Data access: CIFAR-10 `val` split sampling only, `2` batches per shard; all records report `test_accessed=false`.
- Scope: 64 eligible Linear layers, sharded once each across 10 one-GPU jobs on `work`, `ripper`, `foureyes`, and `ai`.
- Recipe: Stage D/E fixed architecture, `vanilla_ste`, `split_routing_output`, `shared_unrouted_frac=0.2`, `route_rows=1`, `leaf_rows=4`, `depth=5`, `route_result_rows=2`, `route_rows_output_count=all`, `route_rows_output_fraction=0.5`, LocoProp-S every 500/one post-loop refit.

## Scheduler And Coverage

- Jobs succeeded: `10/10`
- Excluded slots: `ripper:1` and `foureyes:0` were marked unavailable because pre-launch GPU checks showed anomalous persistent 100% utilization with negligible memory and no visible compute PID.
- Layer index coverage: `0..63`, `64` unique, no missing or duplicate indices.
- `layer_summary.json` contains NMSE/cosine/throughput; route diagnostics including dead leaves are emitted in `layer_metrics.jsonl` and included in the CSV report.

| Slot | Indices | Layers | Mean NMSE | Mean tokens/s | Test accessed |
| --- | --- | ---: | ---: | ---: | --- |
| `work:0` | `0-6` | 7 | 0.102635 | 45724.9 | `false` |
| `work:1` | `7-13` | 7 | 0.233179 | 39008.7 | `false` |
| `ripper:0` | `14-20` | 7 | 0.254172 | 39633.4 | `false` |
| `ripper:2` | `21-27` | 7 | 0.384136 | 39318.7 | `false` |
| `ripper:3` | `28-34` | 7 | 0.356440 | 45644.5 | `false` |
| `foureyes:1` | `35-41` | 7 | 0.460150 | 42408.7 | `false` |
| `foureyes:2` | `42-48` | 7 | 0.395042 | 44065.5 | `false` |
| `foureyes:3` | `49-55` | 7 | 0.322241 | 40924.0 | `false` |
| `ai:0` | `56-60` | 5 | 0.128602 | 50215.0 | `false` |
| `ai:1` | `61-63` | 3 | 0.092762 | 42667.5 | `false` |

## Aggregate Metrics

| Metric | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Final normalized MSE | 0.288707 | 0.188059 | 0.021491 | 0.683897 |
| Final cosine similarity | 0.807726 | 0.915426 | 0.490996 | 0.989873 |
| Tokens/s | 42752.8 | 42512.1 | 26958.6 | 56548.1 |
| Dead leaves | 16.48 | 16.00 | 3 | 27 |
| LocoProp-S MSE before | 4.549086 | 0.051761 | 0.008060 | 171.223694 |
| LocoProp-S MSE after | 1.289520 | 0.024386 | 0.004529 | 46.802467 |

## Hardest Layers By Final NMSE

| Index | Layer | Slot | Final NMSE | Cosine | Tokens/s | Dead leaves |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 39 | `blocks.9.reverse_block.mixer.mixer.out_proj` | `foureyes:1` | 0.683897 | 0.531404 | 44275.0 | 10 |
| 41 | `blocks.10.forward_block.mixer.mixer.out_proj` | `foureyes:1` | 0.681834 | 0.534452 | 49053.6 | 9 |
| 47 | `blocks.11.reverse_block.mixer.mixer.out_proj` | `foureyes:2` | 0.676998 | 0.497438 | 50292.5 | 3 |
| 43 | `blocks.10.reverse_block.mixer.mixer.out_proj` | `foureyes:2` | 0.669105 | 0.538739 | 37998.0 | 10 |
| 45 | `blocks.11.forward_block.mixer.mixer.out_proj` | `foureyes:2` | 0.661269 | 0.490996 | 50362.3 | 5 |
| 35 | `blocks.8.reverse_block.mixer.mixer.out_proj` | `foureyes:1` | 0.658151 | 0.570698 | 28090.3 | 9 |
| 37 | `blocks.9.forward_block.mixer.mixer.out_proj` | `foureyes:1` | 0.654752 | 0.536494 | 47317.1 | 10 |
| 33 | `blocks.8.forward_block.mixer.mixer.out_proj` | `ripper:3` | 0.641370 | 0.566121 | 49213.3 | 10 |
| 31 | `blocks.7.reverse_block.mixer.mixer.out_proj` | `ripper:3` | 0.625663 | 0.584262 | 50720.6 | 14 |
| 25 | `blocks.6.forward_block.mixer.mixer.out_proj` | `ripper:2` | 0.620477 | 0.610693 | 43931.9 | 14 |
| 27 | `blocks.6.reverse_block.mixer.mixer.out_proj` | `ripper:2` | 0.610911 | 0.607922 | 43181.1 | 19 |
| 29 | `blocks.7.forward_block.mixer.mixer.out_proj` | `ripper:3` | 0.598186 | 0.604706 | 39027.7 | 14 |

## Notes

- This is validation-split layerwise distillation evidence, not CIFAR-10 final-test student evaluation.
- The full-layer pass substantially reduced local NMSE versus initialization for every eligible Linear layer, but late/middle `out_proj` layers remain the hardest and need recipe/budget tuning before final student claims.
- LocoProp-S succeeded for every layer and decreased or preserved local MSE in every recorded refit.
- No checkpoints or raw scheduler outputs are committed; this Markdown summary and CSV are the committed compact artifacts.
