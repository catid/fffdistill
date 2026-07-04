# T20 Route-Row Output Contribution Ablation

Status: validation-split single-layer route-output ablation complete; full student accuracy remains downstream T14/T15 work.

## Semantics

`route_rows_output_count` and `route_rows_output_fraction` are interpreted per visited routing node. For a depth-`D` tree, `route_output_rows_per_token = D * route_output_rows_per_node`.

- `routing_only`: route rows choose branches and do not contribute to output.
- `shared_routing_and_output`: the same route rows are used for branch scoring and output contribution.
- `split_routing_output`: routing rows score branches, and separate `route_result_rows` contribute along the visited path.

Reports distinguish the configured request (`route_rows_contribute`) from effective nonzero contribution (`route_output_contributes`). Physical `stored_rows` are reported separately from `effective_stored_rows`, `effective_trainable_rows`, `stored_route_output_rows`, `effective_route_output_rows`, `unused_route_output_rows`, and `unused_stored_route_output_rows`.

## Run Evidence

- Named-case config: `configs/fff_distill_t20_route_output_cases.yaml`
- Base config: `configs/fff_distill_stage_f.yaml`
- Teacher checkpoint: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Layer: eligible index `32`, `blocks.8.forward_block.mixer.mixer.in_proj`
- Data access: CIFAR-10 validation split only; every trial reports `test_accessed=false`.
- Runs: local offsets 0-1 in `t20_route_output_cases_20260704_092832`; remote offsets 2-6 in `t20_route_output_cases_remote_20260704_092948`.
- Scheduler result: 7/7 final named cases succeeded at git commit `9265ec0bf80db186315ce3bc3301ffe82b8932ee`. The first mixed local/remote launch had five infrastructure failures because remote workdirs were still at the Stage F commit; remotes were synced and only missing offsets were rerun successfully.
- Excluded slots: `ripper:1` and `foureyes:0` remained excluded because pre-launch checks showed persistent 100% GPU utilization with negligible memory and no visible compute PID.
- Validation accuracy evaluation: after copying ignored `fff_state.pt` artifacts locally from scheduler outputs/remotes, each one-layer replacement was loaded into the selected teacher and evaluated on the full CIFAR-10 validation split (`val_steps=10`), with `test_accessed=false`.

Recompute command pattern for the validation-accuracy column:

```bash
PYTHONPATH=src .venv/bin/python -m cifar_mamba_fff.t20_recompute_validation \
  --teacher-checkpoint checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt \
  --distill-config configs/fff_distill_stage_f.yaml \
  --replacement-state outputs/scheduler_distill_hpo/<run>/<machine>/<gpu>/trials/trial_000000/layers/blocks__8__forward_block__mixer__mixer__in_proj/fff_state.pt \
  --eligible-index 32 \
  --output-dir outputs/t20_validation_recompute/<case_name> \
  --device cuda \
  --batch-size 512 \
  --max-val-steps 10
```

The recompute tool writes `t20_validation_recompute.json` and `run_context.json`,
records `sample_split=val`, and refuses CUDA execution if CUDA is unavailable. It
does not construct a CIFAR-10 test loader.

## Results

| Case | Role | Output rows/node | Active rows/token | Effective stored rows | Final NMSE | Cosine | Tokens/s | Dead leaves | Validation accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none_routing_only` | `routing_only` | 0 | 108.0 | 232 | 0.300582 | 0.853248 | 36521.3 | 11 | 0.9404 |
| `shared_one_per_node` | `shared_routing_and_output` | 1 | 113.0 | 232 | 0.293318 | 0.857536 | 29078.7 | 13 | 0.9406 |
| `shared_all` | `shared_routing_and_output` | 2 | 118.0 | 232 | 0.287133 | 0.860755 | 28094.1 | 15 | 0.9404 |
| `shared_half_fraction` | `shared_routing_and_output` | 1 | 113.0 | 232 | 0.293337 | 0.857533 | 28063.3 | 13 | 0.9404 |
| `split_one_per_node` | `split_routing_output` | 1 | 113.0 | 263 | 0.292199 | 0.857408 | 34295.6 | 17 | 0.9396 |
| `split_all` | `split_routing_output` | 2 | 118.0 | 294 | 0.284449 | 0.861225 | 28240.1 | 17 | 0.9398 |
| `split_half_fraction` | `split_routing_output` | 1 | 113.0 | 263 | 0.292116 | 0.857402 | 37399.5 | 16 | 0.9396 |

## Observations

- Best local MSE in this one-layer run: `split_all` with NMSE `0.284449` and cosine `0.861225`.
- Fastest case: `split_half_fraction` at `37399.5` tokens/s.
- Route-output contribution improved local MSE over `none_routing_only` for every contributing case. `split_all` was the best quality point at this short budget, while the one-row split/shared cases were close and cheaper in active rows.
- This ablation uses a single representative layer and short validation-split token budget. It is suitable for route-output design ranking, not final model accuracy claims.
- `validation_accuracy_after_replacement` is single-layer replacement accuracy on the full 5k CIFAR-10 validation split (`10` batches at batch size `512`) with CIFAR-10 test disabled. Full assembled-student validation and final-test accuracy remain T14/T15 work.

## Required Result Columns

The committed CSV `docs/t20_route_row_output_ablation_results.csv` includes route contribution flags, route-row role/count/fraction, active rows, stored/effective rows, unused rows, MSE/cosine, throughput, route entropy, dead leaves, leaf occupancy, LocoProp before/after MSE, validation accuracy, validation loss, and validation steps.
