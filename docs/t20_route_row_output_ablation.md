# T20 Route-Row Output Contribution Ablation

Status: implementation/config/report scaffold complete; real distillation metrics pending teacher and layerwise distillation stages.

## Semantics

`route_rows_output_count` and `route_rows_output_fraction` are interpreted per visited routing node. For a depth-`D` tree, `route_output_rows_per_token = D * route_output_rows_per_node`.

- `routing_only`: route rows choose branches and do not contribute to output.
- `shared_routing_and_output`: the same route rows are used for branch scoring and output contribution.
- `split_routing_output`: routing rows score branches, and separate `route_result_rows` contribute along the visited path.

Reports distinguish the configured request (`route_rows_contribute`) from effective
nonzero contribution (`route_output_contributes`). Split-role kernels evaluate only
the selected result rows per visited node, so active-row and throughput metadata match
the selected ablation case rather than the maximum stored `route_result_rows`.

## Named Cases

The reproducible named-case config is `configs/fff_route_output_ablation.yaml`.

| Case | route_rows_contribute | route_row_role | route_rows_output_count | route_rows_output_fraction | route_result_rows | route_output_rows_per_node | Metrics |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| none_routing_only | false | routing_only | 0 | null | 0 | 0 | pending |
| shared_one_per_node | true | shared_routing_and_output | 1 | null | 0 | 1 | pending |
| shared_all | true | shared_routing_and_output | all | null | 0 | 2 | pending |
| shared_half_fraction | true | shared_routing_and_output | all | 0.5 | 0 | 1 | pending |
| split_one_per_node | true | split_routing_output | 1 | null | 2 | 1 | pending |
| split_all | true | split_routing_output | all | null | 2 | 2 | pending |
| split_half_fraction | true | split_routing_output | all | 0.5 | 2 | 1 | pending |

## Required Result Columns

When layerwise distillation and fine-tuning runs are available, report:

- validation MSE and cosine similarity;
- validation accuracy and final-test accuracy only after validation selection;
- throughput and GPU utilization;
- active rows per token and estimated active FLOPs;
- stored rows and parameter count;
- route entropy, dead leaves, and leaf occupancy percentiles;
- route-row role, count, fraction, route-result rows, requested contribution, and
  effective nonzero route-output contribution.
