# Pareto Plots

These rendered plots are generated from committed CSV summaries only.
They are validation and smoke-study artifacts, not final FFF-student claims.

| Plot | Committed source |
| --- | --- |
| [Accuracy vs active rows](accuracy_vs_active_rows.svg) | `docs/t20_route_row_output_ablation_results.csv` |
| [Accuracy vs throughput](accuracy_vs_throughput.svg) | `docs/t20_route_row_output_ablation_results.csv` |
| [MSE vs active rows](mse_vs_active_rows.svg) | `docs/t20_route_row_output_ablation_results.csv`, `docs/t13_stage_f_layerwise_summary.csv` |
| [Dead leaves vs balance](dead_leaves_vs_balance.svg) | `docs/t13_stage_d_arch_summary.csv` |
| [Route entropy vs accuracy](route_entropy_vs_accuracy.svg) | `docs/t20_route_row_output_ablation_results.csv` |
| [STE method vs MSE/throughput](ste_method_vs_mse_throughput.svg) | `docs/t13_stage_c_router_summary.csv`, `docs/t13_stage_d_arch_summary.csv`, `docs/t13_stage_f_layerwise_summary.csv` |

Missing or limited plot-source coverage is documented in
[missing_sources.md](missing_sources.md).

Regenerate and validate:

```bash
.venv/bin/python docs/render_pareto_plots.py
.venv/bin/python docs/render_pareto_plots.py --check
```
