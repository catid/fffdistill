# T18 Remote Artifact Integrity Note

Status: collector hardened for future runs on 2026-07-04.

## Issue

Before `fff-djb`, remote artifact collection copied text through
`read_remote_text(..., max_bytes=...)`, which returned only the tail of files larger
than the byte limit and did not record truncation metadata. Trial collection also
focused on fixed `trial_000000` paths, which was insufficient for multi-trial HPO
jobs.

This does not prove any committed CSV is wrong, but it means summaries generated
from pre-hardening collected remote logs should be treated as provenance-limited
unless they were independently checked against local full artifacts.

## Existing Compact Summaries To Treat Carefully

- `docs/t06_hpo_smoke_summary.md`
- `docs/t06_candidate_prefilter_summary.md`
- `docs/t13_distill_hpo_smoke_summary.md`
- `docs/t13_stage_c_router_summary.md`
- `docs/t13_stage_d_arch_summary.md`
- `docs/t13_stage_e_layer_summary.md`
- `docs/t13_stage_f_layerwise_summary.md`
- `docs/t19_optimizer_ablation_summary.md`
- `docs/t20_route_row_output_ablation.md`

The small committed CSV files themselves are not truncated by Git, but their source
remote artifacts may have been collected by the older tail-only path.

## Fix

Future scheduler artifact collection now records, per file:

- remote source path;
- remote byte size;
- copied byte size;
- local byte size;
- truncation status;
- missing-file status.

It also discovers all remote `trial_*` directories, infers expected trial
directories from non-truncated HPO summaries, and marks collection status as
`artifacts_collected_truncated_metrics` if any metrics artifact is truncated.

Truncated metrics should be used only for failure analysis unless the run is
re-collected or regenerated from full artifacts.
