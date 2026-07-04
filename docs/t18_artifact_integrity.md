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

## Cluster Sync And Preflight Status

As of 2026-07-04, remote GitHub SSH auth is still not the working sync path for
`ripper`, `foureyes`, and `ai`. The working and test-covered fallback is
`scripts/sync_repo_remote.sh`, which rsyncs from the clean `work` checkout after
commits are made. The script:

- refuses any local staged, unstaged, or untracked non-ignored files before
  syncing;
- backs up remote `git status --short` and `git diff` under
  `outputs/remote_sync_backups/` before overwrite;
- rsyncs with `--delete` while excluding heavy/generated directories such as
  `.venv/`, `data/`, `outputs/`, `checkpoints/`, and caches;
- verifies after copy that the remote `git rev-parse HEAD` equals the local
  commit and that `.venv/bin/python` is executable.

Focused unit coverage in `tests/test_sync_repo_remote.py` covers clean local
operation, dirty-local refusal, untracked-local refusal, and the remote rsync
target path with fake `ssh`/`rsync` tools. This makes the fallback testable
without contacting cluster hosts.

Scheduler launch preflight separately records `expected_commit`, `local_commit`,
`remote_commit`, and local/remote worktree cleanliness. It rejects stale remote
commits, dirty local worktrees, dirty remote worktrees, missing remote `.git`,
missing remote Python, and optional missing CIFAR-10 train data before launching
detached jobs.

The scheduler now computes a canonical SHA256 manifest for referenced
config/selection files in job commands and compares that manifest on each remote
before launch. This closes the previous config-content preflight gap for YAML
configs and selection JSON records. Launcher script hashing is still covered
indirectly by the expected git commit and clean-worktree checks rather than by a
separate script manifest. The safe policy is:

- sync with `scripts/sync_repo_remote.sh all` from a clean committed checkout;
- launch with default `--preflight true --expected-commit "$(git rev-parse HEAD)"`;
- require matching local/remote config manifest hashes in preflight records for
  scheduler-launched jobs that reference config or selection files;
- do not promote new remote results as fully preflight-hardened if the launch
  record shows missing or mismatched config manifests.

## Recollection And Regeneration Policy

There is no standalone "collect an already completed remote scheduler run" CLI in
the repository. Hardened artifact collection currently runs through the scheduler
wait path when jobs are launched with `--wait true`. Therefore old summaries can
be repaired in one of three ways:

1. Regenerate from a local full collected root when that root exists under
   `outputs/scheduler_collected/` and no required metrics are truncated.
2. Rerun the original bounded scheduler command with a new run id after rsync
   preflight, then regenerate from the new hardened collection root.
3. Keep the summary marked historical/provenance-limited when neither full local
   artifacts nor an acceptable rerun are available.

Use this integrity probe before regenerating from any launch-result JSONL:

```bash
.venv/bin/python - <<'PY'
import json

launch_jsonl = "outputs/scheduler_launch_results_<run-id>.jsonl"
truncated = []
missing = []
with open(launch_jsonl, encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        for name, record in payload.get("files", {}).items():
            if not isinstance(record, dict):
                continue
            if record.get("truncated"):
                truncated.append((line_no, name))
            if record.get("returncode") == 44:
                missing.append((line_no, name))
print("truncated_files=", truncated)
print("missing_files=", missing)
PY
```

For the distillation-HPO summaries, the local regeneration command pattern is:

```bash
PYTHONPATH=src .venv/bin/python -m cifar_mamba_fff.summarize_distill_hpo \
  --collected-root outputs/scheduler_collected/<run-id> \
  --csv-out docs/<summary>.csv \
  --markdown-out docs/<summary>.md \
  --title "<report title>"
```

Known local roots that can be checked and regenerated without launching jobs:

| Summary | Candidate local root | Command status |
| --- | --- | --- |
| `docs/t13_stage_c_router_summary.md` | `outputs/scheduler_collected/distill_stage_c_router_offsets_20260704_090349` | Regenerate with `summarize_distill_hpo` after integrity probe. |
| `docs/t13_stage_d_arch_summary.md` | `outputs/scheduler_collected/distill_stage_d_arch_20260704_090730` | Regenerate with `summarize_distill_hpo` after integrity probe. |
| `docs/t13_stage_e_layer_summary.md` | `outputs/scheduler_collected/distill_stage_e_layers_20260704_091514` | Regenerate with `summarize_distill_hpo` after integrity probe. |
| `docs/t13_stage_f_layerwise_summary.md` | `outputs/scheduler_collected/distill_stage_f_shards_20260704_091746` | Regenerate with `summarize_distill_hpo` after integrity probe, but keep the validation-capture leakage caveat. |
| `docs/t20_route_row_output_ablation.md` | `outputs/scheduler_collected/t20_route_output_cases_20260704_092832` and `outputs/scheduler_collected/t20_route_output_cases_remote_20260704_092948` plus `outputs/t20_validation_recompute/` | Regenerate distill metrics from collected roots and rerun the documented `t20_recompute_validation` commands for validation accuracy columns if the local replacement states are present. |

`docs/t06_hpo_smoke_summary.md` has local collected artifacts at
`outputs/t06_hpo_smoke_collected`, but there is no committed teacher-HPO summary
regenerator analogous to `summarize_distill_hpo`. Keep it historical unless a
small teacher-HPO summarizer is added or the smoke is rerun with the current
hardened collector.

`docs/t06_candidate_prefilter_summary.md` combines local direct teacher-HPO
outputs and a scheduler smoke; regenerate it manually from the named output dirs
or rerun the exact bounded commands in that document.

`docs/t19_optimizer_ablation_summary.md` was produced from
`outputs/t19_optimizer_ablation_20260704_094958`, not a hardened scheduler
collection root. It should remain provenance-limited until either a recollection
path is added for those per-case outputs or the ablation is rerun under the
current artifact-collection contract.
