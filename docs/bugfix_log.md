# Bugfix Log

## Bootstrap

- Beads was initialized before Git existed, so the fresh database lacked a repository fingerprint. The installed CLI recommended `bd migrate --update-repo-id`; that migration was run before tasks were created.
- The Claude Opus planner wrapper failed schema validation with `Claude JSON did not contain a structured_output object`. The raw artifact at `.git/codex-claude-quality/plan.json.raw.json` shows the underlying `claude-opus-4-8` run returned success and produced an advisory plan. This is recorded as a planning-tool limitation.

Further suspected bugs, fixes, tests, and residual risks will be appended during T18.

## Environment Gate Failure

- `scripts/setup.sh` initially completed but the Mamba install resolver changed torch from nightly `2.14.0.dev20260702+cu130` to `2.12.1+cu130`. `scripts/setup.sh` was patched to install Mamba with `--no-deps`, and `scripts/verify_env.py` now fails if torch is not a `*dev*+cu130` nightly.
- Official `mamba_ssm` from commit `ed6ce09e4d802e274b1ecc7205757b892e180a93` contains `Mamba3`, but import fails in TileLang/TVM on Python 3.12 with `AttributeError: attribute '__dict__' of 'type' objects is not writable`. See `docs/setup_failure_report.md`.

## Mamba-3 Vendor Patch

- Added `scripts/apply_vendor_patches.py` and `scripts/patches/mamba3_python312_tilelang_tvm.patch`.
- Patched `apache-tvm-ffi==0.1.12` to skip unsettable type dunders during TVM FFI class registration.
- Patched `mamba-ssm==2.3.2.post1` so a broken optional Cute inference-step import does not abort `Mamba3` import. The TileLang MIMO training kernel remains verified as available.
- Patched `tilelang==0.1.8` embedded TVM runtime support so decorated Python visitors can initialize wrapper state on Python 3.12.
- Pinned `scripts/setup.sh` to official `state-spaces/mamba` commit `ed6ce09e4d802e274b1ecc7205757b892e180a93` and kept `--no-deps` so the Mamba install cannot silently replace the requested nightly PyTorch cu130 wheel.
- Tightened `scripts/verify_env.py` so Mamba-3 verification requires `is_mimo=True`, `mimo_rank=2`, and `chunk_size=16`; it no longer accepts a non-MIMO constructor path.
- Fixed the official teacher shape test to run a CUDA BF16 autocast MIMO-compatible shape instead of accidentally exercising upstream Triton kernels with CPU tensors.
- Verified `python scripts/verify_env.py --quick-smoke true` passes with torch `2.14.0.dev20260702+cu130`, 2 visible local Blackwell GPUs, official `mamba_ssm.Mamba3`, TileLang MIMO BF16 autocast forward/backward, official `SingleDeviceMuonWithAuxAdam`, and `fastfeedforward.FFF` import.
- `scripts/run_tests.sh` now runs `scripts/verify_env.py --quick-smoke true` before ruff/pytest, so CUDA-hidden runs cannot pass the default local quality gate.
- `scripts/verify_cluster.py` now checks `python3` without `|| true`, records `gpu_available`, `python_available`, and `venv_available` separately, and marks missing Python unavailable. The latest inventory sees all 12 GPUs, but project venvs on `ripper`, `foureyes`, and `ai` are not verified, so long remote jobs remain blocked.
- Closed T02 for honest cluster inventory and created `fff-qb3.26` for remote repo sync/setup plus `verify_env.py` on `ripper`, `foureyes`, and `ai`. Teacher HPO, distillation HPO, and profiling now depend on that remote environment gate.
- Whole-codebase Fable review flagged that cluster checks were not workdir-anchored and could exit 0 with unusable remote envs. Fixed by adding per-machine `workdir`, anchoring repo-specific checks under `cd <workdir>`, validating detected GPU count against config, and making full cluster verification fail unless `--allow-incomplete true` is passed.
- Added `scripts/constraints-py312-cu130.txt` and pinned setup inputs for the observed Stage A environment, including nightly torch/cu130 versions, official Mamba commit, official Muon commit, fastfeedforward, TileLang/TVM, and core Python dependencies.
- `scripts/verify_env.py --quick-smoke true` now includes a `fastfeedforward.FFF` CUDA forward smoke instead of import-only coverage.
- `scripts/verify_env.py` now verifies `direct_url.json` for official `mamba-ssm` and `muon-optimizer`, failing unless the installed packages come from the expected GitHub repositories and pinned commits.
- `scripts/setup.sh` installs the pinned official Muon commit with `--no-deps`, matching the Mamba install discipline and preventing dependency resolver drift after the nightly cu130 torch stack is installed.
- Remote setup exposed two `uv` resolver issues for the torch install step: the PyTorch nightly index alone could not resolve `filelock==3.29.5`, and adding PyPI as an extra index triggered `uv` index-priority protection because PyPI also publishes `torch`. The torch install step now uses PyPI as an extra index plus `--index-strategy unsafe-best-match` while exact-pinning `torch`, `torchvision`, and `torchaudio` to the `+cu130` nightly builds.
- T04 Mamba teacher wrapper review found the original default teacher was below the 9M-11M target. The default was changed to `depth=18`, giving 10,055,078 trainable parameters for the current official Mamba3 CIFAR wrapper, and tests now assert the default parameter count is in range.
- Teacher precision tests now assert parameters remain FP32 before and after CUDA BF16 autocast forward/backward. This protects the documented contract that BF16 comes from autocast, not `model.bfloat16()`, because upstream Mamba3 MIMO biases are not safe under direct BF16 parameter conversion.
- `Mamba3CifarConfig.validate()` now catches official Mamba shape constraints early: `d_model * expand` must be divisible by `headdim`, and CIFAR patch sequence length must be divisible by `chunk_size`.
- `time_cuda_callable` now raises when CUDA is unavailable unless callers explicitly pass `allow_cpu=True`.
- Verified `bash scripts/run_tests.sh` passes: environment verifier, ruff clean, and 126 pytest tests passed. Pytest reports upstream `tvm_ffi` duplicate-field `UserWarning`s during Mamba import; they are non-fatal and recorded as residual dependency noise.

## Subworker Findings Filed

- Subworker A reviewed the Mamba vendor path and found no fake Mamba fallback, but identified three hardening items: strict MIMO construction in `verify_env.py`, pinned Mamba commit in `setup.sh`, and exact patch documentation. These were fixed in T22.
- Subworker B reviewed FFF/LocoProp code and filed follow-up Beads issues.
- `fff-qb3.23` was fixed: route-output empty batches now use an explicit route-row dimension instead of ambiguous `reshape(..., -1)`, and empty leading-dimension tests cover grouped/naive/default paths for routing-only, shared route-output, and split route-output roles.
- `fff-qb3.24` was fixed: LocoProp-S ridge refits detach `A`, `Y`, and `v0`, run the FP32 solve/metrics under `torch.no_grad()`, and tests assert returned weights do not require grad when inputs do.
- `fff-qb3.25` was fixed: CUDA BF16 grouped-vs-naive tests now cover routing-only, shared route-output, and split route-output roles under hard and soft routing with BF16-appropriate tolerance.

## Stage A Component Fixes

- T03 CIFAR-10 data pipeline is implemented with deterministic 45k/5k split, quick-smoke subsets with downloads disabled, crop/flip plus optional RandAugment, explicit `use_test=True` gate for test split access, and offline tests.
- T05 Muon grouping is implemented with module/name-aware routing: hidden 2D matrix parameters go to Muon; biases, norms, embeddings, classifier/head, scalar/1D/non-2D parameters go to AdamW; assignments are returnable/loggable and tests check no overlap/missing parameters.
- T07 FFF correctness coverage now includes shape/routing/grouped-vs-naive/empty-batch/BF16 tests, route diagnostics consistency, gradient flow to applicable parameters, and a tiny overfit sanity test.
- Remote setup T26 is complete: `verify_env.py --quick-smoke true` passed on `ripper`, `foureyes`, and `ai`; strict `scripts/verify_cluster.py` passes for all 12 GPUs. GitHub SSH clone from the remote hosts failed with `publickey` errors, so the remote repo sync used `rsync` from `work`; this remains a scheduler/sync caveat to keep visible before launching jobs.

## Post-Subworker Integration Review

- Linear replacement review found that progressive replacement was not safe to advance on an already modified model, FFF replacements did not preserve eval/train mode, partially failed hook setup could leak earlier hooks, and `distill_linears` quick smoke accepted invalid progressive args. Fixed by making replacement skip already compatible replacements by default, preserving replacement mode, closing captures on constructor failure, validating progressive args at CLI entry, and adding regression tests.
- Linear capture memory safety uses bounded truncating capture instead of streaming shards for now, preserving the existing tensor-return API. `LinearCapture` defaults to 8192 tokens and 64 MiB per layer, copies only the accepted slice to CPU, skips empty forwards, stops appending after limits are reached, and records observed/captured/dropped token counts plus resident captured bytes. `LinearCaptureSet` threads the same per-layer bounds into every hooked linear.
- FFF grouped-path review found that `fallback_leaf=True, region_leak=1.0` gives zero regular-leaf contribution, but the selected-leaf grouped path still evaluated regular leaves and could produce `nan` under `inf * 0` while naive skipped zero-weight leaves. Fixed by returning zero regular-leaf output for that static regime and adding a regression test.
- Benchmark review found that CUDA timing synchronized the default device instead of the requested benchmark device. `time_cuda_callable` now accepts a device and synchronizes that CUDA device; tests monkeypatch `torch.cuda.synchronize` to assert `cuda:1` is passed through.
- Representative FFF profiling should not always run the slow Python-loop naive path. `benchmark_fff.py` now skips naive by default, includes it automatically for quick smoke, and exposes `--include-naive true` for explicit grouped-vs-naive correctness/timing runs.
- Router STE helpers still perform finite validation, but duplicated validation inside utility-targeted and hard-EM call chains was removed by using already validated tensors directly for softmax/argmax. Before long training, T16/T18 should still check whether per-step recipe utilities force unwanted CUDA synchronizations.
- Re-ran `bash scripts/run_tests.sh` after these fixes: environment verification passed, ruff passed, and 148 pytest tests passed. The only warnings were the known upstream TVM/TileLang duplicate-field warnings during Mamba import.

## Teacher Wrapper Review

- Independent review flagged that the default depth-18 CIFAR teacher stacked raw official `Mamba3` mixers without residual/pre-norm structure, while `teacher_default.yaml` already exposed `drop_path`. This would be a bad HPO target before long teacher runs.
- The teacher now wraps official `Mamba3` mixers with upstream `mamba_ssm.modules.block.Block`, keeps the official mixer implementation intact, carries the upstream `(hidden_states, residual)` stream through the stack, and applies final `hidden_states + residual` before the final norm.
- `Mamba3CifarConfig` now validates `drop_path`, `norm_epsilon`, and `residual_in_fp32`; stochastic depth is applied outside the mixer with a linear per-layer schedule from 0 to the configured maximum.
- Default teacher parameter count is now 10,063,142, still inside the required 9M-11M target.
- Verification: `tests/test_mamba3_teacher_shapes.py` covers config validation, residual Block usage, drop-path schedule, default parameter count, CUDA BF16 autocast shape/backward with FP32 parameters, and CUDA BF16 bidirectional shape. Full `bash scripts/run_tests.sh` passed with environment verification, ruff, and 159 pytest tests.

## Detached Scheduler Gate

- T06c added detached local/SSH launch helpers for bounded scheduler smoke jobs. Each job writes a generated `launch.sh`, `status.json`, `stdout.log`, `stderr.log`, `pid.txt`, `exit_code.txt`, and `heartbeat.txt` under its output directory.
- `gpu_scheduler.py --dry-run false` now refuses non-smoke launches unless `--allow-long-jobs true` is explicit. The default teacher scheduler command uses `--quick-smoke true --smoke-mode metadata`, so scheduler readiness does not require CIFAR data or long training.
- Launch preflight now verifies `.git`, the selected Python executable, Python version, and the expected git commit before starting a detached job. Stale remote repos are marked `failed_infra` instead of misclassified as model `failed_logic`.
- Artifact collection copies small status/log/metrics files from each launched local or remote job into a local collection root. Missing train metrics are expected for metadata-only smoke and are logged as missing artifacts.
- Cluster inventory on 2026-07-04 found all configured machines reachable with 12 detected GPUs and project venvs. `foureyes` GPUs 2-3 had only about 15 GiB free, and `ai` GPUs have about 32 GiB total/free, so real HPO scheduling still needs per-job memory-aware slot selection.
- A 12-slot metadata launch before commit preflight hardening caught stale remote workdirs: `ripper`, `foureyes`, and `ai` were at commit `eb67a76a3a2831c70641d76d7ff04d8ffe53bb0d` while `work` expected `6f35d8c1eb7efde0ad1712d32a690743f35045d8`; their old `train_teacher.py` rejected `--smoke-mode metadata`. After preflight hardening, the same stale remotes are skipped as `failed_infra` and both local work slots launch and collect successfully.
- Verification: focused scheduler/cluster tests pass, including local detached success, nonzero return-code classification, stale-commit preflight rejection, dry-run queue generation, and unavailable-slot filtering. Full remote launch should be rerun only after committing/pushing and syncing remote workdirs to the new commit.
