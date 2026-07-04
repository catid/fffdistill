# T19 Optimizer Ablation Summary

This is a short validation-split smoke/ablation gate for optimizer plumbing, not a final accuracy result. CIFAR-10 test was not accessed.

T18 protocol caveat: the recorded 2026-07-04 smoke rows below used one different
seed per optimizer/schedule case (`7331` through `7336`), so they are
seed-confounded and must not be read as optimizer quality comparisons. The runner
and config now support an explicit shared `seeds:` list per case, write
`seed_list`/`seed_count` to artifacts, and require a NorMuon/PACE+NorMuon update-RMS
calibration note before future ablation claims.

- Run id: `t19_optimizer_ablation_20260704_094958`
- Git commit for launch: `c4425c03000273f76cf3a3892e1abf90c1dd8f68`
- Budget per case: `epochs=1`, `max_train_steps=2`, `max_val_steps=2`, batch size `256`, train/validation split only.
- Remote sync: `ripper`, `foureyes`, and `ai` were synced to the launch commit before jobs ran.
- Usable GPU slots used: `work:0`, `work:1`, `ripper:0`, `ripper:2`, `foureyes:1`, `ai:0`.
- Excluded anomalous occupied slots: `ripper:1` and `foureyes:0` showed 100% utilization with negligible memory and no visible compute PID.
- Initial local background launch produced empty sidecars; local cases were rerun in foreground as `work_0_case0_fg` and `work_1_case1_fg`.

| case | slot | optimizer | schedule | seed | val acc | steps | imgs/s | EMA eval | test accessed |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| official_muon_cosine | work_0_case0_fg | muon_adamw | cosine | 7331 | 0.1699 | 2 | 100.83 | False | False |
| official_muon_wsd | work_1_case1_fg | muon_adamw | wsd | 7332 | 0.1094 | 2 | 100.56 | False | False |
| pace_muon_ema_control | ripper_0_case2 | pace_muon | wsd | 7333 | 0.1387 | 2 | 13.11 | True | False |
| pace_muon_c1e3 | ripper_2_case3 | pace_muon | wsd | 7334 | 0.0996 | 2 | 13.22 | True | False |
| normuon_wsd | foureyes_1_case4 | normuon_adamw | wsd | 7335 | 0.1309 | 2 | 13.86 | False | False |
| pace_normuon_c1e3 | ai_0_case5 | pace_normuon | wsd | 7336 | 0.1797 | 2 | 96.41 | True | False |

## Interpretation

- The goal of this gate was correctness/provenance and equal-budget execution, not optimizer ranking. Two training steps are too short for quality claims.
- The recorded rows are also seed-confounded as described above; rerun with the updated shared-seed protocol before comparing optimizer families.
- Reported `imgs/s` includes first-run Mamba/TileLang compile and launch overhead on some remotes; use T16/profiled runs for throughput claims.
- `pace_muon` wraps the official KellerJordan/Muon optimizer; this preserves the official baseline and adds PACE EMA evaluation.
- `normuon_adamw` and `pace_normuon` are explicitly non-official vendored optimizer-experiments ablations using this repo's audited parameter split.
- WSD is trainer-side and reported separately from optimizer family.

## Provenance

- Optimizer-experiments source commit: `689568d71ebe92093e5f5bf433127a5184ef0c35`.
- Vendored files: `src/cifar_mamba_fff/optim/pace.py` and `src/cifar_mamba_fff/optim/normuon.py`.
- Official Muon baseline remains `SingleDeviceMuonWithAuxAdam` from KellerJordan/Muon, verified by `scripts/verify_env.py`.
