# Experiment Plan

See [plan.md](../plan.md) for the durable project plan. This file tracks run-level execution details once experiments begin.

No long experiments have been launched yet. Stage A gates that have passed include:

- environment verification;
- cluster inventory and SSH/job launch sanity;
- unit tests;
- grouped-vs-naive FFF correctness;
- teacher one-batch smoke;
- tiny LocoProp-S refit;
- profiler smoke;
- bounded multi-machine teacher smoke;
- bounded HPO scheduler smoke;
- CUDA BF16 teacher HPO candidate prefilter smoke.

Before launching broad teacher HPO:

- commit and push the CUDA kernel-smoke prefilter code;
- sync remotes to the pushed commit;
- recheck GPU occupancy on `work`, `ripper`, `foureyes`, and `ai`;
- exclude occupied slots, such as the last observed unrelated jobs on `foureyes:2-3`;
- use unique `--run-id`, run-scoped collection paths, and non-overwriting launch records;
- use validation metrics only, keeping CIFAR-10 test untouched until final selected checkpoints.
