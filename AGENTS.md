# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Project-Specific Directives

- Treat `plan.md` as the durable project plan. Keep it consistent with Beads task updates and material plan changes.
- Use Beads for all non-trivial work. Start from `bd ready`, inspect/claim one item, update notes at meaningful checkpoints, and close tasks only after their acceptance criteria and required checks pass.
- Preserve the no-fallback research contract: do not substitute fake Mamba, Mamba-2, Transformer, ResNet, AdamW-only, CPU-only, smaller models, another dataset, or fake dependency shims when official Mamba-3, official Muon, fastfeedforward, CUDA, or CIFAR-10 work fails.
- Keep early optimizer work simple: establish the official Muon + AdamW fallback baseline first. Defer PACE+Muon, PACE+NorMuon, and related optimizer-experiments ablations to the optimizer-ablation task after baseline teacher/distillation paths are stable.
- Add WSD as a learning-rate schedule option when scheduler tuning begins, but report it separately from optimizer-family changes.
- Preserve the FFF route-row output ablation: compare routing-only rows, rows shared for routing and output contribution, and split-role rows where separate introduced rows handle routing versus path output contribution.
- Keep CIFAR-10 test untouched until validation-selected final checkpoints. Clearly separate smoke, validation, and final test results.
- Do not run long experiments until environment verification, unit tests, grouped-vs-naive FFF correctness, SSH/job-launch sanity, GPU utilization checks, and profiler sanity pass.
- Use all usable GPUs across `work`, `ripper`, `foureyes`, and `ai` for nontrivial queued experiments. Prefer concurrent one-GPU trials for CIFAR-10-sized HPO unless a measured DDP benchmark is faster. If any GPU stays idle while runnable jobs are queued, document the measured bottleneck or infrastructure failure.
- Use parallel workers/subagents where work can be split into independent, disjoint scopes. Subworkers may implement or investigate bounded slices, but main Codex must review, test, integrate, and commit coherent code.
- Before launching long-running jobs, optimize and bug-fix the pipeline: autotune batch size and DataLoader settings, verify remote environments, run smoke tests, profile hot paths, check GPU utilization, and fix correctness/performance issues found by the gates.
- Before trusting behavior-changing implementation, run deterministic checks and available independent review/subworker passes. Record limitations honestly if a reviewer or check is unavailable.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds. This repository explicitly requires pushing to `git@github.com:catid/fffdistill.git`; that project requirement overrides any generic Beads guidance about local-only ephemeral branches.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
