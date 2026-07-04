# CIFAR-10 Mamba-3 FFF Distillation

This repository is being built as a reproducible BF16 GPU research project for distilling eligible `nn.Linear` layers from an official Mamba-3 CIFAR-10 teacher into Fast Feedforward Network (FFF) replacements.

Current status: bootstrap skeleton and task tracking. No teacher accuracy, FFF quality, throughput, or final CIFAR-10 test result is claimed until the relevant Beads tasks produce saved logs and reports.

## Ground Rules

- Teacher blocks must come from the official `state-spaces/mamba` codebase and must use official `mamba_ssm` Mamba-3 modules.
- Optimizer baseline starts with official Muon plus AdamW fallback parameter groups. PACE+Muon, PACE+NorMuon, and related optimizer-experiments techniques are deferred to the optimizer-ablation task after the simple baseline is working.
- fastfeedforward is installed and used as a canonical reference/baseline where shape-compatible.
- CIFAR-10 train is split into 45k train / 5k validation. The CIFAR-10 test split is used only for validation-selected final checkpoints.
- Dependency, GPU, SSH, correctness, and profiler gates must pass before long experiments.
- Failed installs, unavailable SSH hosts, collapsed routers, unmet accuracy targets, and reviewer/tool failures are reported as failures or limitations, not hidden behind fallbacks.
- `scripts/setup.sh` applies documented vendor patches for Python 3.12 compatibility in official Mamba-3's TileLang/TVM dependency path. The patches preserve the official TileLang MIMO training kernels and are documented in `scripts/patches/mamba3_python312_tilelang_tvm.patch`.

## Setup

```bash
bd init
git init
git remote add origin git@github.com:catid/fffdistill.git
bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_env.py
python scripts/verify_cluster.py --allow-incomplete true
```

## Tests

```bash
bash scripts/run_tests.sh
```

## Experiment Entry Points

```bash
bash scripts/run_teacher_hpo.sh
bash scripts/run_teacher_final.sh
bash scripts/run_distill_hpo.sh
bash scripts/run_finetune_hpo.sh
bash scripts/make_report.sh
```

Every script supports `--quick-smoke true` when it can run a bounded smoke path.

## Primary References

- Mamba-3 paper: https://arxiv.org/pdf/2603.15569
- Official Mamba repository: https://github.com/state-spaces/mamba
- Muon repository: https://github.com/KellerJordan/Muon
- Muon writeup: https://kellerjordan.github.io/posts/muon/
- Fast Feedforward Networks paper: https://arxiv.org/pdf/2308.14711
- Official fastfeedforward repository: https://github.com/pbelcak/fastfeedforward
- LocoProp paper: https://arxiv.org/pdf/2106.06199
- LocoProp Google Research code: https://github.com/google-research/google-research/tree/master/locoprop
- Proximal Backpropagation: https://arxiv.org/pdf/1706.04638
- Simple Linear Neuron Boosting: https://arxiv.org/pdf/2502.01131
- FOOF / Gradient Descent on Neurons: https://arxiv.org/pdf/2201.12250
- Layerwise preconditioning / feature learning: https://proceedings.mlr.press/v267/zhang25bh/zhang25bh.pdf
- Straight-through estimator theory: https://arxiv.org/pdf/1903.05662
- Gumbel-Softmax: https://arxiv.org/pdf/1611.01144
- Concrete distribution: https://arxiv.org/pdf/1611.00712
- Hard Concrete / L0 gates: https://arxiv.org/pdf/1712.01312
- Expert Choice routing: https://arxiv.org/pdf/2202.09368
- DSelect-k: https://arxiv.org/pdf/2106.03760
- Optimizer experiments: https://github.com/catid/optimizer_experiments
