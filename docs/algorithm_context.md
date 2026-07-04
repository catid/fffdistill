# Algorithm Context

FFF distillation here is a BF16 GPU study. BitNet/CPU deployment is deliberately out of scope until the official Mamba-3 teacher, FFF replacements, router training, and grouped execution are validated.

The conceptual bridge is:

- ProxProp/LocoProp-S local squared losses;
- FOOF and activation-covariance right preconditioning;
- linear-neuron boosting and feature-whitened local solves;
- hard-routed FFF, where each leaf becomes a local teacher-regression subproblem.

Distillation gives router supervision that generic training does not: for a fixed teacher Linear target, candidate branches or leaves can be scored by local prediction error. Vanilla STE is expected to be weak, so utility-targeted STE and hard-EM utility routing are first-class algorithm candidates.

## References

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
