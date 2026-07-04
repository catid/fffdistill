# T18 Region-Leak Inference Policy

Status: fixed and smoke-measured on 2026-07-04.

## Policy

`FFFLinear.region_leak` is now a train-only regularizer. Evaluation and inference
use `effective_region_leak=0.0`, even when the configured training value is
nonzero. This preserves hard-routed conditional inference and avoids silently
turning selected-leaf grouped inference into dense all-leaf evaluation.

Diagnostics now report:

- `region_leak`: configured training value.
- `effective_region_leak`: value actually used for the current train/eval mode.
- `region_leak_policy`: currently `train_only`.
- `grouped_leaf_path`: selected-leaf versus all-leaf execution path.

## CUDA BF16 Eval Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m cifar_mamba_fff.benchmark_fff \
  --quick-smoke false --device cuda:0 --dtype bfloat16 \
  --batch-size 1024 --in-features 192 --out-features 192 \
  --depth 5 --shared-rows 38 --route-rows 1 --route-result-rows 2 \
  --leaf-rows 4 --route-row-role split_routing_output \
  --route-rows-output-count all --region-leak 0.01 --eval-mode true \
  --include-naive true --iterations 10 --warmup 3 --json true
```

Result summary:

| Path | Tokens/s | Region leak | Effective leak | Grouped leaf path |
| --- | ---: | ---: | ---: | --- |
| dense | 135657887.9 | 0.01 | 0.0 | selected_leaf |
| fff_grouped | 2048182.7 | 0.01 | 0.0 | selected_leaf |
| fff_naive | 1964.5 | 0.01 | 0.0 | selected_leaf |

Grouped-vs-naive BF16 max absolute difference was `0.0078125`.

This is a bounded inference-policy smoke, not a full T16 throughput study.
