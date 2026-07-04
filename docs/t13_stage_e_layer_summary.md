# T13 Stage E Representative-Layer Summary

- Run id: `distill_stage_e_layers_20260704_091514`
- Git commit: `c43d243223a05bd648f625299254543b42d6500b`
- Data access: CIFAR-10 validation split sampling only; every summary reports `test_accessed=false`.
- Scope: Stage D winning architecture fixed; early/middle/late eligible Linear indices `[0]`, `[32]`, `[60]`; one layer per GPU job; 150 steps; LocoProp-S enabled.

| Offset | Layer Index | Layer | Slot | Final NMSE | Cosine | Tokens/s | Dead leaves |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | `[0]` | `blocks.0.forward_block.mixer.mixer.in_proj` | `work:0` | 0.004614 | 0.997881 | 35418.4 | 17 |
| 1 | `[32]` | `blocks.8.forward_block.mixer.mixer.in_proj` | `work:1` | 0.072066 | 0.964517 | 38259.2 | 10 |
| 2 | `[60]` | `blocks.15.forward_block.mixer.mixer.in_proj` | `ripper:0` | 0.041259 | 0.977644 | 43985.9 | 21 |

## Notes

- This validates that the Stage D recipe works beyond the first layer, but middle/late layers need additional tuning before full-layer distillation claims.
- No CIFAR-10 test data was accessed.
