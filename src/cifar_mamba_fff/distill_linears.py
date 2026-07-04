from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch import nn

from .models.replacement import (
    discover_linear_layers,
    linear_reports_as_log_records,
    select_progressive_reports,
)
from .utils import RunContext, bool_arg, load_yaml, write_json


def linear_replacement_plan(
    model: nn.Module,
    config: dict[str, Any],
    *,
    progressive_step: int | None = None,
    progressive_step_size: int = 1,
) -> dict[str, Any]:
    eligible_config = config.get("eligible_linear", {})
    if not isinstance(eligible_config, dict):
        raise ValueError("eligible_linear config must be a mapping")

    reports = discover_linear_layers(
        model,
        min_in_features=int(eligible_config.get("min_in_features", 64)),
        min_out_features=int(eligible_config.get("min_out_features", 64)),
    )
    selected = select_progressive_reports(
        reports,
        step=progressive_step,
        step_size=progressive_step_size,
    )
    return {
        "linear_layers": linear_reports_as_log_records(reports),
        "selected_replacements": [report.name for report in selected],
        "progressive": {
            "step": progressive_step,
            "step_size": progressive_step_size,
            "selected_count": len(selected),
            "eligible_count": sum(report.included for report in reports),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fff_distill_default.yaml")
    parser.add_argument("--output-dir", default="outputs/distill")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    parser.add_argument("--progressive-step", type=int, default=None)
    parser.add_argument("--progressive-step-size", type=int, default=1)
    args = parser.parse_args()
    select_progressive_reports(
        [],
        step=args.progressive_step,
        step_size=args.progressive_step_size,
    )
    config = load_yaml(args.config)
    context = RunContext(
        Path(args.output_dir),
        seed=int(config.get("seed", 1337)),
        quick_smoke=args.quick_smoke,
    )
    context.prepare()
    write_json(
        context.output_dir / "run_context.json",
        context.metadata()
        | {
            "config": config,
            "progressive_step": args.progressive_step,
            "progressive_step_size": args.progressive_step_size,
            "progressive_args_validated": True,
        },
    )
    if args.quick_smoke:
        print("distillation quick smoke metadata written")
        return 0
    raise RuntimeError("layerwise distillation is blocked until teacher and FFF gates pass")


if __name__ == "__main__":
    raise SystemExit(main())
