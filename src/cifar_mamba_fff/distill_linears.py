from __future__ import annotations

import argparse
from pathlib import Path

from .utils import RunContext, bool_arg, load_yaml, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fff_distill_default.yaml")
    parser.add_argument("--output-dir", default="outputs/distill")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    config = load_yaml(args.config)
    context = RunContext(Path(args.output_dir), seed=int(config.get("seed", 1337)), quick_smoke=args.quick_smoke)
    context.prepare()
    write_json(context.output_dir / "run_context.json", context.metadata() | {"config": config})
    if args.quick_smoke:
        print("distillation quick smoke metadata written")
        return 0
    raise RuntimeError("layerwise distillation is blocked until teacher and FFF gates pass")


if __name__ == "__main__":
    raise SystemExit(main())
