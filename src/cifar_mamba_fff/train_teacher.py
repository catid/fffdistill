from __future__ import annotations

import argparse
from pathlib import Path

from .utils import RunContext, bool_arg, load_yaml, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher_default.yaml")
    parser.add_argument("--output-dir", default="outputs/teacher")
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 1337))
    context = RunContext(Path(args.output_dir), seed=seed, quick_smoke=args.quick_smoke)
    context.prepare()
    write_json(context.output_dir / "run_context.json", context.metadata() | {"config": config})
    if args.quick_smoke:
        print("teacher quick smoke metadata written")
        return 0
    raise RuntimeError("teacher training is blocked until T01-T05 gates pass")


if __name__ == "__main__":
    raise SystemExit(main())
