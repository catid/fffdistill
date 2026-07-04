from __future__ import annotations

import argparse

from .utils import bool_arg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    if args.quick_smoke:
        print("FFF benchmark quick smoke placeholder")
        return 0
    raise RuntimeError("FFF benchmarking is blocked until FFF correctness tests pass")


if __name__ == "__main__":
    raise SystemExit(main())
