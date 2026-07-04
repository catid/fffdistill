from __future__ import annotations

import argparse
from pathlib import Path

from .utils import bool_arg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    report = Path("docs/final_report.md")
    if not report.exists():
        raise FileNotFoundError("docs/final_report.md is missing")
    print(f"report present: {report}")
    if args.quick_smoke:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
