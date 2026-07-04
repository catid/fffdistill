from __future__ import annotations

import argparse
from pathlib import Path

from .utils import bool_arg

REQUIRED_SNIPPETS = (
    "## References",
    "## Environment",
    "## Beads And Git",
    "## Teacher",
    "## FFF Validation",
    "## STE And Routing",
    "## Architecture Sweep",
    "## Layerwise Distillation",
    "## End-To-End KD",
    "## Fairness",
    "## Pareto Inputs",
    "## Best Recipes",
    "## Limitations",
    "https://arxiv.org/pdf/2603.15569",
    "https://github.com/state-spaces/mamba",
    "https://github.com/KellerJordan/Muon",
    "https://github.com/pbelcak/fastfeedforward",
    "https://arxiv.org/pdf/2106.06199",
    "Stage C compared seven router recipes",
    "Stage D swept 10 one-layer architecture/recipe trials",
    "test_accessed=true",
)


def validate_final_report(report: Path = Path("docs/final_report.md")) -> None:
    if not report.exists():
        raise FileNotFoundError("docs/final_report.md is missing")
    text = report.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        raise ValueError("final report is missing required snippets: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-smoke", type=bool_arg, default=False)
    args = parser.parse_args()
    report = Path("docs/final_report.md")
    validate_final_report(report)
    print(f"report present: {report}")
    if args.quick_smoke:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
