from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from .fairness import (
    EXPECTED_FAIRNESS_ROWS,
    EXPECTED_TEST_ACCESS_ROWS,
    FAIRNESS_COLUMNS,
    build_fairness_rows,
    validate_fairness_rows,
)
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


def _csv_text(value: object) -> str:
    return "" if value is None else str(value)


def _canonical_evidence(value: object) -> str:
    text = _csv_text(value)
    try:
        return str(Path(text).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return text


def _canonical_fairness_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    for row in rows:
        canonical.append(
            {
                column: (
                    _canonical_evidence(row.get(column, ""))
                    if column == "evidence"
                    else _csv_text(row.get(column, ""))
                )
                for column in FAIRNESS_COLUMNS
            }
        )
    return canonical


def validate_final_report(report: Path = Path("docs/final_report.md")) -> None:
    if not report.exists():
        raise FileNotFoundError("docs/final_report.md is missing")
    text = report.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        raise ValueError("final report is missing required snippets: " + ", ".join(missing))
    fairness_csv = report.parent / "t15_fairness_summary.csv"
    if not fairness_csv.exists():
        raise FileNotFoundError(f"fairness CSV is missing: {fairness_csv}")
    with fairness_csv.open("r", encoding="utf-8", newline="") as handle:
        fairness_rows = list(csv.DictReader(handle))
    validate_fairness_rows(fairness_rows)
    if len(fairness_rows) != EXPECTED_FAIRNESS_ROWS:
        raise ValueError(
            f"fairness CSV expected {EXPECTED_FAIRNESS_ROWS} rows, found {len(fairness_rows)}"
        )
    test_rows = [
        row
        for row in fairness_rows
        if str(row.get("test_accessed", "")).strip().lower() == "true"
    ]
    if len(test_rows) != EXPECTED_TEST_ACCESS_ROWS:
        raise ValueError(
            f"fairness CSV expected {EXPECTED_TEST_ACCESS_ROWS} test-access rows, found {len(test_rows)}"
        )
    bad_test_rows = [
        row for row in test_rows if row.get("split") not in {"final_test", "partial_final_test"}
    ]
    if bad_test_rows:
        raise ValueError("fairness CSV has test access outside final/partial-final rows")
    regenerated_rows = _canonical_fairness_rows(build_fairness_rows(report.parent))
    committed_rows = _canonical_fairness_rows(fairness_rows)
    if committed_rows != regenerated_rows:
        raise ValueError("fairness CSV is stale relative to source evidence")


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
