from __future__ import annotations

from pathlib import Path

import pytest

from cifar_mamba_fff.make_report import validate_final_report


def test_validate_final_report_accepts_committed_report() -> None:
    validate_final_report(Path("docs/final_report.md"))


def test_validate_final_report_rejects_missing_sections(tmp_path: Path) -> None:
    report = tmp_path / "final_report.md"
    report.write_text("# Final Report\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required snippets"):
        validate_final_report(report)
