"""CSV exports of advisory comparisons, with literal spreadsheet-safe strings."""

from __future__ import annotations

import csv
from os import PathLike
from pathlib import Path
from typing import Any

from .scenarios import ScenarioComparison


def _literal(value: Any) -> Any:
    # CSV quoting alone does not stop spreadsheet applications evaluating formulas.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def write_comparison_csv(comparison: ScenarioComparison, path: str | PathLike[str]) -> None:
    """Export flat ranked rows. Strings beginning with formula markers stay literal."""
    rows = comparison.rows()
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _literal(value) for key, value in row.items()} for row in rows)
