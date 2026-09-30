"""Strict calibration configuration and spreadsheet-safe candidate exports."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .calibration import PenaltyCandidate
from .export import _literal
from .intake import _error, _json_policy, _read_json
from .optimizer import Policy, validate_inputs

WEIGHT_FIELDS = frozenset(
    {"late_penalty", "substitution_penalty", "disruption_penalty", "unfilled_penalty"}
)


def load_calibration_config(
    path: str | Path,
) -> tuple[tuple[PenaltyCandidate, ...], Policy, dict[str, float]]:
    """Read explicit fixed evaluator weights and candidate search settings."""
    source = Path(path)
    data = _read_json(source)
    try:
        required = {"schema_version", "scoring_weights", "candidates"}
        if (
            not isinstance(data, dict)
            or required - set(data)
            or set(data) - required - {"constraints"}
        ):
            raise ValueError("Invalid calibration configuration fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("schema_version must be integer 1")
        weights = data["scoring_weights"]
        if not isinstance(weights, dict) or set(weights) != WEIGHT_FIELDS:
            raise ValueError("scoring_weights must contain exactly the four penalty fields")
        policy = _json_policy(weights)
        validate_inputs([], [], policy)
        rows = data["candidates"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("candidates must be a nonempty array")
        candidates = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != WEIGHT_FIELDS | {"name"}:
                raise ValueError("Each candidate requires a name and all four penalty fields")
            candidates.append(PenaltyCandidate(**row))
        if len({candidate.name for candidate in candidates}) != len(candidates):
            raise ValueError("Candidate names must be unique")
        constraints = data.get("constraints", {})
        if not isinstance(constraints, dict) or set(constraints) - {
            "min_fill_rate",
            "max_late_rate",
        }:
            raise ValueError("Unknown service constraints")
        for key, value in constraints.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be a finite number between 0 and 1")
        return tuple(candidates), policy, dict(constraints)
    except (ValueError, TypeError) as exc:
        raise _error(source, 1, str(exc)) from exc


def calibration_config_to_dict(
    candidates: tuple[PenaltyCandidate, ...],
    scoring_policy: Policy,
    *,
    min_fill_rate: float = 0.0,
    max_late_rate: float = 1.0,
) -> dict[str, Any]:
    """Describe supplied search settings; the loader validates persisted configs."""
    return {
        "schema_version": 1,
        "scoring_weights": {
            field: getattr(scoring_policy, field) for field in sorted(WEIGHT_FIELDS)
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "constraints": {"min_fill_rate": min_fill_rate, "max_late_rate": max_late_rate},
    }


def write_calibration_csv(report: dict[str, Any], path: str | Path) -> None:
    """Export candidate ranks using the calibration partition only."""
    rows = report["calibration"]["rankings"]
    flattened = []
    for rank, row in enumerate(rows, start=1):
        flat = {
            key: value
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        flat.update({"rank": rank, "selected": row["name"] == report["selected_candidate"]})
        flat.update({f"penalty_{key}": value for key, value in row["penalties"].items()})
        flattened.append(flat)
    if not flattened:
        raise ValueError("Calibration report has no candidate rows")
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _literal(value) for key, value in row.items()} for row in flattened)
