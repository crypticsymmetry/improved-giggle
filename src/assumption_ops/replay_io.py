"""Strict replay-case JSON and spreadsheet-safe, per-snapshot report exports."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

from .events import EvidenceUpdate, _snapshot_updates
from .export import _literal
from .intake import _error, _json_policy, _read_json
from .optimizer import Order, Supply, validate_inputs
from .replay import ReplayBatch, ReplayCase


def _keys(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if required - set(value) or set(value) - required - optional:
        raise ValueError(f"Invalid fields in {label}")


def replay_case_to_dict(case: ReplayCase) -> dict[str, Any]:
    """Return the versioned interchange schema with explicit provenance."""
    if not isinstance(case, ReplayCase):
        raise ValueError("case must be a ReplayCase")
    # Frozen dataclasses can still hold mutable nested policy/update mappings.
    # Validate before JSON normalization could coerce invalid object keys.
    case.__post_init__()
    return json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "name": case.name,
                "synthetic": case.synthetic,
                "seed": case.seed,
                "initial": {
                    "supplies": [asdict(record) for record in case.supplies],
                    "orders": [asdict(record) for record in case.orders],
                    "policy": asdict(case.policy),
                },
                "batches": [
                    {
                        "event_id": batch.event_id,
                        "updates": [asdict(update) for update in batch.updates],
                    }
                    for batch in case.batches
                ],
            },
            allow_nan=False,
        )
    )


def _parse_case(data: Any) -> ReplayCase:
    _keys(data, {"schema_version", "name", "synthetic", "initial", "batches"}, {"seed"}, "case")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("name must be a nonempty string")
    if type(data["synthetic"]) is not bool:
        raise ValueError("synthetic must be an explicit boolean")
    seed = data.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or null")
    initial = data["initial"]
    _keys(initial, {"supplies", "orders", "policy"}, set(), "initial")
    parsed = []
    for name, kind, optional in (("supplies", Supply, "unit_cost"), ("orders", Order, "priority")):
        if not isinstance(initial[name], list):
            raise ValueError(f"initial.{name} must be an array")
        required = {field.name for field in fields(kind)} - {optional}
        records = []
        for index, record in enumerate(initial[name]):
            _keys(record, required, {optional}, f"initial.{name}[{index}]")
            records.append(kind(**record))
        parsed.append(tuple(records))
    supplies, orders = parsed
    policy = _json_policy(initial["policy"])
    validate_inputs(supplies, orders, policy)
    if not isinstance(data["batches"], list):
        raise ValueError("batches must be an array")
    known = {
        f"supply:{record.id}": ({"sku", "quantity", "available_day", "unit_cost"}, record)
        for record in supplies
    }
    known.update(
        {
            f"order:{record.id}": ({"sku", "quantity", "due_day", "priority"}, record)
            for record in orders
        }
    )
    batches = []
    event_ids: set[str] = set()
    for index, batch in enumerate(data["batches"]):
        _keys(batch, {"event_id", "updates"}, set(), f"batches[{index}]")
        if not isinstance(batch["event_id"], str) or not batch["event_id"].strip():
            raise ValueError("event_id must be a nonempty string")
        if batch["event_id"] in event_ids:
            raise ValueError(f"Duplicate replay batch event_id: {batch['event_id']}")
        event_ids.add(batch["event_id"])
        if not isinstance(batch["updates"], list):
            raise ValueError("updates must be an array")
        updates = []
        for update in batch["updates"]:
            _keys(update, {"entity", "field", "value", "source"}, set(), "update")
            updates.append(EvidenceUpdate(**update))
        copied, _ = _snapshot_updates(updates)
        # Validate every chronological absolute replacement using the same domain
        # rules as planning. Loading never solves or accepts into an evidence store.
        for update in copied:
            if update.entity == "policy" and update.field == "config":
                policy = _json_policy(update.value)
                validate_inputs([], [], policy)
                continue
            if update.entity not in known or update.field not in known[update.entity][0]:
                raise ValueError(
                    f"Unknown scenario entity or field: {update.entity}.{update.field}"
                )
            allowed, record = known[update.entity]
            values = asdict(record) | {update.field: update.value}
            replacement = type(record)(**values)
            validate_inputs(
                [replacement] if isinstance(replacement, Supply) else [],
                [replacement] if isinstance(replacement, Order) else [],
                policy,
            )
            known[update.entity] = (allowed, replacement)
        batches.append(ReplayBatch(batch["event_id"], copied))
    # The case retains its initial policy, rather than the final validation state.
    return ReplayCase(
        data["name"],
        supplies,
        orders,
        _json_policy(initial["policy"]),
        tuple(batches),
        data["synthetic"],
        seed,
    )


def load_replay_case(path: str | Path) -> ReplayCase:
    """Load strict UTF-8/BOM JSON, preserving chronological reviewed batches."""
    source = Path(path)
    data = _read_json(source)
    try:
        return _parse_case(data)
    except (ValueError, TypeError) as exc:
        raise _error(source, 1, str(exc)) from exc


def save_replay_case(case: ReplayCase, path: str | Path) -> None:
    """Validate the exact export schema before writing a local JSON file."""
    data = replay_case_to_dict(case)
    _parse_case(data)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_replay_report_csv(report: dict[str, Any], path: str | Path) -> None:
    """Export each strategy at each snapshot, without summing repeated demand.

    Solver statuses, timings, costs, and metrics remain separate columns. A CSV
    row represents a snapshot comparison, never a new executed shipment.
    """
    rows = []
    for step in report["steps"]:
        for strategy in ("optimized", "greedy"):
            result = step[strategy]
            row = {
                "case_name": report.get("case_name", report.get("name", "")),
                "step": step["step"],
                "event_id": step["event_id"],
                "strategy": strategy,
                "synthetic": report.get("synthetic"),
                "seed": report.get("seed"),
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            for name in (
                "confirmed_fields",
                "proposed_decision_lines",
                "previous_decisions_invalidated",
                "changed_allocation_arcs",
                "changed_allocation_units",
                "duplicate_replay_verified",
            ):
                row[name] = step[name]
            for category in ("costs", "metrics"):
                for name, value in result["evaluation"][category].items():
                    if isinstance(value, (dict, list, tuple)):
                        raise ValueError(f"Report {category}.{name} must be a scalar")
                    row[f"{category}_{name}"] = value
            rows.append(row)
    if not rows:
        raise ValueError("Replay report must contain at least one snapshot")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _literal(value) for key, value in row.items()} for row in rows)
