"""Strict local CSV/JSON intake; no inferred columns, ERP access, or solver calls."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re
from typing import Any

from .optimizer import Order, Policy, Supply, validate_inputs


@dataclass(frozen=True)
class ScenarioData:
    supplies: tuple[Supply, ...]
    orders: tuple[Order, ...]
    policy: Policy

    def to_dict(self) -> dict[str, Any]:
        # JSON roundtrip turns the policy's tuple-valued substitutions into arrays.
        return json.loads(
            json.dumps(
                {
                    "supplies": [asdict(s) for s in self.supplies],
                    "orders": [asdict(o) for o in self.orders],
                    "policy": asdict(self.policy),
                },
                allow_nan=False,
            )
        )


def _error(path: Path, line: int, message: str) -> ValueError:
    return ValueError(f"{path}: line {line}: {message}")


def _read_csv(path: Path, kind: type[Supply] | type[Order]) -> tuple:
    optional = "unit_cost" if kind is Supply else "priority"
    default = 0 if kind is Supply else 1
    expected = {f.name for f in fields(kind)}
    records = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise _error(path, 1, "CSV header is required") from exc
        except csv.Error as exc:
            raise _error(path, reader.line_num or 1, str(exc)) from exc
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            raise _error(path, 1, "Duplicate CSV headers")
        missing, unknown = (expected - {optional}) - set(header), set(header) - expected
        if missing or unknown:
            raise _error(
                path, 1, f"Invalid headers: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        try:
            for row in reader:
                line = reader.line_num
                if len(row) != len(header):
                    raise _error(path, line, f"Expected {len(header)} cells, received {len(row)}")
                values: dict[str, Any] = dict(zip(header, (value.strip() for value in row)))
                for name, value in values.items():
                    if not value:
                        raise _error(path, line, f"{name} must not be blank")
                    if name not in {"id", "sku"}:
                        if re.fullmatch(r"[0-9]+", value) is None:
                            raise _error(path, line, f"{name} must be an unsigned decimal integer")
                        try:
                            values[name] = int(value)
                        except ValueError as exc:
                            raise _error(
                                path, line, f"{name} integer is too large to parse"
                            ) from exc
                values.setdefault(optional, default)
                record = kind(**values)
                # Usually validate one record; include the prefix only for a duplicate
                # so the domain validator supplies its canonical error at this row.
                to_validate = records + [record] if record.id in seen_ids else [record]
                try:
                    validate_inputs(
                        to_validate if kind is Supply else [],
                        to_validate if kind is Order else [],
                        Policy(),
                    )
                except ValueError as exc:
                    raise _error(path, line, str(exc)) from exc
                records.append(record)
                seen_ids.add(record.id)
        except csv.Error as exc:
            raise _error(path, reader.line_num or 1, str(exc)) from exc
    return tuple(records)


def load_csv_scenario(
    supplies_path: str | Path, orders_path: str | Path, policy: Policy | None = None
) -> ScenarioData:
    """Load strict UTF-8 CSVs; optional columns default only when absent.

    Day indices and quantities use unsigned decimal integers. A header without
    data rows is valid. Blank records, unknown columns, and implicit coercions
    are rejected. Existing local files are read only.
    """
    supply_path, order_path = Path(supplies_path), Path(orders_path)
    supplies = _read_csv(supply_path, Supply)
    orders = _read_csv(order_path, Order)
    active_policy = policy if policy is not None else Policy()
    if not isinstance(active_policy, Policy):
        raise _error(supply_path, 1, "policy must be a Policy instance")
    try:
        validate_inputs(supplies, orders, active_policy)
    except ValueError as exc:
        raise _error(supply_path, 1, f"Invalid policy or scenario: {exc}") from exc
    return ScenarioData(supplies, orders, active_policy)


def _json_policy(value: Any) -> Policy:
    if not isinstance(value, dict):
        raise ValueError("policy must be an object")
    unknown = set(value) - {f.name for f in fields(Policy)}
    if unknown:
        raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
    config = dict(value)
    substitutions = config.get("substitutions", {})
    if not isinstance(substitutions, dict):
        raise ValueError("policy.substitutions must be an object of string arrays")
    if any(
        not isinstance(items, list) or any(not isinstance(v, str) for v in items)
        for items in substitutions.values()
    ):
        raise ValueError("policy.substitutions values must be arrays of strings")
    config["substitutions"] = {key: tuple(items) for key, items in substitutions.items()}
    return Policy(**config)


def _read_json(source: Path) -> Any:
    """Read strict JSON without silently discarding duplicate object members."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle, object_pairs_hook=object_pairs, parse_constant=nonfinite)
        return data
    except json.JSONDecodeError as exc:
        raise _error(source, exc.lineno, exc.msg) from exc
    except (ValueError, TypeError) as exc:
        raise _error(source, 1, str(exc)) from exc


def load_policy_json(path: str | Path) -> Policy:
    """Load an explicit policy object with strict JSON and domain validation."""
    source = Path(path)
    data = _read_json(source)
    try:
        policy = _json_policy(data)
        validate_inputs([], [], policy)
        return policy
    except (ValueError, TypeError) as exc:
        raise _error(source, 1, str(exc)) from exc


def load_json_scenario(path: str | Path) -> ScenarioData:
    """Load the explicit supplies/orders/policy schema, rejecting duplicate keys."""
    source = Path(path)
    data = _read_json(source)
    try:
        if not isinstance(data, dict) or not {"supplies", "orders"}.issubset(data):
            raise ValueError("Scenario requires supplies and orders arrays")
        if set(data) - {"supplies", "orders", "policy"}:
            raise ValueError("Unknown scenario fields")
        parsed = []
        for name, kind, optional in (
            ("supplies", Supply, "unit_cost"),
            ("orders", Order, "priority"),
        ):
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be an array")
            expected = {f.name for f in fields(kind)}
            records = []
            for index, item in enumerate(data[name]):
                if not isinstance(item, dict):
                    raise ValueError(f"{name}[{index}] must be an object")
                if set(item) - expected or (expected - {optional}) - set(item):
                    raise ValueError(f"Invalid fields in {name}[{index}]")
                records.append(kind(**item))
            parsed.append(tuple(records))
        policy = _json_policy(data.get("policy", {}))
        supplies, orders = parsed
        validate_inputs(supplies, orders, policy)
        return ScenarioData(supplies, orders, policy)
    except json.JSONDecodeError as exc:
        raise _error(source, exc.lineno, exc.msg) from exc
    except (ValueError, TypeError) as exc:
        raise _error(source, 1, str(exc)) from exc
