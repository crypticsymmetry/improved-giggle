"""Strict local business CSV bundles and spreadsheet-safe advisory exports."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

from .business import (
    BusinessPilot,
    CatalogItem,
    OpenOrderObservation,
    PhysicalMovement,
    StockSnapshot,
    compile_business_pilot,
)
from .export import _literal
from .intake import _error, _read_json, load_policy_json
from .pilot import _case_path, _keys


def _rows(path: Path, headers: tuple[str, ...], numeric: set[str]) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise _error(path, 1, "CSV header is required") from exc
        except csv.Error as exc:
            raise _error(path, reader.line_num or 1, str(exc)) from exc
        header = [value.strip() for value in header]
        if len(header) != len(set(header)) or set(header) != set(headers):
            raise _error(path, 1, f"CSV requires exactly the unique headers {list(headers)}")
        try:
            for row in reader:
                if len(row) != len(header):
                    raise _error(path, reader.line_num, f"Expected {len(header)} cells")
                values: dict[str, Any] = dict(zip(header, (cell.strip() for cell in row)))
                for name, value in values.items():
                    if not value:
                        raise _error(path, reader.line_num, f"{name} must not be blank")
                    if name in numeric:
                        if re.fullmatch(r"[0-9]+", value) is None:
                            raise _error(
                                path,
                                reader.line_num,
                                f"{name} must be an unsigned decimal integer",
                            )
                        try:
                            values[name] = int(value)
                        except ValueError as exc:
                            raise _error(path, reader.line_num, f"{name} is too large") from exc
                values["_line"] = reader.line_num
                records.append(values)
        except csv.Error as exc:
            raise _error(path, reader.line_num or 1, str(exc)) from exc
    return records


def _unique_rows(path: Path, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    seen = set()
    for row in rows:
        identifier = tuple(row[key] for key in keys)
        if identifier in seen:
            raise _error(
                path, row["_line"], f"Duplicate record for {', '.join(keys)}: {identifier}"
            )
        seen.add(identifier)


def load_business_pilot(path: str | Path) -> BusinessPilot:
    """Read and validate a version-1 manifest, with no inferred business semantics.

    Every referenced file must reside within the manifest directory. Integer
    quantities and explicit sources are mandatory; timestamps are validated by
    the business compiler. Files are read only and no work is dispatched.
    """
    source = Path(path)
    manifest = _read_json(source)
    try:
        _keys(
            manifest,
            {
                "schema_version",
                "name",
                "group_id",
                "synthetic",
                "catalog_path",
                "stock_snapshots_path",
                "movements_path",
                "open_orders_path",
                "policy_path",
            },
            "business manifest",
        )
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("schema_version must be integer 1")
        directory = source.resolve().parent
        catalog_path = _case_path(directory, manifest["catalog_path"])
        stock_path = _case_path(directory, manifest["stock_snapshots_path"])
        movement_path = _case_path(directory, manifest["movements_path"])
        orders_path = _case_path(directory, manifest["open_orders_path"])
        policy_path = _case_path(directory, manifest["policy_path"])
        catalog_rows = _rows(catalog_path, ("sku", "unit_cost"), {"unit_cost"})
        _unique_rows(catalog_path, catalog_rows, ("sku",))
        catalog = tuple(CatalogItem(row["sku"], row["unit_cost"]) for row in catalog_rows)
        stock_rows = _rows(
            stock_path, ("snapshot_id", "observed_at", "sku", "quantity", "source"), {"quantity"}
        )
        _unique_rows(stock_path, stock_rows, ("snapshot_id", "sku"))
        grouped: dict[str, dict[str, Any]] = {}
        for row in stock_rows:
            identifier = row["snapshot_id"]
            if identifier not in grouped:
                grouped[identifier] = {
                    "observed_at": row["observed_at"],
                    "source": row["source"],
                    "quantities": {},
                }
            group = grouped[identifier]
            if (group["observed_at"], group["source"]) != (row["observed_at"], row["source"]):
                raise _error(stock_path, row["_line"], "Snapshot rows disagree on time or source")
            group["quantities"][row["sku"]] = row["quantity"]
        snapshots = tuple(
            StockSnapshot(identifier, group["observed_at"], group["quantities"], group["source"])
            for identifier, group in grouped.items()
        )
        movement_rows = _rows(
            movement_path,
            ("movement_id", "event_time", "observed_at", "sku", "kind", "quantity", "source"),
            {"quantity"},
        )
        _unique_rows(movement_path, movement_rows, ("movement_id",))
        movements = tuple(
            PhysicalMovement(**{key: value for key, value in row.items() if key != "_line"})
            for row in movement_rows
        )
        order_rows = _rows(
            orders_path,
            (
                "order_id",
                "event_time",
                "observed_at",
                "sku",
                "quantity",
                "due_date",
                "priority",
                "source",
            ),
            {"quantity", "priority"},
        )
        _unique_rows(orders_path, order_rows, ("order_id", "observed_at"))
        orders = tuple(
            OpenOrderObservation(**{key: value for key, value in row.items() if key != "_line"})
            for row in order_rows
        )
        result = BusinessPilot(
            manifest["name"],
            manifest["group_id"],
            manifest["synthetic"],
            catalog,
            snapshots,
            movements,
            orders,
            load_policy_json(policy_path),
        )
        compile_business_pilot(result)
        return result
    except (ValueError, TypeError, OSError) as exc:
        raise _error(source, 1, str(exc)) from exc


def write_business_csv(report: dict[str, Any], path: str | Path) -> None:
    """Export per-snapshot method comparisons; weighted score is not revenue."""
    columns = (
        "snapshot_id",
        "cutoff",
        "method",
        "status",
        "requested_units",
        "filled_units",
        "on_time_units",
        "late_units",
        "unfilled_units",
        "weighted_score",
    )
    rows = []
    for snapshot in report["snapshots"]:
        for result in snapshot["comparison"]["results"]:
            evaluation = result["evaluation"]
            metrics = evaluation["metrics"]
            rows.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "cutoff": snapshot["cutoff"],
                    "method": result["method"],
                    "status": result["status"],
                    "requested_units": metrics["requested_units"],
                    "filled_units": metrics["allocated_units"],
                    "on_time_units": metrics["on_time_units"],
                    "late_units": metrics["late_units"],
                    "unfilled_units": metrics["unfilled_units"],
                    "weighted_score": evaluation["costs"]["total"],
                }
            )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _literal(value) for key, value in row.items()} for row in rows)
