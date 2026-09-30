"""Pinned industrial warehouse data, with explicit consent for inferred columns.

The source supports a storage-assignment study. Our SKU/day projection omits
lane geometry, batch constraints, travel, and dispatch; it is not a replication
of that study. The demand table's physical names do not match its contents.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import md5, sha256
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import urlopen

from .intake import ScenarioData
from .optimizer import Order, Policy, Supply, validate_inputs

WAREHOUSE_URL = (
    "https://zenodo.org/api/records/18229759/files/"
    "Dynamic%20storage%20assignment%20in%20homogeneous%20dual-access%20deep-lane%20SR%20systems_DB.accdb/content"
)
WAREHOUSE_SHA256 = "4db26e1552f495f840e5c61e5d80fc7ed51bd28e1af6e10bd04b606d164f4548"
WAREHOUSE_MD5 = "3844b06c3cb5326084c6dab1bdc284f0"
WAREHOUSE_DOI = "10.5281/zenodo.18229759"
WAREHOUSE_LICENSE = "CC BY 4.0"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


def _verify(path: Path) -> None:
    if path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError("Warehouse file exceeds the 5 MiB size limit")
    digest = sha256()
    published = md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
            published.update(chunk)
    if digest.hexdigest() != WAREHOUSE_SHA256 or published.hexdigest() != WAREHOUSE_MD5:
        raise ValueError("Warehouse file checksum mismatch; expected the pinned Zenodo source")


def download_warehouse(path: str | Path) -> Path:
    """Download only the pinned source, bounded and atomically checksum-verified.

    Existing cache entries are verified and never silently replaced when invalid.
    Temporary files are removed on network, size, or checksum failures.
    """
    destination = Path(path)
    if destination.exists():
        _verify(destination)
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with urlopen(WAREHOUSE_URL, timeout=30) as response:
            with NamedTemporaryFile(
                dir=destination.parent, prefix="warehouse-", suffix=".tmp", delete=False
            ) as output:
                temporary = Path(output.name)
                length = 0
                while chunk := response.read(65536):
                    length += len(chunk)
                    if length > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Warehouse download exceeds the 5 MiB size limit")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        _verify(temporary)
        os.replace(temporary, destination)
        return destination.resolve()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _integer(value: Any, label: str) -> int:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or int(value) != value
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative integral number")
    return int(value)


def _sku(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty SKU string")
    return value


@dataclass(frozen=True)
class WarehouseRecord:
    day: int
    sku: str
    quantity: int

    def __post_init__(self) -> None:
        for field in ("day", "quantity"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        _sku(self.sku, "record.sku")


@dataclass(frozen=True)
class WarehouseData:
    initial_inventory: dict[str, int]
    production: tuple[WarehouseRecord, ...]
    demand: tuple[WarehouseRecord, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_inventory, dict):
            raise ValueError("initial_inventory must be a SKU-to-quantity dictionary")
        for sku, quantity in self.initial_inventory.items():
            _sku(sku, "initial inventory SKU")
            if type(quantity) is not int or quantity < 0:
                raise ValueError("Initial inventory quantities must be nonnegative integers")
        for records in (self.production, self.demand):
            if not isinstance(records, tuple) or any(
                not isinstance(record, WarehouseRecord) for record in records
            ):
                raise ValueError("production and demand must be tuples of WarehouseRecord")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dictionary")
        object.__setattr__(self, "initial_inventory", dict(self.initial_inventory))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))


def _rows(table: Any, expected: set[str], count: int, name: str) -> list[dict[str, Any]]:
    if not isinstance(table, dict) or set(table) != expected:
        raise ValueError(f"Unexpected physical schema in {name}")
    if any(not isinstance(values, list) or len(values) != count for values in table.values()):
        raise ValueError(f"Unexpected row count in {name}; expected {count}")
    return [dict(zip(table, values)) for values in zip(*table.values())]


def _normalize_tables(tables: dict[str, Any]) -> WarehouseData:
    columns = {"T", "Q", "item_SKU", "item_BATCH", "item_MONTH", "item_WEEK"}
    q = _rows(tables["Q"], columns, 368, "Q")
    d = _rows(tables["D"], columns, 1257, "D")
    inventory = _rows(
        tables["h_LAST"],
        {
            "T",
            "Z",
            "SIDE",
            "item_SKU",
            "h_LAST_SKU",
            "item_BATCH",
            "h_LAST_BATCH",
            "item_MONTH",
            "h_LAST_MONTH",
            "item_WEEK",
            "h_LAST_WEEK",
        },
        780,
        "h_LAST",
    )
    sku_rows = _rows(tables["ITEM_SKU"], {"SKU"}, 101, "ITEM_SKU")
    allowed = {_sku(row["SKU"], "ITEM_SKU.SKU") for row in sku_rows}
    if len(allowed) != 101:
        raise ValueError("ITEM_SKU must contain 101 distinct SKU labels")
    initial: dict[str, int] = defaultdict(int)
    lanes = set()
    for row in inventory:
        if _integer(row["T"], "h_LAST.T") != 0:
            raise ValueError("Initial inventory must be at day zero")
        lane = (row["Z"], row["SIDE"])
        if lane in lanes or row["SIDE"] not in ("dx", "sx"):
            raise ValueError("Initial inventory lane/side keys must be unique and valid")
        lanes.add(lane)
        quantity = _integer(row["h_LAST_SKU"], "h_LAST.h_LAST_SKU")
        sku = row["item_SKU"]
        if sku == "NaN" and quantity == 0:
            continue
        if sku not in allowed:
            raise ValueError("Initial inventory references an unknown SKU")
        initial[sku] += quantity
    production = []
    for row in q:
        day = _integer(row["T"], "Q.T")
        if day > 130 or row["item_SKU"] not in allowed:
            raise ValueError("Q references an invalid day or SKU")
        production.append(WarehouseRecord(day, row["item_SKU"], _integer(row["Q"], "Q.Q")))
    demand = []
    for row in d:
        day = _integer(row["T"], "D.T")
        sku = str(_integer(row["Q"], "D.Q (inferred SKU code)"))
        if not 1 <= day <= 130 or sku not in allowed:
            raise ValueError("D inferred mapping references an invalid day or SKU")
        for field in ("item_SKU", "item_BATCH", "item_MONTH"):
            value = row[field]
            if not isinstance(value, str) or "_" not in value or value.split("_", 1)[0] != sku:
                raise ValueError(f"D inferred SKU prefix mismatch in {field}")
        raw_quantity = row["item_WEEK"]
        if not isinstance(raw_quantity, str) or re.fullmatch(r"[0-9]+", raw_quantity) is None:
            raise ValueError("D.item_WEEK inferred quantity must be unsigned decimal text")
        demand.append(WarehouseRecord(day, sku, int(raw_quantity)))
    totals = {
        "initial_inventory": sum(initial.values()),
        "production": sum(record.quantity for record in production),
        "demand": sum(record.quantity for record in demand),
    }
    if totals != {"initial_inventory": 10709, "production": 36610, "demand": 25686}:
        raise ValueError("Parsed warehouse totals do not match the pinned source invariants")
    provenance = {
        "source_url": WAREHOUSE_URL,
        "doi": WAREHOUSE_DOI,
        "license": WAREHOUSE_LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Sirri, G., Accorsi, R., Lupi, G., and Manzini, R. (2026). Dynamic storage assignment in homogeneous dual-access deep-lane SR systems: Dataset. Zenodo. https://doi.org/10.5281/zenodo.18229759",
        "sha256": WAREHOUSE_SHA256,
        "published_md5": WAREHOUSE_MD5,
        "table_rows": {"Q": 368, "D": 1257, "h_LAST": 780, "ITEM_SKU": 101},
        "totals": totals,
        "inferred_demand_mapping": True,
        "author_confirmation": False,
        "demand_mapping": {
            "day": "D.T",
            "sku": "str(D.Q)",
            "quantity": "int(D.item_WEEK)",
            "batch": "D.item_SKU",
            "month": "D.item_BATCH",
            "week": "D.item_MONTH",
        },
        "warning": "Demand column interpretation is inferred from all-row SKU-prefix checks and independent MDB export; it is not author-confirmed. SKU/day projection omits warehouse lane and batch constraints.",
    }
    return WarehouseData(dict(initial), tuple(production), tuple(demand), provenance)


def read_warehouse(path: str | Path, *, allow_inferred_demand: bool = False) -> WarehouseData:
    """Verify the frozen source and require opt-in to its inferred demand mapping."""
    if type(allow_inferred_demand) is not bool:
        raise ValueError("allow_inferred_demand must be a boolean")
    source = Path(path)
    _verify(source)
    if not allow_inferred_demand:
        raise ValueError(
            "D has mismatched physical column names. Set allow_inferred_demand=True to explicitly accept the documented, independently exported but not author-confirmed mapping."
        )
    try:
        from access_parser import AccessParser
    except ImportError as exc:
        raise ImportError(
            "Warehouse reading requires the datasets extra: pip install 'assumption-ops[datasets]'"
        ) from exc
    parser = AccessParser(str(source))
    needed = {"Q", "D", "h_LAST", "ITEM_SKU"}
    if not needed.issubset(parser.catalog):
        raise ValueError("Pinned warehouse database is missing required tables")
    return _normalize_tables({name: parser.parse_table(name) for name in needed})


def warehouse_scenario(
    data: WarehouseData, *, sku_ids: tuple[str, ...] | None = None, stock_fraction: float = 1.0
) -> ScenarioData:
    """Project interchangeable SKU/day lots; fraction is an explicit supply stress.

    The fraction floors all initial and production quantities. No prices, customer
    priorities, substitutions, or actual delivery promises are inferred.
    """
    if not isinstance(data, WarehouseData):
        raise ValueError("data must be a WarehouseData record")
    data = WarehouseData(data.initial_inventory, data.production, data.demand, data.provenance)
    if (
        type(stock_fraction) not in (int, float)
        or not math.isfinite(stock_fraction)
        or not 0 <= stock_fraction <= 1
    ):
        raise ValueError("stock_fraction must be a finite number in [0,1], excluding booleans")
    known = (
        set(data.initial_inventory)
        | {r.sku for r in data.production}
        | {r.sku for r in data.demand}
    )
    if sku_ids is not None:
        if not isinstance(sku_ids, tuple) or any(
            not isinstance(sku, str) or not sku.strip() for sku in sku_ids
        ):
            raise ValueError("sku_ids must be a tuple of nonempty strings")
        if len(set(sku_ids)) != len(sku_ids) or set(sku_ids) - known:
            raise ValueError("sku_ids must be distinct known SKU labels")
    selected = known if sku_ids is None else set(sku_ids)
    supply_groups: dict[tuple[int, str], int] = defaultdict(int)
    for sku, quantity in data.initial_inventory.items():
        if sku in selected:
            supply_groups[(0, sku)] += quantity
    for record in data.production:
        if record.sku in selected:
            supply_groups[(record.day, record.sku)] += record.quantity
    demand_groups: dict[tuple[int, str], int] = defaultdict(int)
    for record in data.demand:
        if record.sku in selected:
            demand_groups[(record.day, record.sku)] += record.quantity
    supplies = tuple(
        Supply(f"supply-{sku}-day-{day}", sku, math.floor(quantity * stock_fraction), day, 0)
        for (day, sku), quantity in sorted(supply_groups.items())
    )
    orders = tuple(
        Order(f"demand-{sku}-day-{day}", sku, quantity, day, 1)
        for (day, sku), quantity in sorted(demand_groups.items())
    )
    policy = Policy(allow_late=True, late_penalty=10, unfilled_penalty=1000, disruption_penalty=0)
    validate_inputs(supplies, orders, policy)
    return ScenarioData(supplies, orders, policy)
