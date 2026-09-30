"""As-of business-pilot compilation, stock reconciliation and planning comparison.

Physical facts and open-order observations remain independent of suggested
allocations. A stock snapshot asserts a ledger balance; it never creates supply.
The fixed catalog and policy are assumed known before the opening snapshot.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from math import isclose
from platform import python_version
import re
from time import perf_counter
from typing import Any

import scipy

from .baselines import compare_allocators
from .evaluation import evaluate_plan
from .optimizer import Order, Policy, Supply, optimize, validate_inputs
from .pilot import _name


@dataclass(frozen=True)
class CatalogItem:
    sku: str
    unit_cost: int


@dataclass(frozen=True)
class StockSnapshot:
    snapshot_id: str
    observed_at: str
    quantities: dict[str, int]
    source: str


@dataclass(frozen=True)
class PhysicalMovement:
    movement_id: str
    event_time: str
    observed_at: str
    sku: str
    kind: str
    quantity: int
    source: str


@dataclass(frozen=True)
class OpenOrderObservation:
    order_id: str
    event_time: str
    observed_at: str
    sku: str
    quantity: int
    due_date: str
    priority: int
    source: str


@dataclass(frozen=True)
class BusinessPilot:
    name: str
    group_id: str
    synthetic: bool
    catalog: tuple[CatalogItem, ...]
    stock_snapshots: tuple[StockSnapshot, ...]
    movements: tuple[PhysicalMovement, ...]
    order_observations: tuple[OpenOrderObservation, ...]
    policy: Policy


@dataclass(frozen=True)
class DecisionSnapshot:
    snapshot_id: str
    cutoff: str
    supplies: tuple[Supply, ...]
    orders: tuple[Order, ...]
    policy: Policy
    metadata: dict[str, Any]


def _timestamp(value: Any, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None
    ):
        raise ValueError(f"{label} must use UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc


def _date(value: Any) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("due_date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("due_date is not a valid calendar date") from exc


def _integer(value: Any, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum or value > 2**53:
        raise ValueError(f"{label} must be an integer between {minimum} and 2**53")


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _checked(pilot: BusinessPilot) -> BusinessPilot:
    if not isinstance(pilot, BusinessPilot):
        raise ValueError("pilot must be a BusinessPilot")
    pilot = deepcopy(pilot)
    _name(pilot.name, "name")
    _name(pilot.group_id, "group_id")
    if type(pilot.synthetic) is not bool:
        raise ValueError("synthetic must be a boolean")
    for field, kind in (
        ("catalog", CatalogItem),
        ("stock_snapshots", StockSnapshot),
        ("movements", PhysicalMovement),
        ("order_observations", OpenOrderObservation),
    ):
        values = getattr(pilot, field)
        if not isinstance(values, tuple) or any(not isinstance(v, kind) for v in values):
            raise ValueError(f"{field} must be a tuple of {kind.__name__} records")
    if not pilot.catalog or not pilot.stock_snapshots:
        raise ValueError("A pilot needs a nonempty catalog and stock snapshots")
    if not isinstance(pilot.policy, Policy):
        raise ValueError("policy must be a Policy")
    validate_inputs([], [], pilot.policy)
    skus: set[str] = set()
    for item in pilot.catalog:
        _name(item.sku, "catalog SKU")
        _integer(item.unit_cost, "unit_cost")
        if item.sku in skus:
            raise ValueError(f"Duplicate catalog SKU: {item.sku}")
        skus.add(item.sku)
    if any(
        sku not in skus or any(s not in skus for s in subs)
        for sku, subs in pilot.policy.substitutions.items()
    ):
        raise ValueError("Policy substitutions must reference catalog SKUs")
    snapshot_ids, times = set(), set()
    for stock in pilot.stock_snapshots:
        _name(stock.snapshot_id, "snapshot_id")
        _name(stock.source, "snapshot source")
        _timestamp(stock.observed_at, "snapshot.observed_at")
        if stock.snapshot_id in snapshot_ids or stock.observed_at in times:
            raise ValueError("Duplicate snapshot ID or timestamp")
        snapshot_ids.add(stock.snapshot_id)
        times.add(stock.observed_at)
        if not isinstance(stock.quantities, dict) or set(stock.quantities) != skus:
            raise ValueError("Each stock snapshot must cover exactly the catalog SKUs")
        for quantity in stock.quantities.values():
            _integer(quantity, "stock quantity")
    opening = min(times)
    seen_movements = set()
    for movement in pilot.movements:
        _name(movement.movement_id, "movement_id")
        _name(movement.sku, "movement SKU")
        _name(movement.source, "movement source")
        _timestamp(movement.event_time, "movement.event_time")
        _timestamp(movement.observed_at, "movement.observed_at")
        if movement.movement_id in seen_movements:
            raise ValueError(f"Duplicate movement_id: {movement.movement_id}")
        seen_movements.add(movement.movement_id)
        if movement.event_time <= opening or movement.observed_at < movement.event_time:
            raise ValueError(
                "Movements must occur after the opening anchor and be observed no earlier than event_time"
            )
        if movement.sku not in skus:
            raise ValueError(f"Unknown movement SKU: {movement.sku}")
        if movement.kind not in ("receipt", "dispatch", "adjustment_in", "adjustment_out"):
            raise ValueError("Invalid physical movement kind")
        _integer(movement.quantity, "movement quantity", 1)
    seen_orders = set()
    order_skus: dict[str, str] = {}
    for order in pilot.order_observations:
        _name(order.order_id, "order_id")
        _name(order.sku, "order SKU")
        _name(order.source, "order source")
        _timestamp(order.event_time, "order.event_time")
        _timestamp(order.observed_at, "order.observed_at")
        if order.observed_at < opening or order.observed_at < order.event_time:
            raise ValueError(
                "Orders must be observed at or after opening and no earlier than event_time"
            )
        key = (order.order_id, order.observed_at)
        if key in seen_orders:
            raise ValueError("Duplicate order observation at the same observed_at")
        seen_orders.add(key)
        if order.sku not in skus:
            raise ValueError(f"Unknown order SKU: {order.sku}")
        if order.order_id in order_skus and order_skus[order.order_id] != order.sku:
            raise ValueError("An existing order cannot change SKU; use a distinct order ID")
        order_skus[order.order_id] = order.sku
        _date(order.due_date)
        _integer(order.quantity, "remaining order quantity")
        _integer(order.priority, "order priority", 1)
    last_event: dict[str, str] = {}
    for order in sorted(pilot.order_observations, key=lambda o: (o.observed_at, o.order_id)):
        if order.order_id in last_event and order.event_time < last_event[order.order_id]:
            raise ValueError("Order event_time regresses across observations")
        last_event[order.order_id] = order.event_time
    return pilot


def _verify_physical_history(opening: StockSnapshot, movements: list[PhysicalMovement]) -> None:
    """Also reject impossible physical history hidden by delayed batch reporting."""
    balance = dict(opening.quantities)
    ordered = sorted(movements, key=lambda m: (m.event_time, m.movement_id))
    index = 0
    while index < len(ordered):
        event_time = ordered[index].event_time
        deltas = dict.fromkeys(balance, 0)
        while index < len(ordered) and ordered[index].event_time == event_time:
            movement = ordered[index]
            sign = 1 if movement.kind in ("receipt", "adjustment_in") else -1
            deltas[movement.sku] += sign * movement.quantity
            index += 1
        for sku, delta in deltas.items():
            balance[sku] += delta
            _integer(balance[sku], f"Physical stock balance for {sku} at {event_time}")


def compile_business_pilot(pilot: BusinessPilot) -> tuple[DecisionSnapshot, ...]:
    """Compile audited, independent planning snapshots using observed facts only.

    Stock observations are complete synchronous assertions as of observed_at.
    Events observed at the same timestamp are applied atomically. An unexplained
    stock mismatch or negative admitted balance blocks the entire pilot. Open
    orders carry absolute remaining quantity, including zero for a closed order.
    Costs and delivery eligibility have UTC calendar-day, not intraday, resolution.
    """
    pilot = _checked(pilot)
    stocks = sorted(pilot.stock_snapshots, key=lambda s: s.observed_at)
    catalog = sorted(pilot.catalog, key=lambda c: c.sku)
    movements = sorted(pilot.movements, key=lambda m: (m.observed_at, m.movement_id))
    orders = sorted(pilot.order_observations, key=lambda o: (o.observed_at, o.order_id))
    balance = dict(stocks[0].quantities)
    opening_date = _timestamp(stocks[0].observed_at, "opening timestamp").date()
    movement_index = order_index = 0
    current_orders: dict[str, OpenOrderObservation] = {}
    admitted_movements: list[PhysicalMovement] = []
    admitted_orders: list[OpenOrderObservation] = []
    result = []
    for stock in stocks:
        cutoff = stock.observed_at
        while movement_index < len(movements) and movements[movement_index].observed_at <= cutoff:
            observed_at = movements[movement_index].observed_at
            deltas = dict.fromkeys(balance, 0)
            while (
                movement_index < len(movements)
                and movements[movement_index].observed_at == observed_at
            ):
                movement = movements[movement_index]
                sign = 1 if movement.kind in ("receipt", "adjustment_in") else -1
                deltas[movement.sku] += sign * movement.quantity
                admitted_movements.append(movement)
                movement_index += 1
            for sku, delta in deltas.items():
                balance[sku] += delta
                _integer(balance[sku], f"Stock balance for {sku} at {observed_at}")
        _verify_physical_history(stocks[0], admitted_movements)
        if balance != stock.quantities:
            differences = {
                sku: {"ledger": balance[sku], "snapshot": stock.quantities[sku]}
                for sku in balance
                if balance[sku] != stock.quantities[sku]
            }
            raise ValueError(f"Stock reconciliation mismatch at {stock.snapshot_id}: {differences}")
        while order_index < len(orders) and orders[order_index].observed_at <= cutoff:
            order = orders[order_index]
            current_orders[order.order_id] = order
            admitted_orders.append(order)
            order_index += 1
        active = sorted(
            (o for o in current_orders.values() if o.quantity), key=lambda o: o.order_id
        )
        origin = min([opening_date] + [_date(o.due_date) for o in active])
        decision_day = (_timestamp(cutoff, "cutoff").date() - origin).days
        supplies = tuple(
            Supply(f"stock:{item.sku}", item.sku, balance[item.sku], decision_day, item.unit_cost)
            for item in catalog
        )
        compiled_orders = tuple(
            Order(o.order_id, o.sku, o.quantity, (_date(o.due_date) - origin).days, o.priority)
            for o in active
        )
        validate_inputs(supplies, compiled_orders, pilot.policy)
        # Preserve exact float representation within the solvers' shared model.
        maximum_cost = max(
            max(s.unit_cost for s in supplies)
            + pilot.policy.late_penalty
            * max(0, decision_day - min((o.due_day for o in compiled_orders), default=decision_day))
            + pilot.policy.substitution_penalty,
            max((o.priority * pilot.policy.unfilled_penalty for o in compiled_orders), default=0),
        )
        if sum(o.quantity for o in compiled_orders) * maximum_cost > 2**53:
            raise ValueError("Pilot objective bound exceeds exact float integer range")
        input_payload = {
            "supplies": [asdict(s) for s in supplies],
            "orders": [asdict(o) for o in compiled_orders],
            "policy": asdict(pilot.policy),
        }
        visible_payload = {
            "catalog": [asdict(c) for c in catalog],
            "stock_snapshots": [asdict(s) for s in stocks if s.observed_at <= cutoff],
            "movements": [asdict(m) for m in admitted_movements],
            "order_observations": [asdict(o) for o in admitted_orders],
        }
        result.append(
            DecisionSnapshot(
                stock.snapshot_id,
                cutoff,
                supplies,
                compiled_orders,
                deepcopy(pilot.policy),
                {
                    "origin_day": origin.isoformat(),
                    "decision_day": decision_day,
                    "stock_reconciled": True,
                    "admitted_movement_ids": [m.movement_id for m in admitted_movements],
                    "admitted_order_observations": len(admitted_orders),
                    "visible_facts_fingerprint": _hash(visible_payload),
                    "input_fingerprint": _hash(input_payload),
                    "stock_source": stock.source,
                },
            )
        )
    return tuple(result)


def compare_business_pilot(pilot: BusinessPilot) -> dict[str, Any]:
    """Compare identical observed snapshots without simulating fulfillment.

    Repeated open backlog is never summed into demand, service or weighted cost.
    Every snapshot is independently optimized, with no previous allocation.
    """
    pilot = _checked(pilot)
    snapshots = compile_business_pilot(pilot)  # Validate all facts before solving.
    records = []
    for snapshot in snapshots:
        comparison = compare_allocators(snapshot.supplies, snapshot.orders, snapshot.policy)
        start = perf_counter()
        lp = optimize(snapshot.supplies, snapshot.orders, snapshot.policy, solver="lp")
        elapsed = perf_counter() - start
        lp_evaluation = evaluate_plan(snapshot.supplies, snapshot.orders, snapshot.policy, lp)
        milp = comparison["results"][0]
        parity = (
            lp.status == "optimal"
            and milp["status"] == "optimal"
            and isclose(
                lp_evaluation.costs.total,
                milp["evaluation"]["costs"]["total"],
                rel_tol=0,
                abs_tol=1e-6,
            )
        )
        if lp.status == "optimal" and milp["status"] == "optimal" and not parity:
            raise RuntimeError("Proven-optimal LP/MILP pilot scores disagree")
        comparison["results"].append(
            {
                "method": "verified_lp",
                "status": lp.status,
                "plan": lp.to_dict(),
                "evaluation": lp_evaluation.to_dict(),
                "elapsed_seconds": elapsed,
                "gap_to_lp_bound": max(
                    0.0, lp_evaluation.costs.total - comparison["lp_relaxation"]["lower_bound"]
                )
                if comparison["lp_relaxation"]["lower_bound"] is not None
                else None,
            }
        )
        comparison["lp_milp_proven_equal_score"] = parity
        comparison["limitations"] = [
            note
            for note in comparison["limitations"]
            if not note.startswith("LP output is a bound only")
        ]
        comparison["limitations"].append(
            "The lp_relaxation field is a continuous bound; verified_lp is separately checked as an integer plan. Timings here are descriptive, not a controlled scaling benchmark."
        )
        records.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "cutoff": snapshot.cutoff,
                "metadata": snapshot.metadata,
                "inventory_units": sum(s.quantity for s in snapshot.supplies),
                "open_order_units": sum(o.quantity for o in snapshot.orders),
                "comparison": comparison,
            }
        )
    canonical = asdict(pilot)
    for field, key in (
        ("catalog", lambda r: r["sku"]),
        ("stock_snapshots", lambda r: r["observed_at"]),
        ("movements", lambda r: (r["observed_at"], r["movement_id"])),
        ("order_observations", lambda r: (r["observed_at"], r["order_id"])),
    ):
        canonical[field] = sorted(canonical[field], key=key)
    return {
        "schema_version": 1,
        "name": pilot.name,
        "group_id": pilot.group_id,
        "synthetic": pilot.synthetic,
        "source_fingerprint": _hash(canonical),
        "units": "Integer interchangeable units; UTC calendar days; configured weighted penalty units",
        "external_dispatch": False,
        "environment": {"python": python_version(), "scipy": scipy.__version__},
        "limitations": [
            "Synthetic flags, source labels, catalog and open-order completeness are caller assertions, not independently verified provenance.",
            "Catalog and policy are fixed and must be known before the opening snapshot; costs are modeled units, not verified financial prices.",
            "Physical stock requires complete movements after opening; snapshots assert balances and never create additive supply.",
            "Open-order quantities are authoritative remaining backlog observations, not historical sales or additive requests.",
            "Only observed physical receipts are admitted; there are no future delivery forecasts or hidden future orders.",
            "Each suggested allocation is a separate plan, without execution, reservations, fulfillment or a previous-plan stability reference.",
            "UTC calendar-day buckets cannot establish exact intraday lateness; current stock cannot be backdated to its historical receipt time.",
            "Repeated backlog snapshots overlap: demand, shortages and costs must not be added across snapshots.",
            "Snapshot group_id must stay together in any calibration/holdout split; this comparison does not calibrate weights.",
            "Predicted service under the model is not measured customer service, ROI or causal improvement.",
        ],
        "snapshots": records,
        "summary": {
            "snapshots": len(records),
            "stock_reconciliations_verified": len(records),
            "lp_milp_proven_equal_snapshots": sum(
                r["comparison"]["lp_milp_proven_equal_score"] for r in records
            ),
            "final_snapshot": records[-1]["snapshot_id"],
            "final_method_evaluations": {
                r["method"]: r["evaluation"] for r in records[-1]["comparison"]["results"]
            },
            "observations_after_final_cutoff": {
                "movements": sum(m.observed_at > snapshots[-1].cutoff for m in pilot.movements),
                "open_orders": sum(
                    o.observed_at > snapshots[-1].cutoff for o in pilot.order_observations
                ),
            },
            "execution_intents": 0,
        },
    }
