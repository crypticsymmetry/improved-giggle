"""Integer allocation with explicit, weighted business costs.

The objective is a weighted sum, not a lexicographic promise: unit purchasing
cost, lateness per unit-day, substitutions, priority-weighted unfilled demand,
and absolute changes from the prior allocation all contribute. Callers must
choose penalties in consistent units and calibrate them to their business.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Supply:
    id: str
    sku: str
    quantity: int
    available_day: int
    unit_cost: int = 0


@dataclass(frozen=True)
class Order:
    id: str
    sku: str
    quantity: int
    due_day: int
    priority: int = 1


@dataclass(frozen=True)
class Policy:
    # Mapping is from requested SKU to acceptable substitute supply SKUs.
    substitutions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allow_late: bool = False
    late_penalty: int = 10
    substitution_penalty: int = 2
    disruption_penalty: int = 1
    unfilled_penalty: int = 1000


@dataclass(frozen=True)
class Allocation:
    order_id: str
    supply_id: str
    quantity: int


@dataclass(frozen=True)
class Plan:
    allocations: tuple[Allocation, ...]
    unfilled: dict[str, int]
    objective: float
    status: str

    def to_dict(self) -> dict:
        return {
            "allocations": [asdict(a) for a in self.allocations],
            "unfilled": dict(self.unfilled),
            "objective": self.objective,
            "status": self.status,
        }


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (booleans excluded)")


def _name(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_inputs(supplies: Sequence[Supply], orders: Sequence[Order], policy: Policy) -> None:
    for records, label in ((supplies, "supply"), (orders, "order")):
        seen: set[str] = set()
        for record in records:
            _name(record.id, f"{label}.id")
            _name(record.sku, f"{label}.sku")
            if record.id in seen:
                raise ValueError(f"duplicate {label} id: {record.id}")
            seen.add(record.id)
            _integer(record.quantity, f"{label}.quantity")
            if isinstance(record, Supply):
                _integer(record.available_day, "supply.available_day")
                _integer(record.unit_cost, "supply.unit_cost")
            else:
                _integer(record.due_day, "order.due_day")
                _integer(record.priority, "order.priority", 1)
    if type(policy.allow_late) is not bool:
        raise ValueError("allow_late must be a boolean")
    for field_name in (
        "late_penalty",
        "substitution_penalty",
        "disruption_penalty",
        "unfilled_penalty",
    ):
        _integer(getattr(policy, field_name), field_name)
    if not isinstance(policy.substitutions, dict):
        raise ValueError("substitutions must be a mapping of requested SKUs to tuples")
    for sku, substitutes in policy.substitutions.items():
        _name(sku, "substitution SKU")
        if not isinstance(substitutes, tuple):
            raise ValueError("substitutions must contain tuples of SKU strings")
        for substitute in substitutes:
            _name(substitute, "substitute SKU")


def _eligible(supply: Supply, order: Order, policy: Policy) -> bool:
    return (supply.sku == order.sku or supply.sku in policy.substitutions.get(order.sku, ())) and (
        policy.allow_late or supply.available_day <= order.due_day
    )


def validate_inputs(supplies: Sequence[Supply], orders: Sequence[Order], policy: Policy) -> None:
    """Validate allocation inputs without running the solver."""
    _validate_inputs(supplies, orders, policy)


def _allocation_map(allocations: Sequence[Allocation]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for allocation in allocations:
        _name(allocation.order_id, "allocation.order_id")
        _name(allocation.supply_id, "allocation.supply_id")
        _integer(allocation.quantity, "allocation.quantity")
        key = (allocation.order_id, allocation.supply_id)
        result[key] = result.get(key, 0) + allocation.quantity
    return result


def validate_plan(
    supplies: Sequence[Supply], orders: Sequence[Order], policy: Policy, plan: Plan
) -> None:
    """Independently verify every arc, capacity, and exact demand balance.

    Raises ValueError on infeasibility; does not trust the solver's status.
    The objective cannot be verified without the previous plan.
    """
    _validate_inputs(supplies, orders, policy)
    supply_map = {s.id: s for s in supplies}
    order_map = {o.id: o for o in orders}
    if set(plan.unfilled) != set(order_map):
        raise ValueError("unfilled must contain exactly the current order IDs")
    used = dict.fromkeys(supply_map, 0)
    filled = dict.fromkeys(order_map, 0)
    for (order_id, supply_id), quantity in _allocation_map(plan.allocations).items():
        if order_id not in order_map or supply_id not in supply_map:
            raise ValueError("allocation references an unknown order or supply")
        if not _eligible(supply_map[supply_id], order_map[order_id], policy):
            raise ValueError("allocation uses an ineligible SKU or delivery date")
        used[supply_id] += quantity
        filled[order_id] += quantity
    for supply in supplies:
        if used[supply.id] > supply.quantity:
            raise ValueError(f"supply capacity exceeded: {supply.id}")
    for order in orders:
        _integer(plan.unfilled[order.id], "unfilled quantity")
        if filled[order.id] + plan.unfilled[order.id] != order.quantity:
            raise ValueError(f"order balance violated: {order.id}")
    if not isinstance(plan.objective, (int, float)) or not isfinite(plan.objective):
        raise ValueError("objective must be finite")


def optimize(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    policy: Policy,
    previous: Plan | None = None,
    *,
    time_limit: float = 30.0,
) -> Plan:
    """Solve a bounded integer transportation problem; return verified solutions.

    A time-limited incumbent is returned as ``feasible_limit`` only after
    independent feasibility verification. Missing prior arcs contribute their
    full retirement cost, including arcs removed by changed eligibility.
    """
    _validate_inputs(supplies, orders, policy)
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    prior = _allocation_map(previous.allocations) if previous is not None else {}
    arcs = [(o, s) for o in orders for s in supplies if _eligible(s, o, policy)]
    supply_arcs: dict[str, list[tuple[int, float]]] = {s.id: [] for s in supplies}
    order_arcs: dict[str, list[tuple[int, float]]] = {o.id: [] for o in orders}
    for i, (order, supply) in enumerate(arcs):
        supply_arcs[supply.id].append((i, 1.0))
        order_arcs[order.id].append((i, 1.0))
    arc_keys = {(o.id, s.id) for o, s in arcs}
    retirement = policy.disruption_penalty * sum(
        q for key, q in prior.items() if key not in arc_keys
    )
    n_arcs, n_orders = len(arcs), len(orders)
    with_disruption = previous is not None and policy.disruption_penalty > 0
    n_variables = n_arcs + n_orders + (n_arcs if with_disruption else 0)
    if n_variables == 0:
        return Plan((), {}, float(retirement), "optimal")
    costs = [
        float(
            s.unit_cost
            + policy.late_penalty * max(0, s.available_day - o.due_day)
            + policy.substitution_penalty * (s.sku != o.sku)
        )
        for o, s in arcs
    ]
    costs.extend(float(policy.unfilled_penalty * o.priority) for o in orders)
    if with_disruption:
        costs.extend([float(policy.disruption_penalty)] * n_arcs)
    upper = [min(o.quantity, s.quantity) for o, s in arcs] + [o.quantity for o in orders]
    upper.extend([np.inf] * (n_arcs if with_disruption else 0))
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower_rows: list[float] = []
    upper_rows: list[float] = []

    def row(coefficients: list[tuple[int, float]], lower: float, upper_bound: float) -> None:
        index = len(lower_rows)
        for column, value in coefficients:
            row_indices.append(index)
            col_indices.append(column)
            values.append(value)
        lower_rows.append(lower)
        upper_rows.append(upper_bound)

    for supply in supplies:
        row(supply_arcs[supply.id], -np.inf, supply.quantity)
    for j, order in enumerate(orders):
        row(order_arcs[order.id] + [(n_arcs + j, 1.0)], order.quantity, order.quantity)
    if with_disruption:
        for i, (order, supply) in enumerate(arcs):
            old = prior.get((order.id, supply.id), 0)
            deviation = n_arcs + n_orders + i
            row([(i, 1.0), (deviation, -1.0)], -np.inf, old)
            row([(i, -1.0), (deviation, -1.0)], -np.inf, -old)
    matrix = coo_matrix(
        (values, (row_indices, col_indices)), shape=(len(lower_rows), n_variables)
    ).tocsc()
    result = milp(
        c=np.asarray(costs),
        integrality=np.asarray(
            [1] * (n_arcs + n_orders) + [0] * (n_arcs if with_disruption else 0)
        ),
        bounds=Bounds(np.zeros(n_variables), np.asarray(upper)),
        constraints=LinearConstraint(matrix, np.asarray(lower_rows), np.asarray(upper_rows)),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    if result.status not in (0, 1) or result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"allocation solver did not produce a usable solution: {result.message}")
    integer_values = result.x[: n_arcs + n_orders]
    rounded = np.rint(integer_values)
    if np.any(np.abs(integer_values - rounded) > 1e-5):
        raise RuntimeError("allocation solver returned nonintegral quantities")
    quantities = [int(value) for value in rounded]
    allocations = tuple(
        Allocation(o.id, s.id, quantities[i]) for i, (o, s) in enumerate(arcs) if quantities[i]
    )
    unfilled = {o.id: quantities[n_arcs + j] for j, o in enumerate(orders)}
    objective = sum(costs[i] * quantities[i] for i in range(n_arcs + n_orders)) + retirement
    if with_disruption:
        objective += policy.disruption_penalty * sum(
            abs(quantities[i] - prior.get((o.id, s.id), 0)) for i, (o, s) in enumerate(arcs)
        )
    plan = Plan(
        allocations,
        unfilled,
        float(objective),
        "optimal" if result.status == 0 else "feasible_limit",
    )
    try:
        validate_plan(supplies, orders, policy, plan)
    except ValueError as exc:
        raise RuntimeError(f"allocation solver returned an infeasible solution: {exc}") from exc
    return plan
