"""Independent allocation accounting and operational service metrics.

Costs use the optimizer's weighted units; metrics remain unweighted quantities.
A prior plan may reference retired orders or supplies and need not be feasible
under the current inputs. Its allocation identifiers and quantities must still
be valid. Initial allocations count as changes against an empty baseline, but
incur no disruption cost unless an explicit prior plan is supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Sequence

from .optimizer import Allocation, Order, Plan, Policy, Supply, validate_plan


@dataclass(frozen=True)
class CostBreakdown:
    acquisition: int
    lateness: int
    substitution: int
    shortage: int
    disruption: int

    @property
    def total(self) -> int:
        return (
            self.acquisition + self.lateness + self.substitution + self.shortage + self.disruption
        )


@dataclass(frozen=True)
class Evaluation:
    costs: CostBreakdown
    metrics: dict[str, int]
    orders: tuple[dict[str, str | int], ...]

    def to_dict(self) -> dict:
        """Return JSON-compatible accounting, including the computed total."""
        return {
            "costs": {**asdict(self.costs), "total": self.costs.total},
            "metrics": dict(self.metrics),
            "orders": [dict(row) for row in self.orders],
        }


def _aggregate(allocations: Sequence[Allocation]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for allocation in allocations:
        for name in ("order_id", "supply_id"):
            identifier = getattr(allocation, name)
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"allocation.{name} must be a nonempty string")
        if type(allocation.quantity) is not int or allocation.quantity < 0:
            raise ValueError(
                "allocation.quantity must be a nonnegative integer (booleans excluded)"
            )
        key = (allocation.order_id, allocation.supply_id)
        result[key] = result.get(key, 0) + allocation.quantity
    return result


def evaluate_plan(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    policy: Policy,
    plan: Plan,
    previous: Plan | None = None,
) -> Evaluation:
    """Verify feasibility and objective, then independently report costs.

    This reconstructs costs from the returned allocations, without trusting
    solver objective values or auxiliary deviation variables. Duplicate prior
    arcs are aggregated; retired arcs remain in the L1 disruption calculation.
    Objective comparisons allow only floating-point rounding tolerance.
    """
    validate_plan(supplies, orders, policy, plan)
    current = _aggregate(plan.allocations)
    prior = _aggregate(previous.allocations) if previous is not None else {}
    supply_map = {supply.id: supply for supply in supplies}
    order_map = {order.id: order for order in orders}
    rows: dict[str, dict[str, str | int]] = {
        order.id: {
            "order_id": order.id,
            "requested": order.quantity,
            "allocated": 0,
            "unfilled": plan.unfilled[order.id],
            "on_time": 0,
            "late": 0,
            "substitute": 0,
            "priority": order.priority,
        }
        for order in orders
    }
    acquisition = 0
    unit_days_late = 0
    substitute_units = 0
    for (order_id, supply_id), quantity in current.items():
        supply, order = supply_map[supply_id], order_map[order_id]
        days_late = max(0, supply.available_day - order.due_day)
        substituted = supply.sku != order.sku
        acquisition += quantity * supply.unit_cost
        unit_days_late += quantity * days_late
        substitute_units += quantity * int(substituted)
        row = rows[order_id]
        row["allocated"] += quantity
        row["late" if days_late else "on_time"] += quantity
        row["substitute"] += quantity * int(substituted)
    changed_units = sum(
        abs(current.get(key, 0) - prior.get(key, 0)) for key in current.keys() | prior.keys()
    )
    priority_weighted_unfilled = sum(order.priority * plan.unfilled[order.id] for order in orders)
    costs = CostBreakdown(
        acquisition=acquisition,
        lateness=unit_days_late * policy.late_penalty,
        substitution=substitute_units * policy.substitution_penalty,
        shortage=priority_weighted_unfilled * policy.unfilled_penalty,
        disruption=changed_units * policy.disruption_penalty if previous is not None else 0,
    )
    if not isclose(costs.total, plan.objective, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            f"plan objective does not match independently reconstructed cost: "
            f"reported={plan.objective}, reconstructed={costs.total}"
        )
    metrics = {
        "requested_units": sum(order.quantity for order in orders),
        "allocated_units": sum(current.values()),
        "unfilled_units": sum(plan.unfilled.values()),
        "priority_weighted_unfilled": priority_weighted_unfilled,
        "on_time_units": sum(int(row["on_time"]) for row in rows.values()),
        "late_units": sum(int(row["late"]) for row in rows.values()),
        "substitute_units": substitute_units,
        "unit_days_late": unit_days_late,
        "changed_units": changed_units,
        "fully_filled_orders": sum(plan.unfilled[order.id] == 0 for order in orders),
        "total_orders": len(orders),
    }
    return Evaluation(costs, metrics, tuple(rows.values()))
