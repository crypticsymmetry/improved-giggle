"""Verified LP backend for the integral, one-for-one transportation model.

Integer supply capacities and demand balances define an integral transport
polytope. HiGHS dual simplex returns a vertex; every returned plan is still
independently checked for integrality and feasibility. A prior-allocation L1
objective is not supported by this initial backend and must use MILP instead.
"""

from __future__ import annotations

from copy import deepcopy
from math import isclose, isfinite
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .evaluation import evaluate_plan
from .optimizer import Allocation, Order, Plan, Policy, Supply, validate_inputs, validate_plan

_EXACT_FLOAT_INTEGER = 2**53


class LPPlanError(RuntimeError):
    """LP backend could not establish a safe, integral, feasible allocation."""


def supports_lp(previous: Plan | None) -> bool:
    """Whether this version supports the requested allocation objective."""
    return previous is None


def _unit_cost(supply: Supply, order: Order, policy: Policy) -> int:
    return (
        supply.unit_cost
        + max(0, supply.available_day - order.due_day) * policy.late_penalty
        + int(supply.sku != order.sku) * policy.substitution_penalty
    )


def _safe_integer(value: int, label: str) -> None:
    if value > _EXACT_FLOAT_INTEGER:
        raise LPPlanError(f"{label} exceeds the exact floating-point integer range (2**53)")


def optimize_lp(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    policy: Policy,
    previous: Plan | None = None,
    *,
    time_limit: float = 30.0,
) -> Plan:
    """Return only independently verified integer LP allocations.

    Malformed inputs raise ValueError. Unsupported previous plans raise
    ValueError. Solver/numerical backend failures raise LPPlanError so a caller
    can explicitly choose a different backend. No fractional incumbent is
    silently rounded into an executable plan.
    """
    if previous is not None:
        raise ValueError("LP transport backend does not support a previous allocation; use MILP")
    supplies, orders, policy = deepcopy((list(supplies), list(orders), policy))
    validate_inputs(supplies, orders, policy)
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    supply_rows = {supply.id: i for i, supply in enumerate(supplies)}
    supply_by_sku: dict[str, list[Supply]] = {}
    for supply in supplies:
        _safe_integer(supply.quantity, "Supply quantity")
        if supply.quantity:
            supply_by_sku.setdefault(supply.sku, []).append(supply)
    arcs: list[tuple[int, Order, Supply]] = []
    coefficients: list[int] = []
    upper: list[int] = []
    worst_total = 0
    for row, order in enumerate(orders):
        _safe_integer(order.quantity, "Order quantity")
        shortage_cost = order.priority * policy.unfilled_penalty
        _safe_integer(shortage_cost, "Priority-weighted shortage coefficient")
        maximum_cost = shortage_cost
        if order.quantity:
            # Deduplicate memberships, including the exact SKU, while preserving
            # deterministic requested-SKU then substitute-list order.
            for sku in dict.fromkeys((order.sku, *policy.substitutions.get(order.sku, ()))):
                for supply in supply_by_sku.get(sku, ()):
                    if not policy.allow_late and supply.available_day > order.due_day:
                        continue
                    cost = _unit_cost(supply, order, policy)
                    _safe_integer(cost, "Allocation cost coefficient")
                    arcs.append((row, order, supply))
                    coefficients.append(cost)
                    upper.append(min(order.quantity, supply.quantity))
                    maximum_cost = max(maximum_cost, cost)
        worst_total += order.quantity * maximum_cost
    _safe_integer(worst_total, "Worst-case integer objective")
    arc_count = len(arcs)
    coefficients.extend(order.priority * policy.unfilled_penalty for order in orders)
    upper.extend(order.quantity for order in orders)
    variables = len(coefficients)
    if not variables:
        result = Plan((), {}, 0.0, "optimal")
        validate_plan(supplies, orders, policy, result)
        evaluate_plan(supplies, orders, policy, result)
        return result
    capacity_rows, capacity_cols = [], []
    demand_rows, demand_cols = [], []
    for column, (row, _, supply) in enumerate(arcs):
        capacity_rows.append(supply_rows[supply.id])
        capacity_cols.append(column)
        demand_rows.append(row)
        demand_cols.append(column)
    for row in range(len(orders)):
        demand_rows.append(row)
        demand_cols.append(arc_count + row)
    capacities = coo_matrix(
        (np.ones(len(capacity_rows)), (capacity_rows, capacity_cols)),
        shape=(len(supplies), variables),
    ).tocsr()
    demands = coo_matrix(
        (np.ones(len(demand_rows)), (demand_rows, demand_cols)), shape=(len(orders), variables)
    ).tocsr()
    costs = np.asarray(coefficients, dtype=float)
    try:
        solution = linprog(
            costs,
            A_ub=capacities if supplies else None,
            b_ub=np.asarray([supply.quantity for supply in supplies], dtype=float)
            if supplies
            else None,
            A_eq=demands,
            b_eq=np.asarray([order.quantity for order in orders], dtype=float),
            bounds=np.column_stack((np.zeros(variables), np.asarray(upper, dtype=float))),
            method="highs-ds",
            options={"time_limit": float(time_limit)},
        )
    except (ValueError, RuntimeError) as exc:
        raise LPPlanError(
            f"LP solver could not process the validated transport model: {exc}"
        ) from exc
    if solution.status not in (0, 1) or solution.x is None:
        raise LPPlanError(
            f"LP solver produced no usable incumbent: {getattr(solution, 'message', solution.status)}"
        )
    try:
        values = np.asarray(solution.x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LPPlanError("LP solver returned nonnumeric quantities") from exc
    if values.shape != (variables,) or not np.all(np.isfinite(values)):
        raise LPPlanError("LP solver returned missing or nonfinite quantities")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > 1e-5):
        raise LPPlanError(
            "LP solver returned a fractional incumbent; integral feasibility is not established"
        )
    quantities = [int(value) for value in rounded]
    allocations = tuple(
        Allocation(order.id, supply.id, quantities[i])
        for i, (_, order, supply) in enumerate(arcs)
        if quantities[i]
    )
    unfilled = {order.id: quantities[arc_count + row] for row, order in enumerate(orders)}
    integer_objective = sum(
        coefficient * quantity for coefficient, quantity in zip(coefficients, quantities)
    )
    reported_objective = getattr(solution, "fun", None)
    if (
        isinstance(reported_objective, bool)
        or not isinstance(reported_objective, (int, float, np.integer, np.floating))
        or not isfinite(reported_objective)
    ):
        raise LPPlanError("LP solver returned a missing or nonfinite objective")
    if not isclose(float(costs @ values), reported_objective, rel_tol=1e-9, abs_tol=1e-6):
        raise LPPlanError("LP solver objective failed independent reconstruction")
    if solution.status == 0 and not isclose(
        integer_objective, reported_objective, rel_tol=0.0, abs_tol=1e-6
    ):
        raise LPPlanError(
            "Rounded integral objective differs from the claimed optimal LP objective"
        )
    plan = Plan(
        allocations,
        unfilled,
        float(integer_objective),
        "optimal" if solution.status == 0 else "feasible_limit",
    )
    try:
        validate_plan(supplies, orders, policy, plan)
        evaluate_plan(supplies, orders, policy, plan)
    except ValueError as exc:
        raise LPPlanError(
            f"LP solver returned an independently infeasible allocation: {exc}"
        ) from exc
    return plan
