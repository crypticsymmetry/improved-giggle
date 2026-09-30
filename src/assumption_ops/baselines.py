"""Transparent allocation baselines and an LP bound for the same transport model.

All methods use identical facts, eligibility and weighted costs, without a prior
allocation. This one-for-one transportation model has an integral polytope for
integer quantities: LP and MILP objective ties are expected. LP is reported as a
bound, never as an independently unverified executable integer plan.
"""

from __future__ import annotations

from copy import deepcopy
from math import isclose, isfinite
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .evaluation import evaluate_plan
from .optimizer import Allocation, Order, Plan, Policy, Supply, optimize, validate_inputs
from .replay import _greedy


def _eligible(supply: Supply, order: Order, policy: Policy) -> bool:
    return (supply.sku == order.sku or supply.sku in policy.substitutions.get(order.sku, ())) and (
        policy.allow_late or supply.available_day <= order.due_day
    )


def _unit_cost(supply: Supply, order: Order, policy: Policy) -> int:
    return (
        supply.unit_cost
        + max(0, supply.available_day - order.due_day) * policy.late_penalty
        + int(supply.sku != order.sku) * policy.substitution_penalty
    )


def _earliest_due_date(supplies: list[Supply], orders: list[Order], policy: Policy) -> Plan:
    remaining = {supply.id: supply.quantity for supply in supplies}
    allocations = []
    unfilled = {}
    objective = 0
    for order in sorted(orders, key=lambda order: (order.due_day, order.id)):
        outstanding = order.quantity
        for supply in sorted(supplies, key=lambda supply: (supply.available_day, supply.id)):
            if not _eligible(supply, order, policy):
                continue
            cost = _unit_cost(supply, order, policy)
            # Skip expensive supplies but keep scanning: an earlier supply is
            # not necessarily cheaper than a later eligible alternative.
            if cost >= order.priority * policy.unfilled_penalty:
                continue
            quantity = min(outstanding, remaining[supply.id])
            if quantity:
                allocations.append(Allocation(order.id, supply.id, quantity))
                remaining[supply.id] -= quantity
                outstanding -= quantity
                objective += quantity * cost
            if not outstanding:
                break
        unfilled[order.id] = outstanding
        objective += outstanding * order.priority * policy.unfilled_penalty
    return Plan(tuple(allocations), unfilled, float(objective), "heuristic")


def _lp_bound(supplies: list[Supply], orders: list[Order], policy: Policy) -> dict[str, Any]:
    start = perf_counter()
    arcs = [
        (order, supply)
        for order in orders
        for supply in supplies
        if _eligible(supply, order, policy)
    ]
    variables = len(arcs) + len(orders)
    if not variables:
        return {
            "status": "optimal",
            "lower_bound": 0.0,
            "fractional_variables": 0,
            "max_integrality_error": 0.0,
            "elapsed_seconds": perf_counter() - start,
        }
    costs = np.asarray(
        [_unit_cost(supply, order, policy) for order, supply in arcs]
        + [order.priority * policy.unfilled_penalty for order in orders],
        dtype=float,
    )
    supply_rows = {supply.id: i for i, supply in enumerate(supplies)}
    order_rows = {order.id: i for i, order in enumerate(orders)}
    capacity_rows, capacity_cols = [], []
    demand_rows, demand_cols = [], []
    for column, (order, supply) in enumerate(arcs):
        capacity_rows.append(supply_rows[supply.id])
        capacity_cols.append(column)
        demand_rows.append(order_rows[order.id])
        demand_cols.append(column)
    for i in range(len(orders)):
        demand_rows.append(i)
        demand_cols.append(len(arcs) + i)
    capacities = coo_matrix(
        (np.ones(len(capacity_rows)), (capacity_rows, capacity_cols)),
        shape=(len(supplies), variables),
    ).tocsr()
    demands = coo_matrix(
        (np.ones(len(demand_rows)), (demand_rows, demand_cols)), shape=(len(orders), variables)
    ).tocsr()
    supply_limits = np.asarray([s.quantity for s in supplies], dtype=float)
    order_demands = np.asarray([o.quantity for o in orders], dtype=float)
    result = linprog(
        costs,
        A_ub=capacities if supplies else None,
        b_ub=supply_limits if supplies else None,
        A_eq=demands if orders else None,
        b_eq=order_demands if orders else None,
        bounds=(0, None),
        method="highs",
        options={"time_limit": 30.0},
    )
    statuses = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    fractional = None
    maximum_error = None
    lower_bound = None
    if result.x is not None and np.all(np.isfinite(result.x)):
        errors = np.abs(result.x - np.rint(result.x))
        fractional = int(np.count_nonzero(errors > 1e-6))
        maximum_error = float(np.max(errors))
    if result.status == 0:
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError("Optimal LP did not return finite primal quantities")
        if (
            np.any(result.x < -1e-6)
            or np.any(capacities @ result.x - supply_limits > 1e-6)
            or not np.allclose(demands @ result.x, order_demands, rtol=1e-9, atol=1e-6)
        ):
            raise RuntimeError("Optimal LP failed independent capacity or demand verification")
        reconstructed = float(costs @ result.x)
        if not isfinite(result.fun) or not isclose(
            reconstructed, result.fun, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise RuntimeError("Optimal LP objective failed independent reconstruction")
        lower_bound = reconstructed
    return {
        "status": statuses.get(result.status, "solver_error"),
        "lower_bound": lower_bound,
        "fractional_variables": fractional,
        "max_integrality_error": maximum_error,
        "elapsed_seconds": perf_counter() - start,
    }


def compare_allocators(
    supplies: Sequence[Supply], orders: Sequence[Order], policy: Policy
) -> dict[str, Any]:
    """Compare verified integer plans and a continuous bound on one frozen input.

    The LP's optimal value is a lower bound, including when MILP returns only a
    verified time-limited incumbent. A nonoptimal LP primal value is NOT a lower
    bound, so gaps are absent unless the LP reports and verifies optimality.
    """
    supplies, orders, policy = deepcopy((list(supplies), list(orders), policy))
    validate_inputs(supplies, orders, policy)
    results = []
    for name, allocator in (
        ("milp", optimize),
        ("earliest_due_date", _earliest_due_date),
        ("priority_cost_greedy", lambda s, o, p: _greedy(s, o, p, None)),
    ):
        start = perf_counter()
        plan = allocator(supplies, orders, policy)
        elapsed = perf_counter() - start
        evaluation = evaluate_plan(supplies, orders, policy, plan)
        results.append(
            {
                "method": name,
                "status": plan.status,
                "plan": plan.to_dict(),
                "evaluation": evaluation.to_dict(),
                "elapsed_seconds": elapsed,
                "gap_to_lp_bound": None,
            }
        )
    bound = _lp_bound(supplies, orders, policy)
    if bound["lower_bound"] is not None:
        for result in results:
            gap = result["evaluation"]["costs"]["total"] - bound["lower_bound"]
            if gap < 0 and not isclose(gap, 0, abs_tol=1e-6):
                raise RuntimeError(
                    "LP lower bound exceeds an independently verified feasible integer plan"
                )
            result["gap_to_lp_bound"] = max(0.0, gap)
    return {
        "results": results,
        "lp_relaxation": bound,
        "milp_proven_optimal": results[0]["status"] == "optimal",
        "units": "weighted penalty units; integer interchangeable units; relative integer days",
        "external_dispatch": False,
        "limitations": [
            "All methods solve the same one-for-one transport model with no previous allocation or disruption reference.",
            "The LP polytope is integral for integer quantities in this model; LP/MILP ties are expected, not evidence of a MILP advantage.",
            "LP output is a bound only; no unverified rounded LP allocation is presented as an executable plan.",
            "Earliest-due-date uses earliest-available supply; priority greedy uses cheapest weighted eligible supply. Neither handles global opportunity costs.",
            "Weights are configured business penalty units, not verified monetary costs or measured savings.",
            "Elapsed times measure allocation computation separately from independent accounting checks; no external writes occur.",
        ],
    }
