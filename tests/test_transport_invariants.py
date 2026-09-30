"""Independent LP parity, safe dispatch, and untrusted-incumbent invariants."""

from copy import deepcopy
from itertools import product
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from assumption_ops.evaluation import evaluate_plan
from assumption_ops.optimizer import (
    Allocation,
    Order,
    Plan,
    Policy,
    Supply,
    optimize,
    validate_plan,
)
from assumption_ops.transport import LPPlanError, optimize_lp


def exhaustive_cost(supplies, orders, policy):
    """Enumerate every integral allocation without importing solver internals."""
    pairs = [(order, supply) for order in orders for supply in supplies]
    choices = []
    for order, supply in pairs:
        eligible = supply.sku == order.sku or supply.sku in policy.substitutions.get(order.sku, ())
        eligible = eligible and (policy.allow_late or supply.available_day <= order.due_day)
        choices.append(range(min(order.quantity, supply.quantity) + 1) if eligible else (0,))
    best = float("inf")
    for quantities in product(*choices):
        used = {s.id: 0 for s in supplies}
        filled = {o.id: 0 for o in orders}
        cost = 0
        for (order, supply), quantity in zip(pairs, quantities):
            used[supply.id] += quantity
            filled[order.id] += quantity
            cost += quantity * (
                supply.unit_cost
                + max(0, supply.available_day - order.due_day) * policy.late_penalty
                + (supply.sku != order.sku) * policy.substitution_penalty
            )
        if any(used[s.id] > s.quantity for s in supplies):
            continue
        if any(filled[o.id] > o.quantity for o in orders):
            continue
        cost += sum(
            (o.quantity - filled[o.id]) * o.priority * policy.unfilled_penalty for o in orders
        )
        best = min(best, cost)
    return best


def problem():
    return (
        [Supply("stock", "A", 1, 0, 2)],
        [Order("customer", "A", 1, 1)],
        Policy(unfilled_penalty=10),
    )


def test_all_dispatch_modes_match_exhaustive_oracle_on_seeded_tiny_models():
    rng = random.Random(29)
    for index in range(12):
        supplies = [
            Supply(f"s-{n}", sku, rng.randrange(3), rng.randrange(3), rng.randrange(5))
            for n, sku in enumerate(("A", "B"))
        ]
        orders = [
            Order(f"o-{n}", sku, rng.randrange(3), rng.randrange(3), rng.randrange(1, 4))
            for n, sku in enumerate(("A", "B" if index % 3 else "missing"))
        ]
        policy = Policy(
            substitutions={"A": ("A", "B", "B")},
            allow_late=bool(index % 2),
            late_penalty=2,
            substitution_penalty=1,
            unfilled_penalty=7,
        )
        before = deepcopy((supplies, orders, policy))
        expected = exhaustive_cost(supplies, orders, policy)
        for mode in ("milp", "lp", "auto"):
            plan = optimize(supplies, orders, policy, solver=mode)
            validate_plan(supplies, orders, policy, plan)
            evaluation = evaluate_plan(supplies, orders, policy, plan)
            assert plan.status == "optimal"
            assert plan.objective == evaluation.costs.total == expected
        assert (supplies, orders, policy) == before


def test_equivalent_tied_allocation_is_accepted_without_identity_requirement():
    supplies = [Supply("first", "A", 1, 0), Supply("second", "A", 1, 0)]
    orders = [Order("customer", "A", 1, 1)]
    policy = Policy()
    reference = optimize(supplies, orders, policy)
    # A different optimal vertex can assign the other interchangeable supplier.
    chosen = "second" if reference.allocations[0].supply_id == "first" else "first"
    values = [int(supply.id == chosen) for supply in supplies] + [0]
    response = SimpleNamespace(
        status=0, x=np.asarray(values, dtype=float), fun=0.0, message="optimal"
    )
    with patch("assumption_ops.transport.linprog", return_value=response):
        alternative = optimize(supplies, orders, policy, solver="lp")
    assert alternative.allocations != reference.allocations
    assert alternative.objective == reference.objective == 0
    validate_plan(supplies, orders, policy, alternative)


def test_auto_falls_back_on_backend_failure_but_never_caller_validation_errors():
    supplies, orders, policy = problem()
    with patch("assumption_ops.transport.optimize_lp", side_effect=LPPlanError("unsafe incumbent")):
        fallback = optimize(supplies, orders, policy, solver="auto")
    assert fallback.objective == exhaustive_cost(supplies, orders, policy)
    for error in (
        ValueError("invalid backend argument"),
        RuntimeError("unexpected implementation error"),
    ):
        with (
            patch("assumption_ops.transport.optimize_lp", side_effect=error),
            patch("assumption_ops.optimizer._optimize_milp") as milp,
        ):
            with pytest.raises(type(error), match=str(error)):
                optimize(supplies, orders, policy, solver="auto")
            milp.assert_not_called()
    with (
        patch("assumption_ops.transport.optimize_lp") as lp,
        patch("assumption_ops.optimizer._optimize_milp") as milp,
    ):
        with pytest.raises(ValueError):
            optimize([Supply("bad", "A", True, 0)], orders, policy, solver="auto")
        with pytest.raises(ValueError):
            optimize(supplies, orders, policy, solver="auto", time_limit=True)
        with pytest.raises(ValueError):
            optimize(supplies, orders, policy, solver="unknown")
        lp.assert_not_called()
        milp.assert_not_called()


def test_default_and_prior_auto_keep_milp_and_explicit_prior_lp_rejects():
    supplies, orders, policy = problem()
    previous = Plan((Allocation("customer", "stock", 1),), {"customer": 0}, 2, "optimal")
    with patch(
        "assumption_ops.transport.optimize_lp", side_effect=AssertionError("LP called")
    ) as lp:
        default = optimize(supplies, orders, policy)
        stable = optimize(supplies, orders, policy, previous, solver="auto")
        assert default.objective == stable.objective == 2
        lp.assert_not_called()
    with pytest.raises(ValueError, match="previous allocation"):
        optimize(supplies, orders, policy, previous, solver="lp")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status=0, x=np.array([0.5, 0.5]), fun=6.0, message="fractional"),
        SimpleNamespace(status=0, x=np.array([2.0, -1.0]), fun=-6.0, message="bad capacity"),
        SimpleNamespace(status=0, x=np.array([1.0, 0.0]), fun=999.0, message="wrong objective"),
        SimpleNamespace(status=1, x=None, fun=None, message="no incumbent"),
        SimpleNamespace(status=1, x=np.array([1.0, 0.0]), fun=None, message="missing objective"),
    ],
)
def test_invalid_incumbents_cannot_become_executable_lp_plans(response):
    supplies, orders, policy = problem()
    with patch("assumption_ops.transport.linprog", return_value=response):
        with pytest.raises(LPPlanError):
            optimize_lp(supplies, orders, policy)
        # The same backend failure is safe to recover through the verified MILP.
        fallback = optimize(supplies, orders, policy, solver="auto")
    assert fallback.objective == 2
    assert fallback.status == "optimal"


def test_verified_time_limited_integer_incumbent_does_not_claim_optimality():
    supplies, orders, policy = problem()
    # Leaving demand unfilled costs10, though the genuine optimum purchases at2.
    response = SimpleNamespace(status=1, x=np.array([0.0, 1.0]), fun=10.0, message="time limit")
    with patch("assumption_ops.transport.linprog", return_value=response):
        result = optimize(supplies, orders, policy, solver="lp")
    assert result.status == "feasible_limit"
    assert result.objective == 10
    assert result.objective > exhaustive_cost(supplies, orders, policy)
    validate_plan(supplies, orders, policy, result)


def test_sparse_empty_supply_still_balances_every_order_and_reports_shortage():
    orders = [Order("requested", "A", 3, 1), Order("empty", "B", 0, 1)]
    policy = Policy(unfilled_penalty=7)
    result = optimize([], orders, policy, solver="lp")
    assert result.unfilled == {"requested": 3, "empty": 0}
    assert result.allocations == ()
    assert result.objective == 21
    assert optimize([], [], policy, solver="lp").objective == 0
