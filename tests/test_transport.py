from copy import deepcopy
from dataclasses import replace
from itertools import product
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest

from assumption_ops.evaluation import evaluate_plan
from assumption_ops.optimizer import Order, Plan, Policy, Supply
from assumption_ops.transport import LPPlanError, optimize_lp, supports_lp


def test_asymmetric_substitutions_and_global_opportunity_cost():
    supplies = [Supply("A", "A", 1, 0, 3), Supply("B", "B", 1, 0)]
    orders = [Order("high", "A", 1, 0, 2), Order("low", "B", 1, 0)]
    policy = Policy(
        substitutions={"A": ("B", "B", "A")}, substitution_penalty=0, unfilled_penalty=100
    )
    plan = optimize_lp(supplies, orders, policy)
    assert plan.objective == 3
    assert plan.unfilled == {"high": 0, "low": 0}
    assert len(plan.allocations) == 2
    evaluate_plan(supplies, orders, policy, plan)
    reverse = optimize_lp([Supply("a", "A", 1, 0)], [Order("b", "B", 1, 0)], policy)
    assert reverse.unfilled == {"b": 1}


def test_late_units_and_substitution_costs_reconstruct_exactly():
    supplies, orders = [Supply("s", "B", 2, 4, 3)], [Order("o", "A", 2, 1)]
    policy = Policy(
        allow_late=True, substitutions={"A": ("B",)}, late_penalty=7, substitution_penalty=5
    )
    plan = optimize_lp(supplies, orders, policy)
    assert plan.objective == 2 * (3 + 3 * 7 + 5)
    assert evaluate_plan(supplies, orders, policy, plan).costs.total == 58
    assert optimize_lp(supplies, orders, replace(policy, allow_late=False)).unfilled == {"o": 2}


def test_priority_shortage_and_high_acquisition_prices():
    supplies = [Supply("s", "A", 3, 0)]
    orders = [Order("low", "A", 3, 0), Order("high", "A", 3, 0, 2)]
    plan = optimize_lp(supplies, orders, Policy())
    assert plan.unfilled == {"low": 3, "high": 0}
    costly = optimize_lp([Supply("s", "A", 3, 0, 2000)], [Order("o", "A", 3, 0)], Policy())
    assert costly.unfilled == {"o": 3}
    assert costly.objective == 3000


@pytest.mark.parametrize(
    "supplies,orders,expected",
    [
        ([], [], 0),
        ([Supply("s", "A", 5, 0)], [], 0),
        ([], [Order("o", "A", 2, 0)], 2000),
        ([Supply("s", "A", 0, 0)], [Order("o", "A", 0, 0)], 0),
    ],
)
def test_empty_and_zero_quantities(supplies, orders, expected):
    plan = optimize_lp(supplies, orders, Policy())
    assert plan.objective == expected
    assert plan.status == "optimal"
    evaluate_plan(supplies, orders, Policy(), plan)


def test_indexed_model_omits_ineligible_and_zero_capacity_arcs(monkeypatch):
    import assumption_ops.transport as module

    original = module.linprog
    widths = []

    def observed(c, **kwargs):
        widths.append(len(c))
        assert kwargs["method"] == "highs-ds"
        return original(c, **kwargs)

    monkeypatch.setattr(module, "linprog", observed)
    supplies = [Supply(f"s{i}", f"SKU{i}", 1, 0) for i in range(100)] + [
        Supply("zero", "SKU0", 0, 0)
    ]
    orders = [Order("o", "SKU0", 1, 0)]
    assert optimize_lp(supplies, orders, Policy()).objective == 0
    assert widths == [2]  # One eligible arc plus one unfilled-demand variable.


def test_independent_exhaustive_oracle():
    rng = Random(20260930)
    for _ in range(20):
        supplies = [
            Supply(
                f"s{i}",
                rng.choice(("A", "B")),
                rng.randrange(3),
                rng.randrange(3),
                rng.randrange(4),
            )
            for i in range(2)
        ]
        orders = [
            Order(
                f"o{i}",
                rng.choice(("A", "B")),
                rng.randrange(3),
                rng.randrange(3),
                rng.randrange(1, 4),
            )
            for i in range(2)
        ]
        policy = Policy(
            substitutions={"A": ("B",)},
            allow_late=rng.choice((False, True)),
            late_penalty=rng.randrange(4),
            substitution_penalty=rng.randrange(4),
            unfilled_penalty=rng.randrange(1, 6),
        )
        arcs = [(order, supply) for order in orders for supply in supplies]
        best = float("inf")
        for quantities in product(*(range(min(o.quantity, s.quantity) + 1) for o, s in arcs)):
            if any(
                sum(q for q, (_, s) in zip(quantities, arcs) if s.id == supply.id) > supply.quantity
                for supply in supplies
            ):
                continue
            if any(
                sum(q for q, (o, _) in zip(quantities, arcs) if o.id == order.id) > order.quantity
                for order in orders
            ):
                continue
            if any(
                q
                and (
                    (s.sku != o.sku and s.sku not in policy.substitutions.get(o.sku, ()))
                    or (s.available_day > o.due_day and not policy.allow_late)
                )
                for q, (o, s) in zip(quantities, arcs)
            ):
                continue
            cost = sum(
                q
                * (
                    s.unit_cost
                    + policy.late_penalty * max(0, s.available_day - o.due_day)
                    + policy.substitution_penalty * int(s.sku != o.sku)
                )
                for q, (o, s) in zip(quantities, arcs)
            )
            cost += sum(
                (o.quantity - sum(q for q, (other, _) in zip(quantities, arcs) if other.id == o.id))
                * o.priority
                * policy.unfilled_penalty
                for o in orders
            )
            best = min(best, cost)
        plan = optimize_lp(supplies, orders, policy)
        assert plan.objective == best
        evaluate_plan(supplies, orders, policy, plan)


def test_fractional_incumbent_fails_closed(monkeypatch):
    import assumption_ops.transport as module

    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.asarray([0.5, 0.5]), fun=500),
    )
    with pytest.raises(LPPlanError, match="fractional"):
        optimize_lp([Supply("s", "A", 1, 0)], [Order("o", "A", 1, 0)], Policy())


def test_time_limited_integral_incumbent_is_feasible_without_optimality_claim(monkeypatch):
    import assumption_ops.transport as module

    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=1, x=np.asarray([1.0]), fun=1000),
    )
    plan = optimize_lp([], [Order("o", "A", 1, 0)], Policy())
    assert plan.status == "feasible_limit"
    assert plan.objective == 1000


@pytest.mark.parametrize(
    "solution",
    [
        SimpleNamespace(status=1, x=None, message="limit"),
        SimpleNamespace(status=2, x=None, message="infeasible"),
        SimpleNamespace(status=0, x=np.asarray([float("nan")]), fun=0),
        SimpleNamespace(status=0, x=np.asarray([0.0]), fun=0),
        SimpleNamespace(status=0, x=np.asarray([1.0]), fun=0),
        SimpleNamespace(status=0, x=np.asarray([1.0, 2.0]), fun=1000),
    ],
)
def test_missing_corrupt_or_infeasible_solver_results_rejected(monkeypatch, solution):
    import assumption_ops.transport as module

    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: solution)
    with pytest.raises(LPPlanError):
        optimize_lp([], [Order("o", "A", 1, 0)], Policy())


def test_numeric_safety_fails_closed():
    with pytest.raises(LPPlanError, match="range"):
        optimize_lp([Supply("s", "A", 2**53 + 1, 0)], [], Policy())
    with pytest.raises(LPPlanError, match="coefficient"):
        optimize_lp([], [Order("o", "A", 1, 0, 2**53)], Policy())
    with pytest.raises(LPPlanError, match="objective"):
        optimize_lp([], [Order("o", "A", 2**53, 0)], Policy(unfilled_penalty=2))
    plan = optimize_lp([], [Order("o", "A", 1, 0)], Policy(unfilled_penalty=2**53))
    assert plan.objective == 2**53


def test_large_objective_cannot_hide_rounding_difference_behind_relative_tolerance(monkeypatch):
    from types import SimpleNamespace
    import assumption_ops.transport as module

    quantity = 1 - 1e-10  # Near an integer, but its high-cost objective differs by one.
    result = SimpleNamespace(status=0, x=np.array([quantity]), fun=quantity * 10**10)
    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: result)
    with pytest.raises(LPPlanError, match="Rounded integral objective"):
        optimize_lp([], [Order("o", "A", 1, 0)], Policy(unfilled_penalty=10**10))


def test_prior_allocation_explicitly_unsupported():
    assert supports_lp(None)
    previous = Plan((), {}, 0, "optimal")
    assert not supports_lp(previous)
    with pytest.raises(ValueError, match="previous"):
        optimize_lp([], [], Policy(), previous)


@pytest.mark.parametrize("time_limit", [True, 0, -1, float("nan"), float("inf"), "30"])
def test_strict_time_limit_validation(time_limit):
    with pytest.raises(ValueError, match="time_limit"):
        optimize_lp([], [], Policy(), time_limit=time_limit)


def test_inputs_immutable_and_strict():
    supplies, orders, policy = (
        [Supply("s", "A", 1, 0)],
        [Order("o", "A", 1, 0)],
        Policy(substitutions={"A": ("B",)}),
    )
    before = deepcopy((supplies, orders, policy))
    optimize_lp(supplies, orders, policy)
    assert (supplies, orders, policy) == before
    with pytest.raises(ValueError):
        optimize_lp([Supply("s", "A", True, 0)], orders, policy)
