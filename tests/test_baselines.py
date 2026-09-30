from copy import deepcopy
from dataclasses import replace
from itertools import product
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest

from assumption_ops.baselines import compare_allocators
from assumption_ops.optimizer import Order, Policy, Supply


def by_method(report):
    return {row["method"]: row for row in report["results"]}


def test_global_opportunity_cost_beats_priority_greedy():
    supplies = [Supply("A", "A", 1, 0, 3), Supply("B", "B", 1, 0)]
    orders = [Order("high", "A", 1, 0, 2), Order("low", "B", 1, 0)]
    policy = Policy(substitutions={"A": ("B",)}, substitution_penalty=0, unfilled_penalty=100)
    report = compare_allocators(supplies, orders, policy)
    methods = by_method(report)
    assert methods["milp"]["evaluation"]["costs"]["total"] == 3
    assert methods["priority_cost_greedy"]["evaluation"]["costs"]["total"] == 100
    assert report["lp_relaxation"]["lower_bound"] == 3
    assert methods["milp"]["gap_to_lp_bound"] == 0
    assert methods["priority_cost_greedy"]["gap_to_lp_bound"] == 97
    assert report["lp_relaxation"]["fractional_variables"] == 0
    assert "plan" not in report["lp_relaxation"]
    assert report["milp_proven_optimal"]


def test_earliest_due_date_ignores_priority_but_accounts_for_its_cost():
    report = compare_allocators(
        [Supply("s", "A", 1, 0)],
        [Order("urgent", "A", 1, 0), Order("important", "A", 1, 1, 3)],
        Policy(unfilled_penalty=10),
    )
    methods = by_method(report)
    assert methods["earliest_due_date"]["evaluation"]["costs"]["shortage"] == 30
    assert methods["priority_cost_greedy"]["evaluation"]["costs"]["shortage"] == 10
    assert methods["milp"]["evaluation"]["costs"]["shortage"] == 10


def test_edf_skips_expensive_early_supply_and_scans_later_options():
    report = compare_allocators(
        [Supply("expensive", "A", 1, 0, 100), Supply("cheap", "A", 1, 1, 1)],
        [Order("o", "A", 1, 1)],
        Policy(unfilled_penalty=10),
    )
    edf = by_method(report)["earliest_due_date"]
    assert edf["plan"]["allocations"] == [{"order_id": "o", "supply_id": "cheap", "quantity": 1}]
    assert edf["evaluation"]["costs"]["total"] == 1


def test_due_dates_substitutions_and_lateness_costs_match_all_methods():
    supplies = [Supply("wrong", "C", 1, 0), Supply("late-sub", "B", 2, 4, 1)]
    orders = [Order("o", "A", 2, 2)]
    policy = Policy(
        allow_late=True,
        substitutions={"A": ("B",)},
        late_penalty=3,
        substitution_penalty=5,
        unfilled_penalty=100,
    )
    report = compare_allocators(supplies, orders, policy)
    for row in report["results"]:
        assert row["evaluation"]["costs"] == {
            "acquisition": 2,
            "lateness": 12,
            "substitution": 10,
            "shortage": 0,
            "disruption": 0,
            "total": 24,
        }
        assert row["gap_to_lp_bound"] == 0
    blocked = compare_allocators(supplies, orders, replace(policy, allow_late=False))
    assert blocked["lp_relaxation"]["lower_bound"] == 200
    assert all(row["evaluation"]["metrics"]["unfilled_units"] == 2 for row in blocked["results"])


@pytest.mark.parametrize(
    "supplies,orders,expected",
    [
        ([], [], 0),
        ([Supply("s", "A", 1, 0)], [], 0),
        ([], [Order("o", "A", 2, 0)], 2000),
        ([Supply("s", "A", 0, 0)], [Order("o", "A", 0, 0)], 0),
    ],
)
def test_empty_supply_and_demand(supplies, orders, expected):
    report = compare_allocators(supplies, orders, Policy())
    assert report["lp_relaxation"]["lower_bound"] == expected
    assert all(row["evaluation"]["costs"]["total"] == expected for row in report["results"])
    assert all(row["gap_to_lp_bound"] == 0 for row in report["results"])


def test_independent_exhaustive_tiny_transport_bound():
    rng = Random(1729)
    for _ in range(15):
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
            unfilled_penalty=rng.randrange(1, 5),
        )
        arcs = [(order, supply) for order in orders for supply in supplies]
        best = float("inf")
        for quantities in product(*(range(min(o.quantity, s.quantity) + 1) for o, s in arcs)):
            if any(
                sum(q for q, (o, s) in zip(quantities, arcs) if s.id == supply.id) > supply.quantity
                for supply in supplies
            ):
                continue
            if any(
                sum(q for q, (o, s) in zip(quantities, arcs) if o.id == order.id) > order.quantity
                for order in orders
            ):
                continue
            cost = 0
            feasible = True
            for quantity, (order, supply) in zip(quantities, arcs):
                if quantity and (
                    (
                        supply.sku != order.sku
                        and supply.sku not in policy.substitutions.get(order.sku, ())
                    )
                    or (supply.available_day > order.due_day and not policy.allow_late)
                ):
                    feasible = False
                    break
                cost += quantity * (
                    supply.unit_cost
                    + policy.late_penalty * max(0, supply.available_day - order.due_day)
                    + policy.substitution_penalty * int(supply.sku != order.sku)
                )
            if not feasible:
                continue
            for order in orders:
                missing = order.quantity - sum(
                    q for q, (o, s) in zip(quantities, arcs) if o.id == order.id
                )
                cost += missing * order.priority * policy.unfilled_penalty
            best = min(best, cost)
        report = compare_allocators(supplies, orders, policy)
        assert report["lp_relaxation"]["lower_bound"] == pytest.approx(best)
        assert by_method(report)["milp"]["evaluation"]["costs"]["total"] == best
        assert all(row["evaluation"]["costs"]["total"] >= best for row in report["results"])


def test_time_limited_lp_primal_value_is_not_reported_as_lower_bound(monkeypatch):
    import assumption_ops.baselines as module

    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=1, x=np.asarray([1.0]), fun=1000),
    )
    report = compare_allocators([], [Order("o", "A", 1, 0)], Policy())
    assert report["lp_relaxation"]["status"] == "limit"
    assert report["lp_relaxation"]["lower_bound"] is None
    assert all(row["gap_to_lp_bound"] is None for row in report["results"])


def test_verified_milp_incumbent_has_bound_gap_without_optimality_claim(monkeypatch):
    import assumption_ops.baselines as module

    original = module.optimize
    monkeypatch.setattr(
        module, "optimize", lambda *args: replace(original(*args), status="feasible_limit")
    )
    report = compare_allocators([], [Order("o", "A", 1, 0)], Policy())
    assert report["milp_proven_optimal"] is False
    assert by_method(report)["milp"]["status"] == "feasible_limit"
    assert by_method(report)["milp"]["gap_to_lp_bound"] == 0


def test_lp_feasibility_and_objective_are_independently_verified(monkeypatch):
    import assumption_ops.baselines as module

    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.asarray([0.0]), fun=0),
    )
    with pytest.raises(RuntimeError, match="capacity or demand"):
        compare_allocators([], [Order("o", "A", 1, 0)], Policy())
    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.asarray([1.0]), fun=0),
    )
    with pytest.raises(RuntimeError, match="objective"):
        compare_allocators([], [Order("o", "A", 1, 0)], Policy())


def test_fractional_lp_diagnostics_are_bound_only(monkeypatch):
    import assumption_ops.baselines as module

    # A feasible nonvertex optimum of a zero-cost degenerate transport LP can
    # be fractional even though integral optimal vertices always exist.
    monkeypatch.setattr(
        module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.asarray([0.5, 0.5]), fun=0),
    )
    report = compare_allocators(
        [Supply("s", "A", 1, 0)], [Order("o", "A", 1, 0)], Policy(unfilled_penalty=0)
    )
    assert report["lp_relaxation"]["fractional_variables"] == 2
    assert report["lp_relaxation"]["max_integrality_error"] == 0.5
    assert report["lp_relaxation"]["lower_bound"] == 0
    assert "plan" not in report["lp_relaxation"]


def test_source_immutable_and_strict_input_validation():
    supplies, orders, policy = (
        [Supply("s", "A", 1, 0)],
        [Order("o", "A", 1, 0)],
        Policy(substitutions={"A": ("B",)}),
    )
    before = deepcopy((supplies, orders, policy))
    report = compare_allocators(supplies, orders, policy)
    report["results"][0]["plan"]["allocations"][0]["quantity"] = 999
    assert (supplies, orders, policy) == before
    with pytest.raises(ValueError):
        compare_allocators([Supply("s", "A", True, 0)], orders, policy)
