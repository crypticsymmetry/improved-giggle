from dataclasses import replace

import pytest

from assumption_ops.optimizer import (
    Allocation,
    Order,
    Plan,
    Policy,
    Supply,
    optimize,
    validate_plan,
)


def test_capacity_and_priority_shortage():
    supplies = [Supply("s", "A", 5, 0)]
    orders = [Order("low", "A", 4, 1), Order("high", "A", 4, 1, 3)]
    plan = optimize(supplies, orders, Policy())
    assert plan.unfilled == {"low": 3, "high": 0}
    assert sum(a.quantity for a in plan.allocations) == 5
    validate_plan(supplies, orders, Policy(), plan)


def test_due_dates_and_directional_substitutions():
    supplies = [Supply("late", "A", 4, 3), Supply("sub", "B", 2, 0), Supply("bad", "C", 4, 0)]
    orders = [Order("o", "A", 4, 1)]
    policy = Policy(substitutions={"A": ("B",)})
    plan = optimize(supplies, orders, policy)
    assert plan.allocations == (Allocation("o", "sub", 2),)
    assert plan.unfilled == {"o": 2}
    assert plan.objective == 2004
    reversed_policy = Policy(substitutions={"B": ("A",)})
    assert optimize(supplies, orders, reversed_policy).unfilled == {"o": 4}


def test_late_policy_charges_unit_days():
    plan = optimize(
        [Supply("s", "A", 3, 5, 2)],
        [Order("o", "A", 3, 2)],
        Policy(allow_late=True, late_penalty=7),
    )
    assert plan.unfilled == {"o": 0}
    assert plan.objective == 3 * (2 + 3 * 7)


def test_stability_and_removed_arc_retirement_cost():
    orders = [Order("o", "A", 4, 1)]
    supplies = [Supply("s1", "A", 4, 0, 1), Supply("s2", "A", 4, 0)]
    previous = Plan((Allocation("o", "s1", 4),), {"o": 0}, 4, "optimal")
    policy = Policy(disruption_penalty=2)
    plan = optimize(supplies, orders, policy, previous)
    assert plan.allocations == previous.allocations
    assert plan.objective == 4
    plan = optimize(supplies[1:], orders, policy, previous)
    assert plan.allocations == (Allocation("o", "s2", 4),)
    # Four old units retired and four newly allocated units.
    assert plan.objective == 16


def test_removed_eligibility_costs_retirement():
    previous = Plan((Allocation("o", "s", 2),), {"o": 0}, 0, "optimal")
    plan = optimize(
        [Supply("s", "A", 2, 3)], [Order("o", "A", 2, 1)], Policy(disruption_penalty=3), previous
    )
    assert plan.unfilled == {"o": 2}
    assert plan.objective == 2006


def test_costs_are_weighted_not_lexicographic():
    plan = optimize([Supply("s", "A", 2, 0, 1200)], [Order("o", "A", 2, 1)], Policy())
    assert plan.unfilled == {"o": 2}
    assert plan.objective == 2000


def test_empty_inputs():
    assert optimize([], [], Policy()).to_dict() == {
        "allocations": [],
        "unfilled": {},
        "objective": 0.0,
        "status": "optimal",
    }
    assert optimize([], [Order("o", "A", 3, 0)], Policy()).unfilled == {"o": 3}
    assert optimize([Supply("s", "A", 3, 0)], [], Policy()).allocations == ()
    previous = Plan((Allocation("gone", "gone", 2),), {}, 0, "optimal")
    assert optimize([], [], Policy(disruption_penalty=3), previous).objective == 6


@pytest.mark.parametrize(
    "supply",
    [
        Supply("s", "A", -1, 0),
        Supply("s", "A", True, 0),
        Supply("s", "A", 1, -1),
        Supply("s", "A", 1, 0, -1),
        Supply("s", "A", 1.0, 0),
    ],
)
def test_rejects_invalid_supply(supply):
    with pytest.raises(ValueError):
        optimize([supply], [], Policy())


@pytest.mark.parametrize(
    "order",
    [
        Order("o", "A", 1, 0, 0),
        Order("o", "A", 1, 0, True),
        Order("o", "A", 1, -1),
        Order("o", "A", -1, 0),
    ],
)
def test_rejects_invalid_order(order):
    with pytest.raises(ValueError):
        optimize([], [order], Policy())


@pytest.mark.parametrize(
    "policy",
    [
        Policy(late_penalty=-1),
        Policy(unfilled_penalty=True),
        Policy(allow_late=1),
        Policy(substitutions={"A": ["B"]}),
    ],
)
def test_rejects_invalid_policy(policy):
    with pytest.raises(ValueError):
        optimize([], [], policy)


def test_unique_ids():
    with pytest.raises(ValueError, match="duplicate supply"):
        optimize([Supply("s", "A", 1, 0), Supply("s", "B", 1, 0)], [], Policy())
    with pytest.raises(ValueError, match="duplicate order"):
        optimize([], [Order("o", "A", 1, 0), Order("o", "B", 1, 0)], Policy())


@pytest.mark.parametrize(
    "plan",
    [
        Plan((Allocation("o", "s", 3),), {"o": 0}, 0, "optimal"),
        Plan((Allocation("o", "s", 1),), {"o": 0}, 0, "optimal"),
        Plan((Allocation("o", "missing", 2),), {"o": 0}, 0, "optimal"),
        Plan((), {"o": True}, 0, "optimal"),
        Plan((), {}, 0, "optimal"),
    ],
)
def test_independent_checker_rejects_bad_plan(plan):
    with pytest.raises(ValueError):
        validate_plan([Supply("s", "A", 2, 0)], [Order("o", "A", 2, 1)], Policy(), plan)


def test_checker_rejects_ineligible_arc():
    with pytest.raises(ValueError, match="ineligible"):
        validate_plan(
            [Supply("s", "B", 2, 0)],
            [Order("o", "A", 2, 1)],
            Policy(),
            Plan((Allocation("o", "s", 2),), {"o": 0}, 0, "optimal"),
        )


def test_solver_failure_has_no_silent_fallback(monkeypatch):
    from types import SimpleNamespace
    import assumption_ops.optimizer as module

    monkeypatch.setattr(
        module, "milp", lambda **kwargs: SimpleNamespace(status=1, x=None, message="limit")
    )
    with pytest.raises(RuntimeError, match="usable solution"):
        optimize([], [Order("o", "A", 1, 0)], Policy())


def test_time_limit_incumbent_is_verified(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import assumption_ops.optimizer as module

    monkeypatch.setattr(
        module,
        "milp",
        lambda **kwargs: SimpleNamespace(status=1, x=np.array([1.0]), message="limit"),
    )
    assert optimize([], [Order("o", "A", 1, 0)], Policy()).status == "feasible_limit"
    monkeypatch.setattr(
        module,
        "milp",
        lambda **kwargs: SimpleNamespace(status=1, x=np.array([0.0]), message="limit"),
    )
    with pytest.raises(RuntimeError, match="infeasible solution"):
        optimize([], [Order("o", "A", 1, 0)], Policy())


def test_public_input_validator():
    from assumption_ops.optimizer import validate_inputs

    validate_inputs([Supply("s", "A", 1, 0)], [Order("o", "A", 1, 0)], Policy())
    with pytest.raises(ValueError, match="priority"):
        validate_inputs([], [Order("o", "A", 1, 0, 0)], Policy())


def test_randomized_tiny_cases_match_independent_brute_force():
    """Enumerate every integer assignment independently of the MILP model."""
    from itertools import product
    from random import Random

    rng = Random(20260929)
    for trial in range(15):
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
            substitutions={"A": ("B",)} if trial % 3 else {},
            allow_late=bool(trial % 2),
            late_penalty=rng.randrange(4),
            substitution_penalty=rng.randrange(4),
            disruption_penalty=rng.randrange(4),
            unfilled_penalty=rng.randrange(1, 6),
        )
        # Start with a feasible prior plan, then change availability and/or
        # supply presence to exercise newly ineligible and retired arcs.
        previous = optimize(supplies, orders, policy) if trial % 3 else None
        if previous is not None and trial % 4 == 0:
            supplies = [replace(supplies[0], available_day=3), supplies[1]]
        if previous is not None and trial % 5 == 0:
            supplies = supplies[1:]
        keys = [(order.id, supply.id) for order in orders for supply in supplies]
        supply_by_id = {supply.id: supply for supply in supplies}
        order_by_id = {order.id: order for order in orders}
        previous_quantities = (
            {(a.order_id, a.supply_id): a.quantity for a in previous.allocations}
            if previous
            else {}
        )
        best = float("inf")
        for assignment in product(
            *(range(min(order_by_id[o].quantity, supply_by_id[s].quantity) + 1) for o, s in keys)
        ):
            quantities = dict(zip(keys, assignment))
            if any(
                sum(q for (o, s), q in quantities.items() if s == supply.id) > supply.quantity
                for supply in supplies
            ):
                continue
            if any(
                sum(q for (o, s), q in quantities.items() if o == order.id) > order.quantity
                for order in orders
            ):
                continue
            cost = 0
            feasible = True
            for (order_id, supply_id), quantity in quantities.items():
                order, supply = order_by_id[order_id], supply_by_id[supply_id]
                if quantity and (
                    supply.sku != order.sku
                    and supply.sku not in policy.substitutions.get(order.sku, ())
                    or supply.available_day > order.due_day
                    and not policy.allow_late
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
                    q for (o, s), q in quantities.items() if o == order.id
                )
                cost += missing * order.priority * policy.unfilled_penalty
            if previous is not None:
                cost += policy.disruption_penalty * sum(
                    abs(quantities.get(key, 0) - previous_quantities.get(key, 0))
                    for key in quantities.keys() | previous_quantities.keys()
                )
            best = min(best, cost)
        actual = optimize(supplies, orders, policy, previous)
        assert actual.objective == pytest.approx(best), f"trial {trial}"
        validate_plan(supplies, orders, policy, actual)
