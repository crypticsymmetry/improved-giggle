from dataclasses import replace
import json

import pytest

from assumption_ops.evaluation import CostBreakdown, evaluate_plan
from assumption_ops.optimizer import Allocation, Order, Plan, Policy, Supply, optimize


def test_all_cost_components_and_service_metrics():
    supplies = [Supply("same", "A", 2, 0, 3), Supply("sub", "B", 3, 4, 5)]
    orders = [Order("o", "A", 6, 1, 2), Order("empty", "A", 0, 1)]
    policy = Policy(
        substitutions={"A": ("B",)},
        allow_late=True,
        late_penalty=7,
        substitution_penalty=4,
        unfilled_penalty=10,
        disruption_penalty=3,
    )
    prior = Plan((Allocation("o", "same", 3), Allocation("retired", "old", 2)), {}, 0, "optimal")
    # acquisition=21, lateness=63, substitution=12, shortage=20;
    # changed units=1 removed same + 3 new sub + 2 retired = 6.
    plan = Plan(
        (Allocation("o", "same", 2), Allocation("o", "sub", 3)),
        {"o": 1, "empty": 0},
        134,
        "optimal",
    )
    result = evaluate_plan(supplies, orders, policy, plan, prior)
    assert result.costs == CostBreakdown(21, 63, 12, 20, 18)
    assert result.costs.total == 134
    assert result.metrics == {
        "requested_units": 6,
        "allocated_units": 5,
        "unfilled_units": 1,
        "priority_weighted_unfilled": 2,
        "on_time_units": 2,
        "late_units": 3,
        "substitute_units": 3,
        "unit_days_late": 9,
        "changed_units": 6,
        "fully_filled_orders": 1,
        "total_orders": 2,
    }
    assert result.orders == (
        {
            "order_id": "o",
            "requested": 6,
            "allocated": 5,
            "unfilled": 1,
            "on_time": 2,
            "late": 3,
            "substitute": 3,
            "priority": 2,
        },
        {
            "order_id": "empty",
            "requested": 0,
            "allocated": 0,
            "unfilled": 0,
            "on_time": 0,
            "late": 0,
            "substitute": 0,
            "priority": 1,
        },
    )
    serialized = result.to_dict()
    assert serialized["costs"]["total"] == 134
    assert json.loads(json.dumps(serialized)) == serialized
    serialized["metrics"]["total_orders"] = 100
    serialized["orders"][0]["allocated"] = 100
    assert result.metrics["total_orders"] == 2
    assert result.orders[0]["allocated"] == 5


def test_no_previous_does_not_charge_disruption():
    supplies, orders = [Supply("s", "A", 3, 0, 2)], [Order("o", "A", 3, 0)]
    policy = Policy(disruption_penalty=100)
    plan = optimize(supplies, orders, policy)
    result = evaluate_plan(supplies, orders, policy, plan)
    assert result.costs == CostBreakdown(6, 0, 0, 0, 0)
    assert result.metrics["changed_units"] == 3
    explicit_empty = Plan((), {}, 0, "optimal")
    with pytest.raises(ValueError, match="objective"):
        evaluate_plan(supplies, orders, policy, plan, explicit_empty)
    charged = replace(plan, objective=306)
    assert evaluate_plan(supplies, orders, policy, charged, explicit_empty).costs.disruption == 300


def test_duplicate_prior_and_current_arcs_aggregate_before_change_accounting():
    supplies, orders, policy = [Supply("s", "A", 3, 0)], [Order("o", "A", 3, 0)], Policy()
    previous = Plan((Allocation("o", "s", 1), Allocation("o", "s", 2)), {"o": 0}, 0, "optimal")
    current = Plan((Allocation("o", "s", 2), Allocation("o", "s", 1)), {"o": 0}, 0, "optimal")
    result = evaluate_plan(supplies, orders, policy, current, previous)
    assert result.metrics["changed_units"] == 0
    assert result.metrics["allocated_units"] == 3


def test_empty_and_removed_arcs():
    policy = Policy(disruption_penalty=4)
    result = evaluate_plan([], [], policy, optimize([], [], policy))
    assert result.costs.total == 0
    assert all(value == 0 for value in result.metrics.values())
    assert result.orders == ()
    previous = Plan((Allocation("gone", "gone", 7),), {}, 0, "optimal")
    plan = optimize([], [], policy, previous)
    result = evaluate_plan([], [], policy, plan, previous)
    assert result.costs.disruption == 28
    assert result.metrics["changed_units"] == 7


@pytest.mark.parametrize(
    "allocation",
    [
        Allocation("o", "s", True),
        Allocation("o", "s", -1),
        Allocation("o", "s", 1.0),
        Allocation("", "s", 1),
        Allocation(" ", "s", 1),
        Allocation("o", "", 1),
        Allocation(True, "s", 1),
    ],
)
def test_invalid_previous_allocations_rejected(allocation):
    previous = Plan((allocation,), {}, 0, "optimal")
    with pytest.raises(ValueError):
        evaluate_plan([], [], Policy(), Plan((), {}, 0, "optimal"), previous)


def test_feasibility_is_checked_before_accounting():
    with pytest.raises(ValueError, match="capacity"):
        evaluate_plan(
            [Supply("s", "A", 1, 0)],
            [Order("o", "A", 2, 0)],
            Policy(),
            Plan((Allocation("o", "s", 2),), {"o": 0}, 0, "optimal"),
        )


def test_bad_objective_and_rounding_tolerance():
    supplies, orders, policy = [Supply("s", "A", 1, 0, 10)], [Order("o", "A", 1, 0)], Policy()
    plan = optimize(supplies, orders, policy)
    with pytest.raises(ValueError, match="reconstructed"):
        evaluate_plan(supplies, orders, policy, replace(plan, objective=11))
    assert (
        evaluate_plan(supplies, orders, policy, replace(plan, objective=10 + 1e-7)).costs.total
        == 10
    )


def test_optimizer_objectives_reconstruct_across_previous_scenarios():
    supplies = [Supply("a", "A", 2, 0, 2), Supply("b", "B", 3, 2, 1)]
    orders = [Order("o", "A", 4, 1, 2)]
    policy = Policy(substitutions={"A": ("B",)}, allow_late=True)
    first = optimize(supplies, orders, policy)
    evaluate_plan(supplies, orders, policy, first)
    second = optimize(supplies[1:], orders, policy, first)
    result = evaluate_plan(supplies[1:], orders, policy, second, first)
    assert result.costs.total == second.objective
    assert result.metrics["changed_units"] == 3
