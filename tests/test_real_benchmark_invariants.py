"""Truthful source projection and independently checked allocation comparisons."""

from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from assumption_ops.baselines import compare_allocators
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.warehouse_data import (
    WarehouseData,
    WarehouseRecord,
    read_warehouse,
    warehouse_scenario,
)


def example_data():
    return WarehouseData(
        {"A": 3, "B": 4},
        (
            WarehouseRecord(0, "A", 2),
            WarehouseRecord(4, "A", 3),
            WarehouseRecord(4, "A", 2),
            WarehouseRecord(1, "B", 2),
        ),
        (WarehouseRecord(2, "A", 3), WarehouseRecord(2, "A", 4), WarehouseRecord(4, "B", 5)),
        {"inferred_demand_mapping": True, "author_confirmation": False},
    )


def independent_minimum(supplies, orders, policy):
    """Enumerate feasible matrices with independent eligibility/cost accounting."""
    pairs = [(order, supply) for order in orders for supply in supplies]
    domains = []
    for order, supply in pairs:
        allowed = supply.sku == order.sku or supply.sku in policy.substitutions.get(order.sku, ())
        allowed = allowed and (policy.allow_late or supply.available_day <= order.due_day)
        domains.append(range(min(order.quantity, supply.quantity) + 1) if allowed else (0,))
    scores = []
    for amounts in product(*domains):
        used, filled, score = {}, {}, 0
        for (order, supply), quantity in zip(pairs, amounts):
            used[supply.id] = used.get(supply.id, 0) + quantity
            filled[order.id] = filled.get(order.id, 0) + quantity
            score += quantity * (
                supply.unit_cost
                + max(0, supply.available_day - order.due_day) * policy.late_penalty
                + (supply.sku != order.sku) * policy.substitution_penalty
            )
        if any(used.get(s.id, 0) > s.quantity for s in supplies):
            continue
        if any(filled.get(o.id, 0) > o.quantity for o in orders):
            continue
        score += sum(
            (o.quantity - filled.get(o.id, 0)) * o.priority * policy.unfilled_penalty
            for o in orders
        )
        scores.append(score)
    return min(scores)


def test_inferred_mapping_requires_explicit_opt_in_before_optional_parser():
    # Source verification is separate from semantic approval. A verified file
    # must still not silently turn a suspect physical column into customer demand.
    with patch("assumption_ops.warehouse_data._verify"):
        with pytest.raises(ValueError, match="allow_inferred_demand=True"):
            read_warehouse("already-checksummed.accdb")


def test_stock_stress_changes_only_aggregated_supply_not_observed_order_inputs():
    data = example_data()
    before = deepcopy((data.initial_inventory, data.production, data.demand, data.provenance))
    baseline = warehouse_scenario(data)
    stressed = warehouse_scenario(data, stock_fraction=0.5)
    assert stressed.orders == baseline.orders
    assert stressed.policy == baseline.policy
    # Initial3 and day0 production2 form one five-unit lot before flooring.
    # The two day4 production rows likewise form a five-unit lot. Floor each
    # aggregated lot once rather than independently flooring every raw row.
    assert {(s.sku, s.available_day): s.quantity for s in baseline.supplies} == {
        ("A", 0): 5,
        ("A", 4): 5,
        ("B", 0): 4,
        ("B", 1): 2,
    }
    assert {(s.sku, s.available_day): s.quantity for s in stressed.supplies} == {
        ("A", 0): 2,
        ("A", 4): 2,
        ("B", 0): 2,
        ("B", 1): 1,
    }
    assert {(o.sku, o.due_day): o.quantity for o in baseline.orders} == {("A", 2): 7, ("B", 4): 5}
    assert (data.initial_inventory, data.production, data.demand, data.provenance) == before
    assert all(s.unit_cost == 0 for s in baseline.supplies)
    assert all(o.priority == 1 for o in baseline.orders)


def test_zero_supply_stress_is_shortage_accounting_not_a_demand_forecast():
    scenario = warehouse_scenario(example_data(), stock_fraction=0)
    report = compare_allocators(scenario.supplies, scenario.orders, scenario.policy)
    assert all(s.quantity == 0 for s in scenario.supplies)
    assert report["lp_relaxation"]["lower_bound"] == 12_000
    for method in report["results"]:
        metrics = method["evaluation"]["metrics"]
        assert metrics["requested_units"] == 12
        assert metrics["allocated_units"] == 0
        assert metrics["unfilled_units"] == 12
        assert method["evaluation"]["costs"]["total"] == 12_000
        assert method["gap_to_lp_bound"] == 0
    assert report["external_dispatch"] is False


def test_methods_share_fixed_economics_and_integral_lp_ties_independent_oracle():
    supplies = [Supply("A-stock", "A", 1, 0, 3), Supply("B-stock", "B", 1, 0, 0)]
    orders = [Order("wants-A", "A", 1, 1, 2), Order("wants-B", "B", 1, 1, 1)]
    policy = Policy(substitutions={"A": ("B",)}, unfilled_penalty=100)
    before = deepcopy((supplies, orders, policy))
    expected = independent_minimum(supplies, orders, policy)
    assert expected == 3
    report = compare_allocators(supplies, orders, policy)
    methods = {row["method"]: row for row in report["results"]}
    assert report["lp_relaxation"]["status"] == "optimal"
    assert report["lp_relaxation"]["lower_bound"] == expected
    assert methods["milp"]["evaluation"]["costs"]["total"] == expected
    assert report["milp_proven_optimal"]
    # Priority+cheapest greedily uses scarce B on A, leaving B demand unfilled.
    assert methods["priority_cost_greedy"]["evaluation"]["costs"]["total"] == 102
    assert methods["priority_cost_greedy"]["gap_to_lp_bound"] == 99
    for method in methods.values():
        assert method["evaluation"]["metrics"]["requested_units"] == 2
        assert method["evaluation"]["costs"]["disruption"] == 0
        assert method["plan"]["objective"] == method["evaluation"]["costs"]["total"]
    assert (supplies, orders, policy) == before


def test_later_production_is_available_later_not_predicted_or_usable_early():
    data = WarehouseData({}, (WarehouseRecord(5, "A", 2),), (WarehouseRecord(1, "A", 2),), {})
    scenario = warehouse_scenario(data)
    assert scenario.supplies[0].available_day == 5
    assert scenario.orders[0].due_day == 1
    report = compare_allocators(scenario.supplies, scenario.orders, scenario.policy)
    optimal = next(row for row in report["results"] if row["method"] == "milp")
    assert optimal["evaluation"]["metrics"]["on_time_units"] == 0
    assert optimal["evaluation"]["metrics"]["late_units"] == 2
    assert optimal["evaluation"]["costs"]["lateness"] == 80
    assert optimal["evaluation"]["costs"]["shortage"] == 0


def test_optional_pinned_source_counts_remain_provisional():
    path = Path(
        os.environ.get(
            "ASSUMPTION_OPS_WAREHOUSE_PATH",
            str(Path(__file__).resolve().parents[1] / "warehouse_results" / "source.accdb"),
        )
    )
    if not path.is_file():
        pytest.skip("Pinned source file not available; no CI network download")
    pytest.importorskip("access_parser", reason="Optional datasets parser unavailable")
    data = read_warehouse(path, allow_inferred_demand=True)
    assert sum(data.initial_inventory.values()) == 10_709
    assert sum(row.quantity for row in data.production) == 36_610
    assert sum(row.quantity for row in data.demand) == 25_686
    assert data.provenance["inferred_demand_mapping"] is True
    assert data.provenance["author_confirmation"] is False
    assert data.provenance["demand_mapping"]["quantity"] == "int(D.item_WEEK)"
    scenario = warehouse_scenario(data)
    assert sum(order.quantity for order in scenario.orders) == 25_686
    assert sum(supply.quantity for supply in scenario.supplies) == 47_319
    assert "not author-confirmed" in data.provenance["warning"]
