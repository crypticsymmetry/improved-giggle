"""Independent business-shaped oracles for offline replay accounting."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from unittest.mock import patch

from assumption_ops.events import EvidenceUpdate
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.pipeline import OperationsPipeline
from assumption_ops.replay import ReplayBatch, ReplayCase, run_replay


def scarce_substitute_case(batches=()):
    return ReplayCase(
        "scarce asymmetric substitute",
        (Supply("A-stock", "A", 1, 0, 3), Supply("B-stock", "B", 1, 0, 0)),
        (Order("wants-A", "A", 1, 1, 2), Order("wants-B", "B", 1, 1, 1)),
        Policy(substitutions={"A": ("B",)}, disruption_penalty=4, unfilled_penalty=100),
        tuple(batches),
        True,
        17,
    )


def brute_force_minimum(supplies, orders, policy, previous=None):
    """Enumerate every feasible integer matrix; no optimizer/accounting imports."""
    pairs = [(order, supply) for order in orders for supply in supplies]
    choices = []
    for order, supply in pairs:
        eligible = (
            supply.sku == order.sku or supply.sku in policy.substitutions.get(order.sku, ())
        ) and (policy.allow_late or supply.available_day <= order.due_day)
        choices.append(range(min(order.quantity, supply.quantity) + 1) if eligible else (0,))
    old = {}
    if previous is not None:
        for allocation in previous["allocations"]:
            key = (allocation["order_id"], allocation["supply_id"])
            old[key] = old.get(key, 0) + allocation["quantity"]
    best = float("inf")
    for matrix in product(*choices):
        used = {supply.id: 0 for supply in supplies}
        filled = {order.id: 0 for order in orders}
        current = {}
        cost = 0
        for (order, supply), quantity in zip(pairs, matrix):
            used[supply.id] += quantity
            filled[order.id] += quantity
            if quantity:
                current[(order.id, supply.id)] = quantity
            cost += quantity * supply.unit_cost
            cost += quantity * max(supply.available_day - order.due_day, 0) * policy.late_penalty
            cost += quantity * (supply.sku != order.sku) * policy.substitution_penalty
        if any(used[s.id] > s.quantity for s in supplies):
            continue
        if any(filled[o.id] > o.quantity for o in orders):
            continue
        cost += sum(
            (o.quantity - filled[o.id]) * o.priority * policy.unfilled_penalty for o in orders
        )
        if previous is not None:
            cost += policy.disruption_penalty * sum(
                abs(current.get(key, 0) - old.get(key, 0)) for key in current.keys() | old.keys()
            )
        best = min(best, cost)
    return best


def test_every_snapshot_matches_independent_exhaustive_business_oracle():
    case = scarce_substitute_case(
        [
            ReplayBatch("price", (EvidenceUpdate("supply:A-stock", "unit_cost", 5, "price list"),)),
            ReplayBatch(
                "cancel", (EvidenceUpdate("supply:A-stock", "quantity", 0, "cancellation"),)
            ),
            ReplayBatch(
                "restore",
                (
                    EvidenceUpdate("supply:A-stock", "quantity", 1, "correction"),
                    EvidenceUpdate("supply:A-stock", "unit_cost", 4, "corrected price"),
                ),
            ),
        ]
    )
    report = run_replay(case)
    supplies = list(case.supplies)
    initial = report["steps"][0]["optimized"]["plan"]
    for index, step in enumerate(report["steps"]):
        if index:
            for update in case.batches[index - 1].updates:
                supplies = [
                    replace(s, **{update.field: update.value})
                    if update.entity == f"supply:{s.id}"
                    else s
                    for s in supplies
                ]
        expected = brute_force_minimum(
            supplies, case.orders, case.policy, initial if index else None
        )
        assert step["optimized"]["evaluation"]["costs"]["total"] == expected
        assert step["optimized"]["status"] == "optimal"
        assert step["optimized_no_worse_when_proven"]
    assert report["summary"]["final_optimized"]["costs"]["total"] == 4


def test_baseline_scoring_uses_common_initial_incumbent_not_its_own_plan():
    case = scarce_substitute_case(
        [
            ReplayBatch("price", (EvidenceUpdate("supply:A-stock", "unit_cost", 5, "new price"),)),
        ]
    )
    steps = run_replay(case)["steps"]
    assert steps[0]["optimized"]["evaluation"]["costs"]["total"] == 3
    assert steps[0]["greedy"]["evaluation"]["costs"]["total"] == 102
    # Greedy moves A to B, retires A->A and B->B: three arc-unit changes at cost4.
    # Scoring against greedy's own previous plan would incorrectly report zero.
    assert steps[1]["greedy"]["evaluation"]["costs"]["disruption"] == 12
    assert steps[1]["greedy"]["evaluation"]["costs"]["total"] == 114
    assert steps[1]["optimized"]["evaluation"]["costs"]["disruption"] == 0


def test_repeated_corrections_count_final_demand_once_and_follow_trace_order():
    case = scarce_substitute_case(
        [
            ReplayBatch(
                f"price-{index}",
                (EvidenceUpdate("supply:A-stock", "unit_cost", price, f"bulletin {index}"),),
            )
            for index, price in enumerate((8, 3, 9, 4), start=1)
        ]
    )
    report = run_replay(case)
    assert report["summary"]["snapshots"] == 5
    assert report["summary"]["final_optimized"]["metrics"]["requested_units"] == 2
    assert report["summary"]["final_greedy"]["metrics"]["requested_units"] == 2
    assert report["summary"]["final_optimized"]["costs"]["acquisition"] == 4
    assert [
        step["optimized"]["evaluation"]["costs"]["acquisition"] for step in report["steps"]
    ] == [3, 8, 3, 9, 4]
    assert [step["event_id"] for step in report["steps"]] == [
        None,
        "price-1",
        "price-2",
        "price-3",
        "price-4",
    ]
    assert report["summary"]["duplicate_replays_verified"] == 4


def test_review_proxy_uses_only_preceding_active_lines_and_exact_provenance():
    case = scarce_substitute_case(
        [
            ReplayBatch(
                f"price-{index}",
                (EvidenceUpdate("supply:A-stock", "unit_cost", 3, f"confirmation {index}"),),
            )
            for index in range(1, 4)
        ]
    )
    report = run_replay(case)
    # Every new source observation replaces A-stock provenance even at the same
    # value; only the immediately preceding wants-A decision is invalidated.
    assert [step["previous_decisions_invalidated"] for step in report["steps"]] == [0, 1, 1, 1]
    assert [step["proposed_decision_lines"] for step in report["steps"]] == [2, 2, 2, 2]
    assert report["summary"]["cumulative_previous_decisions_invalidated"] == 3
    assert report["summary"]["cumulative_changed_allocation_units"] == 0


def test_source_inputs_remain_independent_of_report_and_repeated_execution():
    case = scarce_substitute_case(
        [
            ReplayBatch("price", (EvidenceUpdate("supply:A-stock", "unit_cost", 5, "new price"),)),
        ]
    )
    original = deepcopy(case.to_dict())
    first = run_replay(case)
    assert case.to_dict() == original
    first["steps"][0]["optimized"]["plan"]["unfilled"]["wants-A"] = 99
    case_manifest = case.to_dict()
    case_manifest["initial"]["policy"]["substitutions"]["A"].append("other")
    assert case.to_dict() == original
    second = run_replay(case)
    assert first["source_fingerprint"] == second["source_fingerprint"]
    assert second["steps"][0]["optimized"]["plan"]["unfilled"]["wants-A"] == 0
    assert first["summary"]["final_optimized"] == second["summary"]["final_optimized"]


def test_replay_never_crosses_approval_or_commit_boundaries():
    case = scarce_substitute_case(
        [
            ReplayBatch("price", (EvidenceUpdate("supply:A-stock", "unit_cost", 5, "new price"),)),
        ]
    )
    with (
        patch.object(
            OperationsPipeline, "approve", side_effect=AssertionError("approval")
        ) as approve,
        patch.object(OperationsPipeline, "commit", side_effect=AssertionError("commit")) as commit,
        patch.object(
            OperationsPipeline, "approve_plan", side_effect=AssertionError("approval")
        ) as approve_plan,
        patch.object(
            OperationsPipeline, "commit_plan", side_effect=AssertionError("commit")
        ) as commit_plan,
    ):
        report = run_replay(case)
    assert report["summary"]["execution_intents"] == 0
    for method in (approve, commit, approve_plan, commit_plan):
        method.assert_not_called()
