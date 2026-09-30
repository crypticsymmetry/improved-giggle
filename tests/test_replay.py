from copy import deepcopy
from dataclasses import replace
import json

import pytest

from assumption_ops.events import EvidenceUpdate
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.replay import ReplayBatch, ReplayCase, run_replay


def adversarial():
    return ReplayCase(
        "opportunity-cost",
        (Supply("A", "A", 1, 0, 3), Supply("B", "B", 1, 0)),
        (Order("high", "A", 1, 0, 2), Order("low", "B", 1, 0)),
        Policy(substitutions={"A": ("B",)}, substitution_penalty=0, unfilled_penalty=100),
        (ReplayBatch("price", (EvidenceUpdate("supply:A", "unit_cost", 5, "confirmed price"),)),),
        synthetic=True,
        seed=1729,
    )


def test_priority_scarcity_greedy_gap_and_update_replay():
    result = run_replay(adversarial())
    initial, updated = result["steps"]
    assert initial["optimized"]["evaluation"]["costs"]["total"] == 3
    assert initial["greedy"]["evaluation"]["costs"]["total"] == 100
    assert initial["greedy_minus_optimized"] == 97
    assert updated["optimized"]["evaluation"]["costs"]["total"] == 5
    # Retire two incumbent arcs and add one greedy arc: L1 disruption is 3.
    assert updated["greedy"]["evaluation"]["costs"]["disruption"] == 3
    assert updated["greedy_minus_optimized"] == 100 + 3 - 5
    assert updated["duplicate_replay_verified"] is True
    assert updated["previous_active_decision_count"] == 2
    assert updated["previous_decisions_invalidated"] == 1
    assert updated["changed_allocation_arcs"] == 0
    assert all(step["optimized_no_worse_when_proven"] is True for step in result["steps"])
    assert result["summary"]["duplicate_replays_verified"] == 1
    assert result["summary"]["execution_intents"] == 0
    assert result["summary"]["final_optimized"]["metrics"]["requested_units"] == 2
    assert result["synthetic"] is True
    assert len(result["source_fingerprint"]) == 64
    json.dumps(result, allow_nan=False)


def test_input_immutable_and_fingerprint_stable():
    case = adversarial()
    before = deepcopy(case.to_dict())
    first, second = run_replay(case), run_replay(case)
    assert case.to_dict() == before
    assert first["source_fingerprint"] == second["source_fingerprint"]
    for left, right in zip(first["steps"], second["steps"]):
        assert left["optimized"]["evaluation"] == right["optimized"]["evaluation"]
        assert left["greedy"]["evaluation"] == right["greedy"]["evaluation"]


def test_chronology_and_fixed_initial_reference():
    case = ReplayCase(
        "repair",
        (Supply("first", "A", 1, 0), Supply("backup", "A", 1, 0, 2)),
        (Order("o", "A", 1, 0),),
        Policy(disruption_penalty=3),
        (
            ReplayBatch("delay", (EvidenceUpdate("supply:first", "available_day", 1, "delay"),)),
            ReplayBatch(
                "return", (EvidenceUpdate("supply:first", "available_day", 0, "correction"),)
            ),
        ),
    )
    report = run_replay(case)
    start, delay, restored = report["steps"]
    assert start["optimized"]["evaluation"]["costs"]["total"] == 0
    assert delay["optimized"]["evaluation"]["costs"]["total"] == 8
    assert delay["optimized"]["evaluation"]["costs"]["disruption"] == 6
    assert restored["optimized"]["evaluation"]["costs"]["total"] == 0
    assert restored["optimized"]["evaluation"]["costs"]["disruption"] == 0
    assert [step["changed_allocation_units"] for step in report["steps"]] == [0, 2, 2]
    # At correction the active backup decision does not depend on first's date;
    # nevertheless reoptimization changes the best allocation.
    assert restored["previous_decisions_invalidated"] == 0
    assert restored["changed_allocation_arcs"] == 2
    assert report["summary"]["cumulative_changed_allocation_units"] == 4
    assert report["summary"]["final_optimized"]["metrics"]["requested_units"] == 1
    assert delay["revisions"]["evidence_revision"] > start["revisions"]["evidence_revision"]
    assert restored["revisions"]["operations_revision"] > delay["revisions"]["operations_revision"]


def test_greedy_declines_units_costing_at_least_shortage_penalty():
    case = ReplayCase(
        "cost",
        (Supply("s", "A", 2, 0, 10),),
        (Order("o", "A", 2, 0),),
        Policy(unfilled_penalty=10),
        (),
    )
    report = run_replay(case)
    greedy = report["steps"][0]["greedy"]
    assert greedy["plan"]["allocations"] == []
    assert greedy["evaluation"]["costs"]["total"] == 20


def test_empty_case_is_safe():
    result = run_replay(ReplayCase("empty", (), (), Policy(), ()))
    assert len(result["steps"]) == 1
    assert result["summary"]["final_optimized"]["costs"]["total"] == 0
    assert result["summary"]["final_greedy"]["metrics"]["total_orders"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"synthetic": 1},
        {"seed": True},
        {"seed": -1},
        {"supplies": []},
        {"orders": []},
        {"batches": []},
        {"policy": {}},
    ],
)
def test_strict_case_validation(changes):
    with pytest.raises(ValueError):
        replace(adversarial(), **changes)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "bad"}, ("bad",)])
def test_strict_json_update_values(value):
    with pytest.raises(ValueError):
        ReplayBatch("e", (EvidenceUpdate("supply:A", "sku", value, "source"),))


def test_duplicate_events_and_fields_rejected():
    case = adversarial()
    with pytest.raises(ValueError, match="unique"):
        replace(case, batches=(case.batches[0], case.batches[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        ReplayBatch(
            "e",
            (
                EvidenceUpdate("supply:A", "quantity", 1, "s"),
                EvidenceUpdate("supply:A", "quantity", 2, "s"),
            ),
        )


@pytest.mark.parametrize(
    "update",
    [
        EvidenceUpdate("missing", "quantity", 1, "s"),
        EvidenceUpdate("supply:A", "id", "new", "s"),
        EvidenceUpdate("supply:A", "quantity", True, "s"),
        EvidenceUpdate("order:high", "priority", 0, "s"),
        EvidenceUpdate("policy", "config", {"allow_late": "yes"}, "s"),
    ],
)
def test_domain_update_validation(update):
    with pytest.raises(ValueError):
        replace(adversarial(), batches=(ReplayBatch("e", (update,)),))


def test_mutable_nested_values_revalidated_before_execution():
    config = {"allow_late": True}
    case = replace(
        adversarial(),
        batches=(ReplayBatch("e", (EvidenceUpdate("policy", "config", config, "s"),)),),
    )
    config["allow_late"] = "not boolean"
    with pytest.raises(ValueError):
        run_replay(case)


def test_circular_update_value_rejected():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="circular"):
        ReplayBatch("e", (EvidenceUpdate("policy", "config", value, "s"),))
