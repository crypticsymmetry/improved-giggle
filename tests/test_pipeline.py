"""Integration invariants: evidence lifecycle, persistence, and resource guards."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from assumption_ops import (
    EvidenceConflict,
    OperationsPipeline,
    Order,
    Policy,
    StaleDecisionError,
    Supply,
)


def seeded(path=":memory:"):
    pipeline = OperationsPipeline(str(path))
    pipeline.seed(
        [Supply("stock", "A", 4, 0), Supply("shipment", "A", 6, 2), Supply("spare", "B", 5, 0)],
        [Order("urgent", "A", 5, 3, 3), Order("standard", "A", 5, 4)],
        Policy(substitutions={"A": ("B",)}),
    )
    return pipeline


def test_proposals_do_not_change_plan_support():
    with seeded() as p:
        baseline = p.plan()
        p.propose("supply:shipment", "available_day", 8, "unreviewed message")
        assert p.impact()["affected_decisions"] == []
        assert all(p.explain(d.id)["logical_support"]["supported"] for d in baseline.decisions)


def test_conflict_resolution_replanning_and_stale_approval():
    with seeded() as p:
        baseline = p.plan()
        target = next(
            d for d in baseline.decisions if any(a.supply_id == "shipment" for a in d.allocations)
        )
        p.approve(target.id)
        evidence = p.propose("supply:shipment", "available_day", 8, "supplier update")
        p.accept(evidence)
        assert p.impact()["conflicts"]
        assert target.id in {d["decision_id"] for d in p.impact()["affected_decisions"]}
        with pytest.raises(EvidenceConflict):
            p.plan()
        with pytest.raises(StaleDecisionError):
            p.commit(target.id)
        p.resolve("supply:shipment", "available_day", evidence)
        revised = p.plan(previous=baseline.plan)
        assert all(a.supply_id != "shipment" for a in revised.plan.allocations)
        assert p.explain(target.id)["logical_support"]["supported"] is False
        with pytest.raises(StaleDecisionError):
            p.commit(target.id)


def test_local_commit_is_idempotent_and_replanning_excludes_reservations():
    with seeded() as p:
        result = p.plan()
        decision = result.decisions[0]
        p.approve(decision.id)
        assert p.commit(decision.id)["idempotent"] is False
        assert p.commit(decision.id)["idempotent"] is True
        assert len([e for e in p.events() if e["kind"] == "execution_intent"]) == 1
        updated = p.plan()
        assert not any(a.order_id == decision.order_id for a in updated.plan.allocations)
        assert updated.plan.unfilled[decision.order_id] == 0


def test_approval_required():
    with seeded() as p:
        decision = p.plan().decisions[0]
        with pytest.raises(ValueError, match="approval"):
            p.commit(decision.id)
        assert not p._reserved("supply_id")


def test_competing_plans_cannot_double_reserve():
    with seeded() as p:
        first, second = p.plan(), p.plan()
        d1 = first.decisions[0]
        d2 = next(d for d in second.decisions if d.order_id == d1.order_id)
        p.approve(d1.id)
        p.approve(d2.id)
        p.commit(d1.id)
        with pytest.raises(StaleDecisionError):
            p.commit(d2.id)
        assert p._reserved("order_id")[d1.order_id] == 5


def test_reopen_rebuilds_reasoning_and_keeps_reservations(tmp_path):
    path = tmp_path / "scenario.sqlite"
    p = seeded(path)
    decision = p.plan().decisions[0]
    p.approve(decision.id)
    p.commit(decision.id)
    p.close()
    with OperationsPipeline(str(path)) as restored:
        assert restored.explain(decision.id)["logical_support"]["supported"]
        assert restored.commit(decision.id)["idempotent"]
        assert restored._reserved("order_id")[decision.order_id] == 5
        with pytest.raises(ValueError, match="already"):
            restored.seed([], [], Policy())


def test_historical_identical_support_can_be_reactivated():
    with seeded() as p:
        old = p.store.current("order:urgent", "priority")
        first = next(d for d in p.plan().decisions if d.order_id == "urgent")
        duplicate = p.propose("order:urgent", "priority", 3, "second confirmation")
        p.accept(duplicate)
        second = next(d for d in p.plan().decisions if d.order_id == "urgent")
        p.resolve("order:urgent", "priority", old.id)
        assert p.explain(first.id)["logical_support"]["supported"]
        assert not p.explain(second.id)["logical_support"]["supported"]


@pytest.mark.parametrize(
    "entity,field,value",
    [
        ("supply:shipment", "quantity", -1),
        ("supply:shipment", "available_day", True),
        ("order:urgent", "priority", 0),
        ("policy", "config", {"allow_late": "yes"}),
        ("policy", "config", {"late_penalty": -1}),
        ("order:urgent", "sku", " "),
        ("unknown", "quantity", 1),
    ],
)
def test_invalid_proposals_rejected(entity, field, value):
    with seeded() as p:
        with pytest.raises((ValueError, TypeError)):
            p.propose(entity, field, value, "source")


def test_seed_is_atomic_on_midway_failure(tmp_path):
    with OperationsPipeline(str(tmp_path / "atomic.sqlite")) as p:
        original = p.store.accept
        calls = 0

        def failing_accept(evidence_id):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated crash")
            original(evidence_id)

        with patch.object(p.store, "accept", side_effect=failing_accept):
            with pytest.raises(RuntimeError):
                p.seed([Supply("s", "A", 1, 0)], [Order("o", "A", 1, 1)], Policy())
        assert p.store.list_evidence() == []
        assert p.events() == []
        p.seed([Supply("s", "A", 1, 0)], [Order("o", "A", 1, 1)], Policy())
        assert p.plan().plan.unfilled == {"o": 0}


def test_two_instances_serialize_competing_commits(tmp_path):
    path = str(tmp_path / "shared.sqlite")
    with seeded(path) as p:
        first = p.plan().decisions[0]
        second = next(d for d in p.plan().decisions if d.order_id == first.order_id)
        p.approve(first.id)
        p.approve(second.id)

    def commit(decision_id):
        with OperationsPipeline(path) as p:
            try:
                return p.commit(decision_id)["status"]
            except StaleDecisionError:
                return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(commit, [first.id, second.id]))
    assert sorted(outcomes) == ["committed", "stale"]
    with OperationsPipeline(path) as p:
        assert len([e for e in p.events() if e["kind"] == "execution_intent"]) == 1


def test_committed_capacity_reduction_requires_compensation():
    with seeded() as p:
        d = p.plan().decisions[0]
        p.approve(d.id)
        p.commit(d.id)
        allocated = d.allocations[0]
        changed = p.propose(f"supply:{allocated.supply_id}", "quantity", 0, "stock correction")
        p.accept(changed)
        p.resolve(f"supply:{allocated.supply_id}", "quantity", changed)
        assert p.impact()["affected_decisions"]
        with pytest.raises(StaleDecisionError, match="reservations"):
            p.plan()


def test_unrelated_evidence_does_not_deactivate_allocated_support():
    with seeded() as p:
        d = p.plan().decisions[0]
        unused = p.propose("supply:spare", "unit_cost", 4, "catalog")
        p.accept(unused)
        assert p.explain(d.id)["logical_support"]["supported"]
        # But global unresolved conflicts still block approval conservatively.
        with pytest.raises(EvidenceConflict):
            p.approve(d.id)
        p.resolve("supply:spare", "unit_cost", unused)
        p.approve(d.id)
        p.commit(d.id)


def test_what_if_is_read_only_and_matches_actual_replan():
    with seeded() as p:
        baseline = p.plan()
        revision, event_count, decisions = p.store.revision, len(p.events()), p.decisions()
        simulated = p.what_if({"supply:shipment": {"available_day": 8}}, previous=baseline.plan)
        assert p.store.revision == revision
        assert len(p.events()) == event_count
        assert p.decisions() == decisions
        changed = p.propose("supply:shipment", "available_day", 8, "confirmed delay")
        p.accept(changed)
        p.resolve("supply:shipment", "available_day", changed)
        actual = p.plan(previous=baseline.plan).plan
        assert simulated.objective == actual.objective
        assert simulated.unfilled == actual.unfilled


@pytest.mark.parametrize(
    "overrides",
    [
        {"missing": {"quantity": 1}},
        {"order:urgent": {"id": "changed"}},
        {"supply:stock": {"quantity": -3}},
        {"policy": {"invalid": True}},
    ],
)
def test_what_if_rejects_invalid_overrides(overrides):
    with seeded() as p:
        with pytest.raises((ValueError, TypeError)):
            p.what_if(overrides)


def test_what_if_keeps_committed_reservations():
    with seeded() as p:
        d = p.plan().decisions[0]
        p.approve(d.id)
        p.commit(d.id)
        result = p.what_if({})
        assert not any(a.order_id == d.order_id for a in result.allocations)
