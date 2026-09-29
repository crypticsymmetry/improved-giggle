from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from assumption_ops import OperationsPipeline, Order, Policy, StaleDecisionError, Supply
from assumption_ops.execution import approve_plan, commit_plan


def seeded(path=":memory:"):
    pipeline = OperationsPipeline(str(path))
    pipeline.seed(
        [Supply("s1", "A", 2, 0), Supply("s2", "B", 2, 0)],
        [Order("o1", "A", 2, 0), Order("o2", "B", 2, 0)],
        Policy(),
    )
    return pipeline


def plan_id(result):
    return result.decisions[0].plan_id


def intents(pipeline):
    return [event for event in pipeline.events() if event["kind"] == "execution_intent"]


def test_plan_approval_and_commit_idempotence():
    with seeded() as p:
        result = p.plan()
        identifier = plan_id(result)
        approved = approve_plan(p, identifier)
        assert all(decision.status == "approved" for decision in approved)
        events = p.events()
        assert approve_plan(p, identifier) == approved
        assert p.events() == events
        outcome = commit_plan(p, identifier)
        assert outcome == {
            "plan_id": identifier,
            "status": "committed",
            "decision_ids": [d.id for d in result.decisions],
            "idempotent": False,
        }
        assert len(intents(p)) == 2
        events = p.events()
        assert commit_plan(p, identifier)["idempotent"]
        assert p.events() == events
        assert all(d.status == "committed" for d in approve_plan(p, identifier))
        assert p.events() == events


def test_no_partial_commit_without_all_approvals():
    with seeded() as p:
        result = p.plan()
        p.approve(result.decisions[0].id)
        before = p.events()
        with pytest.raises(ValueError, match="approval"):
            commit_plan(p, plan_id(result))
        assert p.events() == before
        assert p._reserved("supply_id") == {}
        assert [d.status for d in p.decisions()] == ["approved", "proposed"]


def test_stale_second_decision_rolls_back_first_intent():
    with seeded() as p:
        result = p.plan()
        approve_plan(p, plan_id(result))
        evidence = p.propose("supply:s2", "available_day", 1, "supplier revision")
        p.accept(evidence)
        p.resolve("supply:s2", "available_day", evidence)
        before = p.events()
        with pytest.raises(StaleDecisionError):
            commit_plan(p, plan_id(result))
        assert p.events() == before
        assert p._reserved("supply_id") == {}
        assert all(d.status == "approved" for d in p.decisions())


def test_injected_second_intent_failure_rolls_back_group(monkeypatch):
    with seeded() as p:
        result = p.plan()
        approve_plan(p, plan_id(result))
        before = p.events()
        original_event = p._event
        calls = 0

        def failing_event(kind, payload):
            nonlocal calls
            if kind == "execution_intent":
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated second intent insertion failure")
            original_event(kind, payload)

        monkeypatch.setattr(p, "_event", failing_event)
        with pytest.raises(RuntimeError, match="simulated"):
            commit_plan(p, plan_id(result))
        assert calls == 2
        assert p.events() == before
        assert p._reserved("supply_id") == {}
        assert all(d.status == "approved" for d in p.decisions())


def test_failed_approval_has_no_partial_status_or_event_changes():
    with seeded() as p:
        result = p.plan()
        evidence = p.propose("supply:s2", "available_day", 1, "supplier revision")
        p.accept(evidence)
        p.resolve("supply:s2", "available_day", evidence)
        before = p.events()
        with pytest.raises(StaleDecisionError):
            approve_plan(p, plan_id(result))
        assert p.events() == before
        assert all(d.status == "proposed" for d in p.decisions())


def test_group_approval_checks_aggregate_capacity():
    with seeded() as p:
        first, second = p.plan(), p.plan()
        # Simulate a persisted over-capacity group: each decision is feasible
        # alone, but collectively duplicates the same stock and order demand.
        with p._transaction():
            p._db.execute(
                "UPDATE ops_decisions SET plan_id=? WHERE plan_id=?",
                (plan_id(first), plan_id(second)),
            )
        before = p.events()
        with pytest.raises(StaleDecisionError, match="capacity"):
            approve_plan(p, plan_id(first))
        assert p.events() == before
        assert all(d.status == "proposed" for d in p.decisions())


def test_group_approval_checks_aggregate_order_demand():
    with OperationsPipeline() as p:
        p.seed([Supply("s", "A", 10, 0)], [Order("o", "A", 2, 0)], Policy())
        first, second = p.plan(), p.plan()
        with p._transaction():
            p._db.execute(
                "UPDATE ops_decisions SET plan_id=? WHERE plan_id=?",
                (plan_id(first), plan_id(second)),
            )
        with pytest.raises(StaleDecisionError, match="demand"):
            approve_plan(p, plan_id(first))
        assert all(d.status == "proposed" for d in p.decisions())


def test_mixed_individual_and_plan_commit_skips_historical_support():
    with seeded() as p:
        result = p.plan()
        first = result.decisions[0]
        approve_plan(p, plan_id(result))
        p.commit(first.id)
        evidence = p.propose("supply:s1", "available_day", 3, "historical stock correction")
        p.accept(evidence)
        p.resolve("supply:s1", "available_day", evidence)
        approve_plan(p, plan_id(result))
        assert not commit_plan(p, plan_id(result))["idempotent"]
        assert len(intents(p)) == 2
        assert p._reserved("supply_id") == {"s1": 2, "s2": 2}


def test_unknown_and_known_empty_plan():
    with OperationsPipeline() as p:
        p.seed([], [], Policy())
        p.plan()
        identifier = next(
            event["payload"]["plan_id"] for event in p.events() if event["kind"] == "plan_created"
        )
        before = p.events()
        assert approve_plan(p, identifier) == ()
        assert commit_plan(p, identifier) == {
            "plan_id": identifier,
            "status": "committed",
            "decision_ids": [],
            "idempotent": True,
        }
        assert p.events() == before
        with pytest.raises(KeyError):
            approve_plan(p, "unknown")
        with pytest.raises(KeyError):
            commit_plan(p, "unknown")
        assert p.events() == before


def test_reopen_replays_group_lifecycle(tmp_path):
    path = tmp_path / "ops.sqlite"
    with seeded(path) as p:
        result = p.plan()
        identifier = plan_id(result)
        approve_plan(p, identifier)
    with OperationsPipeline(str(path)) as p:
        assert not commit_plan(p, identifier)["idempotent"]
        events = p.events()
    with OperationsPipeline(str(path)) as p:
        assert commit_plan(p, identifier)["idempotent"]
        assert p.events() == events
        assert len(intents(p)) == 2


def test_concurrent_overlapping_plan_commits_have_one_winner(tmp_path):
    path = tmp_path / "shared.sqlite"
    with seeded(path) as p:
        first, second = p.plan(), p.plan()
        identifiers = [plan_id(first), plan_id(second)]
    barrier = Barrier(2)

    def worker(identifier):
        with OperationsPipeline(str(path)) as p:
            approve_plan(p, identifier)
            barrier.wait(timeout=10)
            try:
                commit_plan(p, identifier)
                return "committed"
            except StaleDecisionError:
                return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(worker, identifiers))
    assert sorted(outcomes) == ["committed", "stale"]
    with OperationsPipeline(str(path)) as p:
        assert p._reserved("supply_id") == {"s1": 2, "s2": 2}
        assert len(intents(p)) == 2
        assert sorted(d.status for d in p.decisions()) == [
            "approved",
            "approved",
            "committed",
            "committed",
        ]
