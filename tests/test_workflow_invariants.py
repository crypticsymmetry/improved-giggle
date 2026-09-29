"""Cross-component failure, replay, and snapshot invariants of reviewed work."""

from unittest.mock import patch

import pytest

from assumption_ops.events import EvidenceUpdate, StaleSnapshotError
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.pipeline import OperationsPipeline


def seeded(path=":memory:"):
    pipeline = OperationsPipeline(str(path))
    pipeline.seed(
        [Supply("a", "A", 2, 0), Supply("b", "B", 2, 0)],
        [Order("first", "A", 2, 3), Order("second", "B", 2, 3)],
        Policy(),
    )
    return pipeline


def apply(pipeline, event_id, updates, snapshot=None):
    snapshot = pipeline.revisions() if snapshot is None else snapshot
    return pipeline.apply_updates(
        event_id,
        updates,
        expected_evidence_revision=snapshot["evidence_revision"],
        expected_operations_revision=snapshot["operations_revision"],
    )


def test_reviewed_batch_syncs_only_after_both_winners_are_durable():
    with seeded() as pipeline:
        baseline = pipeline.plan()
        sync = pipeline._sync
        observed = []

        def check_committed_snapshot():
            observed.append(pipeline._db.in_transaction)
            assert not pipeline._db.in_transaction
            assert pipeline.store.current("supply:a", "available_day").value == 5
            assert pipeline.store.current("supply:b", "available_day").value == 6
            sync()

        with patch.object(pipeline, "_sync", side_effect=check_committed_snapshot):
            apply(
                pipeline,
                "both-delayed",
                [
                    EvidenceUpdate("supply:a", "available_day", 5, "supplier A"),
                    EvidenceUpdate("supply:b", "available_day", 6, "supplier B"),
                ],
            )
        assert observed == [False]
        assert all(
            not pipeline.reasoner.is_supported(f"decision:{d.id}") for d in baseline.decisions
        )


def test_failed_second_update_restores_ledger_and_atms_context():
    with seeded() as pipeline:
        baseline = pipeline.plan()
        evidence_before = pipeline.store.events()
        operations_before = pipeline.events()
        revision_before = pipeline.reasoner.revision
        original_resolve = pipeline.store.resolve
        calls = []

        def fail_second_resolution(entity, field, evidence_id):
            calls.append(entity)
            if len(calls) == 2:
                raise RuntimeError("Injected review persistence failure")
            return original_resolve(entity, field, evidence_id)

        with patch.object(pipeline.store, "resolve", side_effect=fail_second_resolution):
            with pytest.raises(RuntimeError, match="Injected"):
                apply(
                    pipeline,
                    "batch-retry",
                    [
                        EvidenceUpdate("supply:a", "available_day", 5, "supplier A"),
                        EvidenceUpdate("supply:b", "available_day", 6, "supplier B"),
                    ],
                )
        assert calls == ["supply:a", "supply:b"]
        assert pipeline.store.events() == evidence_before
        assert pipeline.events() == operations_before
        assert pipeline.reasoner.revision == revision_before
        assert pipeline.store.current("supply:a", "available_day").value == 0
        assert pipeline.store.current("supply:b", "available_day").value == 0
        assert all(pipeline.reasoner.is_supported(f"decision:{d.id}") for d in baseline.decisions)
        # The failed event ID was not consumed and is safe to retry.
        receipt = apply(
            pipeline,
            "batch-retry",
            [
                EvidenceUpdate("supply:a", "available_day", 5, "supplier A"),
                EvidenceUpdate("supply:b", "available_day", 6, "supplier B"),
            ],
        )
        assert not receipt.replayed


def test_replay_after_later_update_does_not_resurrect_old_facts_or_supports():
    with seeded() as pipeline:
        baseline = pipeline.plan()
        first_update = [EvidenceUpdate("supply:a", "available_day", 1, "first bulletin")]
        snapshot = pipeline.revisions()
        first_receipt = apply(pipeline, "bulletin-1", first_update, snapshot)
        middle = pipeline.plan()
        apply(
            pipeline,
            "bulletin-2",
            [EvidenceUpdate("supply:a", "available_day", 2, "later correction")],
        )
        latest = pipeline.plan()
        before = (
            pipeline.revisions(),
            pipeline.store.events(),
            pipeline.events(),
            pipeline.reasoner.revision,
        )
        replay = apply(pipeline, "bulletin-1", first_update, snapshot)
        assert replay.replayed
        assert replay.evidence_ids == first_receipt.evidence_ids
        assert replay.evidence_revision == first_receipt.evidence_revision
        assert replay.operations_revision == first_receipt.operations_revision
        assert pipeline.store.current("supply:a", "available_day").value == 2
        assert before == (
            pipeline.revisions(),
            pipeline.store.events(),
            pipeline.events(),
            pipeline.reasoner.revision,
        )
        first_decision = next(d for d in middle.decisions if d.order_id == "first")
        assert not pipeline.reasoner.is_supported(f"decision:{first_decision.id}")
        latest_decision = next(d for d in latest.decisions if d.order_id == "first")
        assert pipeline.reasoner.is_supported(f"decision:{latest_decision.id}")
        original = next(d for d in baseline.decisions if d.order_id == "first")
        assert not pipeline.reasoner.is_supported(f"decision:{original.id}")


def test_other_instance_planning_invalidates_reviewed_operations_snapshot(tmp_path):
    path = tmp_path / "workflow.db"
    with seeded(path) as pipeline, OperationsPipeline(str(path)) as other:
        comparison = pipeline.compare_scenarios([])
        snapshot = {
            "evidence_revision": comparison.evidence_revision,
            "operations_revision": comparison.operations_revision,
        }
        other.plan()
        evidence_before = pipeline.store.events()
        with pytest.raises(StaleSnapshotError):
            apply(
                pipeline,
                "stale-choice",
                [EvidenceUpdate("supply:a", "available_day", 2, "reviewed old comparison")],
                snapshot,
            )
        assert pipeline.store.events() == evidence_before
        assert pipeline.store.current("supply:a", "available_day").value == 0


@pytest.mark.parametrize(
    "phase,event_kind",
    [
        ("approve", "decision_approved"),
        ("commit", "execution_intent"),
    ],
)
def test_second_member_failure_rolls_back_entire_plan(phase, event_kind):
    with seeded() as pipeline:
        result = pipeline.plan()
        assert len(result.decisions) == 2
        if phase == "commit":
            pipeline.approve_plan(result.plan_id)
        events_before = pipeline.events()
        decisions_before = pipeline.decisions()
        revisions_before = pipeline.revisions()
        emit = pipeline._event
        seen = []

        def fail_second_member(kind, payload):
            if kind == event_kind:
                seen.append(payload["decision_id"])
                if len(seen) == 2:
                    raise RuntimeError("Injected second member write failure")
            return emit(kind, payload)

        with patch.object(pipeline, "_event", side_effect=fail_second_member):
            with pytest.raises(RuntimeError, match="Injected"):
                if phase == "approve":
                    pipeline.approve_plan(result.plan_id)
                else:
                    pipeline.commit_plan(result.plan_id)
        assert len(seen) == 2
        assert pipeline.events() == events_before
        assert pipeline.decisions() == decisions_before
        assert pipeline.revisions() == revisions_before
        assert pipeline._reserved("supply_id") == {}
        assert pipeline._reserved("order_id") == {}
        assert not pipeline._db.in_transaction
        # A subsequent whole-plan retry can succeed without compensating partial work.
        if phase == "approve":
            pipeline.approve_plan(result.plan_id)
        pipeline.commit_plan(result.plan_id)
        assert all(d.status == "committed" for d in pipeline.decisions())
        assert pipeline._reserved("supply_id") == {"a": 2, "b": 2}


def test_opening_legacy_database_adds_receipts_without_rewriting_history(tmp_path):
    import sqlite3
    from assumption_ops import EvidenceUpdate, OperationsPipeline, Order, Policy, Supply

    path = str(tmp_path / "legacy.sqlite")
    with OperationsPipeline(path) as original:
        original.seed([Supply("s", "A", 2, 0)], [Order("o", "A", 2, 2)], Policy())
        plan = original.plan()
        old_events = original.events()
        old_evidence = original.store.events()
    # v0.2 had all these evidence/plan tables but no reviewed-batch receipt table.
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE ops_reviewed_batches")
    with OperationsPipeline(path) as restored:
        assert restored.events() == old_events
        assert restored.store.events() == old_evidence
        assert restored.explain(plan.decisions[0].id)["logical_support"]["supported"]
        snapshot = restored.revisions()
        receipt = restored.apply_updates(
            "legacy-source-confirmation",
            [
                EvidenceUpdate("supply:s", "available_day", 1, "reviewed confirmation"),
            ],
            expected_evidence_revision=snapshot["evidence_revision"],
            expected_operations_revision=snapshot["operations_revision"],
        )
        assert receipt.evidence_ids
        assert restored.store.current("supply:s", "available_day").value == 1
