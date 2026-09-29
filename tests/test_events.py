from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

from assumption_ops.events import (
    EvidenceUpdate,
    IdempotencyConflict,
    StaleSnapshotError,
    apply_updates,
    initialize_event_schema,
)
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.pipeline import OperationsPipeline


def seeded(path=":memory:"):
    pipeline = OperationsPipeline(str(path))
    pipeline.seed([Supply("shipment", "A", 8, 2)], [Order("urgent", "A", 5, 3)], Policy())
    initialize_event_schema(pipeline._db)
    return pipeline


def revisions(pipeline):
    return {
        "expected_evidence_revision": pipeline.store.revision,
        "expected_operations_revision": pipeline.events()[-1]["sequence"],
    }


def update(value=7, source="reviewed supplier confirmation"):
    return EvidenceUpdate("supply:shipment", "available_day", value, source)


def test_multi_field_batch_is_atomic_and_records_one_operation():
    with seeded() as pipeline:
        before = revisions(pipeline)
        updates = [update(), EvidenceUpdate("supply:shipment", "quantity", 4, "reviewed count")]
        with patch.object(pipeline, "_sync", wraps=pipeline._sync) as sync:
            receipt = apply_updates(pipeline, "event-1", updates, **before)
            sync.assert_called_once()
        assert pipeline.store.current("supply:shipment", "available_day").value == 7
        assert pipeline.store.current("supply:shipment", "quantity").value == 4
        assert receipt.evidence_revision == pipeline.store.revision
        assert receipt.operations_revision == before["expected_operations_revision"] + 1
        assert len(receipt.evidence_ids) == 2
        assert receipt.to_dict()["evidence_ids"] == list(receipt.evidence_ids)
        assert pipeline.events()[-1]["kind"] == "reviewed_updates_applied"
        assert pipeline.impact()["conflicts"] == []


def test_replay_uses_original_receipt_even_when_expected_snapshot_is_old():
    with seeded() as pipeline:
        before = revisions(pipeline)
        receipt = apply_updates(pipeline, "one", [update()], **before)
        evidence_events, ops_events = pipeline.store.events(), pipeline.events()
        replay = apply_updates(pipeline, "one", [update()], **before)
        assert replay.replayed is True
        assert replay.evidence_ids == receipt.evidence_ids
        assert replay.evidence_revision == receipt.evidence_revision
        assert replay.operations_revision == receipt.operations_revision
        assert pipeline.store.events() == evidence_events
        assert pipeline.events() == ops_events


def test_replay_after_later_facts_does_not_restore_old_values():
    with seeded() as pipeline:
        original = revisions(pipeline)
        first = apply_updates(pipeline, "one", [update(7)], **original)
        apply_updates(pipeline, "two", [update(9)], **revisions(pipeline))
        newer_revision = pipeline.store.revision
        replay = apply_updates(pipeline, "one", [update(7)], **original)
        assert replay.evidence_revision == first.evidence_revision
        assert pipeline.store.current("supply:shipment", "available_day").value == 9
        assert pipeline.store.revision == newer_revision


@pytest.mark.parametrize("changed", [update(8), update(7, "different source")])
def test_incompatible_reused_id_has_no_mutations(changed):
    with seeded() as pipeline:
        original = revisions(pipeline)
        apply_updates(pipeline, "one", [update()], **original)
        events, facts = pipeline.events(), pipeline.store.events()
        with pytest.raises(IdempotencyConflict):
            apply_updates(pipeline, "one", [changed], **original)
        assert pipeline.events() == events
        assert pipeline.store.events() == facts


def test_payload_order_is_part_of_idempotency_identity():
    with seeded() as pipeline:
        batch = [update(), EvidenceUpdate("order:urgent", "priority", 2, "reviewed priority")]
        original = revisions(pipeline)
        apply_updates(pipeline, "one", batch, **original)
        with pytest.raises(IdempotencyConflict):
            apply_updates(pipeline, "one", list(reversed(batch)), **original)


@pytest.mark.parametrize("dimension", ["evidence", "operations"])
def test_stale_snapshot_rejects_batch(dimension):
    with seeded() as pipeline:
        original = revisions(pipeline)
        if dimension == "evidence":
            pipeline.propose("supply:shipment", "quantity", 8, "unreviewed count")
        else:
            pipeline.plan()
        before = pipeline.store.events(), pipeline.events()
        with pytest.raises(StaleSnapshotError):
            apply_updates(pipeline, "one", [update()], **original)
        assert (pipeline.store.events(), pipeline.events()) == before
        assert pipeline._db.execute("SELECT COUNT(*) FROM ops_reviewed_batches").fetchone()[0] == 0


def test_bad_final_update_rolls_back_entire_batch_without_reasoner_change():
    with seeded() as pipeline:
        pipeline.plan()
        before = pipeline.store.events(), pipeline.events(), pipeline.store.list_evidence()
        batch = [update(), EvidenceUpdate("supply:shipment", "wrong_field", 1, "review")]
        with patch.object(pipeline, "_sync", wraps=pipeline._sync) as sync:
            with pytest.raises(ValueError, match="Unknown scenario"):
                apply_updates(pipeline, "one", batch, **revisions(pipeline))
            sync.assert_not_called()
        assert (
            pipeline.store.events(),
            pipeline.events(),
            pipeline.store.list_evidence(),
        ) == before
        assert pipeline._db.execute("SELECT COUNT(*) FROM ops_reviewed_batches").fetchone()[0] == 0


def test_existing_updated_conflict_resolved_but_unrelated_conflict_remains():
    with seeded() as pipeline:
        for field, value in (("available_day", 9), ("quantity", 3)):
            evidence = pipeline.propose("supply:shipment", field, value, "conflicting report")
            pipeline.accept(evidence)
        apply_updates(pipeline, "one", [update(7)], **revisions(pipeline))
        assert pipeline.store.current("supply:shipment", "available_day").value == 7
        assert pipeline.impact()["conflicts"] == [
            {"entity": "supply:shipment", "field": "quantity"}
        ]


def test_receipt_and_replay_persist_on_reopen(tmp_path):
    path = tmp_path / "ops.sqlite"
    with seeded(path) as pipeline:
        original = revisions(pipeline)
        receipt = apply_updates(pipeline, "one", [update()], **original)
    with OperationsPipeline(str(path)) as reopened:
        replay = apply_updates(reopened, "one", [update()], **original)
        assert replay.replayed is True
        assert replay.evidence_ids == receipt.evidence_ids
        assert reopened.store.current("supply:shipment", "available_day").value == 7


def test_two_instances_competing_for_snapshot_only_one_wins(tmp_path):
    path = tmp_path / "ops.sqlite"
    with seeded(path) as pipeline:
        original = revisions(pipeline)
    barrier = Barrier(2)

    def attempt(number):
        with OperationsPipeline(str(path)) as pipeline:
            barrier.wait(timeout=5)
            try:
                return apply_updates(pipeline, f"event-{number}", [update(7 + number)], **original)
            except StaleSnapshotError:
                return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (0, 1)))
    assert sum(result is not None for result in results) == 1
    with OperationsPipeline(str(path)) as pipeline:
        assert len([e for e in pipeline.events() if e["kind"] == "reviewed_updates_applied"]) == 1


@pytest.mark.parametrize(
    "event_id,batch", [("", [update()]), ("event", []), ("event", [update(), update(8)])]
)
def test_invalid_batch_rejected_without_mutations(event_id, batch):
    with seeded() as pipeline:
        before = pipeline.store.events(), pipeline.events()
        with pytest.raises(ValueError):
            apply_updates(pipeline, event_id, batch, **revisions(pipeline))
        assert (pipeline.store.events(), pipeline.events()) == before


@pytest.mark.parametrize("revision", [-1, True, 1.0, "1"])
@pytest.mark.parametrize("name", ["expected_evidence_revision", "expected_operations_revision"])
def test_invalid_revision_types(revision, name):
    with seeded() as pipeline:
        guards = revisions(pipeline)
        guards[name] = revision
        with pytest.raises(ValueError, match="nonnegative integer"):
            apply_updates(pipeline, "event", [update()], **guards)


def test_external_nested_transaction_is_rejected_without_mutations():
    with seeded() as pipeline:
        before = pipeline.store.events(), pipeline.events()
        with pipeline.store.transaction():
            with patch.object(pipeline, "_sync", wraps=pipeline._sync) as sync:
                with pytest.raises(ValueError, match="top-level transaction"):
                    apply_updates(pipeline, "event", [update()], **revisions(pipeline))
                sync.assert_not_called()
            assert (pipeline.store.events(), pipeline.events()) == before
        assert (pipeline.store.events(), pipeline.events()) == before
