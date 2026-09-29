"""Atomic, explicitly reviewed fact updates with optimistic snapshot guards.

This endpoint is a caller confirmation boundary. It does not establish that a
source is truthful, infer changes, or execute external operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import sqlite3
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .pipeline import OperationsPipeline


@dataclass(frozen=True)
class EvidenceUpdate:
    entity: str
    field: str
    value: Any
    source: str


@dataclass(frozen=True)
class UpdateReceipt:
    event_id: str
    evidence_ids: tuple[str, ...]
    evidence_revision: int
    operations_revision: int
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


class StaleSnapshotError(RuntimeError):
    """Reviewed updates refer to a snapshot that has since changed."""


class IdempotencyConflict(ValueError):
    """An event identifier was already used for a different ordered payload."""


def initialize_event_schema(connection: sqlite3.Connection) -> None:
    """Initialize receipt storage without committing an enclosing transaction."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS ops_reviewed_batches (
            event_id TEXT PRIMARY KEY,
            payload_hash TEXT NOT NULL,
            receipt_json TEXT NOT NULL
        )
    """)


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Update values must be finite JSON values with string object keys")


def _snapshot_updates(updates: Sequence[EvidenceUpdate]) -> tuple[tuple[EvidenceUpdate, ...], str]:
    if not updates:
        raise ValueError("At least one reviewed update is required")
    payload = []
    seen: set[tuple[str, str]] = set()
    for update in updates:
        if not isinstance(update, EvidenceUpdate):
            raise ValueError("updates must contain EvidenceUpdate records")
        for name in ("entity", "field", "source"):
            value = getattr(update, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Update {name} must be a nonempty string")
        key = (update.entity, update.field)
        if key in seen:
            raise ValueError(f"Duplicate updated entity and field: {key}")
        seen.add(key)
        try:
            _validate_json(update.value)
        except RecursionError as exc:
            raise ValueError("Update values must not contain circular structures") from exc
        payload.append(asdict(update))
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    # The same immutable serialized snapshot feeds the hash and database changes.
    copied = tuple(EvidenceUpdate(**record) for record in json.loads(canonical))
    return copied, sha256(canonical.encode("utf-8")).hexdigest()


def _operations_revision(pipeline: OperationsPipeline) -> int:
    return pipeline._db.execute("SELECT COALESCE(MAX(sequence),0) FROM ops_events").fetchone()[0]


def apply_updates(
    pipeline: OperationsPipeline,
    event_id: str,
    updates: Sequence[EvidenceUpdate],
    *,
    expected_evidence_revision: int,
    expected_operations_revision: int,
) -> UpdateReceipt:
    """Apply reviewed winners atomically, or replay the original durable receipt.

    A reused identifier must carry the identical ordered payload, including
    source attribution. Replays do not reapply facts. New events require both
    evidence and operations revisions to match under the SQLite write lock.
    """
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a nonempty string")
    for name, revision in (
        ("expected_evidence_revision", expected_evidence_revision),
        ("expected_operations_revision", expected_operations_revision),
    ):
        if type(revision) is not int or revision < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    copied, payload_hash = _snapshot_updates(updates)
    with pipeline._lock:
        if pipeline._db.in_transaction:
            raise ValueError("Reviewed updates require a top-level transaction")
        with pipeline._transaction():
            existing = pipeline._db.execute(
                "SELECT payload_hash,receipt_json FROM ops_reviewed_batches WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    raise IdempotencyConflict(
                        f"Event ID {event_id!r} already has a different payload"
                    )
                stored = json.loads(existing["receipt_json"])
                stored["evidence_ids"] = tuple(stored["evidence_ids"])
                receipt = replace(UpdateReceipt(**stored), replayed=True)
            else:
                evidence_revision = pipeline.store.revision
                operations_revision = _operations_revision(pipeline)
                if (evidence_revision, operations_revision) != (
                    expected_evidence_revision,
                    expected_operations_revision,
                ):
                    raise StaleSnapshotError(
                        f"Expected snapshot ({expected_evidence_revision}, {expected_operations_revision}); "
                        f"current snapshot is ({evidence_revision}, {operations_revision})"
                    )
                evidence_ids = []
                for update in copied:
                    evidence_id = pipeline.propose(
                        update.entity, update.field, update.value, update.source
                    )
                    pipeline.store.accept(evidence_id)
                    pipeline.store.resolve(update.entity, update.field, evidence_id)
                    evidence_ids.append(evidence_id)
                pipeline._event(
                    "reviewed_updates_applied",
                    {
                        "event_id": event_id,
                        "evidence_ids": evidence_ids,
                        "sources": [update.source for update in copied],
                    },
                )
                receipt = UpdateReceipt(
                    event_id,
                    tuple(evidence_ids),
                    pipeline.store.revision,
                    _operations_revision(pipeline),
                )
                pipeline._db.execute(
                    "INSERT INTO ops_reviewed_batches(event_id,payload_hash,receipt_json) VALUES(?,?,?)",
                    (event_id, payload_hash, json.dumps(receipt.to_dict(), sort_keys=True)),
                )
        pipeline._sync()
        return receipt
