"""Persistent observations and explicitly reviewed facts.

Observations and audit events are append-only. Acceptance does not silently
resolve disagreements: a reviewer must explicitly choose the winning record.
This module deliberately does not infer facts from natural-language text.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import sqlite3
from threading import RLock
from typing import Iterator, Protocol, TypeAlias
from uuid import uuid4


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class Evidence:
    id: str
    entity: str
    field: str
    value: JSONValue
    source: str
    observed_at: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceConflict(ValueError):
    """Distinct accepted observations require explicit human resolution."""

    def __init__(self, entity: str, field: str, records: list[Evidence]):
        self.entity = entity
        self.field = field
        self.records = tuple(records)
        super().__init__(f"Conflicting accepted evidence for {entity}.{field}")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _scalar(value: JSONValue) -> str:
    def validate(item: JSONValue) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("value must be finite")
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict:
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                validate(child)
            return
        raise ValueError("value must be JSON compatible")

    try:
        validate(value)
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except RecursionError as exc:
        raise ValueError("value must not contain circular structures") from exc


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    _nonempty(value, "observed_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("observed_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


class EvidenceStore:
    """SQLite evidence store, safe for repeated operations and reopening.

    Repeated acceptance/resolution of an already active record is a no-op.
    Separate proposals remain separate source observations even when identical.
    One instance serializes calls; SQLite serializes writes across instances.
    ``revision`` is the committed audit event sequence, suitable for cache keys.
    """

    def __init__(self, path: str = ":memory:", *, connection: sqlite3.Connection | None = None):
        self._lock = RLock()
        self._owns_connection = connection is None
        self._connection = (
            connection
            if connection is not None
            else sqlite3.connect(path, timeout=30, check_same_thread=False)
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        schema = """
            CREATE TABLE IF NOT EXISTS evidence_observations (
                id TEXT PRIMARY KEY,
                entity TEXT NOT NULL,
                field TEXT NOT NULL,
                value_json TEXT NOT NULL,
                source TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_events (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT NOT NULL REFERENCES evidence_observations(id),
                action TEXT NOT NULL CHECK(action IN ('proposed','accepted','superseded','resolved')),
                created_at TEXT NOT NULL,
                detail_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS evidence_event_record
                ON evidence_events(evidence_id, revision);
            CREATE INDEX IF NOT EXISTS evidence_entity_field
                ON evidence_observations(entity, field);
        """
        with self.transaction():
            for statement in schema.split(";"):
                if statement.strip():
                    self._connection.execute(statement)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Atomic write scope; nested scopes use savepoints, preserving the owner.

        A borrowed connection must not be concurrently used outside this scope.
        Transactions on one store are serialized by its reentrant lock.
        """
        with self._lock:
            nested = self._connection.in_transaction
            savepoint = f"evidence_{uuid4().hex}"
            if nested:
                self._connection.execute(f"SAVEPOINT {savepoint}")
            else:
                self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if nested:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    self._connection.rollback()
                raise
            else:
                if nested:
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    try:
                        self._connection.commit()
                    except BaseException:
                        self._connection.rollback()
                        raise

    def _event(self, evidence_id: str, action: str, detail: dict | None = None) -> None:
        self._connection.execute(
            "INSERT INTO evidence_events(evidence_id,action,created_at,detail_json) VALUES(?,?,?,?)",
            (evidence_id, action, datetime.now(timezone.utc).isoformat(), json.dumps(detail or {})),
        )

    def _records(self, entity: str | None = None, field: str | None = None) -> list[Evidence]:
        query = """
            SELECT o.*, (
                SELECT CASE WHEN action='resolved' THEN 'accepted' ELSE action END
                FROM evidence_events e WHERE e.evidence_id=o.id
                ORDER BY revision DESC LIMIT 1
            ) AS status
            FROM evidence_observations o
        """
        clauses, parameters = [], []
        if entity is not None:
            clauses.append("o.entity=?")
            parameters.append(entity)
        if field is not None:
            clauses.append("o.field=?")
            parameters.append(field)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY o.rowid"
        return [
            Evidence(
                row["id"],
                row["entity"],
                row["field"],
                json.loads(row["value_json"]),
                row["source"],
                row["observed_at"],
                row["status"],
            )
            for row in self._connection.execute(query, parameters)
        ]

    def _lookup(self, evidence_id: str) -> Evidence:
        _nonempty(evidence_id, "evidence_id")
        row = self._connection.execute(
            "SELECT entity,field FROM evidence_observations WHERE id=?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown evidence ID: {evidence_id}")
        return next(r for r in self._records(row["entity"], row["field"]) if r.id == evidence_id)

    def propose(
        self, entity: str, field: str, value: JSONValue, source: str, observed_at: str | None = None
    ) -> str:
        for name, item in (("entity", entity), ("field", field), ("source", source)):
            _nonempty(item, name)
        value_json = _scalar(value)
        observed_at = _timestamp(observed_at)
        evidence_id = str(uuid4())
        with self.transaction():
            self._connection.execute(
                "INSERT INTO evidence_observations VALUES(?,?,?,?,?,?)",
                (evidence_id, entity, field, value_json, source, observed_at),
            )
            self._event(evidence_id, "proposed")
        return evidence_id

    def accept(self, evidence_id: str) -> None:
        with self.transaction():
            record = self._lookup(evidence_id)
            if record.status == "superseded":
                raise ValueError(
                    "Superseded evidence cannot be reaccepted; propose a new observation"
                )
            if record.status == "proposed":
                self._event(evidence_id, "accepted")

    def resolve(self, entity: str, field: str, evidence_id: str) -> None:
        _nonempty(entity, "entity")
        _nonempty(field, "field")
        with self.transaction():
            winner = self._lookup(evidence_id)
            if (winner.entity, winner.field) != (entity, field):
                raise ValueError(
                    "Resolution evidence must belong to the requested entity and field"
                )
            if winner.status != "accepted":
                raise ValueError("Resolution winner must already be accepted and active")
            competing = [
                r
                for r in self._records(entity, field)
                if r.status == "accepted" and r.id != evidence_id
            ]
            if competing:
                for record in competing:
                    self._event(record.id, "superseded", {"winner_id": evidence_id})
                self._event(evidence_id, "resolved", {"superseded_ids": [r.id for r in competing]})

    def current(self, entity: str, field: str) -> Evidence | None:
        _nonempty(entity, "entity")
        _nonempty(field, "field")
        with self._lock:
            records = [r for r in self._records(entity, field) if r.status == "accepted"]
        if len({_scalar(r.value) for r in records}) > 1:
            raise EvidenceConflict(entity, field, records)
        return records[-1] if records else None

    def list_evidence(self, entity: str | None = None) -> list[Evidence]:
        if entity is not None:
            _nonempty(entity, "entity")
        with self._lock:
            return self._records(entity)

    def events(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "revision": row["revision"],
                    "evidence_id": row["evidence_id"],
                    "action": row["action"],
                    "created_at": row["created_at"],
                    "detail": json.loads(row["detail_json"]),
                }
                for row in self._connection.execute(
                    "SELECT * FROM evidence_events ORDER BY revision"
                )
            ]

    @property
    def revision(self) -> int:
        with self._lock:
            return self._connection.execute(
                "SELECT COALESCE(MAX(revision),0) FROM evidence_events"
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self._owns_connection:
                self._connection.close()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class Extractor(Protocol):
    def extract(self, text: str, store: EvidenceStore) -> list[str]:
        """Propose source-linked observations; never accept them automatically."""
        ...


class JsonExtractor:
    """Parse one explicit JSON record or a list; prose extraction is unsupported.

    Every record requires entity, field, value and source, with optional
    observed_at. The complete batch is validated before any proposal is saved.
    """

    def extract(self, text: str, store: EvidenceStore) -> list[str]:
        def reject_constant(value: str) -> None:
            raise ValueError(f"Nonfinite JSON value: {value}")

        parsed = json.loads(text, parse_constant=reject_constant)
        records = parsed if isinstance(parsed, list) else [parsed]
        required = {"entity", "field", "value", "source"}
        for record in records:
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError("Every record requires entity, field, value and source")
            if set(record) - required - {"observed_at"}:
                raise ValueError("Unexpected JSON record fields")
            for name in ("entity", "field", "source"):
                _nonempty(record[name], name)
            _scalar(record["value"])
            _timestamp(record.get("observed_at"))
        return [store.propose(**record) for record in records]
