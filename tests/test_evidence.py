import json
import sqlite3

import pytest

from assumption_ops.evidence import EvidenceConflict, EvidenceStore, JsonExtractor


def test_review_conflict_resolution_and_append_only_history():
    with EvidenceStore() as store:
        first = store.propose("supplier-1", "arrival_day", 5, "email-original")
        assert store.current("supplier-1", "arrival_day") is None
        store.accept(first)
        revision = store.revision
        store.accept(first)
        assert store.revision == revision
        second = store.propose("supplier-1", "arrival_day", 8, "email-revision")
        store.accept(second)
        with pytest.raises(EvidenceConflict) as failure:
            store.current("supplier-1", "arrival_day")
        assert {r.id for r in failure.value.records} == {first, second}
        assert len(store.list_evidence()) == 2
        previous_events = store.events()
        store.resolve("supplier-1", "arrival_day", second)
        assert store.current("supplier-1", "arrival_day").value == 8
        assert store.events()[: len(previous_events)] == previous_events
        records = {r.id: r for r in store.list_evidence()}
        assert records[first].value == 5
        assert records[first].source == "email-original"
        assert records[first].status == "superseded"
        revision = store.revision
        store.resolve("supplier-1", "arrival_day", second)
        assert store.revision == revision
        with pytest.raises(ValueError):
            store.accept(first)


def test_reopen_preserves_resolutions_and_events(tmp_path):
    path = str(tmp_path / "evidence.db")
    with EvidenceStore(path) as store:
        record = store.propose(
            "policy", "config", {"limit": 5, "allowed": [True, None]}, "reviewer"
        )
        store.accept(record)
        events = store.events()
    with EvidenceStore(path) as store:
        assert store.events() == events
        assert store.current("policy", "config").id == record
        assert store.current("policy", "config").value == {"limit": 5, "allowed": [True, None]}


def test_equal_accepted_values_are_corroboration():
    with EvidenceStore() as store:
        for source in ("a", "b"):
            store.accept(store.propose("policy", "config", {"x": 1, "y": 2}, source))
        assert store.current("policy", "config").value == {"y": 2, "x": 1}
        assert len(store.list_evidence("policy")) == 2


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        {"x": float("inf")},
        [float("nan")],
        {1: "nonstring key"},
        object(),
    ],
)
def test_invalid_json_values_do_not_create_evidence(value):
    with EvidenceStore() as store:
        with pytest.raises(ValueError):
            store.propose("entity", "field", value, "source")
        assert store.revision == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(entity=""),
        dict(field=" "),
        dict(source=None),
        dict(observed_at="yesterday"),
        dict(observed_at="2026-01-01"),
    ],
)
def test_invalid_metadata(kwargs):
    with EvidenceStore() as store:
        args = dict(entity="e", field="f", value=1, source="s") | kwargs
        with pytest.raises(ValueError):
            store.propose(**args)
        assert store.list_evidence() == []


def test_resolution_requires_matching_accepted_observation():
    with EvidenceStore() as store:
        record = store.propose("e", "f", 1, "s")
        with pytest.raises(ValueError):
            store.resolve("e", "f", record)
        store.accept(record)
        with pytest.raises(ValueError):
            store.resolve("other", "f", record)
        with pytest.raises(KeyError):
            store.accept("missing")
        assert store.current("e", "f").id == record


def test_parameterized_strings_and_timestamp_normalization():
    with EvidenceStore() as store:
        entity = "a'; DROP TABLE evidence_observations; --"
        record = store.propose(entity, "f", "text", "source", "2026-01-01T01:00:00+01:00")
        store.accept(record)
        assert store.current(entity, "f").observed_at == "2026-01-01T00:00:00+00:00"


def test_json_extractor_only_proposes_and_validates_entire_batch():
    with EvidenceStore() as store:
        extractor = JsonExtractor()
        records = [{"entity": "e", "field": "f", "value": 5, "source": "document"}]
        ids = extractor.extract(json.dumps(records), store)
        assert len(ids) == 1
        assert store.current("e", "f") is None
        assert store.list_evidence()[0].status == "proposed"
        with pytest.raises(ValueError):
            extractor.extract(json.dumps(records + [{"entity": "e"}]), store)
        assert len(store.list_evidence()) == 1
        with pytest.raises(ValueError):
            extractor.extract("The supplier says Tuesday", store)
        with pytest.raises(ValueError):
            extractor.extract('{"entity":"e","field":"f","value":NaN,"source":"s"}', store)


def test_multiple_connections_observe_committed_revision(tmp_path):
    path = str(tmp_path / "shared.db")
    with EvidenceStore(path) as first, EvidenceStore(path) as second:
        record = first.propose("e", "f", 1, "s")
        assert second.current("e", "f") is None
        first.accept(record)
        assert second.current("e", "f").id == record
        assert first.revision == second.revision


def test_outer_transaction_rolls_back_observations_events_and_external_tables():
    connection = sqlite3.connect(":memory:")
    with EvidenceStore(connection=connection) as store:
        connection.execute("CREATE TABLE external_state (value TEXT)")
        with pytest.raises(RuntimeError):
            with store.transaction():
                connection.execute("INSERT INTO external_state VALUES ('pending')")
                record = store.propose("e", "f", 1, "s")
                store.accept(record)
                assert store.current("e", "f").value == 1
                raise RuntimeError("abort whole operation")
        assert store.list_evidence() == []
        assert store.events() == []
        assert store.revision == 0
        assert connection.execute("SELECT * FROM external_state").fetchall() == []
    # Borrowed connection belongs to its caller.
    assert connection.execute("SELECT 1").fetchone()[0] == 1
    connection.close()


def test_nested_rollback_preserves_outer_transaction():
    with EvidenceStore() as store:
        with store.transaction():
            kept = store.propose("e", "f", 1, "s")
            with pytest.raises(RuntimeError):
                with store.transaction():
                    store.accept(kept)
                    store.propose("other", "f", 2, "s")
                    raise RuntimeError("abort nested scope")
            assert store.current("e", "f") is None
            assert len(store.list_evidence()) == 1
            store.accept(kept)
        assert store.current("e", "f").value == 1
        assert [e["action"] for e in store.events()] == ["proposed", "accepted"]
