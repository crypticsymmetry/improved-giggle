"""Trace projection is tested with small offline schema-faithful gzip fixtures."""

from io import BytesIO
import gzip
import json
import os
from pathlib import Path

import pytest

import assumption_ops.cluster_data as data

START = data.COHORT_START_US


def machine(t=0, identity=1, event=0, cpu="0.5", memory="0.25"):
    return [str(t), str(identity), str(event), "platform", cpu, memory]


def task(
    t=START, job=1, index=0, event=0, priority=2, cpu="0.1", memory="0.2", flag="0", missing=""
):
    return [
        str(t),
        missing,
        str(job),
        str(index),
        "",
        str(event),
        "user",
        "0",
        str(priority),
        cpu,
        memory,
        "0",
        flag,
    ]


def write(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    return path


def snapshot(tmp_path, machines=None, tasks=None, cutoff=START + 10, **kwargs):
    m = write(tmp_path / "m.gz", machines if machines is not None else [machine()])
    t = write(
        tmp_path / "t.gz",
        tasks if tasks is not None else [task(), task(START + 10, job=999, event=4)],
    )
    return data.read_cluster_snapshot(m, t, cutoff_us=cutoff, verify_checksums=False, **kwargs)


def test_projection_priority_and_conservative_rounding(tmp_path):
    result = snapshot(
        tmp_path,
        [machine(cpu="0.5000009")],
        [task(cpu="0.1000001"), task(START + 10, job=999, event=4)],
    )
    assert result.machines[0].cpu == 500000
    assert result.tasks[0].cpu == 100001
    assert result.tasks[0].priority == 3
    assert result.metadata["counts"]["rounded_capacity_values"] == 1
    assert result.metadata["counts"]["rounded_request_values"] == 1
    json.dumps(result.metadata)


def test_no_future_lookahead_and_global_hash_is_separate(tmp_path):
    rows = [task(), task(START + 20, event=7, cpu="0.9")]
    before = snapshot(tmp_path, tasks=rows)
    rows[-1] = task(START + 20, event=7, cpu="0.8")
    after = snapshot(tmp_path, tasks=rows)
    assert before.tasks == after.tasks
    assert before.metadata["input_fingerprint"] == after.metadata["input_fingerprint"]
    assert before.metadata["event_prefix_sha256"] == after.metadata["event_prefix_sha256"]
    assert before.metadata["source_sha256"] != after.metadata["source_sha256"]


def test_time_zero_terminal_and_schedule_eviction_semantics(tmp_path):
    rows = [
        task(0, job=9),
        task(job=1),
        task(job=2),
        task(START + 1, job=1, event=1),
        task(START + 2, job=1, event=2),
        task(START + 3, job=2, event=4),
        task(START + 10, job=999, event=4),
    ]
    result = snapshot(tmp_path, tasks=rows)
    assert [t.id for t in result.tasks] == ["1:0"]


def test_fixed_cohort_not_refilled_after_completion(tmp_path):
    rows = [
        task(job=1),
        task(START + 1, job=1, event=4),
        task(START + 2, job=2),
        task(START + 10, job=999, event=4),
    ]
    assert snapshot(tmp_path, tasks=rows, max_tasks=1).tasks == ()


def test_missing_update_quarantines_and_complete_later_update_restores(tmp_path):
    rows = [task(), task(START + 1, event=7, cpu=""), task(START + 3, event=8, cpu="0.3")]
    assert snapshot(tmp_path, tasks=rows, cutoff=START + 2).tasks == ()
    assert snapshot(tmp_path, tasks=rows, cutoff=START + 3).tasks[0].cpu == 300000


@pytest.mark.parametrize("flag", ["", "1"])
def test_unknown_or_unsupported_restriction_excluded(tmp_path, flag):
    result = snapshot(tmp_path, tasks=[task(flag=flag), task(START + 10, job=999, event=4)])
    assert result.tasks == ()
    assert result.metadata["counts"]["unsupported_restriction_records"] == 1


def test_pretrace_task_update_cannot_enter_cohort(tmp_path):
    assert snapshot(tmp_path, tasks=[task(0), task(START + 10, event=7)]).tasks == ()


def test_machine_missing_update_remove_and_numeric_sort(tmp_path):
    rows = [
        machine(identity=10),
        machine(identity=2),
        machine(identity=3),
        machine(START, identity=2, event=2, memory=""),
        machine(START + 1, identity=3, event=1),
    ]
    assert [m.id for m in snapshot(tmp_path, machines=rows).machines] == ["10"]
    assert [
        m.id
        for m in snapshot(
            tmp_path, machines=[machine(identity=10), machine(identity=2)], max_machines=1
        ).machines
    ] == ["2"]


def test_cutoff_outside_observed_coverage_rejected(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        snapshot(tmp_path, tasks=[task()], cutoff=START + 1)


def test_max_timestamp_not_used_for_coverage(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        snapshot(tmp_path, tasks=[task(), task(data.MAX_TIMESTAMP)], cutoff=START + 1)


def test_duplicates_consistent_dedup_conflicts_reject(tmp_path):
    result = snapshot(tmp_path, tasks=[task(), task(), task(START + 10, job=999, event=4)])
    assert len(result.tasks) == 1
    assert result.metadata["counts"]["consistent_duplicates"] == 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        snapshot(tmp_path, tasks=[task(), task(cpu="0.3"), task(START + 10, job=999, event=4)])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        snapshot(tmp_path, machines=[machine(), machine(cpu="0.3")])


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1e-1000", "1000001", "bad"])
def test_invalid_resource_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        snapshot(tmp_path, tasks=[task(cpu=value), task(START + 10, job=999, event=4)])


@pytest.mark.parametrize(
    "field,value", [(5, "9"), (8, "12"), (1, "3"), (12, "2"), (7, "4"), (2, "-1")]
)
def test_invalid_task_schema_rejected(tmp_path, field, value):
    row = task()
    row[field] = value
    with pytest.raises(ValueError):
        snapshot(tmp_path, tasks=[row, task(START + 10, job=999, event=4)])


def test_chronology_and_field_count_rejected(tmp_path):
    with pytest.raises(ValueError, match="chronological"):
        snapshot(tmp_path, tasks=[task(START + 10), task()])
    with pytest.raises(ValueError, match="fields"):
        snapshot(tmp_path, tasks=[task()[:-1]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tasks": True},
        {"max_tasks": 0},
        {"max_machines": 100001},
        {"cutoff_us": -1},
        {"verify_checksums": 1},
    ],
)
def test_snapshot_arguments_rejected(tmp_path, kwargs):
    m = write(tmp_path / "m.gz", [machine()])
    t = write(tmp_path / "t.gz", [task()])
    with pytest.raises(ValueError):
        data.read_cluster_snapshot(m, t, **kwargs)


def test_default_checksums_required(tmp_path):
    m = write(tmp_path / "m.gz", [machine()])
    t = write(tmp_path / "t.gz", [task()])
    with pytest.raises(ValueError, match="checksum"):
        data.read_cluster_snapshot(m, t)


def test_decompression_and_row_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_DECOMPRESSED_BYTES", 5)
    with pytest.raises(ValueError, match="Decompressed"):
        snapshot(tmp_path)
    monkeypatch.setattr(data, "MAX_DECOMPRESSED_BYTES", 100000)
    monkeypatch.setattr(data, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="Decompressed"):
        snapshot(tmp_path)


def test_download_cache_and_atomic_verified_bytes(tmp_path, monkeypatch):
    payloads = {}
    for name in data.SOURCES:
        payloads[name] = gzip.compress(b"source\n" + name.encode())
    sources = {
        name: {
            "filename": name + ".gz",
            "url": "https://example.test/" + name,
            "size": len(content),
            "sha256": data.sha256(content).hexdigest(),
        }
        for name, content in payloads.items()
    }
    monkeypatch.setattr(data, "SOURCES", sources)
    calls = []

    def opening(url, timeout):
        calls.append(url)
        return BytesIO(payloads[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(data, "urlopen", opening)
    result = data.download_cluster(tmp_path)
    assert result["tasks"].read_bytes() == payloads["tasks"]
    assert data.download_cluster(tmp_path) == result
    assert len(calls) == 2
    result["tasks"].write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="checksum"):
        data.download_cluster(tmp_path)


@pytest.mark.parametrize("payload", [b"too long", b"x"])
def test_failed_download_leaves_no_unverified_file(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(
        data,
        "SOURCES",
        {
            "machines": {
                "filename": "m.gz",
                "url": "https://example.test",
                "size": 2,
                "sha256": "0" * 64,
            }
        },
    )
    monkeypatch.setattr(data, "urlopen", lambda url, timeout: BytesIO(payload))
    with pytest.raises(ValueError):
        data.download_cluster(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_actual_pinned_source_when_available():
    root = os.environ.get("ASSUMPTION_OPS_CLUSTER_DIR")
    if not root:
        pytest.skip("Optional pinned real-data test")
    p = Path(root)
    result = data.read_cluster_snapshot(
        p / data.SOURCES["machines"]["filename"], p / data.SOURCES["tasks"]["filename"]
    )
    assert len(result.machines) == 8
    assert 0 < len(result.tasks) <= 64
    assert result.metadata["counts"]["machine_rows"] == 37780
    assert result.metadata["counts"]["task_rows"] == 450146
