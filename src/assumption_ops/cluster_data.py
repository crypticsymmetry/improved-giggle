"""Bounded, pinned Google 2011 trace projection for controlled placement tests.

Event time is the only clock supplied. This is not a reconstruction of available
production capacity: existing occupancy, other shards, disk, affinity constraints,
and overcommit behavior are deliberately outside this hard-reservation model.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from .placement import Machine, Task

RESOURCE_SCALE = 1_000_000
COHORT_START_US = 600_000_000
MAX_DECOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_TIMESTAMP = 2**63 - 1
SOURCES = {
    "machines": {
        "filename": "google_machine_events.csv.gz",
        "url": "https://storage.googleapis.com/clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz",
        "sha256": "fc44b8d2b33a96a261382488789855be7a0ce338888c99930ff45cb592249664",
        "size": 347211,
    },
    "tasks": {
        "filename": "google_task_events.csv.gz",
        "url": "https://storage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz",
        "sha256": "eae977d521bc6ed5d8deef56f1e06009d41844d4eb9a7e308e610d68e1b10f18",
        "size": 4139742,
    },
}


@dataclass(frozen=True)
class ClusterSnapshot:
    machines: tuple[Machine, ...]
    tasks: tuple[Task, ...]
    metadata: dict


def _digest(path: Path, limit: int = 5 * 1024 * 1024) -> str:
    if path.stat().st_size > limit:
        raise ValueError("Compressed cluster source exceeds size limit")
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, source: dict) -> None:
    if path.stat().st_size != source["size"] or _digest(path) != source["sha256"]:
        raise ValueError("Cluster source size/checksum mismatch")


def download_cluster(directory: str | Path) -> dict[str, Path]:
    """Fetch exactly the two pinned public files; verify before atomic replacement."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, source in SOURCES.items():
        destination = root / source["filename"]
        if destination.exists():
            _verify(destination, source)
            result[name] = destination.resolve()
            continue
        temporary = None
        try:
            with urlopen(source["url"], timeout=30) as response:
                with NamedTemporaryFile(dir=root, prefix="cluster-", delete=False) as out:
                    temporary = Path(out.name)
                    length = 0
                    while chunk := response.read(65536):
                        length += len(chunk)
                        if length > source["size"]:
                            raise ValueError("Cluster download exceeds pinned size")
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            _verify(temporary, source)
            os.replace(temporary, destination)
            result[name] = destination.resolve()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def _integer(value: str, label: str, *, maximum: int = MAX_TIMESTAMP) -> int:
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError(f"Invalid {label}: expected unsigned integer")
    number = int(value)
    if number > maximum:
        raise ValueError(f"Invalid {label}: out of range")
    return number


def _resource(value: str, *, request: bool) -> tuple[int | None, bool]:
    if value == "":
        return None, False
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid resource value") from exc
    if not number.is_finite() or number < 0 or number > 1_000_000:
        raise ValueError("Nonfinite, negative, or oversized resource value")
    if number.as_tuple().exponent < -12:
        raise ValueError("Resource precision exceeds 12 decimal places")
    scaled = number * RESOURCE_SCALE
    rounded = scaled.to_integral_value(rounding=ROUND_CEILING if request else ROUND_FLOOR)
    return int(rounded), scaled != rounded


def _rows(path: Path, fields: int):
    length = count = 0
    with gzip.open(path, "rb") as stream:
        while line := stream.readline(MAX_DECOMPRESSED_BYTES + 1):
            length += len(line)
            count += 1
            if length > MAX_DECOMPRESSED_BYTES or count > MAX_ROWS:
                raise ValueError("Decompressed cluster source exceeds limit")
            try:
                row = next(csv.reader([line.decode("utf-8")], strict=True))
            except (UnicodeError, csv.Error) as exc:
                raise ValueError("Malformed cluster CSV") from exc
            if len(row) != fields:
                raise ValueError(f"Expected {fields} fields at row {count}")
            yield count, row


def read_cluster_snapshot(
    machine_path: str | Path,
    task_path: str | Path,
    *,
    cutoff_us: int = 1_200_000_000,
    max_machines: int = 8,
    max_tasks: int = 64,
    verify_checksums: bool = True,
    cohort_start_us: int = COHORT_START_US,
    cohort_end_us: int | None = None,
    excluded_job_ids: Sequence[str] = (),
) -> ClusterSnapshot:
    """Project a bounded new-submission cohort using only events by ``cutoff_us``.

    Machine IDs are selected numerically; tasks enter in submission order. Missing
    requests or unsupported/unknown different-machine restrictions quarantine a
    task. Updates never borrow later values. Terminal events remove selected work;
    schedule/eviction events do not simulate execution or release modeled capacity.
    A bounded admission window uses complete SUBMIT records in [start, end).
    Selected identities continue replaying through the cutoff, including exits.
    Excluded jobs never occupy slots; reserved slots and their jobs remain in
    provenance after a task exits or is quarantined.
    """
    for name, value in (
        ("cutoff_us", cutoff_us),
        ("max_machines", max_machines),
        ("max_tasks", max_tasks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_machines > 100_000 or max_tasks > 100_000:
        raise ValueError("Snapshot bound exceeds 100000")
    if not isinstance(verify_checksums, bool):
        raise ValueError("verify_checksums must be boolean")
    if (
        isinstance(cohort_start_us, bool)
        or not isinstance(cohort_start_us, int)
        or not COHORT_START_US <= cohort_start_us <= cutoff_us
    ):
        raise ValueError("cohort_start_us must be an integer between trace start and cutoff")
    if cohort_end_us is not None and (
        isinstance(cohort_end_us, bool)
        or not isinstance(cohort_end_us, int)
        or not cohort_start_us < cohort_end_us <= cutoff_us
    ):
        raise ValueError("cohort_end_us must be an integer after start and at or before cutoff")
    if isinstance(excluded_job_ids, (str, bytes)) or not isinstance(excluded_job_ids, Sequence):
        raise ValueError("excluded_job_ids must be a sequence of distinct canonical job IDs")
    excluded = set()
    for value in excluded_job_ids:
        if not isinstance(value, str):
            raise ValueError("Excluded job IDs must be canonical unsigned numeric strings")
        job_id = _integer(value, "excluded job ID")
        if str(job_id) != value or job_id in excluded:
            raise ValueError("Excluded job IDs must be distinct canonical unsigned numeric strings")
        excluded.add(job_id)
    paths = {"machines": Path(machine_path), "tasks": Path(task_path)}
    hashes = {}
    for name, path in paths.items():
        if verify_checksums:
            _verify(path, SOURCES[name])
        hashes[name] = _digest(path)
    stats = {
        "machine_rows": 0,
        "task_rows": 0,
        "consistent_duplicates": 0,
        "missing_machine_records": 0,
        "quarantined_task_records": 0,
        "unsupported_restriction_records": 0,
        "rounded_capacity_values": 0,
        "rounded_request_values": 0,
    }
    machines = {}
    prefixes = {}
    for kind, fields in (("machines", 6), ("tasks", 13)):
        last = -1
        seen_at_time = {}
        prefix = sha256()
        if kind == "tasks":
            selected = {}
            submitted = set()
            last_finite = -1
        for _, row in _rows(paths[kind], fields):
            stats["machine_rows" if kind == "machines" else "task_rows"] += 1
            timestamp = _integer(row[0], "timestamp")
            if timestamp < last:
                raise ValueError("Cluster events must be chronological")
            if timestamp != last:
                seen_at_time.clear()
            last = timestamp
            if kind == "machines":
                identity = _integer(row[1], "machine ID")
                event = _integer(row[2], "machine event", maximum=2)
                key = (identity, event)
                cpu, rc = _resource(row[4], request=False)
                memory, rm = _resource(row[5], request=False)
            else:
                job = _integer(row[2], "job ID")
                index = _integer(row[3], "task index")
                identity = (job, index)
                event = _integer(row[5], "task event", maximum=8)
                key = (identity, event)
                priority = _integer(row[8], "priority", maximum=11)
                if row[1]:
                    _integer(row[1], "missing info", maximum=2)
                if row[4]:
                    _integer(row[4], "machine ID")
                if row[7]:
                    _integer(row[7], "scheduling class", maximum=3)
                if row[12] not in ("", "0", "1"):
                    raise ValueError("Invalid different machines restriction")
                cpu, rc = _resource(row[9], request=True)
                memory, rm = _resource(row[10], request=True)
                _resource(row[11], request=True)
                if timestamp != MAX_TIMESTAMP:
                    last_finite = timestamp
            frozen = tuple(row)
            if key in seen_at_time:
                if seen_at_time[key] != frozen:
                    raise ValueError("Conflicting duplicate cluster event")
                stats["consistent_duplicates"] += 1
                continue
            seen_at_time[key] = frozen
            if timestamp > cutoff_us or timestamp == MAX_TIMESTAMP:
                continue
            prefix.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
            if kind == "machines":
                if event == 1:
                    machines.pop(identity, None)
                elif cpu is None or memory is None or cpu <= 0 or memory <= 0:
                    machines.pop(identity, None)
                    stats["missing_machine_records"] += 1
                else:
                    machines[identity] = Machine(str(identity), cpu, memory)
                    stats["rounded_capacity_values"] += rc + rm
            else:
                if job in excluded:
                    continue
                complete = (
                    not row[1]
                    and row[12] == "0"
                    and cpu is not None
                    and memory is not None
                    and cpu > 0
                    and memory > 0
                )
                in_window = cohort_start_us <= timestamp and (
                    cohort_end_us is None or timestamp < cohort_end_us
                )
                if cohort_end_us is not None and event == 0 and not in_window:
                    continue
                if event == 0 and in_window:
                    submitted.add(identity)
                if event in (3, 4, 5, 6):
                    if identity in selected:
                        selected[identity] = None
                    submitted.discard(identity)
                elif event in (0, 7, 8) and identity in submitted:
                    if not complete:
                        if identity in selected:
                            selected[identity] = None
                        stats["quarantined_task_records"] += 1
                        stats["unsupported_restriction_records"] += row[12] != "0"
                    elif identity in selected or (
                        len(selected) < max_tasks
                        and (cohort_end_us is None or (event == 0 and in_window))
                    ):
                        selected[identity] = Task(f"{job}:{index}", cpu, memory, priority + 1)
                        stats["rounded_request_values"] += rc + rm
        prefixes[kind] = prefix.hexdigest()
    if cutoff_us > last_finite:
        raise ValueError("Cutoff exceeds first task shard's observed finite coverage")
    machine_values = tuple(machines[key] for key in sorted(machines)[:max_machines])
    task_values = tuple(value for value in selected.values() if value is not None)
    payload = {
        "machines": [[m.id, m.cpu, m.memory] for m in machine_values],
        "tasks": [[t.id, t.cpu, t.memory, t.priority] for t in task_values],
    }
    metadata = {
        "source": "Google clusterdata-2011-2",
        "license": "CC BY 4.0",
        "cutoff_us": cutoff_us,
        "cohort_start_us": cohort_start_us,
        "cohort_end_us": cohort_end_us,
        "excluded_job_ids": [str(job) for job in sorted(excluded)],
        "cohort_policy": (
            "First complete eligible SUBMIT identities in [cohort_start_us, cohort_end_us); "
            "updates replay admitted slots through cutoff; terminal exits do not refill slots."
            if cohort_end_us is not None
            else "First eligible task identities; terminal exits do not refill cohort slots."
        ),
        "max_machines": max_machines,
        "max_tasks": max_tasks,
        "resource_scale": RESOURCE_SCALE,
        "rounding": "ceil task requests; floor machine capacities",
        "source_sha256": hashes,
        "event_prefix_sha256": prefixes,
        "input_fingerprint": sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_machine_ids": [m.id for m in machine_values],
        "selected_task_ids": [t.id for t in task_values],
        "cohort_task_ids": [f"{job}:{index}" for job, index in selected],
        "selected_job_ids": [str(job) for job in sorted({job for job, _ in selected})],
        "task_shard_last_finite_us": last_finite,
        "counts": stats,
        "assumptions": [
            "Event time only: no observation timestamps supplied.",
            "Full modeled machine capacity reserved for the selected cohort; existing occupancy excluded.",
            "Only first task-event shard is loaded; submissions before cohort_start_us are excluded.",
            "Disk, task constraints, affinity, anti-affinity, and production overcommit are unmodeled.",
            "Known different-machine restrictions and unknown flags are excluded.",
            "Priorities are mapped to trace priority + 1; weights are modeling choices.",
            "Schedule and eviction events do not execute or release simulated allocations.",
        ],
    }
    return ClusterSnapshot(machine_values, task_values, metadata)
