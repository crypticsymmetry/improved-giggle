"""Controlled resource-placement experiments on bounded public Google traces.

Existing cluster occupancy and unmodeled constraints are not reconstructed.
These plans allocate selected workload against selected modeled full capacities.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from platform import python_version
from typing import Any, Sequence

import scipy

from .atms import ATMS
from .cluster_data import ClusterSnapshot, read_cluster_snapshot
from .export import _literal
from .placement import Placement, PlacementPolicy, compare_placements


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _facts(snapshot: ClusterSnapshot) -> dict[str, tuple[int, int]]:
    return {f"machine:{m.id}": (m.cpu, m.memory) for m in snapshot.machines} | {
        f"task:{t.id}": (t.cpu, t.memory) for t in snapshot.tasks
    }


def _support_model(
    snapshot: ClusterSnapshot, plan: Placement
) -> tuple[ATMS, dict[str, str], dict[str, str]]:
    """Justify joint per-machine resource feasibility, not global optimality.

    All colocated request facts are premises for each placement on that machine.
    A changed neighboring request therefore invalidates the whole shared-capacity
    justification, rather than leaving an apparently independent arc supported.
    """
    model = ATMS()
    facts = {key: "fact:" + _hash((key, value)) for key, value in _facts(snapshot).items()}
    for node in facts.values():
        model.add_assumption(node)
    groups: dict[str, list[str]] = {}
    for task_id, machine_id in plan.assignments.items():
        groups.setdefault(machine_id, []).append(task_id)
    decisions = {}
    for machine_id, task_ids in sorted(groups.items()):
        premises = (
            facts[f"machine:{machine_id}"],
            *(facts[f"task:{tid}"] for tid in sorted(task_ids)),
        )
        for task_id in sorted(task_ids):
            node = "placement:" + _hash((task_id, machine_id))
            model.add_rule(node, premises)
            decisions[task_id] = node
    return model, decisions, facts


def _support_impact(
    old: ClusterSnapshot | None, new: ClusterSnapshot, plan: Placement | None
) -> dict[str, Any]:
    if old is None or plan is None:
        return {
            "changed_feasibility_fact_keys": [],
            "added_feasibility_fact_keys": [],
            "invalidated_previous_decisions": 0,
            "previous_assigned_decisions": 0,
            "invalidated_task_ids": [],
            "previous_support_records": [],
        }
    model, decisions, nodes = _support_model(old, plan)
    prior, current = _facts(old), _facts(new)
    changed = sorted(key for key, value in prior.items() if current.get(key) != value)
    for key in changed:
        model.set_active(nodes[key], False)
    invalidated = sorted(
        task_id for task_id, node in decisions.items() if not model.is_supported(node)
    )
    return {
        "changed_feasibility_fact_keys": changed,
        "added_feasibility_fact_keys": sorted(current.keys() - prior.keys()),
        "invalidated_previous_decisions": len(invalidated),
        "previous_assigned_decisions": len(decisions),
        "invalidated_task_ids": invalidated,
        "previous_support_records": [
            {"task_id": task_id, "machine_id": plan.assignments[task_id], **model.explain(node)}
            for task_id, node in sorted(decisions.items())
        ],
    }


def _plan(value: dict[str, Any]) -> Placement:
    return Placement(
        dict(value["assignments"]), tuple(value["pending"]), value["objective"], value["status"]
    )


def run_cluster_benchmark(
    machine_path: str | Path,
    task_path: str | Path,
    *,
    cutoffs_us: Sequence[int] = (900_000_000, 1_200_000_000, 1_800_000_000),
    max_machines: int = 8,
    max_tasks: int = 64,
    migration_penalty: int = 1,
    time_limit: float = 30,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Compare alternatives using the initial MILP plan as a common cost reference.

    Cutoffs must be strictly increasing and inside the downloaded task shard.
    All snapshots are validated before solving. Each new snapshot is reoptimized
    even if local feasibility supports remain intact. Historical task termination
    is an observed input, not execution caused by any suggested placement.
    """
    if not isinstance(cutoffs_us, (list, tuple)) or not cutoffs_us:
        raise ValueError("cutoffs_us must be a nonempty list or tuple")
    if any(type(value) is not int or value < 600_000_000 for value in cutoffs_us):
        raise ValueError("cutoffs_us must contain integers >= 600000000")
    if any(a >= b for a, b in zip(cutoffs_us, cutoffs_us[1:])):
        raise ValueError("cutoffs_us must be strictly increasing")
    if type(migration_penalty) is not int or migration_penalty < 0:
        raise ValueError("migration_penalty must be a nonnegative integer")
    policy = PlacementPolicy(migration_penalty=migration_penalty)
    snapshots = tuple(
        read_cluster_snapshot(
            machine_path,
            task_path,
            cutoff_us=cutoff,
            max_machines=max_machines,
            max_tasks=max_tasks,
            verify_checksums=verify_checksums,
        )
        for cutoff in cutoffs_us
    )
    initial = previous_plan = previous_snapshot = None
    records = []
    for cutoff, snapshot in zip(cutoffs_us, snapshots, strict=True):
        impact = _support_impact(previous_snapshot, snapshot, previous_plan)
        comparison = compare_placements(
            snapshot.machines, snapshot.tasks, policy, previous=initial, time_limit=time_limit
        )
        optimized = _plan(comparison["results"][0]["placement"])
        support, decision_nodes, _ = _support_model(snapshot, optimized)
        fingerprint = _hash(
            {
                "machines": [asdict(m) for m in snapshot.machines],
                "tasks": [asdict(t) for t in snapshot.tasks],
                "policy": asdict(policy),
                "previous_assignments": dict(initial.assignments) if initial is not None else None,
            }
        )
        records.append(
            {
                "cutoff_us": cutoff,
                "input_fingerprint": fingerprint,
                "machines": len(snapshot.machines),
                "tasks": len(snapshot.tasks),
                "source_metadata": snapshot.metadata,
                "comparison": comparison,
                "support_impact": impact,
                "placement_supports": [
                    {
                        "task_id": task_id,
                        "machine_id": optimized.assignments[task_id],
                        **support.explain(node),
                    }
                    for task_id, node in sorted(decision_nodes.items())
                ],
            }
        )
        if initial is None:
            initial = optimized
        previous_plan, previous_snapshot = optimized, snapshot
    return {
        "schema_version": 1,
        "dataset": "Google ClusterData2011-2: complete machine-events file and first task-events shard",
        "source": "https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md",
        "license": "CC BY 4.0; see publisher documentation",
        "real_source_records": True,
        "controlled_model": True,
        "external_dispatch": False,
        "environment": {"python": python_version(), "scipy": scipy.__version__},
        "policy": asdict(policy),
        "initial_reference": initial.to_dict(),
        "limitations": [
            "A deterministic bounded cohort of submitted tasks is modeled against selected machines' full recorded capacities; existing occupancy is not reconstructed or assumed actually free.",
            "CPU and memory values are publisher-normalized requests/limits and capacities, not literal core counts, bytes or measured consumption.",
            "Task requests are rounded up and capacities down into fixed integer ticks; reported feasibility applies to that conservative model.",
            "Disk, machine attributes, general affinity, networking, service dependencies and task durations are not modeled; flagged different-machine tasks are excluded.",
            "Google may overcommit resources; this experiment imposes hard CPU/memory capacity limits.",
            "Trace priority plus one is an assumed cardinal admission weight, not monetary value or a faithful reproduction of Borg's policy.",
            "Event timestamps determine simulated visibility; the release provides no per-record ingestion timestamps, so actual observation latency is unknown.",
            "The initial optimized plan is the shared frozen migration-cost reference for all later methods; new snapshots do not evolve each baseline's own execution history.",
            "ATMS supports justify joint per-machine resource feasibility using all colocated request facts; they do not certify global optimality. Every snapshot is reoptimized.",
            "Repeated snapshots overlap and are alternative plans, not cumulative completed jobs, priority served or production savings.",
            "LP is a fractional lower bound only when independently verified optimal; it is not rounded or dispatched as a task assignment.",
            "No scheduling writes, approvals, reservations, causal performance claims or comparison against Google's production scheduler occur.",
        ],
        "snapshots": records,
        "summary": {
            "snapshots": len(records),
            "final_cutoff_us": records[-1]["cutoff_us"],
            "final_method_evaluations": {
                r["method"]: r["evaluation"] for r in records[-1]["comparison"]["results"]
            },
            "cumulative_invalidated_previous_decisions": sum(
                r["support_impact"]["invalidated_previous_decisions"] for r in records
            ),
            "execution_intents": 0,
        },
    }


def write_cluster_csv(report: dict[str, Any], path: str | Path) -> None:
    """Export each alternative separately, with spreadsheet-safe identifiers."""
    fields = (
        "cutoff_us",
        "input_fingerprint",
        "machines",
        "tasks",
        "method",
        "status",
        "placed_tasks",
        "pending_tasks",
        "placed_priority",
        "pending_priority",
        "migration_cost",
        "weighted_score",
        "lp_lower_bound",
        "gap_to_lp_bound",
        "elapsed_seconds",
        "invalidated_previous_decisions",
    )
    rows = []
    for snapshot in report["snapshots"]:
        for result in snapshot["comparison"]["results"]:
            metrics, costs = result["evaluation"]["metrics"], result["evaluation"]["costs"]
            rows.append(
                {
                    "cutoff_us": snapshot["cutoff_us"],
                    "input_fingerprint": snapshot["input_fingerprint"],
                    "machines": snapshot["machines"],
                    "tasks": snapshot["tasks"],
                    "method": result["method"],
                    "status": result["status"],
                    "placed_tasks": metrics["placed_tasks"],
                    "pending_tasks": metrics["pending_tasks"],
                    "placed_priority": metrics["placed_priority"],
                    "pending_priority": costs["pending_priority"],
                    "migration_cost": costs["migration"],
                    "weighted_score": costs["total"],
                    "lp_lower_bound": snapshot["comparison"]["lp_relaxation"]["lower_bound"],
                    "gap_to_lp_bound": result["gap_to_lp_bound"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "invalidated_previous_decisions": snapshot["support_impact"][
                        "invalidated_previous_decisions"
                    ],
                }
            )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _literal(value) for key, value in row.items()} for row in rows)
