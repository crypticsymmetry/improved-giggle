"""Predeclared, job-disjoint placement evaluation on one bounded trace shard."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from platform import python_version
import re
from statistics import fmean

import scipy

from .cluster_data import COHORT_START_US, read_cluster_snapshot
from .export import _literal
from .placement import PlacementPolicy, compare_placements, validate_placement_inputs


@dataclass(frozen=True)
class Window:
    id: str
    split: str
    start_us: int
    end_us: int


DEFAULT_WINDOWS = (
    Window("development", "development", 600_000_000, 900_000_000),
    Window("validation", "validation", 1_200_000_000, 1_500_000_000),
    *(
        Window(f"holdout_{i + 1}", "holdout", start, start + 300_000_000)
        for i, start in enumerate(range(1_800_000_000, 4_800_000_001, 600_000_000))
    ),
)
METHODS = ("milp", "priority_first_fit", "best_fit", "stability_best_fit")
SPLITS = ("development", "validation", "holdout")


def _hash(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate_protocol(windows, machine_counts, max_tasks, time_limit, verify_checksums):
    for value, name in ((windows, "windows"), (machine_counts, "machine_counts")):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ValueError(f"{name} must be a nonempty sequence")
    windows, machine_counts = tuple(windows), tuple(machine_counts)
    if len(windows) > 32 or len(machine_counts) > 8:
        raise ValueError("Protocol exceeds 32 windows or 8 machine configurations")
    if type(max_tasks) is not int or not 1 <= max_tasks <= 256:
        raise ValueError("max_tasks must be an integer between 1 and 256")
    if any(type(n) is not int or not 1 <= n <= 64 for n in machine_counts):
        raise ValueError("machine_counts must contain integers between 1 and 64")
    if any(a >= b for a, b in zip(machine_counts, machine_counts[1:])):
        raise ValueError("machine_counts must be strictly increasing")
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    if type(verify_checksums) is not bool:
        raise ValueError("verify_checksums must be boolean")
    seen = set()
    previous_end, previous_split = COHORT_START_US, -1
    for window in windows:
        if not isinstance(window, Window):
            raise ValueError("windows must contain Window records")
        if (
            not isinstance(window.id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", window.id)
            or window.id in seen
        ):
            raise ValueError("Window IDs must be unique safe identifiers")
        if window.split not in SPLITS:
            raise ValueError("Unknown window split")
        rank = SPLITS.index(window.split)
        if rank < previous_split:
            raise ValueError("Development, validation and holdout must occur in order")
        if (
            type(window.start_us) is not int
            or type(window.end_us) is not int
            or not previous_end <= window.start_us < window.end_us
        ):
            raise ValueError("Windows must be chronological, nonoverlapping integer ranges")
        seen.add(window.id)
        previous_end, previous_split = window.end_us, rank
    if {w.split for w in windows} != set(SPLITS):
        raise ValueError("Protocol requires development, validation and holdout windows")
    return windows, machine_counts


def _summarize(cohorts: list[dict], cases: list[dict]) -> dict:
    by_split = {}
    for split in SPLITS:
        rows = [case for case in cases if case["split"] == split]
        methods = {}
        for method in METHODS:
            comparisons = []
            lp_gaps = []
            optimal = wins = ties = losses = 0
            for case in rows:
                results = {r["method"]: r for r in case["comparison"]["results"]}
                reference, alternative = results["milp"], results[method]
                delta = (
                    alternative["evaluation"]["costs"]["total"]
                    - reference["evaluation"]["costs"]["total"]
                )
                total_priority = alternative["evaluation"]["metrics"]["total_priority"]
                comparisons.append((delta, delta / max(1, total_priority)))
                wins += delta > 0
                ties += delta == 0
                losses += delta < 0
                optimal += reference["status"] == "optimal"
                if alternative["gap_to_lp_bound"] is not None:
                    lp_gaps.append(alternative["gap_to_lp_bound"])
            methods[method] = {
                "comparisons": len(comparisons),
                "milp_strict_wins": wins,
                "milp_ties": ties,
                "milp_losses": losses,
                "optimal_milp_comparisons": optimal,
                "mean_absolute_regret_to_milp": fmean(x[0] for x in comparisons)
                if comparisons
                else None,
                "mean_normalized_regret_to_milp": fmean(x[1] for x in comparisons)
                if comparisons
                else None,
                "lp_bound_comparisons": len(lp_gaps),
                "mean_lp_gap": fmean(lp_gaps) if lp_gaps else None,
            }
        by_split[split] = {
            "cases": len(rows),
            "cohorts": sum(c["window"]["split"] == split for c in cohorts),
            "empty_cohorts": sum(
                c["window"]["split"] == split and c["status"] == "empty_cohort" for c in cohorts
            ),
            "methods": methods,
        }
    return {"by_split": by_split}


def run_cluster_holdout(
    machine_path: str | Path,
    task_path: str | Path,
    *,
    windows: Sequence[Window] = DEFAULT_WINDOWS,
    machine_counts: Sequence[int] = (1, 2, 4, 8),
    max_tasks: int = 64,
    time_limit: float = 5,
    verify_checksums: bool = True,
) -> dict:
    """Freeze the protocol, compile every cohort, then compare all configurations.

    No policy is fitted or selected. Earlier admitted jobs (including departed
    cohort slots) are excluded from all later windows. Each case has no previous
    plan, avoiding a reference optimized by one method. Empty cohorts are recorded
    and excluded from numerical comparisons, never counted as successful solves.
    """
    windows, machine_counts = _validate_protocol(
        windows, machine_counts, max_tasks, time_limit, verify_checksums
    )
    protocol = {
        "windows": [asdict(w) for w in windows],
        "machine_counts": list(machine_counts),
        "max_task_identities_per_cohort": max_tasks,
        "time_limit_seconds_per_solver": float(time_limit),
        "selection": "First complete eligible submissions in [start,end); state at end; exclude every job admitted by any earlier cohort.",
        "policy": asdict(PlacementPolicy()),
        "previous_plan": None,
        "calibration": "None; weights and configuration grid are fixed before solving.",
        "checksum_verification": verify_checksums,
    }
    protocol_fingerprint = _hash(protocol)
    compiled, cohorts, excluded = [], [], set()
    source_hashes = None
    for window in windows:
        snapshot = read_cluster_snapshot(
            machine_path,
            task_path,
            cutoff_us=window.end_us,
            cohort_start_us=window.start_us,
            cohort_end_us=window.end_us,
            excluded_job_ids=tuple(sorted(excluded, key=int)),
            max_machines=max(machine_counts),
            max_tasks=max_tasks,
            verify_checksums=verify_checksums,
        )
        admitted = set(snapshot.metadata["selected_job_ids"])
        if admitted & excluded:
            raise RuntimeError("Job leakage across cohort boundaries")
        if any(t.id.split(":", 1)[0] not in admitted for t in snapshot.tasks):
            raise RuntimeError("Active task is outside admitted job provenance")
        hashes = snapshot.metadata["source_sha256"]
        if source_hashes is not None and source_hashes != hashes:
            raise RuntimeError("Source files changed while compiling cohorts")
        source_hashes = dict(hashes)
        excluded.update(admitted)
        if len(snapshot.machines) < max(machine_counts):
            raise ValueError("Requested grid exceeds available selected machines")
        validate_placement_inputs(snapshot.machines, snapshot.tasks, PlacementPolicy())
        compiled.append((window, snapshot))
        cohorts.append(
            {
                "window": asdict(window),
                "status": "evaluated" if snapshot.tasks else "empty_cohort",
                "metadata": deepcopy(snapshot.metadata),
                "machines": len(snapshot.machines),
                "tasks": len(snapshot.tasks),
            }
        )
    cases = []
    for window, snapshot in compiled:
        if not snapshot.tasks:
            continue
        for count in machine_counts:
            machines = snapshot.machines[:count]
            fingerprint = _hash(
                {
                    "machines": [asdict(m) for m in machines],
                    "tasks": [asdict(t) for t in snapshot.tasks],
                    "policy": protocol["policy"],
                    "previous_plan": None,
                }
            )
            comparison = compare_placements(
                machines, snapshot.tasks, PlacementPolicy(), previous=None, time_limit=time_limit
            )
            cases.append(
                {
                    "window_id": window.id,
                    "split": window.split,
                    "machine_count": count,
                    "input_fingerprint": fingerprint,
                    "comparison": comparison,
                }
            )
    return {
        "schema_version": 1,
        "dataset": "Google ClusterData2011-2: first task-events shard and complete machine events",
        "source": "https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md",
        "license": "CC BY 4.0; see publisher documentation",
        "controlled_model": True,
        "real_source_records": True,
        "external_dispatch": False,
        "environment": {"python": python_version(), "scipy": scipy.__version__},
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "source_sha256": source_hashes,
        "cohorts": cohorts,
        "cases": cases,
        "summary": _summarize(cohorts, cases),
        "limitations": [
            "This fixed protocol separates admitted job identities and submission windows; no fitted policy, hyperparameter selection or automatic deployment occurs.",
            "Development includes a previously explored time region. Later windows are a reproducible evaluation extension, not an untouched external test set or preregistered study.",
            "All cohorts share one cluster, machine pool and first task shard; grid cases within each cohort are correlated. No independent-sample confidence intervals or causal/generalization claims are made.",
            "Full recorded capacities are modeled for selected work; existing occupancy, durations, disk, networking, general affinity and overcommit are excluded. Different-machine restrictions and uncertain submissions are excluded.",
            "Resources are normalized requests/limits with conservative integer rounding. Trace priority plus one is an assumed admission weight, not money or actual Borg policy.",
            "Each independent case has no previous plan and therefore zero migration cost. This evaluates admission/packing, not execution history or migration quality.",
            "Regret is alternative score minus verified MILP score, normalized by total request priority. A feasible_limit MILP is an incumbent, not proven optimal; regret may then be negative.",
            "Means give equal weight to each reported nonempty window/configuration case. Empty cohorts are excluded explicitly. Scores and placed tasks are not cumulative jobs completed or business savings.",
            "The LP is a verified optimal fractional lower bound when available; it is never rounded into executable placements. Solver timings are environment dependent.",
        ],
    }


def write_cluster_holdout_csv(report: dict, path: str | Path) -> None:
    """One row per evaluated method/configuration; JSON also retains empty cohorts."""
    fields = (
        "protocol_fingerprint",
        "window_id",
        "split",
        "machine_count",
        "input_fingerprint",
        "method",
        "status",
        "tasks",
        "placed_tasks",
        "pending_priority",
        "score",
        "milp_score",
        "regret_to_milp",
        "normalized_regret_to_milp",
        "lp_lower_bound",
        "gap_to_lp_bound",
        "elapsed_seconds",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            reference = next(r for r in case["comparison"]["results"] if r["method"] == "milp")
            milp_score = reference["evaluation"]["costs"]["total"]
            for result in case["comparison"]["results"]:
                metrics, costs = result["evaluation"]["metrics"], result["evaluation"]["costs"]
                row = {
                    "protocol_fingerprint": report["protocol_fingerprint"],
                    "window_id": case["window_id"],
                    "split": case["split"],
                    "machine_count": case["machine_count"],
                    "input_fingerprint": case["input_fingerprint"],
                    "method": result["method"],
                    "status": result["status"],
                    "tasks": metrics["tasks"],
                    "placed_tasks": metrics["placed_tasks"],
                    "pending_priority": costs["pending_priority"],
                    "score": costs["total"],
                    "milp_score": milp_score,
                    "regret_to_milp": costs["total"] - milp_score,
                    "normalized_regret_to_milp": (costs["total"] - milp_score)
                    / max(1, metrics["total_priority"]),
                    "lp_lower_bound": case["comparison"]["lp_relaxation"]["lower_bound"],
                    "gap_to_lp_bound": result["gap_to_lp_bound"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
                writer.writerow({key: _literal(value) for key, value in row.items()})
