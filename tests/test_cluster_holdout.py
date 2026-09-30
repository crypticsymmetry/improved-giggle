"""Leakage boundaries, independent costs, and fail-closed evaluation reports."""

from copy import deepcopy
import csv
import gzip
import json

import pytest

import assumption_ops.cluster_holdout as holdout
from assumption_ops.cluster_data import ClusterSnapshot
from assumption_ops.placement import Task


START = 600_000_000
WINDOWS = (
    holdout.Window("dev", "development", START, START + 10),
    holdout.Window("val", "validation", START + 20, START + 30),
    holdout.Window("test", "holdout", START + 40, START + 50),
)


def sources(tmp_path):
    machines = tmp_path / "machines.gz"
    tasks = tmp_path / "tasks.gz"
    with gzip.open(machines, "wt") as f:
        f.write("0,1,0,p,0.5,0.5\n0,2,0,p,0.5,0.5\n")
    rows = [
        (START + 1, 1, 0),
        (START + 2, 2, 0),
        (START + 3, 1, 4),
        (START + 20, 1, 0),
        (START + 21, 3, 0),
        (START + 40, 3, 0),
        (START + 41, 4, 0),
        (START + 50, 5, 0),
    ]
    with gzip.open(tasks, "wt") as f:
        for timestamp, job, event in rows:
            f.write(f"{timestamp},,{job},0,,{event},u,0,2,0.3,0.3,0,0\n")
    return machines, tasks


def run(tmp_path, **kwargs):
    return holdout.run_cluster_holdout(
        *sources(tmp_path), windows=WINDOWS, machine_counts=(1, 2), verify_checksums=False, **kwargs
    )


def test_departed_jobs_cannot_reenter_later_windows_and_grid_uses_same_work(tmp_path):
    report = run(tmp_path)
    assert [c["metadata"]["selected_job_ids"] for c in report["cohorts"]] == [
        ["1", "2"],
        ["3"],
        ["4"],
    ]
    assert [c["tasks"] for c in report["cohorts"]] == [1, 1, 1]
    assert report["cohorts"][1]["metadata"]["excluded_job_ids"] == ["1", "2"]
    assert report["cohorts"][2]["metadata"]["excluded_job_ids"] == ["1", "2", "3"]
    assert len(report["cases"]) == 6
    for case in report["cases"]:
        for result in case["comparison"]["results"]:
            assert result["evaluation"]["metrics"]["tasks"] == 1
            assert result["evaluation"]["costs"]["migration"] == 0
    assert report["protocol"]["previous_plan"] is None


def test_repeated_protocol_has_stable_fingerprints_but_different_grid_is_recorded(tmp_path):
    a, b = run(tmp_path), run(tmp_path)
    assert a["protocol_fingerprint"] == b["protocol_fingerprint"]
    assert [c["input_fingerprint"] for c in a["cases"]] == [
        c["input_fingerprint"] for c in b["cases"]
    ]
    c = holdout.run_cluster_holdout(
        *sources(tmp_path), windows=WINDOWS, machine_counts=(1,), verify_checksums=False
    )
    assert a["protocol_fingerprint"] != c["protocol_fingerprint"]


def test_all_cohorts_validate_before_any_solver_call(tmp_path, monkeypatch):
    calls = []
    original = holdout.read_cluster_snapshot

    def read(*args, **kwargs):
        calls.append(kwargs["cohort_start_us"])
        if len(calls) == 3:
            raise ValueError("bad final cohort")
        return original(*args, **kwargs)

    monkeypatch.setattr(holdout, "read_cluster_snapshot", read)
    monkeypatch.setattr(
        holdout,
        "compare_placements",
        lambda *a, **k: pytest.fail("must validate all cohorts before solving"),
    )
    with pytest.raises(ValueError, match="bad final cohort"):
        run(tmp_path)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "fault", ["leak", "source_change", "missing_provenance", "insufficient_machines"]
)
def test_inconsistent_cohort_provenance_fails_before_solving(tmp_path, monkeypatch, fault):
    original = holdout.read_cluster_snapshot
    calls = []

    def read(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        calls.append(snapshot)
        if len(calls) == 2:
            metadata = deepcopy(snapshot.metadata)
            machines, tasks = snapshot.machines, snapshot.tasks
            if fault == "leak":
                metadata["selected_job_ids"].append("1")
            elif fault == "source_change":
                metadata["source_sha256"]["tasks"] = "changed"
            elif fault == "missing_provenance":
                tasks = (Task("999:0", 1, 1, 1),)
            else:
                machines = machines[:1]
            return ClusterSnapshot(machines, tasks, metadata)
        return snapshot

    monkeypatch.setattr(holdout, "read_cluster_snapshot", read)
    monkeypatch.setattr(
        holdout,
        "compare_placements",
        lambda *a, **k: pytest.fail("must not solve invalid provenance"),
    )
    with pytest.raises((ValueError, RuntimeError)):
        run(tmp_path)


def test_empty_cohorts_have_no_comparisons_or_spurious_successes(tmp_path, monkeypatch):
    original = holdout.read_cluster_snapshot

    def read(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        return ClusterSnapshot(snapshot.machines, (), snapshot.metadata)

    monkeypatch.setattr(holdout, "read_cluster_snapshot", read)
    monkeypatch.setattr(
        holdout, "compare_placements", lambda *a, **k: pytest.fail("empty cohort must not solve")
    )
    report = run(tmp_path)
    assert report["cases"] == []
    assert all(c["status"] == "empty_cohort" for c in report["cohorts"])
    for split in report["summary"]["by_split"].values():
        assert split["cases"] == 0 and split["empty_cohorts"] == 1
        for method in split["methods"].values():
            assert method["comparisons"] == 0 and method["mean_normalized_regret_to_milp"] is None
    path = tmp_path / "out.csv"
    holdout.write_cluster_holdout_csv(report, path)
    with path.open() as f:
        assert list(csv.DictReader(f)) == []


def test_summary_keeps_negative_regret_for_a_time_limited_milp_incumbent():
    def result(method, score, priority, status="heuristic", gap=None):
        return {
            "method": method,
            "status": status,
            "evaluation": {"costs": {"total": score}, "metrics": {"total_priority": priority}},
            "gap_to_lp_bound": gap,
        }

    cases = []
    for score, priority, status in ((10, 20, "optimal"), (10, 100, "feasible_limit")):
        cases.append(
            {
                "split": "holdout",
                "comparison": {
                    "results": [
                        result("milp", score, priority, status),
                        result("priority_first_fit", 12 if status == "optimal" else 8, priority),
                        result("best_fit", score, priority),
                        result("stability_best_fit", score, priority),
                    ]
                },
            }
        )
    summary = holdout._summarize([], cases)["by_split"]["holdout"]["methods"]["priority_first_fit"]
    assert summary["milp_strict_wins"] == summary["milp_losses"] == 1
    assert summary["optimal_milp_comparisons"] == 1
    assert summary["mean_absolute_regret_to_milp"] == 0
    assert summary["mean_normalized_regret_to_milp"] == pytest.approx(0.04)
    assert summary["lp_bound_comparisons"] == 0 and summary["mean_lp_gap"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"machine_counts": ()},
        {"machine_counts": (2, 1)},
        {"machine_counts": (True,)},
        {"machine_counts": (65,)},
        {"max_tasks": 0},
        {"max_tasks": True},
        {"time_limit": float("nan")},
        {"time_limit": 0},
        {"time_limit": True},
        {"verify_checksums": 1},
        {"windows": ()},
        {"windows": (holdout.Window("x", "holdout", START, START + 10),)},
        {"windows": (WINDOWS[0], WINDOWS[2], WINDOWS[1])},
        {
            "windows": (
                WINDOWS[0],
                holdout.Window("bad", "validation", START + 9, START + 30),
                WINDOWS[2],
            )
        },
        {
            "windows": (
                WINDOWS[0],
                holdout.Window("dev", "validation", START + 20, START + 30),
                WINDOWS[2],
            )
        },
    ],
)
def test_bad_protocol_rejected_before_source_access(monkeypatch, kwargs):
    monkeypatch.setattr(
        holdout,
        "read_cluster_snapshot",
        lambda *a, **k: pytest.fail("must reject protocol before opening files"),
    )
    arguments = {"windows": WINDOWS, "machine_counts": (1, 2)} | kwargs
    with pytest.raises(ValueError):
        holdout.run_cluster_holdout("missing", "missing", **arguments)


def test_csv_preserves_all_alternatives_and_matches_exact_scores(tmp_path):
    report = run(tmp_path)
    path = tmp_path / "comparison.csv"
    holdout.write_cluster_holdout_csv(report, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    assert {r["split"] for r in rows} == {"development", "validation", "holdout"}
    assert {r["method"] for r in rows} == set(holdout.METHODS)
    assert all(float(r["score"]) == float(r["milp_score"]) for r in rows)
    assert all(r["protocol_fingerprint"] == report["protocol_fingerprint"] for r in rows)
    json.dumps(report, allow_nan=False)


def test_gated_policy_preserves_reference_scores_and_exports_routes(tmp_path):
    base = run(tmp_path)
    gated = run(tmp_path, include_gated=True)
    assert base["protocol_fingerprint"] != gated["protocol_fingerprint"]
    assert [c["input_fingerprint"] for c in base["cases"]] == [
        c["input_fingerprint"] for c in gated["cases"]
    ]
    for case in gated["cases"]:
        results = {r["method"]: r for r in case["comparison"]["results"]}
        assert set(results) == set(holdout.METHODS) | {"lp_gated"}
        assert results["lp_gated"]["evaluation"] == results["milp"]["evaluation"]
        assert results["lp_gated"]["gate"]["route"] == "lp_certificate"
        assert not results["lp_gated"]["gate"]["milp_invoked"]
    for split in gated["summary"]["by_split"].values():
        method = split["methods"]["lp_gated"]
        assert method["routes"] == {"lp_certificate": 2, "milp": 0, "greedy_fallback": 0}
        assert method["milp_invocations"] == 0
        assert method["milp_ties"] == method["certified_optimal_cases"] == 2
    path = tmp_path / "gate.csv"
    holdout.write_cluster_holdout_csv(gated, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert all(r["gate_route"] == "lp_certificate" for r in rows if r["method"] == "lp_gated")
    assert all(r["gate_route"] == "" for r in rows if r["method"] != "lp_gated")


def test_gated_certificate_contradicting_feasible_reference_fails_closed(tmp_path, monkeypatch):
    def invalid(machines, tasks, policy, **kwargs):
        candidate = holdout.compare_placements(machines, tasks, policy)["results"][0]
        candidate["evaluation"]["costs"]["total"] += 1
        return {**candidate, "gate": {"certified_optimal": True}}

    monkeypatch.setattr(holdout, "optimize_placement_gated", invalid)
    with pytest.raises(RuntimeError, match="certificate contradicts"):
        run(tmp_path, include_gated=True)


def test_gated_empty_cohorts_do_not_count_as_avoided_solves(tmp_path, monkeypatch):
    original = holdout.read_cluster_snapshot

    def read(*a, **k):
        snapshot = original(*a, **k)
        return ClusterSnapshot(snapshot.machines, (), snapshot.metadata)

    monkeypatch.setattr(holdout, "read_cluster_snapshot", read)
    monkeypatch.setattr(
        holdout,
        "optimize_placement_gated",
        lambda *a, **k: pytest.fail("empty cohort must not invoke gated solver"),
    )
    report = run(tmp_path, include_gated=True)
    for split in report["summary"]["by_split"].values():
        gated = split["methods"]["lp_gated"]
        assert gated["comparisons"] == gated["milp_invocations"] == 0
        assert gated["routes"] == {"lp_certificate": 0, "milp": 0, "greedy_fallback": 0}


def test_invalid_gate_flag_rejected_before_source_access(monkeypatch):
    monkeypatch.setattr(
        holdout, "read_cluster_snapshot", lambda *a, **k: pytest.fail("flag must be validated")
    )
    with pytest.raises(ValueError, match="include_gated"):
        holdout.run_cluster_holdout("missing", "missing", include_gated=1)
