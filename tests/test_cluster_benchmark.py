"""Integration guards for overlapping snapshots and local support scope."""

import csv

import pytest

from assumption_ops.cluster_benchmark import run_cluster_benchmark, write_cluster_csv
from assumption_ops.cluster_data import ClusterSnapshot
from assumption_ops.placement import Machine, Task


def snapshot(machines=(Machine("m", 5, 5),), tasks=(Task("a", 2, 2, 3), Task("b", 2, 2, 2))):
    return ClusterSnapshot(machines, tasks, {})


def reader(monkeypatch, snapshots):
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return snapshots[len(calls) - 1]

    monkeypatch.setattr("assumption_ops.cluster_benchmark.read_cluster_snapshot", read)
    return calls


def test_neighbor_requirement_changes_invalidate_shared_capacity_supports(monkeypatch):
    calls = reader(
        monkeypatch, [snapshot(), snapshot(tasks=(Task("a", 2, 2, 3), Task("b", 4, 4, 2)))]
    )
    report = run_cluster_benchmark("machines", "tasks", cutoffs_us=(900_000_000, 1_200_000_000))
    impact = report["snapshots"][1]["support_impact"]
    assert len(calls) == 2
    assert impact["invalidated_previous_decisions"] == 2
    assert impact["invalidated_task_ids"] == ["a", "b"]
    assert all(not r["supported"] for r in impact["previous_support_records"])
    assert report["summary"]["final_method_evaluations"]["milp"]["metrics"]["placed_tasks"] == 1
    assert report["summary"]["execution_intents"] == 0


def test_new_machine_reoptimizes_even_with_zero_local_invalidations(monkeypatch):
    tasks = (Task("a", 4, 4, 5), Task("b", 2, 2, 2))
    old = snapshot((Machine("m", 4, 4),), tasks)
    new = snapshot((Machine("m", 4, 4), Machine("n", 2, 2)), tasks)
    reader(monkeypatch, [old, new])
    report = run_cluster_benchmark("machines", "tasks", cutoffs_us=(900_000_000, 1_200_000_000))
    assert report["snapshots"][1]["support_impact"]["invalidated_previous_decisions"] == 0
    assert report["snapshots"][1]["support_impact"]["added_feasibility_fact_keys"] == ["machine:n"]
    assert (
        report["snapshots"][1]["comparison"]["results"][0]["evaluation"]["metrics"]["placed_tasks"]
        == 2
    )


def test_snapshot_priority_is_not_aggregated_as_jobs_completed(monkeypatch, tmp_path):
    reader(monkeypatch, [snapshot(), snapshot()])
    report = run_cluster_benchmark("machines", "tasks", cutoffs_us=(900_000_000, 1_200_000_000))
    assert report["summary"]["final_method_evaluations"]["milp"]["metrics"]["total_priority"] == 5
    assert "cumulative_placed_priority" not in report["summary"]
    path = tmp_path / "report.csv"
    write_cluster_csv(report, path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert set(r["method"] for r in rows) == {
        "milp",
        "priority_first_fit",
        "best_fit",
        "stability_best_fit",
    }


def test_all_snapshots_are_validated_before_any_solver(monkeypatch):
    calls = []

    def read(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("invalid later snapshot")
        return snapshot()

    def unexpected(*args, **kwargs):
        raise AssertionError("Solver ran before validating later input")

    monkeypatch.setattr("assumption_ops.cluster_benchmark.read_cluster_snapshot", read)
    monkeypatch.setattr("assumption_ops.cluster_benchmark.compare_placements", unexpected)
    with pytest.raises(ValueError, match="invalid later"):
        run_cluster_benchmark("machines", "tasks", cutoffs_us=(900_000_000, 1_200_000_000))


@pytest.mark.parametrize(
    "cutoffs",
    [(), (True,), (599_999_999,), (900_000_000, 900_000_000), (1_200_000_000, 900_000_000)],
)
def test_bad_cutoffs_rejected_before_reading(monkeypatch, cutoffs):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid cutoffs must not read sources")

    monkeypatch.setattr("assumption_ops.cluster_benchmark.read_cluster_snapshot", unexpected)
    with pytest.raises(ValueError):
        run_cluster_benchmark("machines", "tasks", cutoffs_us=cutoffs)
