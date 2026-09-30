from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from assumption_ops import placement as p


def oracle(machines, tasks, policy, previous=None):
    best = float("inf")
    for destinations in product([None] + [m.id for m in machines], repeat=len(tasks)):
        assignments = {t.id: m for t, m in zip(tasks, destinations) if m is not None}
        pending = tuple(t.id for t, m in zip(tasks, destinations) if m is None)
        old = previous.assignments if previous else {}
        score = sum(t.priority for t in tasks if t.id in pending) + policy.migration_penalty * sum(
            assignments.get(t) != m for t, m in old.items()
        )
        candidate = p.Placement(assignments, pending, float(score), "oracle")
        try:
            p.evaluate_placement(machines, tasks, policy, candidate, previous)
        except ValueError:
            continue
        best = min(best, score)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_oracle(seed):
    rng = np.random.default_rng(seed)
    machines = [
        p.Machine(str(i), int(rng.integers(1, 7)), int(rng.integers(1, 7))) for i in range(2)
    ]
    tasks = [
        p.Task(str(i), int(rng.integers(0, 5)), int(rng.integers(0, 5)), int(rng.integers(1, 6)))
        for i in range(4)
    ]
    policy = p.PlacementPolicy(2)
    previous = p.Placement({"0": "0", "removed": "gone"}, (), 0, "historical")
    before = deepcopy((machines, tasks, policy, previous))
    result = p.optimize_placement(machines, tasks, policy, previous)
    assert result.objective == oracle(machines, tasks, policy, previous)
    assert (machines, tasks, policy, previous) == before
    report = p.compare_placements(machines, tasks, policy, previous)
    import json

    json.dumps(report)
    assert all(r["gap_to_lp_bound"] >= 0 for r in report["results"])


def test_fractional_bound_is_not_placement():
    machines = [p.Machine("m", 3, 3)]
    tasks = [p.Task("a", 2, 2, 1), p.Task("b", 2, 2, 1)]
    bound = p.placement_lp_bound(machines, tasks)
    plan = p.optimize_placement(machines, tasks)
    assert bound["lower_bound"] == pytest.approx(0.5)
    assert bound["fractional_variables"] > 0
    assert plan.objective == 1


def test_retirement_and_machine_loss():
    previous = p.Placement({"a": "gone", "retired": "gone"}, (), 0, "historical")
    plan = p.optimize_placement(
        [p.Machine("new", 1, 1)], [p.Task("a", 1, 1, 10)], p.PlacementPolicy(3), previous
    )
    assert plan.objective == 6
    assert plan.assignments == {"a": "new"}


def test_empty_models():
    old = p.Placement({"retired": "m"}, (), 0, "old")
    assert p.optimize_placement([], [], p.PlacementPolicy(2), old).objective == 2
    plan = p.optimize_placement([], [p.Task("a", 1, 1, 4)])
    assert plan.pending == ("a",)
    assert plan.objective == 4


@pytest.mark.parametrize(
    "task",
    [
        p.Task("", 1, 1, 1),
        p.Task("a", True, 1, 1),
        p.Task("a", -1, 1, 1),
        p.Task("a", 1, 1, 0),
        p.Task("a", 1, 1, float("nan")),
        p.Task("a", 2**53 + 1, 1, 1),
    ],
)
def test_invalid_inputs(task):
    with pytest.raises(ValueError):
        p.optimize_placement([p.Machine("m", 1, 1)], [task])


@pytest.mark.parametrize("limit", [False, 0, -1, float("nan"), float("inf"), "30"])
def test_invalid_time(limit):
    with pytest.raises(ValueError):
        p.optimize_placement([], [], time_limit=limit)


def test_duplicate_and_unknown_identifiers():
    with pytest.raises(ValueError):
        p.optimize_placement([p.Machine("m", 1, 1)] * 2, [])
    with pytest.raises(ValueError):
        p.optimize_placement([], [p.Task("a", 1, 1, 1)] * 2)
    with pytest.raises(ValueError):
        p.evaluate_placement(
            [],
            [p.Task("a", 1, 1, 1)],
            p.PlacementPolicy(),
            p.Placement({"a": "unknown"}, (), 0, "bad"),
        )


@pytest.mark.parametrize(
    "placement",
    [
        p.Placement({}, (), 0, "bad"),
        p.Placement({}, ("a", "a"), 2, "bad"),
        p.Placement({"a": "m"}, ("a",), 1, "bad"),
        p.Placement({"a": "m"}, (), 9, "bad"),
        p.Placement({}, ("a",), float("nan"), "bad"),
    ],
)
def test_invalid_plans(placement):
    with pytest.raises(ValueError):
        p.evaluate_placement(
            [p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)], p.PlacementPolicy(), placement
        )


def test_capacity_independent_check():
    with pytest.raises(ValueError, match="capacity"):
        p.evaluate_placement(
            [p.Machine("m", 1, 1)],
            [p.Task("a", 2, 1, 1)],
            p.PlacementPolicy(),
            p.Placement({"a": "m"}, (), 0, "bad"),
        )


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(status=1, x=None, fun=None),
        SimpleNamespace(status=0, x=np.array([0.5, 0.5]), fun=0.5),
        SimpleNamespace(status=0, x=np.array([1.0, 0.0]), fun=1.0),
        SimpleNamespace(status=0, x=np.array([float("nan"), 0.0]), fun=0.0),
    ],
)
def test_invalid_incumbents(monkeypatch, result):
    monkeypatch.setattr(p, "milp", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError):
        p.optimize_placement([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])


def test_feasible_time_limit(monkeypatch):
    monkeypatch.setattr(
        p,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(status=1, x=np.array([0.0, 1.0]), fun=1.0),
    )
    plan = p.optimize_placement([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert plan.status == "feasible_limit"
    assert plan.objective == 1


def test_nonoptimal_lp_has_no_bound(monkeypatch):
    monkeypatch.setattr(
        p,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            status=1, message="limit", x=np.array([0.0, 1.0]), fun=1.0
        ),
    )
    report = p.placement_lp_bound([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert report["lower_bound"] is None


def test_zero_resource_task():
    plan = p.optimize_placement([p.Machine("m", 0, 0)], [p.Task("a", 0, 0, 2)])
    assert plan.assignments == {"a": "m"}


def test_score_overflow():
    with pytest.raises(ValueError):
        p.optimize_placement([], [p.Task("a", 0, 0, 2**53), p.Task("b", 0, 0, 1)])
    with pytest.raises(ValueError):
        p.optimize_placement(
            [], [], p.PlacementPolicy(2**53), p.Placement({"a": "m", "b": "m"}, (), 0, "old")
        )


def test_no_machine_lp():
    bound = p.placement_lp_bound([], [p.Task("a", 1, 1, 4)])
    assert bound["lower_bound"] == 4
    assert bound["fractional_variables"] == 0


def test_previous_detached_before_solver(monkeypatch):
    previous = p.Placement({"a": "gone"}, (), 0, "old")

    def solver(*args, **kwargs):
        previous.assignments.clear()
        return SimpleNamespace(status=0, x=np.array([1.0, 0.0]), fun=0.0)

    monkeypatch.setattr(p, "milp", solver)
    plan = p.optimize_placement(
        [p.Machine("new", 1, 1)], [p.Task("a", 1, 1, 1)], p.PlacementPolicy(3), previous
    )
    assert plan.objective == 3


def test_tolerance_cannot_hide_large_cost_disagreement(monkeypatch):
    monkeypatch.setattr(
        p,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=0, x=np.array([1e-10, 1 - 1e-10]), fun=1e10 - 1
        ),
    )
    with pytest.raises(RuntimeError, match="integer objective"):
        p.optimize_placement([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, int(1e10))])


def test_claimed_optimal_worse_than_greedy_rejected(monkeypatch):
    monkeypatch.setattr(
        p, "optimize_placement", lambda *args, **kwargs: p.Placement({}, ("a",), 1, "optimal")
    )
    with pytest.raises(RuntimeError, match="greedy"):
        p.compare_placements([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])


@pytest.mark.parametrize("machines,tasks", [(None, []), ([], None), ("bad", []), ([], "bad")])
def test_records_must_be_sequences(machines, tasks):
    with pytest.raises(ValueError):
        p.optimize_placement(machines, tasks)
    with pytest.raises(ValueError):
        p.compare_placements(machines, tasks)


def test_stability_best_fit_preserves_feasible_incumbent():
    machines = [p.Machine("small", 2, 2), p.Machine("large", 4, 4)]
    tasks = [p.Task("a", 1, 1, 5), p.Task("b", 1, 1, 4)]
    previous = p.Placement({"a": "large", "b": "large"}, (), 0, "old")
    report = p.compare_placements(machines, tasks, p.PlacementPolicy(1), previous)
    results = {r["method"]: r for r in report["results"]}
    assert set(results) == {"milp", "priority_first_fit", "best_fit", "stability_best_fit"}
    assert results["stability_best_fit"]["placement"]["assignments"] == previous.assignments
    assert results["stability_best_fit"]["evaluation"]["costs"]["migration"] == 0
    assert results["best_fit"]["evaluation"]["costs"]["migration"] == 2
    assert results["milp"]["evaluation"]["costs"]["total"] == 0


def test_stability_best_fit_rehomes_missing_machine():
    previous = p.Placement({"a": "removed"}, (), 0, "old")
    report = p.compare_placements(
        [p.Machine("new", 2, 2)], [p.Task("a", 1, 1, 3)], previous=previous
    )
    sticky = next(r for r in report["results"] if r["method"] == "stability_best_fit")
    assert sticky["placement"]["assignments"] == {"a": "new"}
    assert sticky["evaluation"]["costs"]["migration"] == 1
