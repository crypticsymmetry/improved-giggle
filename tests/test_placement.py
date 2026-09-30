from copy import deepcopy
from fractions import Fraction
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


def _lp_without_certificate():
    return {
        "status": "limit",
        "lower_bound": None,
        "fractional_variables": None,
        "exact_lower_bound": None,
        "certified_integer_lower_bound": None,
    }


@pytest.mark.parametrize("seed", range(16))
def test_exact_dual_bound_and_gate_against_enumerated_oracle(seed):
    rng = np.random.default_rng(seed + 60)
    machines = [
        p.Machine(str(i), int(rng.integers(1, 7)), int(rng.integers(1, 7))) for i in range(2)
    ]
    tasks = [
        p.Task(str(i), int(rng.integers(0, 5)), int(rng.integers(0, 5)), int(rng.integers(1, 6)))
        for i in range(4)
    ]
    policy = p.PlacementPolicy(2)
    previous = p.Placement({"0": "0", "retired": "lost"}, (), 0, "historical")
    exact_optimum = oracle(machines, tasks, policy, previous)
    bound = p.placement_lp_bound(machines, tasks, policy, previous)
    assert bound["certified_integer_lower_bound"] <= exact_optimum
    value = bound["exact_lower_bound"]
    assert Fraction(value["numerator"], value["denominator"]) <= exact_optimum
    before = deepcopy((machines, tasks, policy, previous))
    gated = p.optimize_placement_gated(machines, tasks, policy, previous)
    assert gated["evaluation"]["costs"]["total"] == exact_optimum
    assert gated["gate"]["certified_optimal"]
    assert (machines, tasks, policy, previous) == before
    # Arbitrary, nonoptimal and sign-incorrect solver multipliers must still
    # yield a valid bound after nonpositive clipping and exact box correction.
    _, _, _, _, c, a, _, high, constant = p._model(machines, tasks, policy, previous, 30)
    fake_dual = SimpleNamespace(
        eqlin=SimpleNamespace(marginals=rng.normal(0, 4, len(tasks))),
        ineqlin=SimpleNamespace(marginals=rng.normal(0, 4, 2 * len(machines))),
    )
    repaired = p._exact_dual_box_bound(fake_dual, c, a, high, len(tasks), constant)
    assert repaired <= exact_optimum
    assert p._certificate_fields(repaired)["certified_integer_lower_bound"] <= exact_optimum


def test_exact_ceiling_does_not_use_near_integer_tolerance():
    from scipy.sparse import csr_matrix

    dual = SimpleNamespace(
        eqlin=SimpleNamespace(marginals=[1.000001]), ineqlin=SimpleNamespace(marginals=[])
    )
    bound = p._exact_dual_box_bound(dual, np.array([2.0]), csr_matrix([[1]]), np.array([1.0]), 1, 0)
    assert bound == Fraction(1_000_001, 1_000_000)
    assert p._certificate_fields(bound)["certified_integer_lower_bound"] == 2


def test_box_bound_repairs_positive_resource_dual_and_variable_bounds():
    from scipy.sparse import csr_matrix

    dual = SimpleNamespace(
        eqlin=SimpleNamespace(marginals=[2.0]), ineqlin=SimpleNamespace(marginals=[10.0])
    )
    bound = p._exact_dual_box_bound(
        dual, np.array([0.0, 1.0]), csr_matrix([[1, 1], [1, 0]]), np.array([1, 1]), 1, 0
    )
    # With z clipped to zero, y=2 and the residuals (-2,-1) give -1.
    assert bound == -1


@pytest.mark.parametrize(
    "equality,inequality",
    [([float("nan")], [0, 0]), ([0], [float("inf"), 0]), ([], [0, 0]), ([[0]], [0, 0]), ([0], [0])],
)
def test_malformed_duals_leave_lp_diagnostic_but_no_certificate(monkeypatch, equality, inequality):
    monkeypatch.setattr(
        p,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            status=0,
            x=np.array([1.0, 0.0]),
            fun=0.0,
            eqlin=SimpleNamespace(marginals=equality),
            ineqlin=SimpleNamespace(marginals=inequality),
        ),
    )
    bound = p.placement_lp_bound([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert bound["status"] == "optimal"
    assert bound["lower_bound"] == 0
    assert bound["exact_lower_bound"] is None
    assert bound["certified_integer_lower_bound"] is None


def test_missing_dual_does_not_certify_float_objective(monkeypatch):
    monkeypatch.setattr(
        p,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.array([1.0, 0.0]), fun=0.0),
    )
    called = []
    real_milp = p.milp

    def counted(*args, **kwargs):
        called.append(True)
        return real_milp(*args, **kwargs)

    monkeypatch.setattr(p, "milp", counted)
    gated = p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert called == [True]
    assert gated["gate"]["route"] == "milp"
    assert gated["gate"]["integer_lower_bound"] is None


@pytest.mark.parametrize(
    "machines,tasks,policy,previous,score",
    [
        ([p.Machine("m", 10, 10)], [p.Task("a", 2, 2, 1)], p.PlacementPolicy(), None, 0),
        (
            [p.Machine("m", 3, 3)],
            [p.Task("a", 2, 2, 1), p.Task("b", 2, 2, 1)],
            p.PlacementPolicy(),
            None,
            1,
        ),
        ([], [], p.PlacementPolicy(3), p.Placement({"retired": "gone"}, (), 0, "old"), 3),
        ([], [p.Task("a", 2, 2, 4)], p.PlacementPolicy(), None, 4),
    ],
)
def test_gate_certificate_skips_milp(monkeypatch, machines, tasks, policy, previous, score):
    monkeypatch.setattr(p, "milp", lambda *args, **kwargs: pytest.fail("MILP must be skipped"))
    gated = p.optimize_placement_gated(machines, tasks, policy, previous)
    assert gated["gate"]["route"] == "lp_certificate"
    assert not gated["gate"]["milp_invoked"]
    assert gated["gate"]["certified_optimal"]
    assert gated["evaluation"]["costs"]["total"] == score
    assert gated["placement"]["status"] == "optimal"


def test_gate_narrow_no_incumbent_fallback(monkeypatch):
    monkeypatch.setattr(p, "placement_lp_bound", lambda *args, **kwargs: _lp_without_certificate())
    monkeypatch.setattr(p, "milp", lambda *args, **kwargs: SimpleNamespace(status=1, x=None))
    gated = p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert gated["gate"]["route"] == "greedy_fallback"
    assert gated["gate"]["milp_invoked"]
    assert not gated["gate"]["certified_optimal"]
    assert gated["placement"]["assignments"] == {"a": "m"}


def test_gate_retains_greedy_if_time_limited_incumbent_is_worse(monkeypatch):
    monkeypatch.setattr(p, "placement_lp_bound", lambda *args, **kwargs: _lp_without_certificate())
    monkeypatch.setattr(
        p,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(status=1, x=np.array([0.0, 1.0]), fun=1.0),
    )
    gated = p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])
    assert gated["evaluation"]["costs"]["total"] == 0
    assert gated["placement"]["status"] == "heuristic"
    assert gated["gate"]["route"] == "milp"
    assert not gated["gate"]["certified_optimal"]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(status=0, x=np.array([0.5, 0.5]), fun=0.5),
        SimpleNamespace(status=0, x=np.array([0.0, 1.0]), fun=1.0),
        SimpleNamespace(status=1, x=np.array([1.0, 0.0]), fun=1.0),
        SimpleNamespace(status=4, x=None, message="numerical failure"),
    ],
)
def test_gate_does_not_hide_solver_verification_failures(monkeypatch, result):
    monkeypatch.setattr(p, "placement_lp_bound", lambda *args, **kwargs: _lp_without_certificate())
    monkeypatch.setattr(p, "milp", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError):
        p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])


def test_gate_rejects_malformed_lp_primal(monkeypatch):
    monkeypatch.setattr(
        p,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.array([1.0, 1.0]), fun=1.0),
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])


def test_gate_rejects_certificate_above_verified_score(monkeypatch):
    bad = _lp_without_certificate()
    bad["certified_integer_lower_bound"] = 1
    monkeypatch.setattr(p, "placement_lp_bound", lambda *args, **kwargs: bad)
    with pytest.raises(RuntimeError, match="exceeds"):
        p.optimize_placement_gated([p.Machine("m", 1, 1)], [p.Task("a", 1, 1, 1)])


def test_gate_invalid_input_is_not_a_fallback():
    with pytest.raises(ValueError):
        p.optimize_placement_gated([], [p.Task("a", -1, 2, 1)])


def test_gate_solves_a_remaining_integer_gap():
    machines = [p.Machine("m", 4, 4)]
    tasks = [p.Task("a", 3, 3, 3), p.Task("b", 2, 2, 2), p.Task("c", 2, 2, 2)]
    gated = p.optimize_placement_gated(machines, tasks)
    assert gated["gate"]["route"] == "milp"
    assert gated["gate"]["milp_invoked"]
    assert gated["gate"]["selected_heuristic"] == "priority_first_fit"
    assert gated["gate"]["integer_lower_bound"] == 3
    assert gated["evaluation"]["costs"]["total"] == 3
    assert gated["placement"]["assignments"] == {"b": "m", "c": "m"}


def test_gate_certifies_a_time_limited_incumbent_that_reaches_exact_bound(monkeypatch):
    machines = [p.Machine("m", 4, 4)]
    tasks = [p.Task("a", 3, 3, 3), p.Task("b", 2, 2, 2), p.Task("c", 2, 2, 2)]
    monkeypatch.setattr(
        p,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1, x=np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]), fun=3.0
        ),
    )
    gated = p.optimize_placement_gated(machines, tasks)
    assert gated["gate"]["route"] == "milp"
    assert gated["gate"]["certified_optimal"]
    assert gated["placement"]["status"] == "optimal"
    assert gated["evaluation"]["costs"]["total"] == 3
