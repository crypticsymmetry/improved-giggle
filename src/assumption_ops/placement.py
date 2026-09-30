"""Indivisible CPU/memory placement, independently verified before reporting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isclose, isfinite
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Machine:
    id: str
    cpu: int
    memory: int


@dataclass(frozen=True)
class Task:
    id: str
    cpu: int
    memory: int
    priority: int


@dataclass(frozen=True)
class PlacementPolicy:
    migration_penalty: int = 1


@dataclass(frozen=True)
class Placement:
    assignments: dict[str, str]
    pending: tuple[str, ...]
    objective: float
    status: str

    def to_dict(self) -> dict:
        return {
            "assignments": dict(self.assignments),
            "pending": list(self.pending),
            "objective": self.objective,
            "status": self.status,
        }


def _integer(value: object, label: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**53:
        raise ValueError(f"{label} must be an integer between {minimum} and 2**53")


def validate_placement_inputs(
    machines: Sequence[Machine], tasks: Sequence[Task], policy: PlacementPolicy
) -> None:
    """Reject malformed identifiers, quantities, and numerically unsafe scores."""
    if not isinstance(policy, PlacementPolicy):
        raise ValueError("policy must be PlacementPolicy")
    _integer(policy.migration_penalty, "migration_penalty")
    for records, kind in ((machines, Machine), (tasks, Task)):
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("machines and tasks must be sequences")
        seen: set[str] = set()
        for item in records:
            if not isinstance(item, kind):
                raise ValueError(f"records must contain {kind.__name__}")
            if not isinstance(item.id, str) or not item.id.strip() or item.id in seen:
                raise ValueError("identifiers must be nonempty and unique")
            seen.add(item.id)
            _integer(item.cpu, "cpu")
            _integer(item.memory, "memory")
            if isinstance(item, Task):
                _integer(item.priority, "priority", 1)
    if sum(t.priority for t in tasks) > 2**53:
        raise ValueError("total priority exceeds exact floating-point range")


def _previous(
    previous: Placement | None, policy: PlacementPolicy, tasks: Sequence[Task]
) -> dict[str, str]:
    if previous is None:
        return {}
    if not isinstance(previous, Placement) or not isinstance(previous.assignments, dict):
        raise ValueError("previous must be Placement with assignments dictionary")
    old = dict(previous.assignments)
    if any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in old.items()
    ):
        raise ValueError("previous assignments require nonempty string identifiers")
    if len(old) * policy.migration_penalty + sum(t.priority for t in tasks) > 2**53:
        raise ValueError("worst possible objective exceeds exact floating-point range")
    return old


def evaluate_placement(
    machines: Sequence[Machine],
    tasks: Sequence[Task],
    policy: PlacementPolicy,
    placement: Placement,
    previous: Placement | None = None,
) -> dict:
    """Reconstruct exact integer capacity, coverage, service, and score checks."""
    machines, tasks, policy, placement, previous = deepcopy(
        (machines, tasks, policy, placement, previous)
    )
    validate_placement_inputs(machines, tasks, policy)
    old = _previous(previous, policy, tasks)
    if not isinstance(placement, Placement) or not isinstance(placement.assignments, dict):
        raise ValueError("placement must contain assignments dictionary")
    if not isinstance(placement.pending, tuple) or any(
        not isinstance(x, str) for x in placement.pending
    ):
        raise ValueError("pending must be tuple of task identifiers")
    by_machine = {m.id: m for m in machines}
    by_task = {t.id: t for t in tasks}
    assigned = placement.assignments
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in assigned.items()):
        raise ValueError("assignment identifiers must be strings")
    pending = set(placement.pending)
    if (
        len(pending) != len(placement.pending)
        or set(assigned) & pending
        or set(assigned) | pending != set(by_task)
    ):
        raise ValueError("each task must be assigned or pending exactly once")
    cpu = dict.fromkeys(by_machine, 0)
    memory = dict.fromkeys(by_machine, 0)
    for task_id, machine_id in assigned.items():
        if machine_id not in by_machine:
            raise ValueError("unknown machine")
        cpu[machine_id] += by_task[task_id].cpu
        memory[machine_id] += by_task[task_id].memory
    if any(cpu[m.id] > m.cpu or memory[m.id] > m.memory for m in machines):
        raise ValueError("machine capacity exceeded")
    pending_priority = sum(by_task[t].priority for t in pending)
    migration = policy.migration_penalty * sum(assigned.get(t) != m for t, m in old.items())
    total = pending_priority + migration
    if (
        isinstance(placement.objective, bool)
        or not isinstance(placement.objective, (int, float))
        or not isfinite(placement.objective)
        or not isclose(placement.objective, total, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("placement objective does not match independently reconstructed score")
    return {
        "costs": {"pending_priority": pending_priority, "migration": migration, "total": total},
        "metrics": {
            "tasks": len(tasks),
            "placed_tasks": len(assigned),
            "pending_tasks": len(pending),
            "placed_priority": sum(t.priority for t in tasks if t.id in assigned),
            "total_priority": sum(t.priority for t in tasks),
            "used_cpu": sum(cpu.values()),
            "used_memory": sum(memory.values()),
            "cpu_capacity": sum(m.cpu for m in machines),
            "memory_capacity": sum(m.memory for m in machines),
        },
    }


def _model(machines, tasks, policy, previous, time_limit):
    validate_placement_inputs(machines, tasks, policy)
    machines, tasks = tuple(machines), tuple(tasks)
    old = _previous(previous, policy, tasks)
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive")
    arcs = [
        (ti, mi)
        for ti, t in enumerate(tasks)
        for mi, m in enumerate(machines)
        if t.cpu <= m.cpu and t.memory <= m.memory
    ]
    coefficients = [
        -policy.migration_penalty if old.get(tasks[ti].id) == machines[mi].id else 0
        for ti, mi in arcs
    ]
    coefficients.extend(t.priority for t in tasks)
    rows, cols, values = [], [], []
    for col, (ti, mi) in enumerate(arcs):
        for row, value in (
            (ti, 1),
            (len(tasks) + 2 * mi, tasks[ti].cpu),
            (len(tasks) + 2 * mi + 1, tasks[ti].memory),
        ):
            rows.append(row)
            cols.append(col)
            values.append(value)
    for ti in range(len(tasks)):
        rows.append(ti)
        cols.append(len(arcs) + ti)
        values.append(1)
    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(tasks) + 2 * len(machines), len(coefficients))
    ).tocsr()
    upper = np.array(
        [1] * len(tasks) + [x for m in machines for x in (m.cpu, m.memory)], dtype=float
    )
    lower = np.array([1] * len(tasks) + [-np.inf] * (2 * len(machines)))
    constant = len(old) * policy.migration_penalty
    return (
        machines,
        tasks,
        old,
        arcs,
        np.array(coefficients, dtype=float),
        matrix,
        lower,
        upper,
        constant,
    )


def _verified_primal(result, coefficients, matrix, lower, upper):
    if result.x is None:
        raise RuntimeError("solver returned no incumbent")
    primal = np.asarray(result.x, dtype=float)
    if (
        primal.shape != coefficients.shape
        or not np.all(np.isfinite(primal))
        or np.any(primal < -1e-7)
        or np.any(primal > 1 + 1e-7)
    ):
        raise RuntimeError("solver returned invalid variable bounds")
    activity = matrix @ primal
    if np.any(activity < lower - 1e-7) or np.any(activity > upper + 1e-7):
        raise RuntimeError("solver returned infeasible incumbent")
    if (
        result.fun is None
        or not isfinite(result.fun)
        or not isclose(float(coefficients @ primal), float(result.fun), rel_tol=0, abs_tol=1e-6)
    ):
        raise RuntimeError("solver objective inconsistent with primal")
    return primal


def optimize_placement(
    machines, tasks, policy=PlacementPolicy(), previous=None, *, time_limit=30.0
) -> Placement:
    """Solve binary placement; accept only independently verified integer incumbents."""
    machines, tasks, policy, previous = deepcopy((machines, tasks, policy, previous))
    machines, tasks, old, arcs, c, a, low, high, constant = _model(
        machines, tasks, policy, previous, time_limit
    )
    if not len(tasks):
        return Placement({}, (), float(constant), "optimal")
    result = milp(
        c,
        integrality=np.ones(len(c)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, low, high),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    if result.status not in (0, 1):
        raise RuntimeError(f"placement solver failed: {result.message}")
    x = _verified_primal(result, c, a, low, high)
    integer = np.rint(x)
    if np.any(np.abs(integer - x) > 1e-5):
        raise RuntimeError("placement solver returned fractional incumbent")
    assignments = {
        tasks[ti].id: machines[mi].id for col, (ti, mi) in enumerate(arcs) if integer[col] == 1
    }
    pending = tuple(t.id for ti, t in enumerate(tasks) if integer[len(arcs) + ti] == 1)
    total = sum(t.priority for t in tasks if t.id in pending) + policy.migration_penalty * sum(
        assignments.get(t) != m for t, m in old.items()
    )
    if result.status == 0 and not isclose(
        total - constant, float(result.fun), rel_tol=0, abs_tol=1e-6
    ):
        raise RuntimeError("optimal integer objective does not match reported objective")
    placement = Placement(
        assignments, pending, float(total), "optimal" if result.status == 0 else "feasible_limit"
    )
    evaluate_placement(machines, tasks, policy, placement, previous)
    return placement


def placement_lp_bound(
    machines, tasks, policy=PlacementPolicy(), previous=None, *, time_limit=30.0
) -> dict:
    """Fractional relaxation for a bound only; never convert fractions into placements."""
    machines, tasks, policy, previous = deepcopy((machines, tasks, policy, previous))
    machines, tasks, old, arcs, c, a, low, high, constant = _model(
        machines, tasks, policy, previous, time_limit
    )
    if not len(tasks):
        return {"status": "optimal", "lower_bound": float(constant), "fractional_variables": 0}
    result = linprog(
        c,
        A_eq=a[: len(tasks)],
        b_eq=high[: len(tasks)],
        A_ub=a[len(tasks) :],
        b_ub=high[len(tasks) :],
        bounds=(0, 1),
        method="highs",
        options={"time_limit": float(time_limit)},
    )
    if result.status != 0:
        return {
            "status": "limit" if result.status == 1 else "failed",
            "lower_bound": None,
            "fractional_variables": None,
            "message": result.message,
        }
    x = _verified_primal(result, c, a, low, high)
    return {
        "status": "optimal",
        "lower_bound": float(result.fun + constant),
        "fractional_variables": int(np.count_nonzero(np.abs(x - np.rint(x)) > 1e-5)),
    }


def _greedy(machines, tasks, policy, previous, best_fit, stability=False):
    old = _previous(previous, policy, tasks)
    available = {m.id: [m.cpu, m.memory] for m in machines}
    assignments = {}
    machines = sorted(machines, key=lambda m: m.id)
    for t in sorted(tasks, key=lambda t: (-t.priority, t.id)):
        choices = [
            m
            for m in machines
            if available[m.id][0] >= t.cpu
            and available[m.id][1] >= t.memory
            and (policy.migration_penalty * int(t.id in old and old[t.id] != m.id))
            < t.priority + policy.migration_penalty * int(t.id in old)
        ]
        if not choices:
            continue
        if best_fit:
            choices.sort(
                key=lambda m: (
                    policy.migration_penalty * int(t.id in old and old[t.id] != m.id)
                    if stability
                    else 0,
                    (available[m.id][0] - t.cpu) / max(1, m.cpu)
                    + (available[m.id][1] - t.memory) / max(1, m.memory),
                    m.id,
                )
            )
        chosen = choices[0]
        assignments[t.id] = chosen.id
        available[chosen.id][0] -= t.cpu
        available[chosen.id][1] -= t.memory
    pending = tuple(t.id for t in tasks if t.id not in assignments)
    total = sum(t.priority for t in tasks if t.id in pending) + policy.migration_penalty * sum(
        assignments.get(t) != m for t, m in old.items()
    )
    return Placement(assignments, pending, float(total), "heuristic")


def compare_placements(
    machines, tasks, policy=PlacementPolicy(), previous=None, *, time_limit=30.0
) -> dict:
    """Compare identical inputs with exact verification and an optimal LP bound."""
    validate_placement_inputs(machines, tasks, policy)
    machines, tasks, policy, previous = deepcopy((tuple(machines), tuple(tasks), policy, previous))
    _model(machines, tasks, policy, previous, time_limit)
    started = perf_counter()
    bound = placement_lp_bound(machines, tasks, policy, previous, time_limit=time_limit)
    bound["elapsed_seconds"] = perf_counter() - started
    results = []
    for method in ("milp", "priority_first_fit", "best_fit", "stability_best_fit"):
        started = perf_counter()
        placement = (
            optimize_placement(machines, tasks, policy, previous, time_limit=time_limit)
            if method == "milp"
            else _greedy(
                machines,
                tasks,
                policy,
                previous,
                method in ("best_fit", "stability_best_fit"),
                method == "stability_best_fit",
            )
        )
        elapsed = perf_counter() - started
        evaluation = evaluate_placement(machines, tasks, policy, placement, previous)
        lower = bound["lower_bound"]
        if lower is not None and placement.objective < lower - 1e-6:
            raise RuntimeError("verified placement objective is below LP bound")
        results.append(
            {
                "method": method,
                "status": placement.status,
                "placement": placement.to_dict(),
                "evaluation": evaluation,
                "elapsed_seconds": elapsed,
                "gap_to_lp_bound": None if lower is None else max(0.0, placement.objective - lower),
            }
        )
    if results[0]["status"] == "optimal" and any(
        results[0]["evaluation"]["costs"]["total"] > r["evaluation"]["costs"]["total"]
        for r in results[1:]
    ):
        raise RuntimeError("claimed optimal MILP score exceeds a feasible greedy score")
    return {"results": results, "lp_relaxation": bound}
