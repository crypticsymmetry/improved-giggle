"""Chronological, local-only planning replay with an independently scored baseline.

Weighted penalty units are not money, human effort, or measured business savings.
Both strategies see identical accepted facts and the same initial incumbent.
Repeated snapshots are alternative plans, never cumulative demand or cost.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isclose, isfinite
from platform import python_version
from statistics import median
from time import perf_counter
from typing import Any

import scipy

from .evaluation import evaluate_plan
from .events import EvidenceUpdate
from .optimizer import Allocation, Order, Plan, Policy, Supply, validate_inputs
from .pipeline import OperationsPipeline


def _name(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _json_value(value: Any) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Update values must be finite JSON values with string object keys")


@dataclass(frozen=True)
class ReplayBatch:
    event_id: str
    updates: tuple[EvidenceUpdate, ...]

    def __post_init__(self) -> None:
        _name(self.event_id, "event_id")
        if not isinstance(self.updates, tuple) or not self.updates:
            raise ValueError("updates must be a nonempty tuple of EvidenceUpdate records")
        seen: set[tuple[str, str]] = set()
        for update in self.updates:
            if not isinstance(update, EvidenceUpdate):
                raise ValueError("updates must contain EvidenceUpdate records")
            for field in ("entity", "field", "source"):
                _name(getattr(update, field), f"update.{field}")
            key = (update.entity, update.field)
            if key in seen:
                raise ValueError(f"Duplicate updated field: {key}")
            seen.add(key)
            try:
                _json_value(update.value)
            except RecursionError as exc:
                raise ValueError("Update values must not be circular") from exc
        try:
            json.dumps([asdict(update) for update in self.updates], allow_nan=False)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("updates must contain finite JSON-compatible values") from exc


@dataclass(frozen=True)
class ReplayCase:
    name: str
    supplies: tuple[Supply, ...]
    orders: tuple[Order, ...]
    policy: Policy
    batches: tuple[ReplayBatch, ...]
    synthetic: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        _name(self.name, "case name")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a boolean")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if not isinstance(self.supplies, tuple) or any(
            not isinstance(s, Supply) for s in self.supplies
        ):
            raise ValueError("supplies must be a tuple of Supply records")
        if not isinstance(self.orders, tuple) or any(not isinstance(o, Order) for o in self.orders):
            raise ValueError("orders must be a tuple of Order records")
        if not isinstance(self.policy, Policy):
            raise ValueError("policy must be a Policy record")
        if not isinstance(self.batches, tuple) or any(
            not isinstance(b, ReplayBatch) for b in self.batches
        ):
            raise ValueError("batches must be a tuple of ReplayBatch records")
        event_ids = [batch.event_id for batch in self.batches]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Replay batch event IDs must be unique")
        validate_inputs(self.supplies, self.orders, self.policy)
        known = {f"supply:{s.id}": s for s in self.supplies} | {
            f"order:{o.id}": o for o in self.orders
        }
        policy = self.policy
        for batch in self.batches:
            batch.__post_init__()
            for update in batch.updates:
                if update.entity == "policy" and update.field == "config":
                    try:
                        policy = OperationsPipeline._policy(update.value)
                    except TypeError as exc:
                        raise ValueError("Invalid policy update") from exc
                    continue
                record = known.get(update.entity)
                allowed = (
                    {"sku", "quantity", "available_day", "unit_cost"}
                    if isinstance(record, Supply)
                    else {"sku", "quantity", "due_day", "priority"}
                )
                if record is None or update.field not in allowed:
                    raise ValueError(
                        f"Unknown scenario entity or field: {update.entity}.{update.field}"
                    )
                replacement = replace(record, **{update.field: update.value})
                validate_inputs(
                    [replacement] if isinstance(replacement, Supply) else [],
                    [replacement] if isinstance(replacement, Order) else [],
                    policy,
                )
                known[update.entity] = replacement

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible case manifest."""
        from .replay_io import replay_case_to_dict

        return replay_case_to_dict(self)


def _quantities(plan: Plan | None) -> dict[tuple[str, str], int]:
    quantities: dict[tuple[str, str], int] = {}
    if plan is not None:
        for allocation in plan.allocations:
            key = (allocation.order_id, allocation.supply_id)
            quantities[key] = quantities.get(key, 0) + allocation.quantity
    return quantities


def _objective(
    supplies: list[Supply],
    orders: list[Order],
    policy: Policy,
    allocations: tuple[Allocation, ...],
    unfilled: dict[str, int],
    previous: Plan | None,
) -> int:
    supply_map, order_map = {s.id: s for s in supplies}, {o.id: o for o in orders}
    cost = sum(
        a.quantity
        * (
            supply_map[a.supply_id].unit_cost
            + max(0, supply_map[a.supply_id].available_day - order_map[a.order_id].due_day)
            * policy.late_penalty
            + int(supply_map[a.supply_id].sku != order_map[a.order_id].sku)
            * policy.substitution_penalty
        )
        for a in allocations
    )
    cost += sum(unfilled[o.id] * o.priority * policy.unfilled_penalty for o in orders)
    if previous is not None:
        current = {(a.order_id, a.supply_id): a.quantity for a in allocations}
        prior = _quantities(previous)
        cost += policy.disruption_penalty * sum(
            abs(current.get(key, 0) - prior.get(key, 0)) for key in current.keys() | prior.keys()
        )
    return cost


def _greedy(
    supplies: list[Supply], orders: list[Order], policy: Policy, previous: Plan | None
) -> Plan:
    remaining = {s.id: s.quantity for s in supplies}
    allocations: list[Allocation] = []
    unfilled: dict[str, int] = {}
    for order in sorted(orders, key=lambda o: (-o.priority, o.due_day, o.id)):

        def cost(supply: Supply) -> int:
            return (
                supply.unit_cost
                + policy.late_penalty * max(0, supply.available_day - order.due_day)
                + policy.substitution_penalty * int(supply.sku != order.sku)
            )

        eligible = [
            s
            for s in supplies
            if (s.sku == order.sku or s.sku in policy.substitutions.get(order.sku, ()))
            and (policy.allow_late or s.available_day <= order.due_day)
        ]
        demand = order.quantity
        for supply in sorted(eligible, key=lambda s: (cost(s), s.id)):
            if cost(supply) >= policy.unfilled_penalty * order.priority:
                break
            quantity = min(demand, remaining[supply.id])
            if quantity:
                allocations.append(Allocation(order.id, supply.id, quantity))
                remaining[supply.id] -= quantity
                demand -= quantity
            if demand == 0:
                break
        unfilled[order.id] = demand
    result = tuple(allocations)
    return Plan(
        result,
        unfilled,
        float(_objective(supplies, orders, policy, result, unfilled, previous)),
        "heuristic",
    )


def run_replay(case: ReplayCase) -> dict[str, Any]:
    """Replay confirmed changes in isolation, verifying each duplicate receipt.

    The greedy baseline prioritizes orders and chooses cheapest eligible units;
    it does not optimize opportunity cost or disruption. Optimized timing includes
    persistence and support bookkeeping, so timings are not solver speedup claims.
    """
    if not isinstance(case, ReplayCase):
        raise ValueError("case must be a ReplayCase")
    # Revalidate after copying because frozen records can contain mutable values.
    case = deepcopy(case)
    case.__post_init__()
    for batch in case.batches:
        batch.__post_init__()
    manifest = case.to_dict()
    fingerprint = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    start = perf_counter()
    steps: list[dict[str, Any]] = []
    initial: Plan | None = None
    active_result = None
    with OperationsPipeline() as pipeline:
        pipeline.seed(case.supplies, case.orders, case.policy)
        for index in range(len(case.batches) + 1):
            step_start = perf_counter()
            batch = case.batches[index - 1] if index else None
            duplicate_verified = None
            receipt_data = None
            event_apply_seconds = 0.0
            duplicate_replay_seconds = 0.0
            if batch is not None:
                revisions = pipeline.revisions()
                apply_start = perf_counter()
                receipt = pipeline.apply_updates(
                    batch.event_id,
                    batch.updates,
                    expected_evidence_revision=revisions["evidence_revision"],
                    expected_operations_revision=revisions["operations_revision"],
                )
                event_apply_seconds = perf_counter() - apply_start
                after = pipeline.revisions()
                replay_start = perf_counter()
                replay = pipeline.apply_updates(
                    batch.event_id,
                    batch.updates,
                    expected_evidence_revision=revisions["evidence_revision"],
                    expected_operations_revision=revisions["operations_revision"],
                )
                duplicate_replay_seconds = perf_counter() - replay_start
                original_receipt, replay_receipt = receipt.to_dict(), replay.to_dict()
                replay_receipt["replayed"] = False
                duplicate_verified = (
                    replay.replayed
                    and not receipt.replayed
                    and original_receipt == replay_receipt
                    and after == pipeline.revisions()
                )
                if not duplicate_verified:
                    raise RuntimeError(
                        "Duplicate replay changed revisions or failed receipt verification"
                    )
                receipt_data = receipt.to_dict()
            invalidated = 0
            if active_result is not None:
                invalidated = sum(
                    bool(pipeline._current_support(d.supports)[1]) for d in active_result.decisions
                )
            supplies, orders, policy, _ = pipeline._snapshot()
            opt_start = perf_counter()
            result = pipeline.plan(previous=initial)
            optimized_elapsed = perf_counter() - opt_start
            optimized_eval = evaluate_plan(supplies, orders, policy, result.plan, initial)
            greedy_start = perf_counter()
            greedy = _greedy(supplies, orders, policy, initial)
            greedy_elapsed = perf_counter() - greedy_start
            greedy_eval = evaluate_plan(supplies, orders, policy, greedy, initial)
            gap = greedy_eval.costs.total - optimized_eval.costs.total
            proven = result.plan.status == "optimal"
            if proven and gap < 0 and not isclose(gap, 0, abs_tol=1e-6):
                raise RuntimeError(
                    "Proven-optimal allocation is worse than independently feasible greedy baseline"
                )
            current = _quantities(result.plan)
            previous = _quantities(active_result.plan) if active_result is not None else current
            union = current.keys() | previous.keys()
            steps.append(
                {
                    "step": index,
                    "event_id": batch.event_id if batch else None,
                    "confirmed_fields": len(batch.updates) if batch else 0,
                    "proposed_decision_lines": len(result.decisions),
                    "previous_decisions_invalidated": invalidated,
                    "previous_active_decision_count": len(active_result.decisions)
                    if active_result is not None
                    else 0,
                    "event_apply_seconds": event_apply_seconds,
                    "duplicate_replay_seconds": duplicate_replay_seconds,
                    "changed_allocation_arcs": sum(
                        current.get(key, 0) != previous.get(key, 0) for key in union
                    ),
                    "changed_allocation_units": sum(
                        abs(current.get(key, 0) - previous.get(key, 0)) for key in union
                    ),
                    "revisions": pipeline.revisions(),
                    "duplicate_replay_verified": duplicate_verified,
                    "update_receipt": receipt_data,
                    "optimized": {
                        "plan": result.plan.to_dict(),
                        "evaluation": optimized_eval.to_dict(),
                        "status": result.plan.status,
                        "elapsed_seconds": optimized_elapsed,
                    },
                    "greedy": {
                        "plan": greedy.to_dict(),
                        "evaluation": greedy_eval.to_dict(),
                        "status": greedy.status,
                        "elapsed_seconds": greedy_elapsed,
                    },
                    "greedy_minus_optimized": gap,
                    "optimized_no_worse_when_proven": gap >= 0 if proven else None,
                    "step_elapsed_seconds": perf_counter() - step_start,
                }
            )
            if initial is None:
                initial = result.plan
            active_result = result
        if pipeline._reserved("supply_id") or any(
            event["kind"] == "execution_intent" for event in pipeline.events()
        ):
            raise RuntimeError(
                "Planning replay unexpectedly produced reservations or execution intents"
            )
    optimized_times = [step["optimized"]["elapsed_seconds"] for step in steps]
    greedy_times = [step["greedy"]["elapsed_seconds"] for step in steps]
    return {
        "case_name": case.name,
        "synthetic": case.synthetic,
        "seed": case.seed,
        "source_fingerprint": fingerprint,
        "units": "weighted penalty units; integer interchangeable supply units; relative integer days",
        "limitations": [
            "No human review time, monetary savings, or customer outcome is measured.",
            "Synthetic cases are fixtures, not actual business data; supplied cases retain caller provenance.",
            "All updated snapshots use the frozen initial optimized allocation as their common cost reference.",
            "Changed arcs and invalidated decisions compare only the preceding active optimized snapshot.",
            "Source-support impacts describe feasibility justification; zero invalidations does not establish unchanged global optimality or remove the need to reoptimize.",
            "Greedy ignores opportunity cost and disruption when selecting supply; no optimality guarantee.",
            "Optimized timing includes proposal persistence and support bookkeeping; greedy timing covers allocation only.",
            "Repeated snapshots are alternatives: demand, shortages, and costs are reported only for the final snapshot in summaries.",
            "No approvals, commits, reservations, or external dispatch occur.",
        ],
        "environment": {"python": python_version(), "scipy": scipy.__version__},
        "steps": steps,
        "summary": {
            "snapshots": len(steps),
            "confirmed_fields": sum(s["confirmed_fields"] for s in steps),
            "proposed_decision_lines": sum(s["proposed_decision_lines"] for s in steps),
            "cumulative_changed_allocation_arcs": sum(s["changed_allocation_arcs"] for s in steps),
            "cumulative_changed_allocation_units": sum(
                s["changed_allocation_units"] for s in steps
            ),
            "cumulative_previous_decisions_invalidated": sum(
                s["previous_decisions_invalidated"] for s in steps
            ),
            "cumulative_previous_active_decision_count": sum(
                s["previous_active_decision_count"] for s in steps
            ),
            "event_apply_seconds": sum(s["event_apply_seconds"] for s in steps),
            "duplicate_replay_seconds": sum(s["duplicate_replay_seconds"] for s in steps),
            "duplicate_replays_verified": sum(
                s["duplicate_replay_verified"] is True for s in steps
            ),
            "final_optimized": steps[-1]["optimized"]["evaluation"],
            "final_greedy": steps[-1]["greedy"]["evaluation"],
            "execution_intents": 0,
            "runtime_seconds": perf_counter() - start,
            "optimized_pipeline_median_seconds": median(optimized_times),
            "greedy_allocator_median_seconds": median(greedy_times),
        },
    }
