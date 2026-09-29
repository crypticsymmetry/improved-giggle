"""Atomic local approval and reservation for an entire recorded plan.

These operations record local execution intents only. Committed decisions are
historical and idempotent; no remote dispatch or reservation compensation occurs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import Decision, OperationsPipeline


def _plan_decisions(pipeline: OperationsPipeline, plan_id: str) -> tuple[Decision, ...]:
    decisions = tuple(decision for decision in pipeline.decisions() if decision.plan_id == plan_id)
    if decisions:
        return decisions
    # Empty optimization results still have a durable plan_created event.
    for row in pipeline._db.execute("SELECT payload FROM ops_events WHERE kind='plan_created'"):
        if json.loads(row[0]).get("plan_id") == plan_id:
            return ()
    raise KeyError(plan_id)


def _approval_preflight(pipeline: OperationsPipeline, pending: tuple[Decision, ...]) -> None:
    from .pipeline import StaleDecisionError

    for decision in pending:
        if decision.status not in {"proposed", "approved"}:
            raise ValueError(f"Decision {decision.id} cannot be approved from {decision.status}")
        pipeline._guard(decision)
    if not pending:
        return
    supplies, orders, _, _ = pipeline._snapshot()
    # Detect globally inconsistent historical reservations as well as the new
    # group's cumulative capacity and demand. Approval itself reserves nothing.
    pipeline._remaining(supplies, orders)
    used = pipeline._reserved("supply_id")
    filled = pipeline._reserved("order_id")
    for decision in pending:
        for allocation in decision.allocations:
            used[allocation.supply_id] = used.get(allocation.supply_id, 0) + allocation.quantity
            filled[allocation.order_id] = filled.get(allocation.order_id, 0) + allocation.quantity
    if any(used.get(supply.id, 0) > supply.quantity for supply in supplies):
        raise StaleDecisionError("Plan allocations exceed unreserved supply capacity")
    if any(filled.get(order.id, 0) > order.quantity for order in orders):
        raise StaleDecisionError("Plan allocations exceed outstanding order demand")


def approve_plan(pipeline: OperationsPipeline, plan_id: str) -> tuple[Decision, ...]:
    """Validate the whole pending group before atomically promoting proposals."""
    with pipeline._lock, pipeline._transaction():
        decisions = _plan_decisions(pipeline, plan_id)
        pending = tuple(decision for decision in decisions if decision.status != "committed")
        _approval_preflight(pipeline, pending)
        changed = [decision.id for decision in pending if decision.status == "proposed"]
        for decision_id in changed:
            pipeline._db.execute(
                "UPDATE ops_decisions SET status='approved' WHERE id=?", (decision_id,)
            )
            pipeline._event("decision_approved", {"decision_id": decision_id})
        if changed:
            pipeline._event("plan_approved", {"plan_id": plan_id, "decision_ids": changed})
        return tuple(pipeline._decision(decision.id) for decision in decisions)


def commit_plan(pipeline: OperationsPipeline, plan_id: str) -> dict[str, Any]:
    """Reserve and record all pending decisions in one SQLite transaction.

    Nested per-decision commits use savepoints. A stale later decision or event
    insertion failure rolls back earlier reservations, statuses, and intents.
    Existing committed members remain untouched and produce no duplicate events.
    """
    with pipeline._lock, pipeline._transaction():
        decisions = _plan_decisions(pipeline, plan_id)
        pending = tuple(decision for decision in decisions if decision.status != "committed")
        if any(decision.status != "approved" for decision in pending):
            raise ValueError("Explicit approval is required for every decision before plan commit")
        for decision in pending:
            pipeline.commit(decision.id)
        return {
            "plan_id": plan_id,
            "status": "committed",
            "decision_ids": [decision.id for decision in decisions],
            "idempotent": not pending,
        }
