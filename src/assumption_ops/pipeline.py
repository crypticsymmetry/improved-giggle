"""Supplier-exception orchestration, persisted decisions, and local execution guards.

No ERP writes occur here. A commit reserves inventory in this local ledger and records
an execution intent. Integrators must implement an idempotent transactional outbox
consumer for external side effects; a database transaction cannot make a remote API atomic.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import sqlite3
from threading import RLock
from typing import Any, Iterator, Sequence
from uuid import uuid4

from .atms import ATMS
from .evidence import EvidenceConflict, EvidenceStore
from .optimizer import Allocation, Order, Plan, Policy, Supply, optimize, validate_inputs
from .scenarios import Scenario, ScenarioComparison, compare_scenarios as compare_interventions


class StaleDecisionError(RuntimeError):
    """A decision lost its support or cannot reserve its planned inventory."""


@dataclass(frozen=True)
class Decision:
    id: str
    order_id: str
    status: str
    supports: dict[str, str]
    allocations: tuple[Allocation, ...]
    plan_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningResult:
    plan: Plan
    decisions: tuple[Decision, ...]
    evidence_revision: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "evidence_revision": self.evidence_revision,
        }


class OperationsPipeline:
    """One bounded supplier scenario, backed by SQLite.

    Quantities are integral interchangeable units; substitutions are one-for-one.
    Days are relative integers, and costs are integer business penalty units.
    Multiple instances can share a file; commit checks and reservations are protected
    by SQLite's write lock. Each instance is used on its creating thread.
    """

    def __init__(self, db_path: str = ":memory:", *, max_environments: int = 10000):
        self._db = sqlite3.connect(db_path, timeout=30)
        self._db.row_factory = sqlite3.Row
        self.store = EvidenceStore(db_path, connection=self._db)
        self._lock = RLock()
        self._max_environments = max_environments
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ops_state (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ops_decisions (
                id TEXT PRIMARY KEY, order_id TEXT NOT NULL, status TEXT NOT NULL,
                plan_id TEXT NOT NULL, supports TEXT NOT NULL, allocations TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ops_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE TABLE IF NOT EXISTS ops_reservations (
                decision_id TEXT NOT NULL, order_id TEXT NOT NULL,
                supply_id TEXT NOT NULL, quantity INTEGER NOT NULL CHECK(quantity > 0),
                PRIMARY KEY(decision_id, supply_id)
            );
        """)
        self._db.commit()
        self.reasoner = ATMS(max_environments=max_environments)
        self._sync()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self.store.transaction():
            yield

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO ops_events(kind,payload) VALUES (?,?)",
            (kind, json.dumps(payload, sort_keys=True, allow_nan=False)),
        )

    @property
    def is_seeded(self) -> bool:
        return (
            self._db.execute("SELECT 1 FROM ops_state WHERE key='scenario'").fetchone() is not None
        )

    def _state(self) -> dict[str, Any]:
        row = self._db.execute("SELECT value FROM ops_state WHERE key='scenario'").fetchone()
        if row is None:
            raise ValueError("Seed a scenario before planning")
        return json.loads(row[0])

    def seed(self, supplies: Sequence[Supply], orders: Sequence[Order], policy: Policy) -> None:
        """Initialize once. Subsequent facts enter through propose/accept/resolve."""
        with self._lock, self._transaction():
            if self._db.execute("SELECT 1 FROM ops_state WHERE key='scenario'").fetchone():
                raise ValueError("Scenario already seeded")
            # Validate once without solving: initialization does not allocate anything.
            validate_inputs(supplies, orders, policy)
            for prefix, records in (("supply", supplies), ("order", orders)):
                for record in records:
                    for field, value in asdict(record).items():
                        if field != "id":
                            evidence_id = self.store.propose(
                                f"{prefix}:{record.id}", field, value, "scenario seed"
                            )
                            self.store.accept(evidence_id)
            policy_id = self.store.propose(
                "policy", "config", json.loads(json.dumps(asdict(policy))), "scenario seed"
            )
            self.store.accept(policy_id)
            with self._transaction():
                self._db.execute(
                    "INSERT INTO ops_state(key,value) VALUES ('scenario',?)",
                    (
                        json.dumps(
                            {"supplies": [s.id for s in supplies], "orders": [o.id for o in orders]}
                        ),
                    ),
                )
                self._event("scenario_seeded", {"supplies": len(supplies), "orders": len(orders)})
            self._sync()

    def propose(self, entity: str, field: str, value: Any, source: str) -> str:
        """Propose an observation. It has no effect until explicitly accepted."""
        with self._lock:
            state = self._state()
            fields = {
                **{
                    f"supply:{i}": {"sku", "quantity", "available_day", "unit_cost"}
                    for i in state["supplies"]
                },
                **{
                    f"order:{i}": {"sku", "quantity", "due_day", "priority"}
                    for i in state["orders"]
                },
                "policy": {"config"},
            }
            if field not in fields.get(entity, set()):
                raise ValueError("Unknown scenario entity or field")
            # Validate a hypothetical replacement before accepting it into the inbox.
            if entity == "policy":
                self._policy(value)
            elif field == "sku":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("SKU must be a nonempty string")
            elif type(value) is not int or value < (1 if field == "priority" else 0):
                raise ValueError(f"{field} must be a valid nonnegative integer")
            return self.store.propose(entity, field, value, source)

    def accept(self, evidence_id: str) -> None:
        with self._lock:
            self.store.accept(evidence_id)
            self._sync()

    def resolve(self, entity: str, field: str, evidence_id: str) -> None:
        with self._lock:
            self.store.resolve(entity, field, evidence_id)
            self._sync()

    @staticmethod
    def _policy(value: dict[str, Any]) -> Policy:
        if not isinstance(value, dict):
            raise ValueError("Policy config must be an object")
        config = dict(value)
        substitutions = config.get("substitutions", {})
        if not isinstance(substitutions, dict) or any(
            not isinstance(values, (list, tuple)) for values in substitutions.values()
        ):
            raise ValueError("Substitutions must map SKU strings to lists or tuples of SKUs")
        config["substitutions"] = {key: tuple(values) for key, values in substitutions.items()}
        policy = Policy(**config)
        validate_inputs([], [], policy)
        return policy

    def _snapshot(self) -> tuple[list[Supply], list[Order], Policy, dict[str, str]]:
        state = self._state()
        supports: dict[str, str] = {}

        def record(entity: str, fields: tuple[str, ...]) -> dict[str, Any]:
            values = {}
            for field in fields:
                evidence = self.store.current(entity, field)
                if evidence is None:
                    raise ValueError(f"Missing accepted evidence for {entity}.{field}")
                supports[f"{entity}|{field}"] = evidence.id
                values[field] = evidence.value
            return values

        supplies = [
            Supply(id=i, **record(f"supply:{i}", ("sku", "quantity", "available_day", "unit_cost")))
            for i in state["supplies"]
        ]
        orders = [
            Order(id=i, **record(f"order:{i}", ("sku", "quantity", "due_day", "priority")))
            for i in state["orders"]
        ]
        policy = self._policy(record("policy", ("config",))["config"])
        validate_inputs(supplies, orders, policy)
        return supplies, orders, policy, supports

    def _reserved(self, column: str) -> dict[str, int]:
        if column not in {"supply_id", "order_id"}:
            raise ValueError("Invalid reservation dimension")
        return {
            row[0]: row[1]
            for row in self._db.execute(
                f"SELECT {column}, SUM(quantity) FROM ops_reservations GROUP BY {column}"
            )
        }

    def _remaining(
        self, supplies: Sequence[Supply], orders: Sequence[Order]
    ) -> tuple[list[Supply], list[Order]]:
        supply_used, order_filled = self._reserved("supply_id"), self._reserved("order_id")
        remaining_supplies, remaining_orders = [], []
        for supply in supplies:
            if supply_used.get(supply.id, 0) > supply.quantity:
                raise StaleDecisionError(
                    f"Accepted supply {supply.id} is below committed reservations"
                )
            remaining_supplies.append(
                Supply(
                    **{
                        **asdict(supply),
                        "quantity": supply.quantity - supply_used.get(supply.id, 0),
                    }
                )
            )
        for order in orders:
            if order_filled.get(order.id, 0) > order.quantity:
                raise StaleDecisionError(
                    f"Accepted order {order.id} is below committed reservations"
                )
            remaining_orders.append(
                Order(
                    **{**asdict(order), "quantity": order.quantity - order_filled.get(order.id, 0)}
                )
            )
        return remaining_supplies, remaining_orders

    def what_if(
        self, overrides: dict[str, dict[str, Any]], *, previous: Plan | None = None
    ) -> Plan:
        """Solve hypothetical replacements without changing evidence or decisions.

        Example: {"supply:shipment": {"available_day": 8},
                  "policy": {"config": {"allow_late": True}}}.
        Policy config replaces the full policy; absent policy keys use defaults.
        The base accepted snapshot must be conflict-free. Scenario inputs are
        validated and committed reservations remain binding.
        """
        with self._lock, self._transaction():
            supplies, orders, policy, _ = self._snapshot()
            replacements = dict(overrides)

            def replace_records(prefix: str, records: Sequence[Any]) -> list[Any]:
                result = []
                for record in records:
                    changes = replacements.pop(f"{prefix}:{record.id}", {})
                    if not isinstance(changes, dict) or "id" in changes:
                        raise ValueError("Overrides must be field mappings and cannot change IDs")
                    result.append(type(record)(**{**asdict(record), **changes}))
                return result

            supplies = replace_records("supply", supplies)
            orders = replace_records("order", orders)
            changes = replacements.pop("policy", {})
            if changes:
                if set(changes) != {"config"}:
                    raise ValueError("Policy overrides must contain only config")
                policy = self._policy(changes["config"])
            if replacements:
                raise ValueError(f"Unknown scenario entities: {sorted(replacements)}")
            validate_inputs(supplies, orders, policy)
            supplies, orders = self._remaining(supplies, orders)
            return optimize(supplies, orders, policy, previous)

    def compare_scenarios(
        self, scenarios: Sequence[Scenario], *, previous: Plan | None = None
    ) -> ScenarioComparison:
        """Rank proposed interventions against one consistent accepted snapshot.

        Every solve uses the same prior allocation and cost weights. No observation,
        decision, approval, or reservation is created. Returned revisions identify
        the snapshot; selecting a candidate does not apply it to accepted evidence.
        """
        with self._lock, self._transaction():
            supplies, orders, policy, _ = self._snapshot()
            row = self._db.execute("SELECT COALESCE(MAX(sequence),0) FROM ops_events").fetchone()
            return compare_interventions(
                supplies,
                orders,
                policy,
                scenarios,
                previous,
                reserved_supply=self._reserved("supply_id"),
                reserved_order=self._reserved("order_id"),
                evidence_revision=self.store.revision,
                operations_revision=row[0],
            )

    def plan(self, *, previous: Plan | None = None) -> PlanningResult:
        """Optimize outstanding demand against unreserved supply, recording proposals.

        Decision support covers its order, allocated supplies, and policy. This is a
        feasibility/justification guard, not a claim that a decision remains globally
        optimal after unrelated demand changes. Capacity is rechecked at commit time.
        """
        with self._lock, self._transaction():
            supplies, orders, policy, supports = self._snapshot()
            revision = self.store.revision
            remaining_supplies, remaining_orders = self._remaining(supplies, orders)
            plan = optimize(remaining_supplies, remaining_orders, policy, previous)
            plan_id = uuid4().hex
            decisions = []
            for order in orders:
                allocations = tuple(a for a in plan.allocations if a.order_id == order.id)
                if not allocations:
                    continue
                entities = {f"order:{order.id}", "policy"} | {
                    f"supply:{a.supply_id}" for a in allocations
                }
                decision = Decision(
                    uuid4().hex,
                    order.id,
                    "proposed",
                    {k: v for k, v in supports.items() if k.rsplit("|", 1)[0] in entities},
                    allocations,
                    plan_id,
                )
                self._db.execute(
                    "INSERT INTO ops_decisions VALUES (?,?,?,?,?,?)",
                    (
                        decision.id,
                        decision.order_id,
                        decision.status,
                        plan_id,
                        json.dumps(decision.supports),
                        json.dumps([asdict(a) for a in allocations]),
                    ),
                )
                decisions.append(decision)
            self._event(
                "plan_created",
                {
                    "plan_id": plan_id,
                    "evidence_revision": revision,
                    "decision_ids": [d.id for d in decisions],
                    "plan": plan.to_dict(),
                },
            )
        self._sync()
        return PlanningResult(plan, tuple(decisions), revision)

    def decisions(self) -> tuple[Decision, ...]:
        return tuple(
            Decision(
                row["id"],
                row["order_id"],
                row["status"],
                json.loads(row["supports"]),
                tuple(Allocation(**a) for a in json.loads(row["allocations"])),
                row["plan_id"],
            )
            for row in self._db.execute("SELECT * FROM ops_decisions ORDER BY rowid")
        )

    def _decision(self, decision_id: str) -> Decision:
        for decision in self.decisions():
            if decision.id == decision_id:
                return decision
        raise KeyError(decision_id)

    def _current_support(self, supports: dict[str, str]) -> tuple[set[str], list[dict[str, Any]]]:
        active, problems = set(), []
        for key, evidence_id in supports.items():
            entity, field = key.rsplit("|", 1)
            try:
                current = self.store.current(entity, field)
            except EvidenceConflict:
                problems.append(
                    {"entity": entity, "field": field, "reason": "conflicting evidence"}
                )
                continue
            if current is not None and current.id == evidence_id:
                active.add(evidence_id)
            else:
                problems.append(
                    {
                        "entity": entity,
                        "field": field,
                        "reason": "support superseded or missing",
                        "expected_evidence": evidence_id,
                        "current_evidence": None if current is None else current.id,
                    }
                )
        return active, problems

    def _sync(self) -> None:
        # Rebuild on restart/new decisions; evidence-only changes update active context.
        decisions = self.decisions()
        supports = {k: v for d in decisions for k, v in d.supports.items()}
        # Keep all historical assumption IDs, even when the same field was superseded.
        all_ids = {v for d in decisions for v in d.supports.values()}
        active = set()
        for key in supports:
            entity, field = key.rsplit("|", 1)
            try:
                current = self.store.current(entity, field)
            except EvidenceConflict:
                continue
            if current is not None:
                active.add(current.id)
        for evidence_id in sorted(all_ids):
            node = f"evidence:{evidence_id}"
            if node not in self.reasoner.nodes:
                self.reasoner.add_assumption(node, active=evidence_id in active)
            else:
                self.reasoner.set_active(node, evidence_id in active)
        for decision in decisions:
            node = f"decision:{decision.id}"
            if node not in self.reasoner.nodes:
                self.reasoner.add_rule(
                    node, tuple(f"evidence:{v}" for v in decision.supports.values())
                )

    def impact(self) -> dict[str, Any]:
        """Return unsupported decisions plus conflicts across the entire scenario."""
        with self._lock:
            self._sync()
            affected = []
            for decision in self.decisions():
                _, problems = self._current_support(decision.supports)
                if problems:
                    affected.append(
                        {
                            "decision_id": decision.id,
                            "order_id": decision.order_id,
                            "status": decision.status,
                            "reasons": problems,
                        }
                    )
            conflicts = []
            state = self._state()
            for entity, fields in (
                *[
                    (f"supply:{i}", ("sku", "quantity", "available_day", "unit_cost"))
                    for i in state["supplies"]
                ],
                *[
                    (f"order:{i}", ("sku", "quantity", "due_day", "priority"))
                    for i in state["orders"]
                ],
                ("policy", ("config",)),
            ):
                for field in fields:
                    try:
                        self.store.current(entity, field)
                    except EvidenceConflict:
                        conflicts.append({"entity": entity, "field": field})
            return {
                "evidence_revision": self.store.revision,
                "conflicts": conflicts,
                "affected_decisions": affected,
                "affected_order_ids": sorted({a["order_id"] for a in affected}),
            }

    def explain(self, decision_id: str) -> dict[str, Any]:
        with self._lock:
            self._sync()
            decision = self._decision(decision_id)
            ids = set(decision.supports.values())
            return {
                "decision": decision.to_dict(),
                "logical_support": self.reasoner.explain(f"decision:{decision_id}"),
                "evidence": [e.to_dict() for e in self.store.list_evidence() if e.id in ids],
                "limitations": "Logical support tracks current evidence behind the original plan. Approval and commit recheck live capacity; source truth and global optimality are not proved.",
            }

    def _guard(self, decision: Decision) -> None:
        _, problems = self._current_support(decision.supports)
        if problems:
            raise StaleDecisionError(json.dumps(problems))
        supplies, orders, policy, _ = (
            self._snapshot()
        )  # Fail closed on any unresolved scenario conflict.
        supply_by_id, order_by_id = {s.id: s for s in supplies}, {o.id: o for o in orders}
        used, filled = self._reserved("supply_id"), self._reserved("order_id")
        order = order_by_id[decision.order_id]
        if filled.get(order.id, 0) + sum(a.quantity for a in decision.allocations) > order.quantity:
            raise StaleDecisionError("Order demand already reserved by another decision")
        for allocation in decision.allocations:
            supply = supply_by_id[allocation.supply_id]
            if used.get(supply.id, 0) + allocation.quantity > supply.quantity:
                raise StaleDecisionError("Supply already reserved by another decision")
            if supply.sku != order.sku and supply.sku not in policy.substitutions.get(
                order.sku, ()
            ):
                raise StaleDecisionError("Substitution is no longer permitted")
            if not policy.allow_late and supply.available_day > order.due_day:
                raise StaleDecisionError("Supply misses order due date")

    def approve(self, decision_id: str) -> Decision:
        with self._lock, self._transaction():
            decision = self._decision(decision_id)
            if decision.status == "committed":
                return decision
            self._guard(decision)
            if decision.status != "approved":
                self._db.execute(
                    "UPDATE ops_decisions SET status='approved' WHERE id=?", (decision_id,)
                )
                self._event("decision_approved", {"decision_id": decision_id})
        return self._decision(decision_id)

    def commit(self, decision_id: str) -> dict[str, Any]:
        """Atomically recheck support, reserve units, and record one local intent.

        Idempotent for a decision ID. An already committed decision is historical;
        later evidence changes are reported by impact(), not silently compensated.
        """
        with self._lock, self._transaction():
            decision = self._decision(decision_id)
            if decision.status == "committed":
                return {"decision_id": decision_id, "status": "committed", "idempotent": True}
            if decision.status != "approved":
                raise ValueError("Explicit approval is required before local commit")
            self._guard(decision)
            for allocation in decision.allocations:
                self._db.execute(
                    "INSERT INTO ops_reservations VALUES (?,?,?,?)",
                    (decision_id, decision.order_id, allocation.supply_id, allocation.quantity),
                )
            self._db.execute(
                "UPDATE ops_decisions SET status='committed' WHERE id=?", (decision_id,)
            )
            self._event(
                "execution_intent",
                {
                    "decision_id": decision_id,
                    "allocations": [asdict(a) for a in decision.allocations],
                    "external_execution": False,
                },
            )
        return {"decision_id": decision_id, "status": "committed", "idempotent": False}

    def events(self) -> list[dict[str, Any]]:
        return [
            {
                "sequence": r["sequence"],
                "kind": r["kind"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in self._db.execute("SELECT * FROM ops_events ORDER BY sequence")
        ]

    def close(self) -> None:
        self.store.close()
        self._db.close()

    def __enter__(self) -> OperationsPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
