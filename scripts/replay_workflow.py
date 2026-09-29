"""Run a mock confirmed-update workflow with duplicate delivery and restart.

Everything stays local. The confirmations below are synthetic fixtures, not
actual supplier messages. Run: python scripts/replay_workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_ops import (
    EvidenceUpdate,
    OperationsPipeline,
    Scenario,
    StaleDecisionError,
    load_json_scenario,
)

ROOT = Path(__file__).resolve().parents[1]


def run():
    data = load_json_scenario(ROOT / "examples/sample_scenario.json")
    with TemporaryDirectory() as folder:
        db_path = str(Path(folder) / "workflow.sqlite")
        with OperationsPipeline(db_path) as ops:
            ops.seed(data.supplies, data.orders, data.policy)
            original = ops.plan()
            ops.approve_plan(original.plan_id)
            delay_snapshot = ops.revisions()
            delay = [
                EvidenceUpdate(
                    "supply:shipment",
                    "available_day",
                    8,
                    "demo: reviewed supplier delay confirmation",
                )
            ]
            receipt = ops.apply_updates(
                "supplier-delay-001",
                delay,
                **{
                    "expected_evidence_revision": delay_snapshot["evidence_revision"],
                    "expected_operations_revision": delay_snapshot["operations_revision"],
                },
            )
            try:
                ops.commit_plan(original.plan_id)
            except StaleDecisionError:
                pass
            else:
                raise AssertionError("An outdated plan must not commit")
            comparison = ops.compare_scenarios(
                [
                    Scenario("expedite", {"supply:shipment": {"available_day": 3}}, 40),
                ],
                previous=original.plan,
            )
            confirmed = [
                EvidenceUpdate(
                    "supply:shipment",
                    "available_day",
                    3,
                    "demo: reviewed expedited arrival confirmation",
                ),
                EvidenceUpdate(
                    "supply:shipment",
                    "unit_cost",
                    1,
                    "demo: reviewed purchase unit-cost confirmation",
                ),
            ]
            applied = ops.apply_updates(
                "supplier-expedite-002",
                confirmed,
                expected_evidence_revision=comparison.evidence_revision,
                expected_operations_revision=comparison.operations_revision,
            )
            replayed = ops.apply_updates(
                "supplier-expedite-002",
                confirmed,
                expected_evidence_revision=comparison.evidence_revision,
                expected_operations_revision=comparison.operations_revision,
            )
            assert replayed.replayed and replayed.evidence_ids == applied.evidence_ids
            current = ops.plan(previous=original.plan)
            ops.approve_plan(current.plan_id)
            commit = ops.commit_plan(current.plan_id)
            before_replay = ops.events()
            assert ops.commit_plan(current.plan_id)["idempotent"]
            assert ops.events() == before_replay
            assert sum(ops._reserved("order_id").values()) == 10
            result = {
                "delay_receipt": receipt.to_dict(),
                "comparison": comparison.rows(),
                "confirmed_receipt": applied.to_dict(),
                "duplicate_receipt": replayed.to_dict(),
                "commit": commit,
                "intent_count": len([e for e in ops.events() if e["kind"] == "execution_intent"]),
            }
        with OperationsPipeline(db_path) as restored:
            restored_replay = restored.apply_updates(
                "supplier-delay-001",
                delay,
                expected_evidence_revision=delay_snapshot["evidence_revision"],
                expected_operations_revision=delay_snapshot["operations_revision"],
            )
            assert restored_replay.replayed
            assert restored.store.current("supply:shipment", "available_day").value == 3
            assert restored.commit_plan(current.plan_id)["idempotent"]
            result["restart_verified"] = True
            result["external_dispatch"] = False
        return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
