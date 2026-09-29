"""Business intervention comparisons use one fixed task and scoring contract."""

from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

from assumption_ops.optimizer import Allocation, Order, Plan, Policy, Supply
from assumption_ops.scenarios import Scenario, apply_overrides, compare_scenarios


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.supplies = [Supply("shipment", "widget", 2, 5)]
        self.orders = [Order("customer", "widget", 2, 2)]
        self.policy = Policy(late_penalty=7, unfilled_penalty=100)

    def compare(self, scenarios, **kwargs):
        return compare_scenarios(self.supplies, self.orders, self.policy, scenarios, **kwargs)

    def test_rank_accounts_for_intervention_cost(self):
        comparison = self.compare(
            [
                Scenario("expedite", {"supply:shipment": {"available_day": 1}}, 20),
                Scenario("allow late", {"policy": {"config": {"allow_late": True}}}),
                Scenario("overpriced expedite", {"supply:shipment": {"available_day": 1}}, 250),
            ],
            evidence_revision=13,
            operations_revision=4,
        )
        self.assertEqual(comparison.baseline.total_score, 200)
        outcomes = {result.name: result for result in comparison.alternatives}
        self.assertEqual(outcomes["expedite"].total_score, 20)
        self.assertEqual(outcomes["expedite"].delta_vs_baseline, -180)
        self.assertEqual(outcomes["allow late"].total_score, 42)
        self.assertEqual(
            [row["name"] for row in comparison.rows()],
            ["expedite", "allow late", "baseline", "overpriced expedite"],
        )
        row = comparison.rows()[1]
        self.assertEqual(row["late_units"], 2)
        self.assertEqual(row["lateness"], 42)
        payload = json.loads(json.dumps(comparison.to_dict()))
        self.assertEqual(payload["evidence_revision"], 13)
        self.assertEqual(payload["operations_revision"], 4)
        self.assertEqual(payload["baseline"]["delta_vs_baseline"], 0)

    def test_comparison_partial_policy_keeps_fixed_weights(self):
        policy = Policy(
            late_penalty=13, substitution_penalty=8, disruption_penalty=9, unfilled_penalty=17
        )
        result = compare_scenarios(
            self.supplies,
            self.orders,
            policy,
            [Scenario("late", {"policy": {"config": {"allow_late": True}}})],
        )
        # Lateness costs 78, so the unchanged shortage penalty makes leaving demand
        # unfilled (34) preferable. A reset to default weights would change ranking.
        self.assertEqual(result.alternatives[0].evaluation.costs.shortage, 34)
        self.assertEqual(result.alternatives[0].plan.unfilled["customer"], 2)

    def test_generic_policy_replacement_preserves_existing_what_if_semantics(self):
        _, _, policy = apply_overrides(
            self.supplies, self.orders, self.policy, {"policy": {"config": {"allow_late": True}}}
        )
        self.assertEqual(policy, Policy(allow_late=True))
        self.assertEqual(self.policy.late_penalty, 7)

    def test_changed_scoring_or_demand_rejected(self):
        cases = [
            {"policy": {"config": {"unfilled_penalty": 0}}},
            {"policy": {"config": {"late_penalty": True}}},
            {"order:customer": {"quantity": 0}},
            {"order:customer": {"priority": 2}},
            {"order:customer": {"sku": "other"}},
            {"supply:shipment": {"sku": "other"}},
            {"supply:shipment": {"id": "other"}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.compare([Scenario("candidate", overrides)])
        # A full policy config is acceptable if every objective weight is unchanged.
        result = self.compare([Scenario("same", {"policy": {"config": asdict(self.policy)}})])
        self.assertEqual(result.alternatives[0].delta_vs_baseline, 0)

    def test_order_deadline_can_change(self):
        result = self.compare([Scenario("renegotiate", {"order:customer": {"due_day": 6}}, 8)])
        self.assertEqual(result.alternatives[0].total_score, 8)
        self.assertEqual(result.alternatives[0].evaluation.metrics["on_time_units"], 2)

    def test_reservations_subtracted_after_absolute_overrides(self):
        supplies = [Supply("shipment", "widget", 3, 0)]
        orders = [Order("customer", "widget", 6, 2)]
        result = compare_scenarios(
            supplies,
            orders,
            Policy(),
            [Scenario("buy stock", {"supply:shipment": {"quantity": 6}}, 30)],
            reserved_supply={"shipment": 2},
            reserved_order={"customer": 2},
        )
        self.assertEqual(result.baseline.evaluation.metrics["allocated_units"], 1)
        self.assertEqual(result.baseline.evaluation.metrics["unfilled_units"], 3)
        self.assertEqual(result.alternatives[0].evaluation.metrics["allocated_units"], 4)
        self.assertEqual(result.alternatives[0].total_score, 30)

    def test_invalid_reservations_and_overrides_fail_explicitly(self):
        for kwargs in (
            {"reserved_supply": {"unknown": 1}},
            {"reserved_order": {"customer": 3}},
            {"reserved_supply": {"shipment": -1}},
            {"reserved_supply": {"shipment": True}},
            {"reserved_order": {"customer": 0.5}},
            {"reserved_supply": {None: 0}},
            {"reserved_supply": [1]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compare([], **kwargs)
        with self.assertRaises(ValueError):
            self.compare(
                [Scenario("shrink stock", {"supply:shipment": {"quantity": 0}})],
                reserved_supply={"shipment": 1},
            )

    def test_all_candidates_validated_before_any_solver(self):
        with patch("assumption_ops.scenarios.optimize") as solver:
            with self.assertRaises(ValueError):
                self.compare(
                    [
                        Scenario("good", {"supply:shipment": {"available_day": 0}}),
                        Scenario("bad", {"supply:missing": {"quantity": 1}}),
                    ]
                )
            solver.assert_not_called()

    def test_snapshot_and_outcomes_are_independent(self):
        self.policy.substitutions["widget"] = ("alternate",)
        scenarios = [Scenario("candidate", {"supply:shipment": {"available_day": 1}})]
        previous = Plan((Allocation("customer", "shipment", 2),), {"customer": 0}, 0, "optimal")
        before = deepcopy((self.supplies, self.orders, self.policy, scenarios, previous))
        result = self.compare(scenarios, previous=previous)
        self.assertEqual((self.supplies, self.orders, self.policy, scenarios, previous), before)
        result.alternatives[0].plan.unfilled["customer"] = 99
        self.assertEqual(result.baseline.plan.unfilled["customer"], 2)
        self.assertEqual(previous.unfilled["customer"], 0)

    def test_override_provenance_is_original_independent_and_serializable(self):
        overrides = {
            "supply:shipment": {"available_day": 1},
            "policy": {"config": {"allow_late": True}},
        }
        expected = deepcopy(overrides)
        candidate = Scenario("expedite", overrides, 5)
        result = self.compare([candidate])
        candidate.overrides["supply:shipment"]["available_day"] = 99
        candidate.overrides["policy"]["config"]["allow_late"] = False
        outcome = result.alternatives[0]
        self.assertEqual(outcome.overrides, expected)
        self.assertEqual(result.baseline.overrides, {})
        serialized = result.to_dict()
        self.assertEqual(serialized["alternatives"][0]["overrides"], expected)
        self.assertEqual(serialized["baseline"]["overrides"], {})
        # Serialization never gives callers access to the outcome's provenance.
        serialized["alternatives"][0]["overrides"]["policy"]["config"]["allow_late"] = False
        self.assertEqual(outcome.overrides, expected)
        self.assertNotIn("late_penalty", outcome.overrides["policy"]["config"])
        self.assertEqual(json.loads(json.dumps(outcome.to_dict()))["overrides"], expected)

    def test_all_candidates_use_same_previous_plan(self):
        previous = Plan((Allocation("customer", "shipment", 2),), {"customer": 0}, 0, "optimal")
        result = self.compare(
            [
                Scenario("expedite", {"supply:shipment": {"available_day": 0}}, 3),
                Scenario("unchanged", {}),
            ],
            previous=previous,
        )
        self.assertEqual(result.baseline.evaluation.costs.disruption, 2)
        self.assertEqual(result.alternatives[0].evaluation.costs.disruption, 0)
        self.assertEqual(result.alternatives[1].evaluation.costs.disruption, 2)
        self.assertEqual(result.alternatives[1].delta_vs_baseline, 0)

    def test_ties_are_stable_by_name_not_submission_order(self):
        result = self.compare([Scenario("zeta", {}), Scenario("alpha", {})])
        self.assertEqual([row["name"] for row in result.rows()], ["alpha", "baseline", "zeta"])
        self.assertEqual([row["rank"] for row in result.rows()], [1, 2, 3])
        self.assertEqual([o.name for o in result.alternatives], ["zeta", "alpha"])

    def test_bad_names_costs_and_configuration(self):
        for candidates in (
            [Scenario("baseline", {})],
            [Scenario("same", {}), Scenario("same", {})],
            [Scenario(" ", {})],
            [Scenario("bad", {}, -1)],
            [Scenario("bad", {}, True)],
            [Scenario("bad", {}, 0.1)],
            [Scenario("bad", {"policy": {"config": {"nonexistent": 1}}})],
            [Scenario("bad", {"supply:shipment": {"quantity": -1}})],
            [Scenario("bad", {"supply:shipment": {"quantity": True}})],
            [Scenario("bad", {"policy": {"config": {"substitutions": {"widget": "other"}}}})],
        ):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.compare(candidates)

    def test_generic_helper_preserves_other_fields_and_validates_entities(self):
        supplies, orders, policy = apply_overrides(
            self.supplies,
            self.orders,
            self.policy,
            {"supply:shipment": {"available_day": 1}, "order:customer": {"priority": 3}},
        )
        self.assertEqual(supplies[0], Supply("shipment", "widget", 2, 1))
        self.assertEqual(orders[0], Order("customer", "widget", 2, 2, 3))
        self.assertEqual(policy, self.policy)
        self.assertIsNot(policy.substitutions, self.policy.substitutions)
        for overrides in (
            {"unknown": {}},
            {"supply:shipment": {"id": "new"}},
            {"order:customer": {"nonexistent": 1}},
            {"policy": {"other": {}}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                apply_overrides(self.supplies, self.orders, self.policy, overrides)


if __name__ == "__main__":
    unittest.main()
