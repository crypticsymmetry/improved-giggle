"""Exact support and update semantics for the bounded ATMS."""

import itertools
import json
import random
import unittest

from assumption_ops.atms import ATMS, EnvironmentOverflow


class ATMSTests(unittest.TestCase):
    def test_conjunctive_alternative_truth_table(self):
        model = ATMS()
        for name in "abc":
            model.add_assumption(name)
        model.add_rule("decision", ("a", "b"))
        model.add_rule("decision", ("c",))
        expected = (frozenset({"c"}), frozenset({"a", "b"}))
        self.assertEqual(model.supports("decision"), expected)
        for a, b, c in itertools.product((False, True), repeat=3):
            for name, active in zip("abc", (a, b, c)):
                model.set_active(name, active)
            self.assertEqual(model.is_supported("decision"), (a and b) or c)
            self.assertEqual(model.supports("decision"), expected)

    def test_antichain_removes_dominated_environment(self):
        model = ATMS()
        model.add_assumption("a")
        model.add_assumption("b")
        model.add_rule("x", ("a", "b"))
        model.add_rule("x", ("a",))
        self.assertEqual(model.supports("x"), (frozenset({"a"}),))
        model.add_rule("x", ())
        self.assertEqual(model.supports("x"), (frozenset(),))
        model.set_active("a", False)
        self.assertTrue(model.is_supported("x"))

    def test_unfounded_and_grounded_cycles(self):
        model = ATMS()
        model.add_rule("x", ("y",))
        model.add_rule("y", ("x",))
        self.assertFalse(model.is_supported("x"))
        model.add_assumption("seed")
        model.add_rule("x", ("seed",))
        self.assertEqual(model.supports("y"), (frozenset({"seed"}),))
        self.assertEqual(model.set_active("seed", False), {"seed", "x", "y"})
        model.add_nogood(("seed",))
        self.assertEqual(model.supports("x"), ())
        self.assertEqual(model.supports("y"), ())

    def test_nogood_filters_union_and_preserves_alternative(self):
        model = ATMS()
        for name in "abc":
            model.add_assumption(name)
        model.add_rule("x", ("a",))
        model.add_rule("y", ("b",))
        model.add_rule("z", ("x", "y"))
        model.add_rule("z", ("c",))
        model.add_rule("downstream", ("z",))
        changed = model.add_nogood(("a", "b"))
        self.assertEqual(changed, {"z", "downstream"})
        self.assertEqual(model.supports("z"), (frozenset({"c"}),))
        model.set_active("c", False)
        self.assertFalse(model.is_supported("z"))
        self.assertTrue(model.is_supported("x"))

    def test_empty_nogood_invalidates_facts(self):
        model = ATMS()
        model.add_rule("fact", ())
        model.add_rule("derived", ("fact",))
        model.add_nogood(())
        self.assertFalse(model.is_supported("fact"))
        model.add_rule("later", ())
        self.assertFalse(model.is_supported("later"))

    def test_context_changes_report_only_truth_changes(self):
        model = ATMS()
        model.add_assumption("a")
        model.add_assumption("b")
        model.add_rule("x", ("a",))
        model.add_rule("x", ("b",))
        model.add_rule("unrelated", ())
        labels = model._labels
        self.assertEqual(model.set_active("a", False), {"a"})
        self.assertIs(model._labels, labels)
        self.assertEqual(model.set_active("b", False), {"b", "x"})

    def test_incremental_update_preserves_unrelated_label_object(self):
        model = ATMS()
        model.add_assumption("a")
        model.add_rule("unrelated", ())
        unrelated = model._labels["unrelated"]
        model.add_rule("x", ("a",))
        self.assertIs(model._labels["unrelated"], unrelated)

    def test_overflow_is_explicit_and_atomic(self):
        model = ATMS(max_environments=1)
        model.add_assumption("a")
        model.add_assumption("b")
        model.add_rule("x", ("a",))
        model.add_rule("y", ("x",))
        revision = model.revision
        with self.assertRaises(EnvironmentOverflow):
            model.add_rule("x", ("b",))
        self.assertEqual(model.revision, revision)
        self.assertEqual(model.supports("y"), (frozenset({"a"}),))
        self.assertEqual(model.explain("x")["justifications"], [["a"]])
        self.assertEqual(model.set_active("b", False), {"b"})

    def test_conjunctive_cartesian_supports(self):
        model = ATMS()
        for name in "abcd":
            model.add_assumption(name)
        for name in "ab":
            model.add_rule("left", (name,))
        for name in "cd":
            model.add_rule("right", (name,))
        model.add_rule("joined", ("left", "right"))
        self.assertEqual(
            set(model.supports("joined")),
            {frozenset({left, right}) for left in "ab" for right in "cd"},
        )

    def test_existing_node_becomes_assumption_and_duplicate_noops(self):
        model = ATMS()
        model.add_rule("x", ("a",))
        model.add_assumption("a", active=False)
        self.assertEqual(model.supports("x"), (frozenset({"a"}),))
        self.assertFalse(model.is_supported("x"))
        revision = model.revision
        model.add_rule("x", ("a",))
        model.add_assumption("a", active=False)
        self.assertEqual(model.revision, revision)
        self.assertEqual(model.nodes, frozenset({"x", "a"}))

    def test_explanation_is_deterministic_json(self):
        model = ATMS()
        model.add_assumption("a", False)
        model.add_rule("x", ("a",))
        explanation = json.loads(json.dumps(model.explain("x")))
        self.assertEqual(explanation["supports"], [["a"]])
        self.assertEqual(explanation["active_supports"], [])
        self.assertEqual(explanation["inactive_assumptions"], ["a"])
        self.assertFalse(model.is_supported("missing"))

    def test_input_validation(self):
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ATMS(limit)
        model = ATMS()
        with self.assertRaises(ValueError):
            model.add_assumption("")
        with self.assertRaises(KeyError):
            model.set_active("unknown", False)
        with self.assertRaises(KeyError):
            model.add_nogood(("unknown",))
        with self.assertRaises(ValueError):
            model.add_rule("x", "abc")

    def test_random_cyclic_rules_match_forward_chaining_truth_tables(self):
        """Independent Boolean oracle checks symbolic labels on cyclic graphs."""
        rng = random.Random(42)
        assumptions = ("a", "b", "c")
        derived = ("w", "x", "y", "z")
        for _ in range(20):
            model = ATMS()
            for assumption in assumptions:
                model.add_assumption(assumption)
            rules = []
            for _ in range(12):
                conclusion = rng.choice(derived)
                premises = tuple(rng.sample(assumptions + derived, rng.randrange(3)))
                rules.append((conclusion, premises))
                model.add_rule(conclusion, premises)
            for context in itertools.product((False, True), repeat=len(assumptions)):
                truth = {name for name, active in zip(assumptions, context) if active}
                for name, active in zip(assumptions, context):
                    model.set_active(name, active)
                while True:
                    updated = truth | {
                        conclusion for conclusion, premises in rules if set(premises) <= truth
                    }
                    if updated == truth:
                        break
                    truth = updated
                for node in assumptions + derived:
                    self.assertEqual(model.is_supported(node), node in truth)


if __name__ == "__main__":
    unittest.main()
