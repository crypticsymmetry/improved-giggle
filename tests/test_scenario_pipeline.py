"""Integration of frozen comparisons, safe exports, and the real intake CLI."""

import csv
import json
from pathlib import Path

import pytest

from assumption_ops import (
    EvidenceConflict,
    OperationsPipeline,
    Order,
    Policy,
    Scenario,
    Supply,
    load_csv_scenario,
    write_comparison_csv,
)
from assumption_ops.cli import main

ROOT = Path(__file__).resolve().parents[1]


def seeded():
    p = OperationsPipeline()
    p.seed(
        [
            Supply("stock", "A", 3, 0, 1),
            Supply("shipment", "A", 7, 2, 1),
            Supply("substitute", "B", 4, 0, 3),
        ],
        [Order("urgent", "A", 5, 3, 3), Order("standard", "A", 5, 4)],
        Policy(substitutions={"A": ("B",)}),
    )
    return p


def test_comparison_does_not_change_evidence_decisions_or_history():
    with seeded() as p:
        baseline = p.plan()
        evidence, events, decisions = p.store.events(), p.events(), p.decisions()
        comparison = p.compare_scenarios(
            [Scenario("delay", {"supply:shipment": {"available_day": 8}})], previous=baseline.plan
        )
        assert p.store.events() == evidence
        assert p.events() == events
        assert p.decisions() == decisions
        assert comparison.evidence_revision == p.store.revision
        assert comparison.operations_revision == events[-1]["sequence"]
        delay = comparison.alternatives[0]
        expected = p.what_if({"supply:shipment": {"available_day": 8}}, previous=baseline.plan)
        assert delay.plan.objective == expected.objective
        assert delay.plan.unfilled == expected.unfilled
        assert delay.evaluation.costs.total == delay.plan.objective


def test_operational_candidates_recover_service_with_comparable_scores():
    with seeded() as p:
        previous = p.plan().plan
        change = p.propose("supply:shipment", "available_day", 8, "supplier confirmation")
        p.accept(change)
        p.resolve("supply:shipment", "available_day", change)
        c = p.compare_scenarios(
            [
                Scenario("expedite", {"supply:shipment": {"available_day": 3}}, 40),
                Scenario("more substitute", {"supply:substitute": {"quantity": 7}}, 20),
                Scenario("allow late", {"policy": {"config": {"allow_late": True}}}),
            ],
            previous=previous,
        )
        assert c.baseline.evaluation.metrics["unfilled_units"] == 3
        assert all(o.evaluation.metrics["unfilled_units"] == 0 for o in c.alternatives)
        assert c.rows()[0]["name"] == "expedite"
        assert next(o for o in c.alternatives if o.name == "expedite").total_score == 50
        assert all(
            o.total_score == o.evaluation.costs.total + o.action_cost for o in c.alternatives
        )


def test_comparison_respects_local_reservations():
    with seeded() as p:
        d = p.plan().decisions[0]
        p.approve(d.id)
        p.commit(d.id)
        c = p.compare_scenarios([])
        assert c.baseline.evaluation.metrics["requested_units"] == 5
        assert all(a.order_id != d.order_id for a in c.baseline.plan.allocations)
        before = p._reserved("supply_id")
        with pytest.raises(ValueError):
            p.compare_scenarios(
                [
                    Scenario(
                        "remove reserved stock",
                        {f"supply:{d.allocations[0].supply_id}": {"quantity": 0}},
                    )
                ]
            )
        assert p._reserved("supply_id") == before


def test_comparison_fails_closed_on_conflict():
    with seeded() as p:
        e = p.propose("supply:shipment", "available_day", 8, "contradiction")
        p.accept(e)
        with pytest.raises(EvidenceConflict):
            p.compare_scenarios([])


def test_csv_report_preserves_numeric_values_and_escapes_formulas(tmp_path):
    with seeded() as p:
        c = p.compare_scenarios([Scenario("=1+1", {}, 1)])
        output = tmp_path / "report.csv"
        write_comparison_csv(c, output)
        with output.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert any(row["name"] == "'=1+1" for row in rows)
        assert all(float(row["total_score"]) >= 0 for row in rows)


def test_csv_cli_matches_json_cli(capsys):
    main([str(ROOT / "examples/sample_scenario.json")])
    json_result = json.loads(capsys.readouterr().out)
    main(
        [
            "--supplies-csv",
            str(ROOT / "examples/supplies.csv"),
            "--orders-csv",
            str(ROOT / "examples/orders.csv"),
            "--policy-json",
            str(ROOT / "examples/policy.json"),
        ]
    )
    csv_result = json.loads(capsys.readouterr().out)
    assert csv_result["baseline"]["plan"] == json_result["baseline"]["plan"]


def test_cli_delays_then_compares_and_exports(tmp_path, capsys):
    report = tmp_path / "actions.csv"
    main(
        [
            str(ROOT / "examples/sample_scenario.json"),
            "--delay-supply",
            "shipment",
            "--available-day",
            "8",
            "--compare",
            str(ROOT / "examples/interventions.json"),
            "--report-csv",
            str(report),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["comparison_rows"][0]["name"] == "expedite shipment"
    assert report.exists()
    assert sum(result["revised"]["plan"]["unfilled"].values()) == 3


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--supplies-csv", "missing"],
        ["anything.json", "--supplies-csv", "s.csv", "--orders-csv", "o.csv"],
        ["anything.json", "--report-csv", "out.csv"],
    ],
)
def test_cli_requires_unambiguous_inputs(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_csv_default_policy_has_no_implicit_substitution():
    data = load_csv_scenario(ROOT / "examples/supplies.csv", ROOT / "examples/orders.csv")
    assert data.policy.substitutions == {}


@pytest.mark.parametrize(
    "bad_policy",
    [
        {"substitutions": {"A": "B"}},
        {"substitutions": []},
    ],
)
def test_pipeline_policy_rejects_implicit_substitution_coercion(bad_policy):
    with seeded() as p:
        with pytest.raises(ValueError):
            p.propose("policy", "config", bad_policy, "bad policy")
        with pytest.raises(ValueError):
            p.what_if({"policy": {"config": bad_policy}})


def test_csv_cli_rejects_invalid_policy_file(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text('{"substitutions":{"A":"B"}}')
    with pytest.raises(ValueError):
        main(
            [
                "--supplies-csv",
                str(ROOT / "examples/supplies.csv"),
                "--orders-csv",
                str(ROOT / "examples/orders.csv"),
                "--policy-json",
                str(policy),
            ]
        )
