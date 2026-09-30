from copy import deepcopy
import csv
import json

import pytest

from assumption_ops.replay import run_replay
from assumption_ops.replay_fixtures import synthetic_cases
from assumption_ops.replay_io import (
    load_replay_case,
    replay_case_to_dict,
    save_replay_case,
    write_replay_report_csv,
)


def fixture():
    return replay_case_to_dict(synthetic_cases(order_count=4, seed=17)[0])


def test_deterministic_generator_has_explicit_provenance_and_four_regimes():
    first = synthetic_cases(order_count=7, seed=31)
    assert first == synthetic_cases(order_count=7, seed=31)
    assert first != synthetic_cases(order_count=7, seed=32)
    assert {case.name for case in first} == {
        "supplier_delay",
        "stock_loss",
        "demand_surge",
        "cost_shock",
    }
    for case in first:
        assert case.synthetic is True
        assert case.seed == 31
        assert len(case.orders) == 7
        assert {order.sku for order in case.orders} == {"A", "B", "C"}
        assert len(case.batches) == 3
        assert all(batch.updates for batch in case.batches)
        assert all(
            "SYNTHETIC" in update.source for batch in case.batches for update in batch.updates
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order_count": -1},
        {"order_count": 0},
        {"order_count": True},
        {"order_count": 2.5},
        {"seed": -1},
        {"seed": True},
        {"seed": 1.5},
    ],
)
def test_generator_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        synthetic_cases(**kwargs)


def test_all_synthetic_cases_roundtrip_without_mutation(tmp_path):
    for case in synthetic_cases(order_count=5, seed=41):
        original = deepcopy(case)
        path = tmp_path / f"{case.name}.json"
        save_replay_case(case, path)
        assert load_replay_case(path) == case
        assert case == original
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        assert data["synthetic"] is True


def test_json_bom_and_optional_seed(tmp_path):
    data = fixture()
    del data["seed"]
    path = tmp_path / "case.json"
    path.write_text("\ufeff" + json.dumps(data), encoding="utf-8")
    assert load_replay_case(path).seed is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(synthetic="yes"),
        lambda d: d.pop("synthetic"),
        lambda d: d.update(seed=True),
        lambda d: d.update(seed=-1),
        lambda d: d.update(name=" "),
        lambda d: d.update(extra=1),
        lambda d: d["initial"].update(extra=1),
        lambda d: d["initial"]["supplies"][0].update(quantity=True),
        lambda d: d["initial"]["supplies"][0].update(quantity="1"),
        lambda d: d["initial"]["policy"].update(substitutions={"A": "B"}),
        lambda d: d["batches"][0].update(extra=1),
        lambda d: d["batches"][0].update(updates=[]),
        lambda d: d["batches"][0]["updates"][0].update(field="unknown"),
        lambda d: d["batches"][0]["updates"][0].update(entity="supply:missing"),
        lambda d: d["batches"][0]["updates"][0].update(value=-1),
        lambda d: d["batches"][0]["updates"][0].update(source=""),
        lambda d: d["batches"][0]["updates"].append(deepcopy(d["batches"][0]["updates"][0])),
        lambda d: d["batches"][1].update(event_id=d["batches"][0]["event_id"]),
    ],
)
def test_case_schema_rejects_invalid_inputs(tmp_path, change):
    data = fixture()
    change(data)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.json: line"):
        load_replay_case(path)


@pytest.mark.parametrize(
    "text",
    ['{"name":"a","name":"b"}', '{"unrelated":NaN}', '{"unrelated":Infinity}', "{\n broken\n}"],
)
def test_json_duplicate_keys_and_nonfinite_values_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json: line"):
        load_replay_case(path)


def test_csv_reports_one_row_per_strategy_snapshot_and_safe_names(tmp_path):
    case = synthetic_cases(order_count=4, seed=17)[0]
    report = run_replay(case)
    report["case_name"] = "=EXECUTE()"
    path = tmp_path / "report.csv"
    write_replay_report_csv(report, path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # Initial snapshot + three batches, two alternatives each.
    assert {row["strategy"] for row in rows} == {"optimized", "greedy"}
    assert all(row["case_name"] == "'=EXECUTE()" for row in rows)
    assert all("costs_total" in row for row in rows)
    assert all("elapsed_seconds" in row for row in rows)
    assert [row["step"] for row in rows] == ["0", "0", "1", "1", "2", "2", "3", "3"]


def test_empty_report_rejected_before_creating_file(tmp_path):
    path = tmp_path / "report.csv"
    with pytest.raises(ValueError, match="at least one snapshot"):
        write_replay_report_csv({"steps": []}, path)
    assert not path.exists()


def test_export_revalidates_mutated_nested_values_before_json_coercion(tmp_path):
    from assumption_ops import EvidenceUpdate, ReplayBatch, ReplayCase

    original = synthetic_cases(order_count=4, seed=17)[0]
    config = {"substitutions": {"A": ["A-alt"]}}
    case = ReplayCase(
        "mutated",
        original.supplies,
        original.orders,
        original.policy,
        (ReplayBatch("policy", (EvidenceUpdate("policy", "config", config, "confirmed"),)),),
        synthetic=True,
    )
    config["substitutions"][123] = ["A-alt"]
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="string object keys"):
        save_replay_case(case, path)
    assert not path.exists()
