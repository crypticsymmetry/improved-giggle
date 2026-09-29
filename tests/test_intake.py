import json
from pathlib import Path

import pytest

from assumption_ops.intake import load_csv_scenario, load_json_scenario, load_policy_json
from assumption_ops.optimizer import Policy


def _files(
    tmp_path,
    supplies="id,sku,quantity,available_day\ns,A,2,0\n",
    orders="id,sku,quantity,due_day\no,A,1,3\n",
):
    s, o = tmp_path / "supplies.csv", tmp_path / "orders.csv"
    s.write_text(supplies, encoding="utf-8")
    o.write_text(orders, encoding="utf-8")
    return s, o


def test_csv_defaults_bom_quoted_commas_and_whitespace(tmp_path):
    supplies, orders = _files(
        tmp_path, '\ufeffid,sku,quantity,available_day\n"supplier,west", A , 02 , 0 \n'
    )
    result = load_csv_scenario(supplies, orders)
    assert result.supplies[0].id == "supplier,west"
    assert result.supplies[0].sku == "A"
    assert result.supplies[0].quantity == 2
    assert result.supplies[0].unit_cost == 0
    assert result.orders[0].priority == 1
    assert isinstance(result.supplies, tuple)
    assert isinstance(result.orders, tuple)


def test_empty_data_is_valid_but_header_required(tmp_path):
    paths = _files(tmp_path, "id,sku,quantity,available_day\n", "id,sku,quantity,due_day\n")
    assert load_csv_scenario(*paths).supplies == ()
    paths[0].write_text("")
    with pytest.raises(ValueError, match="line 1.*header"):
        load_csv_scenario(*paths)


@pytest.mark.parametrize(
    "header",
    [
        "id,id,quantity,available_day",
        "id,sku,quantity",
        "id,sku,quantity,available_day,mystery",
        "",
    ],
)
def test_reject_invalid_headers(tmp_path, header):
    paths = _files(tmp_path, header + "\ns,A,2,0\n")
    with pytest.raises(ValueError, match="supplies.csv: line 1"):
        load_csv_scenario(*paths)


@pytest.mark.parametrize(
    "row",
    [
        "s,A,1.2,0",
        "s,A,true,0",
        "s,A,1e2,0",
        "s,A,-1,0",
        "s,A,+1,0",
        "s,A,١,0",
        "s,A,,0",
        ",A,1,0",
        "s, ,1,0",
        "s,A,1",
        "s,A,1,0,extra",
        "",
    ],
)
def test_invalid_supply_rows_have_source_line(tmp_path, row):
    paths = _files(tmp_path, "id,sku,quantity,available_day\n" + row + "\n")
    with pytest.raises(ValueError, match="supplies.csv: line 2"):
        load_csv_scenario(*paths)


def test_blank_optional_value_does_not_use_default(tmp_path):
    paths = _files(tmp_path, "id,sku,quantity,available_day,unit_cost\ns,A,2,0,\n")
    with pytest.raises(ValueError, match="unit_cost must not be blank"):
        load_csv_scenario(*paths)


def test_priority_zero_is_invalid(tmp_path):
    paths = _files(tmp_path, orders="id,sku,quantity,due_day,priority\no,A,1,3,0\n")
    with pytest.raises(ValueError, match="orders.csv: line 2.*priority"):
        load_csv_scenario(*paths)


@pytest.mark.parametrize("kind", ["supply", "order"])
def test_duplicate_ids_show_second_row_line(tmp_path, kind):
    supplies = "id,sku,quantity,available_day\ns,A,2,0\ns,B,3,1\n"
    orders = "id,sku,quantity,due_day\no,A,1,3\no,B,1,4\n"
    paths = _files(
        tmp_path,
        supplies=supplies if kind == "supply" else "id,sku,quantity,available_day\ns,A,2,0\n",
        orders=orders if kind == "order" else "id,sku,quantity,due_day\no,A,1,3\n",
    )
    with pytest.raises(ValueError, match=f"line 3.*duplicate {kind} id"):
        load_csv_scenario(*paths)


def test_unclosed_csv_quote_reports_line(tmp_path):
    paths = _files(tmp_path, 'id,sku,quantity,available_day\n"broken,A,2,0\n')
    with pytest.raises(ValueError, match="supplies.csv: line 2"):
        load_csv_scenario(*paths)


def test_policy_validation_without_solver(tmp_path):
    paths = _files(tmp_path)
    with pytest.raises(ValueError, match="allow_late"):
        load_csv_scenario(*paths, policy=Policy(allow_late="yes"))


def test_fixtures_match_json_sample():
    examples = Path(__file__).resolve().parents[1] / "examples"
    from_json = load_json_scenario(examples / "sample_scenario.json")
    from_csv = load_csv_scenario(
        examples / "supplies.csv", examples / "orders.csv", from_json.policy
    )
    assert from_csv == from_json
    assert from_json.to_dict()["policy"]["substitutions"] == {"A": ["B"]}


def test_json_roundtrip(tmp_path):
    scenario = load_csv_scenario(*_files(tmp_path), policy=Policy(substitutions={"A": ("B",)}))
    target = tmp_path / "scenario.json"
    target.write_text(json.dumps(scenario.to_dict()), encoding="utf-8")
    assert load_json_scenario(target) == scenario


@pytest.mark.parametrize(
    "data",
    [
        {"supplies": [], "orders": [], "extra": 1},
        {"supplies": [], "orders": [], "policy": {"extra": 1}},
        {"supplies": [], "orders": [], "policy": {"substitutions": {"A": "B"}}},
        {"supplies": [{"id": "s", "sku": "A", "quantity": True, "available_day": 0}], "orders": []},
        {"supplies": [{"id": "s", "sku": "A", "quantity": "1", "available_day": 0}], "orders": []},
        {"supplies": [{"id": "s", "sku": "A", "quantity": 1.0, "available_day": 0}], "orders": []},
        {
            "supplies": [{"id": "s", "sku": "A", "quantity": 1, "available_day": 0, "extra": 1}],
            "orders": [],
        },
        {"supplies": {}, "orders": []},
        {"supplies": []},
    ],
)
def test_json_rejects_bad_schema_and_implicit_coercion(tmp_path, data):
    target = tmp_path / "scenario.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="scenario.json: line"):
        load_json_scenario(target)


@pytest.mark.parametrize(
    "text",
    [
        '{"supplies":[],"supplies":[],"orders":[]}',
        '{"supplies":[],"orders":[],"policy":{"unfilled_penalty":NaN}}',
        "{\n broken json\n}",
    ],
)
def test_json_duplicate_keys_nonfinite_and_parse_errors(tmp_path, text):
    target = tmp_path / "scenario.json"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="scenario.json: line"):
        load_json_scenario(target)


def test_policy_json_bom_and_defaults(tmp_path):
    target = tmp_path / "policy.json"
    target.write_text('\ufeff{"substitutions":{"A":["B"]},"allow_late":true}', encoding="utf-8")
    result = load_policy_json(target)
    assert result == Policy(substitutions={"A": ("B",)}, allow_late=True)


@pytest.mark.parametrize(
    "text",
    [
        '{"allow_late":false,"allow_late":true}',
        '{"late_penalty":NaN}',
        '{"late_penalty":Infinity}',
        '{"late_penalty":true}',
        '{"late_penalty":1.0}',
        '{"late_penalty":"1"}',
        '{"late_penalty":-1}',
        '{"allow_late":"yes"}',
        '{"unknown":1}',
        '{"substitutions":{"A":"B"}}',
        '{"substitutions":{"A":[true]}}',
        '{"substitutions":{"A":[" "]}}',
        "[]",
        "null",
        "{broken}",
    ],
)
def test_policy_json_rejects_ambiguous_or_invalid_policy(tmp_path, text):
    target = tmp_path / "policy.json"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="policy.json: line"):
        load_policy_json(target)
