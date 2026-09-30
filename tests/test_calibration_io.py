"""Strict config boundaries and CLI validation without solving."""

import csv
import json

import pytest

from assumption_ops.calibration_io import load_calibration_config, write_calibration_csv


def configuration():
    weights = dict(
        late_penalty=20, substitution_penalty=8, disruption_penalty=3, unfilled_penalty=500
    )
    return {
        "schema_version": 1,
        "scoring_weights": weights,
        "candidates": [dict(name="baseline", **weights)],
    }


@pytest.mark.parametrize("value", [-1, True, 1.5, "100"])
def test_evaluator_weights_are_validated_at_intake(tmp_path, value):
    data = configuration()
    data["scoring_weights"]["unfilled_penalty"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="config.json: line"):
        load_calibration_config(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(extra=1),
        lambda d: d.update(candidates=[]),
        lambda d: d["candidates"].append(dict(d["candidates"][0])),
        lambda d: d["scoring_weights"].update(allow_late=True),
        lambda d: d.update(constraints={"min_fill_rate": True}),
        lambda d: d.update(constraints={"max_late_rate": 1.1}),
        lambda d: d.update(constraints={"unknown": 0}),
    ],
)
def test_invalid_configuration_rejected(tmp_path, change):
    data = configuration()
    change(data)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_calibration_config(path)


def test_config_and_csv_preserve_fixed_weights_and_safe_labels(tmp_path):
    data = configuration()
    data["constraints"] = {"min_fill_rate": 0.9, "max_late_rate": 0.2}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    candidates, policy, bounds = load_calibration_config(path)
    assert policy.unfilled_penalty == 500
    assert candidates[0].unfilled_penalty == 500
    assert bounds == data["constraints"]
    report = {
        "selected_candidate": "=unsafe",
        "calibration": {
            "rankings": [
                {
                    "name": "=unsafe",
                    "eligible": True,
                    "mean_final_fixed_score_per_requested_unit": 2,
                    "penalties": data["scoring_weights"],
                    "episode_results": [],
                }
            ]
        },
    }
    output = tmp_path / "ranking.csv"
    write_calibration_csv(report, output)
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["name"] == "'=unsafe"
    assert row["penalty_unfilled_penalty"] == "500"
    assert row["selected"] == "True"
