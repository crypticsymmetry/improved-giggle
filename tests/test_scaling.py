"""Check paired timing records and exclusion of unproven quality comparisons."""

from dataclasses import replace

import pytest

from assumption_ops.scaling import benchmark_solvers


def test_scaling_records_pair_inputs_and_alternate_timing_order():
    report = benchmark_solvers((8,), (19, 23), repeats=2)
    assert report["synthetic"] is True
    assert report["external_dispatch"] is False
    assert len(report["runs"]) == 8
    for index in range(4):
        pair = [row for row in report["runs"] if row["pair"] == index]
        assert pair[0]["input_fingerprint"] == pair[1]["input_fingerprint"]
        assert pair[0]["objective"] == pair[1]["objective"]
        assert {row["solver"] for row in pair} == {"milp", "lp"}
        assert pair[0]["solver"] != report["runs"][2 * ((index + 1) % 4)]["solver"]
    assert report["summary"][0]["verified_pairs"] == 4
    assert report["summary"][0]["unverified_pairs"] == 0


def test_feasible_incumbents_are_excluded_from_timing_quality_summary(monkeypatch):
    import assumption_ops.scaling as module

    real = module.optimize

    def limited(*args, **kwargs):
        plan = real(*args, **kwargs)
        return replace(plan, status="feasible_limit") if kwargs["solver"] == "lp" else plan

    monkeypatch.setattr(module, "optimize", limited)
    report = benchmark_solvers((3,), (7,), repeats=1)
    assert report["summary"][0]["verified_pairs"] == 0
    assert report["summary"][0]["unverified_pairs"] == 1
    assert report["summary"][0]["ratio_milp_to_lp_medians"] is None
    assert not any(row["proven_equal_quality"] for row in report["runs"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order_counts": ()},
        {"order_counts": (True,)},
        {"order_counts": (0,)},
        {"seeds": (-1,)},
        {"seeds": (3, 3)},
        {"repeats": True},
        {"repeats": 0},
    ],
)
def test_invalid_measurement_grid_rejected(kwargs):
    with pytest.raises(ValueError):
        benchmark_solvers(**kwargs)
