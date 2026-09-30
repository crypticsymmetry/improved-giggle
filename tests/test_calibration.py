from copy import deepcopy
from dataclasses import replace
import json

import pytest

from assumption_ops.calibration import PenaltyCandidate, calibrate
from assumption_ops.events import EvidenceUpdate
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.pilot import PilotDataset, PilotEpisode
from assumption_ops.replay import ReplayBatch, ReplayCase


def make_case(name, *, quantity=1, cost=7, batches=(), allow_late=False):
    return ReplayCase(
        name,
        (Supply("s", "A", quantity, 0, cost),),
        (Order("o", "A", quantity, 0),),
        Policy(allow_late=allow_late, unfilled_penalty=100),
        batches,
        synthetic=True,
    )


def dataset(holdout=None):
    return PilotDataset(
        "pilot",
        (
            PilotEpisode("cal", "supplier-cal", "calibration", make_case("cal")),
            PilotEpisode(
                "hold", "supplier-hold", "holdout", holdout or make_case("hold", quantity=2, cost=9)
            ),
        ),
    )


CANDIDATES = (PenaltyCandidate("service", 10, 2, 3, 100), PenaltyCandidate("zero", 0, 0, 0, 0))
SCORING = Policy(
    late_penalty=20, substitution_penalty=4, disruption_penalty=30, unfilled_penalty=200
)


def test_zero_raw_objective_cannot_cheat_fixed_score():
    report = calibrate(dataset(), CANDIDATES, SCORING)
    assert report["selected_candidate"] == "service"
    rows = {row["name"]: row for row in report["calibration"]["rankings"]}
    assert rows["zero"]["episode_results"][0]["final"]["own_objective"] == 0
    assert rows["zero"]["mean_final_fixed_score_per_requested_unit"] == 200
    assert rows["service"]["mean_final_fixed_score_per_requested_unit"] == 7
    assert report["holdout"]["selected_mean_final_fixed_score_per_requested_unit"] == 9
    assert report["external_dispatch"] is False
    json.dumps(report, allow_nan=False)


def test_holdout_changes_cannot_change_selection_or_calibration_scores():
    first = calibrate(dataset(), CANDIDATES, SCORING)
    second = calibrate(
        dataset(make_case("changed-holdout", quantity=4, cost=500)), CANDIDATES, SCORING
    )
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["calibration"] == second["calibration"]
    assert first["holdout"] != second["holdout"]


def test_constraints_are_calibration_only_and_holdout_does_not_reselect():
    holdout = make_case("expensive-holdout", quantity=2, cost=500)
    report = calibrate(dataset(holdout), CANDIDATES, SCORING, min_fill_rate=1.0)
    rows = {row["name"]: row for row in report["calibration"]["rankings"]}
    assert not rows["zero"]["eligible"]
    assert rows["service"]["eligible"]
    assert report["selected_candidate"] == "service"
    assert not report["holdout"]["episodes"][0]["selected"]["final"]["constraints_passed"]
    with pytest.raises(ValueError, match="No candidate"):
        calibrate(dataset(), (CANDIDATES[1],), SCORING, min_fill_rate=1.0)


def test_fixed_reference_fairness_and_chronological_cost_rescoring():
    batch = ReplayBatch(
        "increase", (EvidenceUpdate("supply:s", "unit_cost", 11, "supplier quote"),)
    )
    case = make_case("cal", cost=7, batches=(batch,))
    pilot = PilotDataset(
        "reference",
        (
            PilotEpisode("cal", "calgroup", "calibration", case),
            PilotEpisode("hold", "holdgroup", "holdout", make_case("hold", quantity=2)),
        ),
    )
    report = calibrate(pilot, CANDIDATES, SCORING)
    runs = [row["episode_results"][0] for row in report["calibration"]["rankings"]]
    reference = runs[0]["initial_reference"]
    assert all(run["initial_reference"] == reference for run in runs)
    assert reference["allocations"] == [{"order_id": "o", "supply_id": "s", "quantity": 1}]
    zero = next(run for run in runs if run["own_weights"]["unfilled_penalty"] == 0)
    assert zero["steps"][0]["common_reference_used"] is False
    assert zero["steps"][1]["common_reference_used"] is True
    assert zero["steps"][1]["fixed_evaluation"]["costs"]["disruption"] == 30
    assert zero["final"]["fixed_score"] == 230
    assert zero["final"]["requested_units"] == 1
    service = next(run for run in runs if run["own_weights"]["unfilled_penalty"] == 100)
    assert service["final"]["fixed_score"] == 11
    assert service["final"]["fixed_evaluation"]["costs"]["disruption"] == 0


def test_scoring_policy_does_not_override_case_eligibility():
    late = ReplayBatch("late", (EvidenceUpdate("supply:s", "available_day", 1, "shipment delay"),))
    pilot = PilotDataset(
        "eligibility",
        (
            PilotEpisode(
                "cal",
                "calgroup",
                "calibration",
                make_case("cal", cost=0, batches=(late,), allow_late=True),
            ),
            PilotEpisode(
                "hold",
                "holdgroup",
                "holdout",
                make_case("hold", quantity=2, cost=0, batches=(late,), allow_late=True),
            ),
        ),
    )
    result = calibrate(pilot, (CANDIDATES[0],), SCORING)
    final = result["calibration"]["rankings"][0]["episode_results"][0]["final"]
    assert final["fixed_evaluation"]["metrics"]["late_units"] == 1
    assert final["fill_rate"] == 1
    assert final["late_rate"] == 1
    with pytest.raises(ValueError, match="No candidate"):
        calibrate(pilot, (CANDIDATES[0],), SCORING, max_late_rate=0)


def test_equal_episode_weighting_and_final_snapshot_only():
    batch = ReplayBatch("price", (EvidenceUpdate("supply:s", "unit_cost", 2, "quote"),))
    pilot = PilotDataset(
        "weighting",
        (
            PilotEpisode(
                "small", "small", "calibration", make_case("small", cost=10, batches=(batch,))
            ),
            PilotEpisode("large", "large", "calibration", make_case("large", quantity=10, cost=8)),
            PilotEpisode("hold", "hold", "holdout", make_case("hold", quantity=2, cost=9)),
        ),
    )
    report = calibrate(pilot, (CANDIDATES[0],), SCORING)
    assert report["calibration"]["rankings"][0]["mean_final_fixed_score_per_requested_unit"] == 5
    assert [
        run["final"]["requested_units"]
        for run in report["calibration"]["rankings"][0]["episode_results"]
    ] == [1, 10]


def test_empty_demand_denominator_and_rates():
    empty = ReplayCase("empty", (), (), Policy(), (), synthetic=True)
    other = ReplayCase(
        "different-empty", (Supply("s", "A", 1, 0),), (), Policy(), (), synthetic=True
    )
    pilot = PilotDataset(
        "empty",
        (
            PilotEpisode("cal", "cal", "calibration", empty),
            PilotEpisode("hold", "hold", "holdout", other),
        ),
    )
    final = calibrate(pilot, (CANDIDATES[0],), SCORING, min_fill_rate=1.0, max_late_rate=0)[
        "calibration"
    ]["rankings"][0]["episode_results"][0]["final"]
    assert final["normalized_fixed_score"] == 0
    assert final["fill_rate"] == 1
    assert final["late_rate"] == 0


def test_source_inputs_immutable():
    pilot = dataset()
    before = deepcopy(pilot)
    scoring_before = deepcopy(SCORING)
    report = calibrate(pilot, CANDIDATES, SCORING)
    report["calibration"]["rankings"][0]["episode_results"][0]["initial_reference"]["allocations"][
        0
    ]["quantity"] = 999
    assert pilot == before
    assert SCORING == scoring_before


def test_ties_break_by_name_not_candidate_input_order():
    a, z = PenaltyCandidate("a", 10, 2, 3, 100), PenaltyCandidate("z", 10, 2, 3, 100)
    assert calibrate(dataset(), (z, a), SCORING)["selected_candidate"] == "a"


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " "},
        {"late_penalty": True},
        {"substitution_penalty": -1},
        {"disruption_penalty": 1.0},
        {"unfilled_penalty": False},
    ],
)
def test_candidate_strict_validation(changes):
    with pytest.raises(ValueError):
        replace(CANDIDATES[0], **changes)


@pytest.mark.parametrize("candidates", [(), [], (CANDIDATES[0], CANDIDATES[0]), ("bad",)])
def test_invalid_candidate_grid(candidates):
    with pytest.raises(ValueError):
        calibrate(dataset(), candidates, SCORING)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_fill_rate": True},
        {"min_fill_rate": -1},
        {"max_late_rate": 2},
        {"max_late_rate": float("nan")},
        {"max_late_rate": float("inf")},
    ],
)
def test_invalid_constraints(kwargs):
    with pytest.raises(ValueError):
        calibrate(dataset(), CANDIDATES, SCORING, **kwargs)


def test_nonoptimal_incumbent_rejected_for_calibration(monkeypatch):
    import assumption_ops.calibration as module

    original = module.optimize
    monkeypatch.setattr(
        module,
        "optimize",
        lambda *args, **kwargs: replace(original(*args, **kwargs), status="feasible_limit"),
    )
    with pytest.raises(RuntimeError, match="proven-optimal"):
        calibrate(dataset(), CANDIDATES, SCORING)


def test_all_candidate_calibration_runs_precede_holdout(monkeypatch):
    import assumption_ops.calibration as module

    original = module._run
    sequence = []

    def tracked(context, candidate, *args):
        sequence.append(
            (context["episode"].split, candidate.name if candidate else "original-baseline")
        )
        return original(context, candidate, *args)

    monkeypatch.setattr(module, "_run", tracked)
    calibrate(dataset(), CANDIDATES, SCORING)
    first_holdout = next(i for i, (split, _) in enumerate(sequence) if split == "holdout")
    assert all(split == "calibration" for split, _ in sequence[:first_holdout])
    assert sequence[first_holdout:] == [("holdout", "original-baseline"), ("holdout", "service")]


def test_uniform_penalty_rescaling_cannot_win_by_smaller_raw_objective():
    scaled_down = PenaltyCandidate("scaled-down", 1, 1, 1, 1)
    scaled_up = PenaltyCandidate("scaled-up", 100, 100, 100, 100)
    report = calibrate(dataset(), (scaled_down, scaled_up), SCORING)
    by_name = {row["name"]: row for row in report["calibration"]["rankings"]}
    low = by_name["scaled-down"]["episode_results"][0]["final"]
    high = by_name["scaled-up"]["episode_results"][0]["final"]
    assert low["own_objective"] == 1 < high["own_objective"] == 7
    assert low["fixed_score"] == 200 > high["fixed_score"] == 7
    assert report["selected_candidate"] == "scaled-up"
