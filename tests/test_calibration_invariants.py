"""Independent holdout, scoring, and provenance checks for pilot calibration."""

from copy import deepcopy
from dataclasses import replace

import pytest

from assumption_ops.calibration import PenaltyCandidate, calibrate
from assumption_ops.events import EvidenceUpdate
from assumption_ops.optimizer import Order, Policy, Supply
from assumption_ops.pilot import PilotDataset, PilotEpisode, dataset_summary
from assumption_ops.replay import ReplayBatch, ReplayCase


FIXED = Policy(late_penalty=1, substitution_penalty=2, disruption_penalty=1, unfilled_penalty=100)
CANDIDATES = (
    PenaltyCandidate("a_zero", 0, 0, 0, 0),
    PenaltyCandidate("b_high", 1, 2, 1, 10_000),
)


def episode(identifier, split, cost, quantity=1, *, synthetic=True, prices=None):
    prices = (cost,) if prices is None else prices
    case = ReplayCase(
        f"episode {identifier}",
        (Supply("stock", "A", quantity, 0, cost),),
        (Order("customer", "A", quantity, 1),),
        FIXED,
        tuple(
            ReplayBatch(
                f"price-{index}",
                (EvidenceUpdate("supply:stock", "unit_cost", price, f"source {index}"),),
            )
            for index, price in enumerate(prices)
        ),
        synthetic,
        17 if synthetic else None,
    )
    return PilotEpisode(identifier, f"group {identifier}", split, case)


def dataset(calibration, holdout):
    return PilotDataset("independent scoring check", tuple(calibration) + tuple(holdout))


def by_name(report):
    return {row["name"]: row for row in report["calibration"]["rankings"]}


def brute_force_single_source_final(quantity, initial_cost, final_cost, candidate):
    """Enumerate units and costs without calling optimizer or its evaluators."""
    reference = min(
        range(quantity + 1),
        key=lambda filled: filled * initial_cost + (quantity - filled) * FIXED.unfilled_penalty,
    )

    def own_score(filled):
        return (
            filled * final_cost
            + (quantity - filled) * candidate.unfilled_penalty
            + abs(filled - reference) * candidate.disruption_penalty
        )

    filled = min(range(quantity + 1), key=own_score)
    fixed_score = (
        filled * final_cost
        + (quantity - filled) * FIXED.unfilled_penalty
        + abs(filled - reference) * FIXED.disruption_penalty
    )
    return reference, filled, own_score(filled), fixed_score


def test_zero_raw_objective_cannot_cheat_fixed_score_or_common_reference():
    data = dataset([episode("cheap", "calibration", 1)], [episode("expensive", "holdout", 200)])
    report = calibrate(data, CANDIDATES, FIXED)
    rows = by_name(report)
    assert report["selected_candidate"] == "b_high"
    common = rows["b_high"]["episode_results"][0]["initial_reference"]
    assert common["unfilled"]["customer"] == 0
    for candidate in CANDIDATES:
        result = rows[candidate.name]["episode_results"][0]
        reference, filled, raw, score = brute_force_single_source_final(1, 1, 1, candidate)
        assert reference == 1
        assert result["initial_reference"] == common
        assert result["final"]["own_objective"] == raw
        assert result["final"]["fixed_score"] == score
        assert result["final"]["fixed_evaluation"]["metrics"]["allocated_units"] == filled
        assert result["steps"][-1]["common_reference_used"]
    zero = rows["a_zero"]["episode_results"][0]
    assert zero["final"]["own_objective"] == 0
    assert zero["final"]["fixed_score"] == 101
    assert zero["final"]["fixed_evaluation"]["costs"]["disruption"] == 1


def test_opposing_holdout_outcomes_cannot_change_calibration_ranking():
    calibration = [episode("cheap", "calibration", 1)]
    small = dataset(calibration, [episode("holdout-200", "holdout", 200)])
    large = dataset(
        calibration, [episode(f"holdout-{cost}", "holdout", cost) for cost in range(200, 205)]
    )
    first = calibrate(small, CANDIDATES, FIXED)
    second = calibrate(large, CANDIDATES, FIXED)
    assert first["calibration"] == second["calibration"]
    assert first["selected_candidate"] == second["selected_candidate"] == "b_high"
    for result in second["holdout"]["episodes"]:
        cost = int(result["episode_id"].split("-")[-1])
        _, _, _, predicted = brute_force_single_source_final(1, cost, cost, CANDIDATES[1])
        assert result["selected"]["final"]["fixed_score"] == predicted
        # Holdout favors leaving demand unfilled under fixed weights, yet the
        # calibration winner remains selected rather than silently re-ranking.
        assert (
            result["selected"]["final"]["fixed_score"] > result["baseline"]["final"]["fixed_score"]
        )
    assert first["holdout"]["delta_selected_minus_baseline"] > 0


def test_episode_weighting_is_equal_and_only_final_snapshots_supply_scores():
    data = dataset(
        [
            episode("small", "calibration", 1, 1),
            episode("large", "calibration", 3, 4, prices=(9, 7, 3)),
        ],
        [episode("holdout", "holdout", 200)],
    )
    report = calibrate(data, CANDIDATES, FIXED)
    high = by_name(report)["b_high"]
    assert high["mean_final_fixed_score_per_requested_unit"] == 2
    results = {result["episode_id"]: result for result in high["episode_results"]}
    assert results["small"]["final"]["fixed_score"] == 1
    assert results["large"]["final"]["fixed_score"] == 12
    assert results["large"]["final"]["requested_units"] == 4
    assert len(results["large"]["steps"]) == 4
    assert results["large"]["final"]["normalized_fixed_score"] == 3
    # Pooling all five units would give 2.6; summing intermediate snapshots would
    # also differ. The declared selection rule uses two equally weighted episodes.
    assert high["mean_final_fixed_score_per_requested_unit"] != 13 / 5


def test_fixed_evaluator_weights_preserve_case_late_and_substitution_eligibility():
    case = ReplayCase(
        "allowed late substitute",
        (Supply("substitute", "B", 1, 3, 1),),
        (Order("customer", "A", 1, 1),),
        replace(FIXED, allow_late=True, substitutions={"A": ("B",)}),
        (
            ReplayBatch(
                "confirmed", (EvidenceUpdate("supply:substitute", "unit_cost", 1, "supplier"),)
            ),
        ),
        True,
        11,
    )
    data = dataset(
        [PilotEpisode("eligible", "eligible group", "calibration", case)],
        [episode("holdout", "holdout", 200)],
    )
    report = calibrate(data, (CANDIDATES[1],), FIXED)
    result = report["calibration"]["rankings"][0]["episode_results"][0]
    assert not FIXED.allow_late
    assert FIXED.substitutions == {}
    assert result["initial_reference"]["unfilled"] == {"customer": 0}
    assert result["final"]["fixed_evaluation"]["metrics"]["late_units"] == 1
    assert result["final"]["fixed_evaluation"]["metrics"]["substitute_units"] == 1
    assert result["final"]["fixed_score"] == 5  # acquisition1 + late2 + substitute2


def test_renamed_trace_and_reordered_semantic_metadata_cannot_cross_splits():
    base = episode("source", "calibration", 1).case
    base = replace(base, policy=replace(base.policy, substitutions={"A": ("B", "C")}))
    changed = replace(
        base,
        name="renamed holdout",
        synthetic=False,
        seed=None,
        policy=replace(base.policy, substitutions={"A": ("C", "B", "B")}),
        batches=(
            ReplayBatch(
                "renamed event", (replace(base.batches[0].updates[0], source="renamed source"),)
            ),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate business trace"):
        dataset(
            [PilotEpisode("cal", "cal group", "calibration", base)],
            [PilotEpisode("test", "test group", "holdout", changed)],
        )


def test_provenance_labels_and_inputs_are_preserved_without_truth_claims():
    data = dataset(
        [episode("cal", "calibration", 1, synthetic=True)],
        [episode("test", "holdout", 200, synthetic=False)],
    )
    before = deepcopy([(e.episode_id, e.case.to_dict()) for e in data.episodes])
    candidates_before = deepcopy(CANDIDATES)
    fixed_before = deepcopy(FIXED)
    report = calibrate(data, CANDIDATES, FIXED)
    assert [(e.episode_id, e.case.to_dict()) for e in data.episodes] == before
    assert CANDIDATES == candidates_before
    assert FIXED == fixed_before
    assert report["calibration"]["rankings"][0]["episode_results"][0]["synthetic"] is True
    assert report["holdout"]["episodes"][0]["selected"]["synthetic"] is False
    summary = dataset_summary(data)
    assert summary["splits"]["calibration"]["synthetic_episodes"] == 1
    assert summary["splits"]["holdout"]["caller_labeled_nonsynthetic_episodes"] == 1
    assert "caller claims" in summary["provenance_note"]
    report["calibration"]["rankings"][0]["episode_results"][0]["initial_reference"]["unfilled"][
        "customer"
    ] = 99
    assert [(e.episode_id, e.case.to_dict()) for e in data.episodes] == before


def test_policy_config_events_cannot_modify_scoring_contract():
    base = episode("source", "calibration", 1).case
    changed = replace(
        base,
        batches=(
            ReplayBatch(
                "policy",
                (EvidenceUpdate("policy", "config", {"unfilled_penalty": 0}, "policy update"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="policy config"):
        PilotEpisode("policy episode", "policy group", "calibration", changed)
