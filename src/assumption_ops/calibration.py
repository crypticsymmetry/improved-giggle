"""Penalty-grid selection on calibration groups, followed by one held-out evaluation.

Candidate objectives cannot be compared across different weights. Selection uses
independent, fixed business weights, equal episode weighting, and only the FINAL
snapshot. All candidates share a case-specific initial reference optimized under
those fixed weights. The holdout is evaluated only after selection is frozen.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from platform import python_version
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING, Any

import scipy

from .evaluation import evaluate_plan
from .optimizer import Order, Plan, Policy, Supply, optimize, validate_inputs
from .scenarios import apply_overrides

if TYPE_CHECKING:
    from .pilot import PilotDataset, PilotEpisode

_WEIGHT_FIELDS = ("late_penalty", "substitution_penalty", "disruption_penalty", "unfilled_penalty")


@dataclass(frozen=True)
class PenaltyCandidate:
    name: str
    late_penalty: int
    substitution_penalty: int
    disruption_penalty: int
    unfilled_penalty: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Candidate name must be a nonempty string")
        for field in _WEIGHT_FIELDS:
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"Candidate {field} must be a nonnegative integer (booleans excluded)"
                )


def _weights(policy: Policy | PenaltyCandidate) -> dict[str, int]:
    return {field: getattr(policy, field) for field in _WEIGHT_FIELDS}


def _rate(value: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite number between 0 and 1")


def _optimal(
    supplies: list[Supply], orders: list[Order], policy: Policy, previous: Plan | None = None
) -> Plan:
    plan = optimize(supplies, orders, policy, previous)
    if plan.status != "optimal":
        raise RuntimeError(
            "Calibration requires proven-optimal solves; a time-limited incumbent cannot rank candidates"
        )
    return plan


def _context(episode: PilotEpisode, scoring_policy: Policy) -> dict[str, Any]:
    case = episode.case
    case.__post_init__()
    snapshots = [(list(case.supplies), list(case.orders))]
    supplies, orders = list(case.supplies), list(case.orders)
    for batch in case.batches:
        overrides: dict[str, dict[str, Any]] = {}
        for update in batch.updates:
            if update.entity == "policy":
                raise ValueError(
                    "Pilot calibration permits fact updates only; policy-config events are forbidden"
                )
            overrides.setdefault(update.entity, {})[update.field] = deepcopy(update.value)
        supplies, orders, _ = apply_overrides(supplies, orders, case.policy, overrides)
        snapshots.append((supplies, orders))
    fixed_policy = replace(case.policy, **_weights(scoring_policy))
    reference = _optimal(snapshots[0][0], snapshots[0][1], fixed_policy)
    evaluate_plan(snapshots[0][0], snapshots[0][1], fixed_policy, reference)
    return {
        "episode": episode,
        "snapshots": snapshots,
        "fixed_policy": fixed_policy,
        "reference": reference,
    }


def _run(
    context: dict[str, Any],
    candidate: PenaltyCandidate | None,
    min_fill_rate: float,
    max_late_rate: float,
) -> dict[str, Any]:
    episode = context["episode"]
    case = episode.case
    own_policy = (
        deepcopy(case.policy) if candidate is None else replace(case.policy, **_weights(candidate))
    )
    fixed_policy = context["fixed_policy"]
    steps: list[dict[str, Any]] = []
    for index, (supplies, orders) in enumerate(context["snapshots"]):
        previous = context["reference"] if index else None
        plan = _optimal(supplies, orders, own_policy, previous)
        own = evaluate_plan(supplies, orders, own_policy, plan, previous)
        metrics = own.metrics
        fixed_objective = (
            own.costs.acquisition
            + metrics["unit_days_late"] * fixed_policy.late_penalty
            + metrics["substitute_units"] * fixed_policy.substitution_penalty
            + metrics["priority_weighted_unfilled"] * fixed_policy.unfilled_penalty
            + (
                metrics["changed_units"] * fixed_policy.disruption_penalty
                if previous is not None
                else 0
            )
        )
        # Rescoring changes only the reported objective, never the allocation or
        # eligibility, then passes the independent accounting verifier again.
        rescored = replace(plan, objective=float(fixed_objective))
        fixed = evaluate_plan(supplies, orders, fixed_policy, rescored, previous)
        steps.append(
            {
                "step": index,
                "event_id": case.batches[index - 1].event_id if index else None,
                "own_plan": plan.to_dict(),
                "own_evaluation": own.to_dict(),
                "fixed_evaluation": fixed.to_dict(),
                "status": plan.status,
                "common_reference_used": previous is not None,
            }
        )
    last = steps[-1]
    final_metrics = last["fixed_evaluation"]["metrics"]
    requested = final_metrics["requested_units"]
    fill_rate = final_metrics["allocated_units"] / requested if requested else 1.0
    late_rate = final_metrics["late_units"] / requested if requested else 0.0
    fixed_score = last["fixed_evaluation"]["costs"]["total"]
    return {
        "episode_id": episode.episode_id,
        "group_id": episode.group_id,
        "case_name": case.name,
        "synthetic": case.synthetic,
        "initial_reference": context["reference"].to_dict(),
        "own_weights": _weights(own_policy),
        "steps": steps,
        "final": {
            "own_objective": last["own_plan"]["objective"],
            "fixed_score": fixed_score,
            "normalized_fixed_score": fixed_score / max(1, requested),
            "requested_units": requested,
            "fill_rate": fill_rate,
            "late_rate": late_rate,
            "constraints_passed": fill_rate >= min_fill_rate and late_rate <= max_late_rate,
            "own_evaluation": last["own_evaluation"],
            "fixed_evaluation": last["fixed_evaluation"],
            "status": last["status"],
        },
    }


def calibrate(
    dataset: PilotDataset,
    candidates: tuple[PenaltyCandidate, ...],
    scoring_policy: Policy,
    *,
    min_fill_rate: float = 0.0,
    max_late_rate: float = 1.0,
) -> dict[str, Any]:
    """Select penalty weights without inspecting held-out allocation results.

    Rates are calculated per final episode against requested units. Empty demand
    has fill rate 1, late rate 0, and normalized-score denominator 1. Every
    calibration episode must satisfy constraints. Holdout failures are reported
    without reselection. Eligibility always comes from each case, never from the
    scoring policy or candidates. No approvals or execution intents are created.
    """
    from .pilot import PilotDataset

    if not isinstance(dataset, PilotDataset):
        raise ValueError("dataset must be a PilotDataset")
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("candidates must be a nonempty tuple of PenaltyCandidate records")
    if any(not isinstance(candidate, PenaltyCandidate) for candidate in candidates):
        raise ValueError("candidates must contain PenaltyCandidate records")
    for candidate in candidates:
        candidate.__post_init__()
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    if not isinstance(scoring_policy, Policy):
        raise ValueError("scoring_policy must be a Policy")
    validate_inputs([], [], scoring_policy)
    _rate(min_fill_rate, "min_fill_rate")
    _rate(max_late_rate, "max_late_rate")
    # Frozen DTOs contain mutable policy/update dictionaries; snapshot all source
    # values so neither replay nor the report can modify the caller's dataset.
    dataset, candidates, scoring_policy = deepcopy((dataset, candidates, scoring_policy))
    dataset.__post_init__()
    start = perf_counter()
    calibration_episodes = tuple(e for e in dataset.episodes if e.split == "calibration")
    holdout_episodes = tuple(e for e in dataset.episodes if e.split == "holdout")
    contexts = [_context(episode, scoring_policy) for episode in calibration_episodes]
    rows = []
    for candidate in sorted(candidates, key=lambda candidate: candidate.name):
        results = [_run(context, candidate, min_fill_rate, max_late_rate) for context in contexts]
        rows.append(
            {
                "name": candidate.name,
                "penalties": _weights(candidate),
                "mean_final_fixed_score_per_requested_unit": mean(
                    r["final"]["normalized_fixed_score"] for r in results
                ),
                "eligible": all(r["final"]["constraints_passed"] for r in results),
                "episode_results": results,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["eligible"],
            row["mean_final_fixed_score_per_requested_unit"],
            row["name"],
        )
    )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError(
            "No candidate satisfies the final per-episode calibration fill/late constraints"
        )
    selected_name = eligible[0]["name"]
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    baseline_calibration = [
        _run(context, None, min_fill_rate, max_late_rate) for context in contexts
    ]
    # Selection is irrevocably complete before any holdout optimization occurs.
    holdout_results = []
    for episode in holdout_episodes:
        context = _context(episode, scoring_policy)
        holdout_results.append(
            {
                "episode_id": episode.episode_id,
                "group_id": episode.group_id,
                "baseline": _run(context, None, min_fill_rate, max_late_rate),
                "selected": _run(context, selected, min_fill_rate, max_late_rate),
            }
        )
    baseline_holdout_mean = mean(
        r["baseline"]["final"]["normalized_fixed_score"] for r in holdout_results
    )
    selected_holdout_mean = mean(
        r["selected"]["final"]["normalized_fixed_score"] for r in holdout_results
    )
    return {
        "dataset_name": dataset.name,
        "selected_candidate": selected_name,
        "selected_penalties": _weights(selected),
        "scoring_weights": _weights(scoring_policy),
        "eligibility": "Each case's substitutions and allow_late are preserved; scoring_policy supplies weights only.",
        "selection_rule": "Lowest arithmetic mean FINAL fixed-score / max(1, requested units) over calibration episodes; eligible ties by candidate name.",
        "constraints": {
            "min_fill_rate": min_fill_rate,
            "max_late_rate": max_late_rate,
            "denominator": "final requested units",
            "empty_fill_rate": 1.0,
            "empty_late_rate": 0.0,
        },
        "calibration": {
            "episode_ids": [e.episode_id for e in calibration_episodes],
            "rankings": rows,
            "baseline": {
                "mean_final_fixed_score_per_requested_unit": mean(
                    r["final"]["normalized_fixed_score"] for r in baseline_calibration
                ),
                "episode_results": baseline_calibration,
            },
        },
        "holdout": {
            "episode_ids": [e.episode_id for e in holdout_episodes],
            "episodes": holdout_results,
            "baseline_mean_final_fixed_score_per_requested_unit": baseline_holdout_mean,
            "selected_mean_final_fixed_score_per_requested_unit": selected_holdout_mean,
            "delta_selected_minus_baseline": selected_holdout_mean - baseline_holdout_mean,
        },
        "external_dispatch": False,
        "environment": {"python": python_version(), "scipy": scipy.__version__},
        "runtime_seconds": perf_counter() - start,
        "limitations": [
            "Fixed scores are weighted penalty units, not monetary savings or measured customer outcomes.",
            "Every snapshot solve must be proven optimal; grid selection is optimal only among supplied eligible candidates.",
            "Only calibration episodes rank candidates; the holdout compares original-case baseline and the selected candidate once.",
            "Fact snapshots are replayed as pure validated replacements; no evidence ledger, approvals, reservations, or external dispatch is produced.",
            "All updated candidate snapshots use the same initial reference generated under fixed scoring weights; initial snapshots have no disruption reference.",
            "Each episode receives equal weight after final requested-unit normalization; repeated snapshots are never summed as new demand or cost.",
            "Holdout constraint failures are reported without changing selection.",
        ],
    }
