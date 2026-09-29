"""Pure, reservation-aware comparison of business interventions.

Every candidate is solved from the same frozen snapshot and previous plan.
Scores retain the baseline's objective weights; intervention costs use the same
business penalty units. Nothing is accepted, persisted, approved, or executed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from .evaluation import Evaluation, evaluate_plan
from .optimizer import Order, Plan, Policy, Supply, optimize, validate_inputs


@dataclass(frozen=True)
class Scenario:
    """Named absolute field replacements and an additional intervention cost."""

    name: str
    overrides: dict[str, dict[str, Any]]
    action_cost: int = 0


@dataclass(frozen=True)
class ScenarioOutcome:
    """One independently evaluated proposal; a negative delta improves score."""

    name: str
    plan: Plan
    evaluation: Evaluation
    action_cost: int
    total_score: float
    delta_vs_baseline: float
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan": self.plan.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "action_cost": self.action_cost,
            "total_score": self.total_score,
            "delta_vs_baseline": self.delta_vs_baseline,
            "overrides": deepcopy(self.overrides),
        }


@dataclass(frozen=True)
class ScenarioComparison:
    """A result envelope tied to optional source revision identifiers."""

    baseline: ScenarioOutcome
    alternatives: tuple[ScenarioOutcome, ...]
    evidence_revision: int | None = None
    operations_revision: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "alternatives": [outcome.to_dict() for outcome in self.alternatives],
            "evidence_revision": self.evidence_revision,
            "operations_revision": self.operations_revision,
        }

    def rows(self) -> list[dict[str, Any]]:
        """Return all options ranked by score then name, including the baseline."""
        return [
            {
                "rank": rank,
                "name": outcome.name,
                "objective": outcome.plan.objective,
                "action_cost": outcome.action_cost,
                "total_score": outcome.total_score,
                "delta_vs_baseline": outcome.delta_vs_baseline,
                "status": outcome.plan.status,
                "acquisition": outcome.evaluation.costs.acquisition,
                "lateness": outcome.evaluation.costs.lateness,
                "substitution": outcome.evaluation.costs.substitution,
                "shortage": outcome.evaluation.costs.shortage,
                "disruption": outcome.evaluation.costs.disruption,
                "allocated_units": outcome.evaluation.metrics["allocated_units"],
                "unfilled_units": outcome.evaluation.metrics["unfilled_units"],
                "on_time_units": outcome.evaluation.metrics["on_time_units"],
                "late_units": outcome.evaluation.metrics["late_units"],
                "substitute_units": outcome.evaluation.metrics["substitute_units"],
                "changed_units": outcome.evaluation.metrics["changed_units"],
            }
            for rank, outcome in enumerate(
                sorted((self.baseline, *self.alternatives), key=lambda o: (o.total_score, o.name)),
                start=1,
            )
        ]


def apply_overrides(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    policy: Policy,
    overrides: dict[str, dict[str, Any]],
) -> tuple[list[Supply], list[Order], Policy]:
    """Apply validated absolute replacements to independent copies.

    Record fields may be replaced except IDs. ``policy: {config: {...}}``
    replaces the complete policy: omitted fields receive Policy defaults. This
    preserves the existing single-scenario what-if contract. Comparison callers
    merge policy eligibility changes with their baseline before using this helper.
    """
    validate_inputs(supplies, orders, policy)
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be an entity-to-field mapping")
    replacements = deepcopy(overrides)
    for entity, changes in replacements.items():
        if not isinstance(entity, str) or not isinstance(changes, dict):
            raise ValueError("Overrides must be entity names mapped to field mappings")
    new_supplies, new_orders = [], []
    for prefix, records, result in (
        ("supply", supplies, new_supplies),
        ("order", orders, new_orders),
    ):
        for record in records:
            changes = replacements.pop(f"{prefix}:{record.id}", {})
            allowed = set(asdict(record)) - {"id"}
            if set(changes) - allowed:
                raise ValueError(f"Unknown or immutable fields for {prefix}:{record.id}")
            result.append(type(record)(**{**asdict(record), **changes}))
    changes = replacements.pop("policy", {})
    new_policy = deepcopy(policy)
    if changes:
        if set(changes) != {"config"} or not isinstance(changes["config"], dict):
            raise ValueError("Policy overrides must contain only a config mapping")
        config = changes["config"]
        if set(config) - set(asdict(policy)):
            raise ValueError("Unknown policy configuration fields")
        if "substitutions" in config:
            substitutions = config["substitutions"]
            if not isinstance(substitutions, dict):
                raise ValueError("substitutions must be a mapping of SKU names to sequences")
            normalized = {}
            for sku, values in substitutions.items():
                if not isinstance(values, (list, tuple)):
                    raise ValueError("Each substitution entry must be a list or tuple of SKUs")
                normalized[sku] = tuple(values)
            config["substitutions"] = normalized
        new_policy = Policy(**config)
    if replacements:
        raise ValueError(f"Unknown scenario entities: {sorted(replacements)}")
    validate_inputs(new_supplies, new_orders, new_policy)
    return new_supplies, new_orders, new_policy


def _remaining(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    reserved_supply: Mapping[str, int],
    reserved_order: Mapping[str, int],
) -> tuple[list[Supply], list[Order]]:
    """Deduct frozen reservations after applying absolute scenario quantities."""
    result = []
    for label, records, reservations in (
        ("supply", supplies, reserved_supply),
        ("order", orders, reserved_order),
    ):
        if any(not isinstance(key, str) or not key.strip() for key in reservations):
            raise ValueError("Reservation IDs must be nonempty strings")
        unknown = set(reservations) - {record.id for record in records}
        if unknown:
            raise ValueError(f"Unknown reserved {label} IDs: {sorted(unknown)}")
        remaining = []
        for record in records:
            used = reservations.get(record.id, 0)
            if type(used) is not int or used < 0:
                raise ValueError(f"Reserved {label} quantity must be a nonnegative integer")
            if used > record.quantity:
                raise ValueError(f"Scenario {label} quantity is below committed reservations")
            remaining.append(replace(record, quantity=record.quantity - used))
        result.append(remaining)
    return result[0], result[1]


def _comparison_overrides(scenario: Scenario, policy: Policy) -> dict[str, dict[str, Any]]:
    if not isinstance(scenario.name, str) or not scenario.name.strip():
        raise ValueError("Scenario names must be nonempty strings")
    if type(scenario.action_cost) is not int or scenario.action_cost < 0:
        raise ValueError("action_cost must be a nonnegative integer in objective units")
    if not isinstance(scenario.overrides, dict):
        raise ValueError("Scenario overrides must be entity-to-field mappings")
    overrides = deepcopy(scenario.overrides)
    for entity, fields in overrides.items():
        if not isinstance(entity, str) or not isinstance(fields, dict):
            raise ValueError("Scenario overrides must map entity names to field mappings")
        if entity.startswith("order:") and set(fields) - {"due_day"}:
            raise ValueError("Comparisons may change order due_day only; demand/scoring are fixed")
        if entity.startswith("supply:") and set(fields) - {
            "quantity",
            "available_day",
            "unit_cost",
        }:
            raise ValueError(
                "Comparisons may change supply quantity, available_day, unit_cost only"
            )
        if entity == "policy" and fields:
            if set(fields) != {"config"} or not isinstance(fields["config"], dict):
                raise ValueError("Policy overrides must contain only a config mapping")
            config = fields["config"]
            baseline = asdict(policy)
            if set(config) - set(baseline):
                raise ValueError("Unknown policy configuration fields")
            for key, value in config.items():
                if key not in {"allow_late", "substitutions"} and (
                    type(value) is not int or value != baseline[key]
                ):
                    raise ValueError("Comparison objective penalty weights must match the baseline")
            fields["config"] = {**baseline, **config}
    return overrides


def compare_scenarios(
    supplies: Sequence[Supply],
    orders: Sequence[Order],
    policy: Policy,
    scenarios: Sequence[Scenario],
    previous: Plan | None = None,
    *,
    reserved_supply: Mapping[str, int] | None = None,
    reserved_order: Mapping[str, int] | None = None,
    evidence_revision: int | None = None,
    operations_revision: int | None = None,
) -> ScenarioComparison:
    """Compare interventions without changing evidence or operational state.

    Order quantities, priorities, identities, SKUs, and objective penalty weights
    are held fixed so candidates cannot improve their rank by changing the task.
    Comparison policy config is merged with the baseline, unlike the generic
    ``apply_overrides`` helper's complete policy replacement semantics.
    Supply quantity overrides are absolute totals, then reservations are deducted.
    Invalid candidates abort explicitly rather than being silently omitted.
    """
    supplies, orders, policy, previous, scenarios, reserved_supply, reserved_order = deepcopy(
        (
            list(supplies),
            list(orders),
            policy,
            previous,
            tuple(scenarios),
            {} if reserved_supply is None else reserved_supply,
            {} if reserved_order is None else reserved_order,
        )
    )
    validate_inputs(supplies, orders, policy)
    if not isinstance(reserved_supply, Mapping) or not isinstance(reserved_order, Mapping):
        raise ValueError("Reservations must map IDs to nonnegative integer quantities")
    names = {"baseline"}
    candidates = []
    for scenario in scenarios:
        if not isinstance(scenario, Scenario):
            raise ValueError("Candidates must be Scenario instances")
        overrides = _comparison_overrides(scenario, policy)
        if scenario.name in names:
            raise ValueError(f"Duplicate or reserved scenario name: {scenario.name!r}")
        names.add(scenario.name)
        # Validate all candidate inputs before running any solver.
        candidate_supplies, candidate_orders, candidate_policy = apply_overrides(
            supplies, orders, policy, overrides
        )
        candidate_supplies, candidate_orders = _remaining(
            candidate_supplies, candidate_orders, reserved_supply, reserved_order
        )
        candidates.append((scenario, candidate_supplies, candidate_orders, candidate_policy))
    base_supplies, base_orders = _remaining(supplies, orders, reserved_supply, reserved_order)

    def outcome(
        name: str,
        candidate_supplies: Sequence[Supply],
        candidate_orders: Sequence[Order],
        candidate_policy: Policy,
        action_cost: int,
        base_score: float,
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> ScenarioOutcome:
        plan = optimize(candidate_supplies, candidate_orders, candidate_policy, deepcopy(previous))
        evaluation = evaluate_plan(
            candidate_supplies, candidate_orders, candidate_policy, plan, deepcopy(previous)
        )
        score = plan.objective + action_cost
        return ScenarioOutcome(
            name,
            plan,
            evaluation,
            action_cost,
            score,
            score - base_score,
            {} if overrides is None else deepcopy(overrides),
        )

    baseline = outcome("baseline", base_supplies, base_orders, deepcopy(policy), 0, 0)
    baseline = replace(baseline, delta_vs_baseline=0.0)
    alternatives = tuple(
        outcome(
            scenario.name,
            candidate_supplies,
            candidate_orders,
            candidate_policy,
            scenario.action_cost,
            baseline.total_score,
            scenario.overrides,
        )
        for scenario, candidate_supplies, candidate_orders, candidate_policy in candidates
    )
    return ScenarioComparison(baseline, alternatives, evidence_revision, operations_revision)
