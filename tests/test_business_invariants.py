"""Adversarial as-of visibility and physical-stock accounting checks."""

from dataclasses import replace

import pytest

from assumption_ops.business import (
    BusinessPilot,
    CatalogItem,
    OpenOrderObservation,
    PhysicalMovement,
    StockSnapshot,
    compare_business_pilot,
    compile_business_pilot,
)
from assumption_ops.evaluation import evaluate_plan
from assumption_ops.optimizer import Policy, optimize


OPEN = "2026-01-01T00:00:00Z"
DAY2 = "2026-01-02T00:00:00Z"
DAY3 = "2026-01-03T00:00:00Z"


def pilot(**changes):
    values = dict(
        name="adversarial pilot",
        group_id="independent-episode",
        synthetic=True,
        catalog=(CatalogItem("A", 0),),
        stock_snapshots=(StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),),
        movements=(),
        order_observations=(
            OpenOrderObservation("customer", OPEN, OPEN, "A", 4, "2026-01-01", 1, "orders"),
        ),
        policy=Policy(allow_late=True, late_penalty=2, unfilled_penalty=100),
    )
    values.update(changes)
    return BusinessPilot(**values)


def movement(identifier, kind, quantity, event_time=DAY2, observed_at=DAY2):
    return PhysicalMovement(identifier, event_time, observed_at, "A", kind, quantity, "warehouse")


def quantity(snapshot):
    return sum(supply.quantity for supply in snapshot.supplies)


def test_later_receipts_and_order_updates_cannot_change_opening_inputs():
    original = compile_business_pilot(pilot())[0]
    augmented = compile_business_pilot(
        pilot(
            movements=(movement("future receipt", "receipt", 100),),
            order_observations=pilot().order_observations
            + (OpenOrderObservation("customer", DAY3, DAY3, "A", 50, "2026-01-04", 3, "orders"),),
        )
    )[0]
    assert augmented.supplies == original.supplies
    assert augmented.orders == original.orders
    assert augmented.policy == original.policy


def test_late_observed_receipt_is_not_known_at_its_physical_event_time():
    snapshots = compile_business_pilot(
        pilot(
            movements=(movement("late receipt", "receipt", 5, DAY2, DAY3),),
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("day two", DAY2, {"A": 10}, "warehouse"),
                StockSnapshot("day three", DAY3, {"A": 15}, "warehouse"),
            ),
            order_observations=pilot().order_observations
            + (OpenOrderObservation("other", DAY2, DAY2, "A", 1, "2026-01-03", 1, "orders"),),
        )
    )
    assert [quantity(snapshot) for snapshot in snapshots] == [10, 10, 15]


def test_matching_stock_snapshots_assert_stock_without_adding_it():
    snapshots = compile_business_pilot(
        pilot(
            movements=(movement("receipt", "receipt", 5),),
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("audit", DAY2, {"A": 15}, "warehouse"),
            ),
        )
    )
    assert [quantity(snapshot) for snapshot in snapshots] == [10, 15]


def test_stock_mismatch_is_not_silently_converted_to_receipt():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                stock_snapshots=(
                    StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                    StockSnapshot("unexplained", DAY2, {"A": 12}, "warehouse"),
                )
            )
        )


def test_same_observation_batch_is_atomic_and_input_order_independent():
    records = (movement("out", "dispatch", 15), movement("in", "receipt", 5))
    stocks = (
        StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
        StockSnapshot("day two", DAY2, {"A": 0}, "warehouse"),
    )
    left = compile_business_pilot(pilot(movements=records, stock_snapshots=stocks))
    right = compile_business_pilot(
        pilot(movements=tuple(reversed(records)), stock_snapshots=stocks)
    )
    assert [quantity(snapshot) for snapshot in left] == [10, 0]
    assert [(s.supplies, s.orders) for s in left] == [(s.supplies, s.orders) for s in right]


def test_negative_stock_is_rejected_without_future_receipt_borrowing():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                stock_snapshots=(
                    StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                    StockSnapshot("day three", DAY3, {"A": 19}, "warehouse"),
                ),
                movements=(
                    movement("too much outbound", "dispatch", 11),
                    movement("tomorrow receipt", "receipt", 20, DAY3, DAY3),
                ),
            )
        )


def test_same_observation_batch_cannot_hide_negative_physical_history():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                movements=(
                    movement("earlier outbound", "dispatch", 15, DAY2, DAY3),
                    movement("later receipt", "receipt", 5, DAY3, DAY3),
                ),
                stock_snapshots=(
                    StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                    StockSnapshot("day three", DAY3, {"A": 0}, "warehouse"),
                ),
            )
        )


def test_delayed_observation_order_can_differ_from_valid_physical_event_order():
    snapshots = compile_business_pilot(
        pilot(
            movements=(
                movement("later outbound", "dispatch", 6, "2026-01-02T12:00:00Z", DAY3),
                movement("earlier receipt", "receipt", 5, DAY2, "2026-01-03T01:00:00Z"),
            ),
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("final", "2026-01-03T02:00:00Z", {"A": 9}, "warehouse"),
            ),
        )
    )
    assert [quantity(snapshot) for snapshot in snapshots] == [10, 9]


def test_physical_dispatch_does_not_implicitly_fulfill_remaining_order():
    snapshots = compile_business_pilot(
        pilot(
            movements=(movement("outbound", "dispatch", 4),),
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("day two", DAY2, {"A": 6}, "warehouse"),
            ),
        )
    )
    assert quantity(snapshots[-1]) == 6
    assert snapshots[-1].orders[0].quantity == 4


def test_zero_remaining_order_does_not_implicitly_remove_physical_stock():
    snapshots = compile_business_pilot(
        pilot(
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("day two", DAY2, {"A": 10}, "warehouse"),
            ),
            order_observations=pilot().order_observations
            + (OpenOrderObservation("customer", DAY2, DAY2, "A", 0, "2026-01-01", 1, "orders"),),
        )
    )
    assert snapshots[-1].orders == ()
    assert quantity(snapshots[-1]) == 10


def test_authoritative_remaining_quantity_replaces_prior_quantity():
    snapshots = compile_business_pilot(
        pilot(
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("day two", DAY2, {"A": 10}, "warehouse"),
            ),
            order_observations=pilot().order_observations
            + (OpenOrderObservation("customer", DAY2, DAY2, "A", 3, "2026-01-01", 1, "orders"),),
        )
    )
    assert snapshots[-1].orders[0].quantity == 3


def test_historical_stock_cannot_make_new_past_due_plan_look_on_time():
    snapshots = compile_business_pilot(
        pilot(
            stock_snapshots=(
                StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
                StockSnapshot("day three", DAY3, {"A": 10}, "warehouse"),
            )
        )
    )
    snapshot = snapshots[-1]
    assert all(supply.available_day == 2 for supply in snapshot.supplies)
    assert snapshot.orders[0].due_day == 0
    result = optimize(snapshot.supplies, snapshot.orders, snapshot.policy)
    score = evaluate_plan(snapshot.supplies, snapshot.orders, snapshot.policy, result)
    assert score.costs.total == 16


def test_initial_past_due_order_preserves_lateness_without_negative_day_indices():
    case = pilot(
        order_observations=(
            OpenOrderObservation("customer", OPEN, OPEN, "A", 4, "2025-12-30", 1, "orders"),
        )
    )
    snapshot = compile_business_pilot(case)[0]
    assert snapshot.orders[0].due_day == 0
    assert all(supply.available_day == 2 for supply in snapshot.supplies)
    result = optimize(snapshot.supplies, snapshot.orders, snapshot.policy)
    score = evaluate_plan(snapshot.supplies, snapshot.orders, snapshot.policy, result)
    assert score.costs.total == 16


def test_future_past_due_observation_cannot_shift_opening_time_origin():
    original = compile_business_pilot(pilot())[0]
    augmented = compile_business_pilot(
        pilot(
            order_observations=pilot().order_observations
            + (OpenOrderObservation("other", DAY3, DAY3, "A", 1, "2025-01-01", 1, "orders"),)
        )
    )[0]
    assert augmented.supplies == original.supplies
    assert augmented.orders == original.orders


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-01",
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00.000Z",
        "2026-02-30T00:00:00Z",
    ],
)
def test_timestamp_syntax_and_calendar_are_strict(timestamp):
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(stock_snapshots=(StockSnapshot("opening", timestamp, {"A": 10}, "warehouse"),))
        )


@pytest.mark.parametrize("bad_quantity", [True, 1.5, "10", -1])
def test_stock_quantity_does_not_coerce(bad_quantity):
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                stock_snapshots=(StockSnapshot("opening", OPEN, {"A": bad_quantity}, "warehouse"),)
            )
        )


def test_unknown_and_missing_snapshot_skus_are_rejected():
    for quantities in ({}, {"A": 10, "unknown": 3}):
        with pytest.raises(ValueError):
            compile_business_pilot(
                pilot(stock_snapshots=(StockSnapshot("opening", OPEN, quantities, "warehouse"),))
            )


def test_conflicting_movement_ids_are_rejected():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                movements=(
                    movement("same", "receipt", 2),
                    movement("same", "receipt", 3, DAY3, DAY3),
                )
            )
        )


def test_same_order_observation_timestamp_is_rejected():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                order_observations=pilot().order_observations
                + (OpenOrderObservation("customer", OPEN, OPEN, "A", 2, "2026-01-01", 1, "orders"),)
            )
        )


@pytest.mark.parametrize("changes", [{"observed_at": 1}, {"order_id": None}, {"sku": []}])
def test_malformed_order_fields_are_validated_before_sorting_or_set_lookup(changes):
    broken = replace(pilot().order_observations[0], **changes)
    with pytest.raises(ValueError):
        compile_business_pilot(pilot(order_observations=pilot().order_observations + (broken,)))


def test_malformed_movement_sku_is_rejected_with_domain_error():
    broken = replace(movement("bad sku", "receipt", 1), sku=[])
    with pytest.raises(ValueError):
        compile_business_pilot(pilot(movements=(broken,)))


def test_order_event_time_cannot_regress_with_later_observation():
    with pytest.raises(ValueError):
        compile_business_pilot(
            pilot(
                order_observations=(
                    OpenOrderObservation("customer", DAY2, DAY2, "A", 4, "2026-01-03", 1, "orders"),
                    OpenOrderObservation("customer", OPEN, DAY3, "A", 2, "2026-01-03", 1, "orders"),
                )
            )
        )


def test_mutated_nested_snapshot_is_revalidated_before_compilation():
    case = pilot()
    case.stock_snapshots[0].quantities["A"] = -1
    with pytest.raises(ValueError):
        compile_business_pilot(case)


def test_compiled_snapshot_inputs_do_not_alias_mutable_source_policy():
    case = pilot(policy=Policy(substitutions={"A": ()}))
    snapshots = compile_business_pilot(case)
    case.policy.substitutions["A"] = ("unknown",)
    assert snapshots[0].policy.substitutions == {"A": ()}


def test_dispatch_quantity_and_observed_before_event_are_invalid():
    for record in (movement("bad", "dispatch", 0), movement("bad", "receipt", 1, DAY3, DAY2)):
        with pytest.raises(ValueError):
            compile_business_pilot(pilot(movements=(record,)))


def test_non_boolean_synthetic_provenance_is_rejected():
    with pytest.raises(ValueError):
        compile_business_pilot(replace(pilot(), synthetic="false"))


def test_comparison_keeps_repeated_backlog_snapshots_separate_and_reports_no_execution():
    case = pilot(
        stock_snapshots=(
            StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
            StockSnapshot("day two", DAY2, {"A": 10}, "warehouse"),
        )
    )
    report = compare_business_pilot(case)
    assert [record["open_order_units"] for record in report["snapshots"]] == [4, 4]
    assert report["summary"]["final_snapshot"] == "day two"
    assert report["summary"]["execution_intents"] == 0
    assert report["external_dispatch"] is False
    assert "total_demand_units" not in report["summary"]


def test_comparison_source_hash_is_invariant_to_record_order():
    case = pilot(
        stock_snapshots=(
            StockSnapshot("opening", OPEN, {"A": 10}, "warehouse"),
            StockSnapshot("day two", DAY2, {"A": 15}, "warehouse"),
        ),
        movements=(movement("in", "receipt", 5),),
    )
    reordered = replace(case, stock_snapshots=tuple(reversed(case.stock_snapshots)))
    left = compare_business_pilot(case)
    right = compare_business_pilot(reordered)
    assert left["source_fingerprint"] == right["source_fingerprint"]
    assert [record["metadata"]["input_fingerprint"] for record in left["snapshots"]] == [
        record["metadata"]["input_fingerprint"] for record in right["snapshots"]
    ]


def test_order_identity_cannot_switch_catalog_sku():
    with pytest.raises(ValueError, match="cannot change SKU"):
        compile_business_pilot(
            pilot(
                catalog=(CatalogItem("A", 0), CatalogItem("B", 0)),
                stock_snapshots=(StockSnapshot("opening", OPEN, {"A": 10, "B": 0}, "warehouse"),),
                order_observations=pilot().order_observations
                + (
                    OpenOrderObservation("customer", DAY2, DAY2, "B", 4, "2026-01-03", 1, "orders"),
                ),
            )
        )
