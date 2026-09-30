# Business pilot intake and snapshot comparison

This adapter accepts documented business exports and builds frozen allocation inputs at observation cutoffs. It does not connect to an ERP, infer receipts from inventory differences, reconstruct requested demand from fulfilled sales, or execute plans. The included `examples/business_pilot` bundle is explicitly synthetic. Passing it is evidence that the intake path works, not evidence of performance on real business data.

Run `notebooks/business_pilot.ipynb` in Colab, or call:

```python
from assumption_ops import (
    load_business_pilot, compile_business_pilot, compare_business_pilot,
)

pilot = load_business_pilot("examples/business_pilot/manifest.json")
snapshots = compile_business_pilot(pilot)
report = compare_business_pilot(pilot)
```

## Files and exact headers

The manifest has exactly these keys:

```json
{
  "schema_version": 1,
  "name": "Synthetic recorded business pilot",
  "group_id": "synthetic-business-pilot-001",
  "synthetic": true,
  "catalog_path": "catalog.csv",
  "stock_snapshots_path": "stock_snapshots.csv",
  "movements_path": "movements.csv",
  "open_orders_path": "open_orders.csv",
  "policy_path": "policy.json"
}
```

Use `synthetic: false` for actual business observations. `group_id` identifies the business episode/group; it does not automatically establish an independent holdout. All paths must remain within the manifest directory. Absolute paths, parent traversal, and symlink escapes are rejected.

| File | Exact CSV header or contents | Meaning |
| --- | --- | --- |
| Catalog | `sku,unit_cost` | Unique SKU and its nonnegative integer allocation cost per interchangeable unit. |
| Stock observations | `snapshot_id,observed_at,sku,quantity,source` | Complete physical on-hand inventory for every catalog SKU at each snapshot. All rows of a snapshot share timestamp and source. |
| Movements | `movement_id,event_time,observed_at,sku,kind,quantity,source` | A unique physical receipt, dispatch, or explicitly recorded adjustment, with strictly positive integer quantity. |
| Open-order updates | `order_id,event_time,observed_at,sku,quantity,due_date,priority,source` | Absolute remaining requested backlog after the update; zero means closed/cancelled. |
| Policy JSON | `substitutions`, `allow_late`, `late_penalty`, `substitution_penalty`, `unfilled_penalty`, `disruption_penalty` | Explicit eligibility and weighted penalty settings under the existing `Policy` contract. |

Quantities and costs use unsigned integer text. There is no implicit conversion of decimals, signed numbers, blank cells, or booleans. `priority` is a positive integer. SKU names and order identifiers are stable identifiers; an existing order cannot change SKU. The allowed movement kinds are `receipt`, `dispatch`, `adjustment_in`, and `adjustment_out`. Receipts and inward adjustments add stock; dispatches and outward adjustments remove stock. CSV quantities themselves are positive. Sources and identifiers must be nonempty.

Timestamps are UTC in exact `YYYY-MM-DDTHH:MM:SSZ` form. `event_time` records when a fact happened; `observed_at` records when the system knew it. Observation cannot precede the event. Due dates are calendar dates `YYYY-MM-DD`; past-due backlog is valid and must not be silently moved to the planning cutoff. Resolve local time zones and daylight-saving ambiguities before exporting. All units must be consistent and interchangeable: do not silently round kilograms, currency amounts, or fractional pack quantities into integer units.

## Conservation and observation-time visibility

The earliest complete snapshot initializes physical inventory. Later stock snapshots are reconciliation observations, not receipts. Each SKU's admitted balance is opening stock plus admitted receipts and inward adjustments minus admitted dispatches and outward adjustments. Movements must occur after the opening snapshot. Unknown SKUs, duplicate movement IDs, negative resulting stock, inconsistent snapshot metadata, and reconciliation differences are errors. This intentionally requires a complete physical movement ledger; it does not guess adjustments, transfers, or returns. Explicitly map those business transactions into documented movement kinds before exporting.

The compiler checks both observation-time balances and the admitted physical history ordered by event time. Negative stock hidden by delayed batch reporting is rejected even if a later receipt makes the final snapshot reconcile. Movements sharing the same timestamp are applied atomically, so this contract does not establish their intrasecond ordering.

Only events whose event time and observation time are both at or before a cutoff can enter that cutoff's state. Late-observed history cannot appear in earlier plans. Current on-hand units become available on the cutoff's calendar date, so an overdue order incurs the corresponding late penalty rather than receiving stock backdated to opening day. The compiler chooses a relative-day origin that preserves any past-due date.

Open orders are maintained independently of physical inventory. The latest admitted update replaces an order's remaining quantity. For example, updates from 6 units to 5 units yield backlog 5, not 11. A dispatch changes physical stock; it does not automatically reduce backlog without an explicit order update. A cancellation changes backlog; it does not create a physical receipt. Coordinate both exports using the business system's actual conventions.

## What the included fixture demonstrates

Three daily cutoffs show opening A=4/B=4, then A=6/B=3 after an A receipt and B dispatch, then A=5/B=3 after an A dispatch. One order event occurs on day two but is first observed on day three; it is intentionally hidden from the day-two plan. Another order is explicitly cancelled by an absolute zero remaining quantity.

A costs 0 and B costs 10 in configured penalty units. A constrained order needs A, while a higher-priority flexible B order accepts A as a substitute at penalty 2. Priority/cost greedy can consume inexpensive A before considering the constrained order's alternative-free need. The optimal solver handles this shared-capacity opportunity cost. This is a constructed demonstration, not a measured prevalence or return on investment.

## Reading comparison reports

Each snapshot compares MILP, verified LP, earliest-due-date, and priority/cost greedy on the same admitted inventory, open backlog, eligibility, and weights. Independent accounting verifies demand balance, capacity, and objective. An optimal verified LP bound provides a reference; it is not an unverified executable allocation. Reports retain source fingerprints, snapshot cutoffs, provenance metadata, solver status, service quantities, configured weighted costs, and timings.

Snapshot results are repeated current-state decisions. Do not sum their requested/allocated units as newly arriving demand or fulfilled sales; the same backlog can appear at several cutoffs. No returned plan is committed, reserved, or dispatched. Timings depend on the machine and workload. Weighted penalties are business assumptions until their economic meanings are validated.

For a genuine pilot, agree on column semantics with the data owner, reconcile a complete movement ledger, inspect late observations, and compare methods across representative episodes. Tune policy weights only on calibration groups and evaluate once on independent holdout groups. Retrospective snapshots can assess feasibility and modeled decisions; they do not measure causal fulfillment improvements because alternative allocations can change subsequent inventory and order histories.

## Requesting actual exports

Create a new unfilled bundle without copying the synthetic observations:

```bash
assumption-ops-business --init-dir /path/to/new/business_export
```

The destination must not already exist. The command creates the four CSV headers,
manifest, policy file, and a concrete `DATA_REQUEST.md` for the data owner. It does
not fetch records or run a solver. The provenance flag and policy fields are
`null` so they must be set explicitly; validation rejects the unfilled template.
Fill the episode/group identifier as well. Customer identities and contact or
payment information are unnecessary; stable pseudonymous identifiers suffice.

The final two sections of `business_pilot.ipynb` create a downloadable template
ZIP and provide a separate `REAL_MANIFEST` path for uploaded records. Keep the
synthetic demo assertions unchanged; actual records use a separate comparison
without hard-coded expected scores. The generated request explains the exact
semantics and source-completeness information needed from the data owner. Review
it before sharing; no message is sent by the framework.
