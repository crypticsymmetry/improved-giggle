# Assumption Ops

A runnable supplier-disruption framework: accepted evidence supports an allocation
proposal; a changed supplier fact identifies the proposals that need reconsideration;
an integer optimizer replans outstanding demand; approval and local commit recheck
source support and inventory capacity.

[**Run the notebook in Google Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/colab_demo.ipynb)

The notebook runs on a CPU, needs no API keys, and installs the package from this
repository. It includes a supplier delay, explicit conflict resolution, what-if
comparison, replanning, stale approval rejection, and local execution intent.

## Run locally

Python 3.11 or newer. SciPy/HiGHS is the only direct runtime dependency.

```bash
git clone https://github.com/crypticsymmetry/improved-giggle.git
cd improved-giggle
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
assumption-ops examples/sample_scenario.json --delay-supply shipment --available-day 8
python -m pytest -q
python scripts/validate_notebook.py
```

To persist a scenario, add `--db scenario.sqlite`. A JSON file initializes a new
database; an existing database retains its accepted evidence and reservations.
Run the notebook from Colab, Jupyter, or VS Code. Its bootstrap skips installation
when the package is already importable.

## A real pipeline

```python
from assumption_ops import OperationsPipeline, Supply, Order, Policy, StaleDecisionError

with OperationsPipeline("scenario.sqlite") as ops:
    if not ops.is_seeded:
        ops.seed(
            supplies=[Supply("stock", "A", 3, 0), Supply("shipment", "A", 7, 2),
                      Supply("substitute", "B", 4, 0, unit_cost=3)],
            orders=[Order("urgent", "A", 5, 3, priority=3),
                    Order("standard", "A", 5, 4)],
            policy=Policy(substitutions={"A": ("B",)}),
        )
    baseline = ops.plan()
    hypothetical = ops.what_if(
        {"supply:shipment": {"available_day": 8}}, previous=baseline.plan
    )  # No evidence, decisions, or reservations are written.

    observation = ops.propose("supply:shipment", "available_day", 8, "supplier confirmation")
    ops.accept(observation)
    print(ops.impact())  # Conflicting accepted dates remain explicit.
    # ops.plan() raises EvidenceConflict until a reviewer chooses a source.
    ops.resolve("supply:shipment", "available_day", observation)
    revised = ops.plan(previous=baseline.plan)

    if revised.decisions:
        decision = revised.decisions[0]
        print(ops.explain(decision.id))
        ops.approve(decision.id)
        ops.commit(decision.id)  # Atomic LOCAL reservation and execution-intent event.
```

`propose` creates an observation only. `accept` marks it reviewed, without silently
replacing contradictory observations. `resolve` selects an accepted observation
and appends supersession events. Source observations are never rewritten.
Identical accepted values do not conflict; their distinct provenance is retained,
and changing the current evidence ID conservatively invalidates earlier support.

## Implementation map

| File | Implemented responsibility |
|---|---|
| `src/assumption_ops/evidence.py` | SQLite evidence, review lifecycle, canonical JSON comparisons, append-only audit events, nested transactions, structured extractor protocol |
| `src/assumption_ops/atms.py` | Minimal assumption-support antichains, conjunctive rules, alternative justifications, grounded cyclic fixed points, nogoods, context updates |
| `src/assumption_ops/optimizer.py` | Sparse integer allocation model, shortages, substitutions, dates, stability penalties, independent solution verification |
| `src/assumption_ops/pipeline.py` | Snapshot-based planning, read-only what-if solves, support explanations, persisted decisions, approvals, inventory reservations, execution intents |
| `src/assumption_ops/cli.py` | Reproducible JSON scenario runner |
| `src/assumption_ops/replay.py` | Chronological evidence replay, common-reference allocation comparison, service accounting and active-plan review proxies |
| `src/assumption_ops/replay_io.py` | Versioned replay JSON intake and spreadsheet-safe snapshot CSV export |
| `src/assumption_ops/replay_fixtures.py` | Fixed-seed synthetic supplier-delay, stock-loss, demand-surge and cost-shock traces |
| `src/assumption_ops/replay_cli.py` | Multi-case replay runner with reusable inputs, raw reports and per-case manifest |
| `notebooks/colab_demo.ipynb` | End-to-end executable walkthrough plus an alternative-support reasoning example |
| `tests/` | Component, independent-oracle, persistence, failure, and concurrent-commit checks |

The implementation separates **source evidence**, **logical support**, and
**resource feasibility**. A language model may propose a structured observation;
it does not set solver constraints, certify source truth, or approve an action.

## Algorithms and contracts

### Assumption-based reasoning

A node's label is an antichain of minimal assumption sets. For a conjunctive
justification, the engine takes unions from the Cartesian product of premise
labels, rejects sets containing a nogood, and eliminates dominated supersets.
Multiple justifications give alternative supports. A node is supported when at
least one consistent support is entirely active in the current context.

The reverse dependency index identifies descendants for incremental updates.
Rule changes recompute the affected region to its least fixed point; unsupported
cycles do not create facts. Evidence activation changes context without rebuilding
symbolic labels. `EnvironmentOverflow` raises on a per-node label limit with atomic
rollback. This limit bounds stored environments, **not exponential computation time**.

The supplier pipeline creates one conjunctive evidence support per allocation
decision. The standalone ATMS supports alternative proofs and nogoods, demonstrated
separately in the notebook and tests. Logical alternatives alone cannot arbitrate
shared stock: that remains the optimizer's responsibility.

### Integer allocation

For every eligible order/supply pair, integer `x[order,supply]` is the assigned
quantity. Integer `u[order]` is unmet demand. Constraints enforce:

- `sum_supply x[order,supply] + u[order] == outstanding demand`.
- `sum_order x[order,supply] <= unreserved supply`.
- Only exact SKUs or explicitly permitted one-for-one substitutes are eligible.
- Late supply is ineligible unless the policy explicitly permits it.

The objective is the sum of acquisition cost, lateness per unit-day,
substitution cost, priority-weighted unfilled penalties, and absolute deviations
from a supplied previous allocation. Absolute deviations are linearized with
auxiliary variables; disappearing prior arcs retain their retirement cost.

This is a **weighted objective**, not lexicographic service priority. A sufficiently
expensive unit can legitimately remain unallocated. Calibrate all costs to
consistent business units. HiGHS solves the MILP through `scipy.optimize.milp`.
Independent validation checks integer quantities, every arc, capacities, exact
order balances, and finite objective values before a result is returned. An
independently verified time-limited incumbent is labeled `feasible_limit`, not
`optimal`; missing or infeasible incumbents raise.

### Persistence and execution

Planning reads a consistent accepted snapshot under a SQLite transaction. All
observations, decisions, approvals, and reservations share one connection; nested
operations use savepoints. Initialization and each local commit are atomic.

A decision records the evidence IDs for its order, allocated supplies, and policy.
`impact()` distinguishes unresolved source conflicts from superseded support.
An unrelated changed fact need not deactivate its logical label, although any
unresolved scenario conflict conservatively blocks planning, approval, and commit.

Before local commit, the framework checks:

1. Explicit approval exists.
2. Supporting evidence IDs are still current and conflicts are resolved.
3. The selected units still meet SKU/date rules.
4. Other committed decisions have not consumed the stock or fulfilled the demand.

SQLite write serialization protects competing commits across process/connection
instances. Repeated commit of the same decision ID is idempotent. Replanning
subtracts committed supply reservations and fulfilled demand. Different proposals
for the same stock cannot both over-reserve it.

`logical_support.supported` describes whether the original evidence remains
current. It does **not** certify remaining inventory, source truth, or continued
global optimality. Live capacity is checked at approval and commit.

## Structured ingestion

`JsonExtractor` implements the `Extractor` protocol and accepts explicit JSON:

```python
from assumption_ops import EvidenceStore, JsonExtractor

with EvidenceStore() as store:
    ids = JsonExtractor().extract(
        '[{"entity":"supply:shipment","field":"available_day",'
        '"value":8,"source":"supplier message 42"}]', store
    )
    # Records are proposals, not accepted facts.
```

Use an adapter to map ERP records, document extraction, or model output into this
schema, preserve a source reference, and route proposals through explicit review.
Natural-language email extraction and live ERP connectors are not implemented.
Standalone stores accept arbitrary JSON; the scenario pipeline additionally
validates known entities and typed domain fields. The framework's public store
allows low-level use; snapshots validate accepted domain inputs again.

## Verification

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python scripts/validate_notebook.py
```

Tests include exhaustive active-context comparisons with an independent forward
reasoner on random cyclic graphs; tiny integer allocations checked against a
brute-force objective oracle; explicit overflow rollback; conflict/resolution
history; atomic initialization failure; stale approvals; database reopening;
read-only what-if equivalence; idempotent local commits; and simultaneous competing
commits using separate connections. CI runs on Python 3.11, 3.12, and 3.13 and
executes the notebook in a real Jupyter kernel.

## Scope and next integration steps

This is a tested implementation for a single bounded supplier scenario, suitable
for experimentation and a narrowly scoped pilot. Quantities are interchangeable
integer units; substitution ratios are one-for-one; days are relative integers.
The planner permits partial fulfillment and explicitly reports unmet quantities;
an allocation decision is not a promise that an entire order will ship.

To connect a business system, implement an authenticated intake/review service,
source-specific adapters, and an idempotent outbox dispatcher. The current
`execution_intent` audit event is **local only**: no supplier, customer, or ERP API
is called, and a local SQLite transaction cannot make a remote action atomic.
Production use also needs tenant isolation, permissions, external inventory
reconciliation, operational metrics, and compensation for already committed
allocations when later facts change. Such changes appear in impact reports;
they are not silently undone. Past reservations must be reconciled explicitly.

Automatic learned procedures, probabilistic diagnosis, free-form LLM extraction,
and a hosted UI are extension points, rather than simulated capabilities.

## Operational comparisons and CSV intake (v0.2)

[**Run the operational comparison notebook in Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/operational_scenarios.ipynb)

The original walkthrough establishes the evidence/reasoning/commit invariants.
The second notebook loads CSV data and compares business responses with compact
service and cost tables. Both need no API keys. Bootstrap checks for the current
comparison API and upgrades an older framework installation when necessary.

```python
from assumption_ops import load_csv_scenario, OperationsPipeline, Policy, Scenario

data = load_csv_scenario(
    "examples/supplies.csv", "examples/orders.csv",
    Policy(substitutions={"A": ("B",)}),
)
with OperationsPipeline() as ops:
    ops.seed(data.supplies, data.orders, data.policy)
    original = ops.plan()
    update = ops.propose("supply:shipment", "available_day", 8, "supplier confirmation")
    ops.accept(update)
    ops.resolve("supply:shipment", "available_day", update)
    comparison = ops.compare_scenarios([
        Scenario("expedite", {"supply:shipment": {"available_day": 3}}, action_cost=40),
        Scenario("more substitutes", {"supply:substitute": {"quantity": 7}}, action_cost=20),
        Scenario("late fulfillment", {"policy": {"config": {"allow_late": True}}}),
    ], previous=original.plan)
    print(comparison.rows())
```

Every comparison freezes accepted evidence, reservations, and the previous plan.
Its results include evidence and operation revision identifiers and each candidate's explicit hypothetical overrides. It creates no
observations, decisions, approvals, or reservations. Selecting the first ranked
row does **not** apply that candidate. New availability or permissions must enter
through the reviewed evidence lifecycle before a plan is approved.

The independent `evaluate_plan` recomputes all objective terms from the allocation
and rejects an objective mismatch. It reports requested/allocated/unfilled units,
priority-weighted shortages, on-time/late/substitute units, unit-days late, allocation
changes, fully fulfilled orders, and per-order service rows. With no previous plan,
changed units are measured against an empty allocation; disruption cost remains
zero, matching the optimizer. When using a previous plan, all options use the same
reference. The original notebook's hypothetical and confirmed delay now use the
same previous-plan penalty and have comparable objectives.

Comparison guards hold order demand, priority, SKUs, and penalty weights fixed.
They permit hypothetical supply quantity/date/cost changes, negotiated due dates,
and eligibility-policy changes. Policy configuration in a comparison **merges**
with the baseline so an eligibility change cannot accidentally reset cost weights.
The existing standalone `what_if` contract still replaces a full supplied policy
configuration and applies defaults to omitted fields.

`action_cost` is a separate nonnegative fixed cost in the same units as the
objective. Per-unit costs are already included: avoid counting them twice. These
scores are modeled penalty units, not calibrated currency or demonstrated ROI.
Negotiated dates, added stock, and action prices are hypothetical inputs needing
confirmation. A time-limited solver result is explicitly labeled `feasible_limit`;
ranked achieved scores then do not establish a globally optimal ranking.

```bash
assumption-ops --supplies-csv examples/supplies.csv --orders-csv examples/orders.csv \
  --policy-json examples/policy.json --delay-supply shipment --available-day 8 \
  --compare examples/interventions.json --report-csv /tmp/interventions.csv
python scripts/benchmark.py --orders 25 100 --repeats 3
```

CSV intake rejects duplicate/unknown headers, missing fields, malformed rows,
invalid integers, and duplicate IDs, with source-line errors. `unit_cost` and
`priority` columns are optional and default to zero and one respectively when
absent. An existing blank cell is an error. UTF-8 BOM and quoted commas work.
JSON intake rejects extra fields, duplicate keys, nonfinite values, and implicit
numeric coercion. Inputs are validated without running a solver.

`write_comparison_csv` produces flat ranked rows and keeps spreadsheet formula-like
string values literal. Its fixed action fees are advisory estimates, not booked
payments or evidence-backed financial commitments. The benchmark script generates
seeded synthetic cases, keeps individual timings, and reports medians for initialization,
planning, and a four-solve comparison. It is a reproducible scaling probe, not a
business-outcome study; timing thresholds are not used as flaky test assertions.

New modules: `evaluation.py`, `scenarios.py`, `intake.py`, and `export.py`.

## Reviewed events and atomic plan execution (v0.3)

[**Run the transactional workflow notebook in Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/transactional_workflow.ipynb)

The third notebook covers the transition from an advisory comparison to confirmed
facts and local allocation intent. The examples are mock supplier confirmations;
no external accounts or API keys are required.

```python
from assumption_ops import EvidenceUpdate

snapshot = ops.revisions()
receipt = ops.apply_updates(
    "supplier-message-123",
    [EvidenceUpdate("supply:shipment", "available_day", 3,
                    "reviewed supplier confirmation 123")],
    expected_evidence_revision=snapshot["evidence_revision"],
    expected_operations_revision=snapshot["operations_revision"],
)
reviewed_plan = ops.plan(previous=original.plan)
ops.approve_plan(reviewed_plan.plan_id)
local_commit = ops.commit_plan(reviewed_plan.plan_id)
```

`apply_updates` is an explicit **trusted caller review boundary**. It proposes,
accepts, and resolves the submitted values as one transaction. It must not be
wired directly to unreviewed email parsing or model output. Use `propose` for
incoming observations and a separate authorized review action for confirmations.
The library does not authenticate a reviewer or prove source truth.

The batch and every evidence/audit write share one SQLite transaction. A bad later
field rolls back earlier updates. A source event ID maps to a durable payload hash
and receipt in `ops_reviewed_batches`. Payload identity includes ordered fields,
values, and source attribution; canonical JSON makes dictionary-key ordering
irrelevant. A reused event ID with different content raises `IdempotencyConflict`.

New event IDs must match **both** evidence and operations revisions under the
write lock. `StaleSnapshotError` requires refreshing the review snapshot; it does
not retry a changed decision automatically. Unaccepted proposals also advance the
evidence revision, so this guard is conservative. Existing event IDs with identical
payloads return their original receipt with `replayed=True` before stale checks.
They never restore earlier facts, even after a later correction or process restart.
Receipt revisions describe the original application, not the current database.

Batches must be called at the top level, outside an existing database transaction.
This ensures a returned receipt represents committed SQL and ATMS synchronization
occurs after commit. The complete accepted snapshot is reflected in the logical
engine only after the batch succeeds. Existing unresolved conflicts on unrelated
fields remain visible; the batch resolves only its explicitly named fields.

`PlanningResult.plan_id` identifies a recorded allocation group, including an
empty group. `approve_plan` validates every pending decision and the group's
aggregate capacity/demand before promoting anything. Approval reserves no stock.
`commit_plan` requires every pending decision to be approved, then rechecks and
reserves all new decisions within one outer transaction. A later failure rolls
back earlier reservations, statuses, and intent events from that call. Previously
committed members remain historical; retries skip them without duplicate intents.
A known empty plan returns an idempotent no-op. Committing an allocation group does
not imply full order fulfillment or physical dispatch.

Opening an older v0.2 database creates the receipt table without rewriting existing
observations or plan history. The replay and reservation checks remain durable
across reopening. New tests cover migration, duplicate delivery, changed payloads,
competing snapshots, full-batch rollback, ATMS consistency, injected approval/commit
failures, and overlapping concurrent plan commits.

```bash
python scripts/replay_workflow.py
python scripts/validate_notebook.py   # Executes all three notebooks.
```

The replay script creates a temporary persistent database, applies a mock delay,
compares an expedite hypothesis, applies mock reviewed confirmations, reserves the
replacement plan, retries deliveries, and reopens the database. It asserts that an
old delay cannot overwrite the newer arrival and that intents are not duplicated.
The hypothetical fixed intervention fee remains an advisory estimate; it is not
booked by local plan commits. External dispatch, authenticated review, release,
and compensation of already committed inventory remain integration work.

New modules: `events.py` and `execution.py`.


## v0.4: chronological business-scenario replay

[**Run the replay benchmark in Google Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/replay_benchmark.ipynb)

This milestone measures allocation quality and operational review proxies before
connecting execution to an external system. The supplied cases are synthetic,
with fixed seeds and explicit source labels. They cover supplier delays and
corrections, stock losses and replenishment, demand revisions, and cost shocks.
They are executable examples, not evidence of production performance or savings.

```bash
python -m pip install -e '.[dev]'
assumption-ops-replay --orders 20 --seeds 1729 1730 1731 --output-dir replay_results
# Or supply an explicitly labeled, chronological case:
assumption-ops-replay --case examples/replay_case.json --output-dir replay_results
```

Each invocation exports a manifest plus an input JSON, detailed report JSON, and
snapshot CSV for every case. Numbered filenames avoid interpreting case names as
paths. The manifest lists only files from the current invocation; unrelated or
older files in the output directory are retained. Input cases are read without
modification. Successful outputs are written after all case solves complete;
filesystem writes themselves are not a distributed transaction.

```python
from assumption_ops.replay import run_replay
from assumption_ops.replay_fixtures import synthetic_cases
from assumption_ops.replay_io import (
    load_replay_case, save_replay_case, write_replay_report_csv,
)

case = synthetic_cases(order_count=20, seed=1729)[0]
report = run_replay(case)
save_replay_case(case, "my_replay_case.json")
assert load_replay_case("my_replay_case.json") == case
write_replay_report_csv(report, "my_replay_snapshots.csv")
print(report["summary"]["final_optimized"])
```

The replay seeds a fresh local evidence store, plans the initial snapshot, and
applies only the next explicit confirmation batch at each step. Each delivery is
immediately retried with the identical event ID and payload to check that the
receipt is replayed without new evidence or operational events. No approval,
commit, reservation, shipment, or payment is performed. Initial entity identities
remain fixed; updates change fields on existing orders and supplies or replace
policy configuration. New order/lot creation and reservation release are not yet
supported by this replay format.

At each snapshot the integer optimizer and deterministic greedy allocator see
identical current demand, supply, eligibility and objective weights. Both are
scored against the same **initial optimized allocation** after the initial step.
This deliberately fixed reference makes snapshot comparisons interpretable; it
is not a comparison of two independently executing strategies with their own
previous allocations. Greedy orders by priority, then due date and identifier,
and selects eligible supply by weighted unit cost. It is a transparent baseline,
not a claim about the quality of a particular company's current process.

The reports separate these measurements:

| Measurement | Meaning |
|---|---|
| Final service and cost | Final projected fill, shortage, on-time units and independently reconstructed cost for each strategy |
| Per-snapshot score gap | Greedy score minus optimizer score, under the same reference and weights; solver status accompanies each result |
| Allocation changes | Changed arcs and L1 movement relative to the preceding optimized snapshot; moving one unit between lots counts two allocation-arc units |
| Review proxy | Source-support invalidations among the preceding active plan's decisions, excluding historical proposals |
| Confirmations and retries | Explicit updated fields, revision guards and duplicate-delivery verification |
| Runtime | Machine-specific wall times and software environment; no hard latency guarantee |

Repeated snapshots contain many of the same orders. The reports therefore do
**not** sum their demand, shortage or projected acquisition costs. Cumulative
allocation changes and event/review counts describe adaptation across the trace;
final service describes the last snapshot only. Cost scores are configured
penalty units and are not automatically dollars or ROI. Decision counts are not
measured human review time. Support guards establish source justification and
feasibility: zero affected decisions does not imply that a plan remains globally
optimal after unrelated facts change.

The versioned JSON schema requires `schema_version: 1`, a nonempty `name`, an
explicit boolean `synthetic`, an `initial` scenario and ordered `batches`.
`seed` is an optional nonnegative integer or null. Each batch has a unique
`event_id` and nonempty `updates` with `entity`, `field`, `value` and `source`.
Quantities and dates use the package's strict integer contracts. Unknown keys,
identities and fields, duplicate JSON keys, nonfinite values and implicit numeric
coercions are rejected. Setting `synthetic: false` is a caller provenance label,
not an independent verification of the dataset's origin.

For a business pilot, export a fixed planning snapshot plus time-ordered,
explicitly confirmed revisions in this format. Calibrate all penalties in
consistent units and compare final service, shortages, adaptation and review
counts by case. The next integration should consume these local reports and
explicit approvals before dispatching an external action.
