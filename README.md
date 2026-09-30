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


## v0.5: pilot intake and holdout policy calibration

[**Run pilot calibration in Google Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/pilot_calibration.ipynb)

The pilot workflow accepts a directory containing a manifest and reusable replay
cases. Episodes are assigned explicitly to `calibration` or `holdout`; their
`group_id` keeps related business episodes together. The loader rejects groups
that appear in both splits and identical business traces renamed or relabeled
across splits. Input provenance remains explicit. The included pilot is synthetic;
no actual historical business dataset is bundled or inferred.

```bash
python -m pip install -e '.[dev]'
assumption-ops-pilot --manifest examples/pilot/pilot.json --validate-only
assumption-ops-pilot --manifest examples/pilot/pilot.json \
  --config examples/pilot/calibration_config.json --output-dir pilot_results
```

Validation does not run the solver. Calibration evaluates every configured
penalty candidate on calibration episodes, selects one policy, and then evaluates
only that selected policy and the existing baseline on holdout episodes. The
holdout report can show a regression or failed service constraint; it never
silently chooses a different policy from holdout performance.

**Candidate objective values are not comparable across different penalty weights.**
Every candidate is independently verified under its own optimization policy, then
rescored with the same explicit `scoring_weights`. Selection minimizes the mean
final fixed score per requested unit across calibration episodes, subject to the
configured final fill and late-rate constraints on every calibration episode.
Empty-demand episodes use denominator one, fill rate one and late rate zero.
This gives each episode equal weight rather than letting the largest case dominate.
It is a constrained policy search with supplied business values, not automatic
inference of true costs or proof of financial savings.

All candidates preserve each case's substitution eligibility and `allow_late`
setting. They vary only lateness, substitution, disruption and shortage penalties.
For each episode, a single initial allocation is solved using the fixed scoring
weights, and all updated candidate and baseline snapshots use that common
reference. Baseline plans use the episode's original penalty weights. Initial
plans have no disruption reference; subsequent plans use the fixed initial one.
The objective comparison therefore describes provisional snapshot planning,
not independently executing strategies with their own reservation histories.
All solves must report optimal status for a calibration result to be produced.

Policy-configuration events are rejected in pilot episodes so events cannot
silently overwrite the search policy. Confirmed facts on existing supply and
order entities can still change chronologically. No approval, commit, inventory
reservation or external action occurs. Final service and cost are measured once
per episode; repeated snapshots are not counted as additional demand.

The pilot manifest has this versioned structure:

```json
{
  "schema_version": 1,
  "name": "my offline pilot",
  "episodes": [
    {"episode_id": "period-1", "group_id": "account-period-1",
     "split": "calibration", "case_path": "case-001.json"},
    {"episode_id": "period-2", "group_id": "account-period-2",
     "split": "holdout", "case_path": "case-002.json"}
  ]
}
```

Case paths must stay within the manifest directory; absolute paths, directory
traversal and symlink escapes are rejected. Each case uses the v0.4 replay schema.
Both splits must be nonempty and episode IDs must be unique. Group labels are
caller supplied: the loader cannot detect undisclosed relationships between
otherwise different traces. A caller's `synthetic: false` label does not establish
that a file is authentic production history.

Calibration configuration requires `schema_version: 1`, `scoring_weights`
containing all four integer penalties, and a nonempty `candidates` array. Every
candidate supplies a unique name and all four penalties. Optional `constraints`
contains `min_fill_rate` and `max_late_rate`, each a finite number in [0, 1].
Constraints apply to final units requested, with late units divided by requested
units. The CLI writes a dataset summary, complete calibration/holdout JSON and
spreadsheet-safe calibration ranking CSV. Holdout results remain diagnostic and
are separate from candidate ranking. Filesystem report writes are local writes,
not a distributed transaction.

A practical pilot starts by selecting disjoint business periods or accounts,
exporting their starting planning snapshots and explicitly confirmed revisions,
and documenting the units and rationale for fixed business penalties and service
bounds. Inspect the holdout report before using the selected settings operationally.
The framework does not infer chronological split boundaries, tune business
weights from holdout data, or dispatch orders.


## Public industrial data and allocation-method comparison

[**Run the real warehouse benchmark in Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/real_warehouse_benchmark.ipynb)

The public source is **Dynamic storage assignment in homogeneous dual-access
deep-lane S/R systems: Dataset**, by Gabriele Sirri, Riccardo Accorsi, Giacomo Lupi
and Riccardo Manzini (2026), [DOI 10.5281/zenodo.18229759](https://doi.org/10.5281/zenodo.18229759),
[Zenodo record](https://zenodo.org/records/18229759), CC BY 4.0. The authors describe
an industrial food-and-beverage case with a 130-day horizon, production quantities,
daily demand, and initial inventory. Our conversion adapts those data into a SKU
allocation problem; it does not reproduce the original storage-assignment model.

```bash
python -m pip install -e '.[dev,datasets]'
assumption-ops-warehouse --download --allow-inferred-demand \
  --stock-fractions 1.0 0.5 --output-dir warehouse_results
```

The downloaded Access file is pinned to SHA-256
`4db26e1552f495f840e5c61e5d80fc7ed51bd28e1af6e10bd04b606d164f4548`.
The adapter verifies the checksum, item catalogs and data invariants before using
its quantities. `access-parser==0.0.6` is an optional pure-Python reader; the core
framework remains usable without it. Native table exports were independently
checked with MDB Tools during adapter development.

**This benchmark is provisional because the published D table has mismatched
field roles.** In that table, the named `Q` column contains values matching SKU
codes and the named `item_WEEK` column contains apparent numeric quantities.
Both Access readers agree on the stored values; the adapter validates SKU catalog
membership and identifier prefixes across all rows, but the authors have not
confirmed our interpretation. Reading the dataset fails by default. The explicit
`--allow-inferred-demand` flag acknowledges the documented mapping; it does not
turn that interpretation into independently verified demand semantics. Every
report retains this limitation and the physical-to-modeled field mapping.

The projection aggregates demand by SKU/day, production into dated supply lots,
and initial inventory by SKU. Production day is treated as availability day;
demand day is treated as requested due day. Acquisition costs are zero because
purchase costs are absent; lateness/shortage penalties are illustrative. No
substitution relationships, customer priorities, lane geometry, batch compatibility,
warehouse capacity or holding costs are inferred. All methods see the same full
horizon, so this is an **offline planning benchmark**, not an online forecast or
an estimate of realized business outcomes.

The comparison reports independently validated integer allocations from MILP,
earliest-due-date allocation, and priority/cost greedy allocation. A separate
continuous LP solves the same objective and resource balance to provide a lower
bound only after an optimal LP result; no unvalidated fractional plan is described
as executable. Solver statuses, objective gaps, service metrics and wall times
remain explicit. This pure transportation formulation has integral structure,
so LP/MILP ties are expected. A tie is evidence against claiming a unique MILP
advantage on these inputs.

A stock fraction of one uses the projected source quantities. Fractions below
one scale both initial inventory and production quantities downward with integer
rounding. They are **modeled supply stress**, not observed stock losses or supplier
delays. Reports separate these runs and export full allocation JSON, method CSV,
and a compact provenance-aware summary. No source database is redistributed in
git; the checksum-pinned downloader reproduces it locally.


A measured development run is retained in
[`examples/warehouse_benchmark_summary.json`](examples/warehouse_benchmark_summary.json),
including source checksums, exact weights, run settings, solver statuses and
machine-specific timings. All methods filled 25,686 interpreted demand units on
time in the native projection (score zero). With all projected supply reduced to
50%, MILP allocated 18,366 units on time versus 17,305 for both greedy methods;
all had 5,126 unfilled units. Weighted scores were 5,544,350 and 6,056,700,
respectively. LP matched MILP at every tested supply fraction (1, .75, .5, .25).
These are results of the documented provisional projection and modeled stress,
not measured improvements to the original industrial operation.


## v0.6: verified LP transport backend and paired scaling measurements

[**Run solver scaling in Colab**](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/solver_scaling.ipynb)

`optimize` now accepts an explicit backend. The default remains `milp` to preserve
existing behavior, including allocation tie-breaking and prior-plan stability.

```python
from assumption_ops import optimize, evaluate_plan

integer_plan = optimize(supplies, orders, policy, solver="lp")
evaluate_plan(supplies, orders, policy, integer_plan)
# Automatic selection handles supported LP failures and unsupported prior plans:
revised = optimize(supplies, orders, policy, previous=integer_plan, solver="auto")
```

The specialized LP backend supports the current one-for-one transportation model
with integer quantities and **no previous allocation**. It indexes eligible lots
by SKU and builds sparse capacity and exact demand-balance matrices. HiGHS dual
simplex solves the continuous formulation. An integral transport polytope makes
integer vertices possible, but solver output is still checked independently:
finite quantities, near-integrality, exact integer capacity/demand feasibility,
objective reconstruction and accounting. Fractional incumbents are rejected, not
arbitrarily rounded. Claimed optimal integer objectives must match the LP result
with an absolute tolerance; a large total cannot conceal rounding differences
through a relative tolerance.

Explicit `lp` with a previous plan raises `ValueError`; this version does not
implement prior-plan L1 stability in LP. `auto` routes prior-plan requests to
MILP, otherwise tries LP and catches only `LPPlanError` for a MILP fallback.
Malformed caller input is rejected before dispatch. A verified time-limited
integer incumbent retains `feasible_limit` status; it is not called optimal.
The LP backend fails closed if quantities, coefficients or its conservative
worst-case objective exceed the exact floating-point integer range. Automatic
fallback uses the existing MILP implementation and its existing numerical limits.
`OperationsPipeline` continues using its unchanged default MILP path; this is an
explicit stateless backend option, not an automatic change to existing workflows.

```bash
python -m pip install -e '.[dev]'
assumption-ops-solvers --orders 25 100 500 1000 --seeds 1729 1730 1731 \
  --repeats 3 --output-dir solver_results
```

The scaling runner warms both backends outside measured runs, alternates execution
order, and retains paired input fingerprints, statuses, objectives, metrics and
wall times. It independently checks feasibility and accounting after each timed
call. Allocation arcs may differ under ties; equivalent quality means equal
proven-optimal objective values on identical inputs. Pairs without proven equal
quality are retained in raw reports but excluded from timing summaries. Both
backends are supplied the same time limit per attempt; `auto` fallback may consume
a second solver attempt and therefore is not a combined wall-time SLA.

The measured development run in
[`examples/solver_scaling_summary.json`](examples/solver_scaling_summary.json)
contains 36 verified pairs across four sizes, three seeds and three repetitions.
At 500 orders, medians were approximately 141 ms MILP and 101 ms LP; at 1,000
orders, 587 ms and 356 ms. At 25 orders MILP was faster, and 100 orders were close.
These synthetic workloads measure this machine and implementation, not production
latency or guaranteed speedups. The CLI exports full JSON and per-run CSV records.

Public data investigation is recorded in
[`docs/data_source_audit.md`](docs/data_source_audit.md). No newly investigated
source was admitted as verified observed inventory plus uncensored demand. The
existing warehouse projection retains its provisional flag. The next data adapter
requires documented inventory timing and receipt/order semantics; snapshots,
sales and stockout flags will not be converted into fictional receipts or latent
demand.
