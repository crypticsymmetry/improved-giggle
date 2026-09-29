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
