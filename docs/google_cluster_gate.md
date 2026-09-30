# Certify greedy placement before invoking MILP

The gate evaluates priority-first fit, best fit and stability-aware best fit on the same CPU/memory placement problem. It independently verifies each integer proposal and selects the lowest score, with deterministic method-order tie breaking. An LP-derived exact lower bound can certify that proposal as optimal. When certification fails, the gate invokes MILP and retains the best verified candidate. This is useful when greedy often reaches the optimum but difficult packing cases still require optimization.

[Run in Colab](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/google_cluster_gate.ipynb), or install the package and run:

```bash
assumption-ops-cluster --holdout --lp-gated --machines-grid 1 2 4 8 \
  --tasks 64 --time-limit 5 --output-dir cluster_gate_results
```

No private dataset, API key or GPU is required. Downloads use the same two bounded, checksum-verified Google trace objects as the [placement benchmark](google_cluster_benchmark.md). To reuse them, supply `--machine-events cluster_data/google_machine_events.csv.gz` and `--task-events cluster_data/google_task_events.csv.gz`. The notebook accepts `ASSUMPTION_OPS_CLUSTER_DIR` while preserving checksum verification.

## Certificate and routes

The gate never certifies optimality from a floating-point near-equality test. The placement model has integer resource coefficients and an integer objective, including migration penalties. Write its LP objective as `K + c·x`, with task equalities `E x = d`, resource inequalities `R x ≤ b`, and variable bounds `0 ≤ x ≤ 1`. For any equality multipliers `y` and nonpositive resource multipliers `z`, a valid lower bound is:

```text
B = K + y·d + z·b + Σ_j min(0, c_j − (Eᵀy + Rᵀz)_j)
```

The implementation obtains multipliers from the LP solver, converts finite values to rational numbers, clamps resource multipliers to nonpositive values, and evaluates this expression using exact integer/rational arithmetic. The residual term minimizes each variable over its box and repairs any dual infeasibility introduced by rationalization. It therefore does not depend on floating-point solver tolerances to establish the bound. The integer optimum is at least `ceil(B)`. If this integer lower bound equals an independently verified greedy score, that greedy proposal is optimal for the model.

The report records three possible routes:

| Route | Meaning |
|---|---|
| `lp_certificate` | Integer lower bound equals the best greedy score; the gated policy avoids MILP and returns a certified optimum. |
| `milp` | A gap remains or the certificate is unavailable; MILP is invoked and the best verified candidate is retained. |
| `greedy_fallback` | MILP reaches its limit without an incumbent; the verified greedy proposal is retained without claiming optimality. |

Malformed, fractional or infeasible solver proposals fail validation rather than entering the fallback. A verified time-limited MILP incumbent can be worse than greedy; the gate retains greedy in that case. A claimed MILP optimum worse than a verified greedy proposal, or a lower bound above a verified feasible score, is an inconsistency and fails closed. Fractional LP assignments are never rounded into plans or dispatched.

`gate.certified_optimal` distinguishes a proven optimum from a feasible result, regardless of route. Gate diagnostics preserve the reason, selected heuristic, MILP invocation flag, exact rational lower bound, its integer ceiling, and LP solver status. The time limit applies separately to LP and MILP calls; it is not a total wall-clock deadline.

For standalone use, `optimize_placement_gated(machines, tasks, policy, previous, time_limit=...)` supports previous placements and integer migration penalties. It returns the serialized placement, independent evaluation, gate diagnostics and elapsed time. The holdout experiment deliberately uses no previous placement, so its migration costs are zero and stability-aware best fit equals ordinary best fit.

## Evaluation and outputs

The notebook retains the [job-disjoint protocol](google_cluster_holdout.md): one development, one validation and six holdout submission windows; the fixed 1/2/4/8-machine grid; and at most 64 admitted task identities per window. Later cohorts exclude every job admitted earlier, including departed tasks. Empty cohorts remain explicit and do not enter case denominators. Settings do not change in response to scores.

The evaluation retains a separate MILP reference alongside all greedy methods and the gated method. Split summaries report gate routes, gated MILP invocations, certificate counts and quality relative to that reference. Avoided MILP calls describe the gated policy only. The entire evaluation still invokes the reference MILP and LP checks for other methods, so these counts do not represent all solver calls avoided by the notebook. Elapsed times are descriptive; a single pass with ordered calls and warm caches does not establish a wall-time speedup.

With `--lp-gated`, the CLI exports `cluster_gate_report.json` and `cluster_gate_comparison.csv`, preserving the existing holdout filenames for runs without the gate. The notebook exports `gate.json` and `gate.csv` in `cluster_gate_results`. JSON preserves source hashes, settings, cohort metadata, input fingerprints, all proposals and evaluations, and gate certificate diagnostics. CSV includes gate route, invocation and certificate columns. Alternative scores are not an additive execution ledger.

## Checked public-data results

The fixed default protocol produced the following route counts:

| Split | Evaluated cases | LP certificate route | Gated MILP invocations |
|---|---:|---:|---:|
| Development | 4 | 1 | 3 |
| Validation | 4 | 0 | 4 |
| Holdout | 20 | 12 | 8 |

All 28 gated scores matched the separately solved optimal MILP reference. No case used the no-incumbent fallback. On the 20 evaluated holdout cases, certificates avoided 12 of the gated policy's 20 potential MILP calls, or 60%. This describes policy route counts; the complete benchmark still solved its reference cases and does not establish a wall-time speedup. Empty cohorts remain outside these denominators.

## Interpretation

Certification establishes optimality only for the explicitly modeled CPU/memory integer problem. Publisher values are normalized requests/limits, not measured consumption. Selected machines contribute full recorded capacity; existing production occupancy is not reconstructed. Unsupported affinity, disk, networking, durations, general machine attributes, ingestion delay and overcommit remain excluded. No live infrastructure action is performed.

Windows share one trace shard and infrastructure, and capacity cases reuse their window's tasks. Job separation does not establish statistical independence or cross-cluster generalization. Scores measure feasible proposed allocations, not observed completion, production latency, superiority to Borg or business ROI. See the [source details and attribution](google_cluster_benchmark.md) for the public data provenance.
