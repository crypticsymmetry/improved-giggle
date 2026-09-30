# Controlled Google-cluster benchmark

This extension tests indivisible CPU/memory task placement on bounded records from Google's public 2011 production cluster trace. It provides a second real-data domain without requiring private inventory exports. It evaluates alternative proposed placements on a deliberately restricted model; it does not reproduce Borg, dispatch jobs, or measure production savings.

Run the [Colab notebook](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/google_cluster_benchmark.ipynb), or install the repository and run:

```bash
assumption-ops-cluster --machines 4 --tasks 64 \
  --cutoffs-us 900000000 1200000000 1800000000 \
  --migration-penalty 1 --time-limit 30 \
  --output-dir cluster_results
```

The CLI downloads missing pinned files into `cluster_data`. Existing files must match exact sizes and SHA-256 digests. To use previously downloaded files:

```bash
assumption-ops-cluster \
  --machine-events cluster_data/google_machine_events.csv.gz \
  --task-events cluster_data/google_task_events.csv.gz
```

The outputs are `cluster_report.json` and `cluster_comparison.csv`. JSON preserves input fingerprints, source metadata, integer placements, independent evaluations, solver status, LP bounds and assumption-support explanations. CSV contains one row per method and cutoff; rows are alternatives and must not be summed as cumulative service. The notebook writes equivalent outputs named `benchmark.json` and `benchmark.csv`.

## Source and bounded download

Publisher documentation: [Google ClusterData2011-2](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md). The publisher licenses the trace and documentation under CC-BY; consult its [repository](https://github.com/google/cluster-data) for attribution requirements and schema documentation.

| Input | Public object | Compressed bytes |
|---|---|---:|
| Machine events | `machine_events/part-00000-of-00001.csv.gz` | 347,211 |
| Task events | `task_events/part-00000-of-00500.csv.gz` | 4,139,742 |

Objects are hosted under `https://storage.googleapis.com/clusterdata-2011-2/`. The downloader accepts only these pinned objects, checks expected size and SHA-256, caps the download, and atomically replaces a temporary file after verification. Parsing independently limits decompressed bytes and row count. This approximately 4.5 MB selection does not download the full approximately 41 GB release. No authentication or GPU is needed.

For offline notebook validation, set `ASSUMPTION_OPS_CLUSTER_DIR` to a directory containing the two named, checksum-matching files. This setting does not disable checksum verification.

## Snapshot semantics

The parser processes source events at or before each cutoff. Machines are selected deterministically by numeric identifier. A bounded task cohort starts with new submissions at or after 600,000,000 microseconds. Tasks require known positive CPU and memory requests; updates cannot borrow later values. Terminal fail, finish, kill and lost events deactivate selected tasks. The cohort remains bounded by the first admitted identities; termination does not refill it with later tasks. Flagged different-machine constraints are excluded rather than silently modeled as unconstrained placement. Metadata records the selection, exclusions and source digests.

The first task shard covers approximately the first 93.5 minutes; default cutoffs of 900, 1,200 and 1,800 seconds are within that coverage (5, 10 and 20 minutes after the trace start at 600 seconds). Cutoffs must increase strictly and remain within the parsed source range. Timestamps determine simulated event-time visibility. The release has no per-record ingestion timestamps, so this does not reconstruct actual observation delay.

CPU and memory values are publisher-normalized requests/limits and capacities. They are not literal CPU-core counts, byte counts, or measured usage. Requests round up and capacities round down at a scale of 1,000,000 integer ticks; reported feasibility applies to this conservative projection. Google may overcommit resources. This model reserves requested CPU and memory under hard limits and performs no overcommit prediction.

The selected machines' full recorded capacity is made available to the modeled cohort. Existing production occupancy is not reconstructed, and that capacity is not claimed to have been actually free. Disk, networking, machine attributes, general affinity, service dependencies and durations are outside the model.

## Optimization and baselines

A task is assigned wholly to one eligible machine or remains pending. Each machine must satisfy both CPU and memory capacity constraints. The score is pending admission priority plus migration cost. Trace priority plus one is an explicit assumed cardinal weight; it is not money or a reproduction of Google's admission policy.

All methods receive the same validated snapshot and policy:

- MILP optimizes indivisible placements with SciPy/HiGHS. `optimal` indicates a solver-certified optimum; `feasible_limit` is an independently verified incumbent without an optimality claim.
- Priority-first greedy and best-fit greedy provide executable integer baselines. A stability-aware best-fit baseline prefers the prior machine when feasible, explicitly accounting for the migration-reference objective. The original heuristics provide comparison points that do not make that preference.
- The LP relaxation can split assignments fractionally. Its objective is reported as a lower bound only after optimal status and independent primal/objective checks. It is never rounded or dispatched as a task plan.

The comparison independently reconstructs integer coverage, CPU/memory use, pending priority, migration cost and total score. No task can be both assigned and pending, and no task can be assigned to multiple machines. A gap to an available LP bound measures remaining room relative to the fractional relaxation; it does not imply the LP solution is executable or that the integer optimum equals that bound.

The first MILP proposal becomes the shared frozen migration reference for later snapshots. Each method therefore receives the same history assumption. These comparisons do not evolve each baseline's own execution history, and historical task completion remains observed input independent of proposed placements. Different optimal placements can tie in cost.

## Evidence and replanning

ATMS supports justify local resource feasibility. Each assigned task's support contains its machine-capacity fact and every colocated task-request fact. If a neighboring task's request changes, all assignments sharing that capacity justification can become unsupported. Removal or capacity changes also deactivate the affected prior facts.

The impact report compares the preceding optimized proposal with the next snapshot, while the migration-cost reference stays frozen at the first proposal. These are distinct histories for distinct questions. Evidence support does not certify global optimality; a newly available machine or changed priority can improve a plan without invalidating its local resource support. Every snapshot is reoptimized.

No operational approvals, reservations or external scheduling writes occur. Snapshot task counts and admitted priority overlap and are not cumulative jobs completed. The report summarizes final method evaluations and separately records support invalidation counts as reasoning activity, not service.

## Interpreting the benchmark

The result establishes whether methods produce feasible proposals and how their configured scores and LP gaps differ on identical real-source projections. It can expose an integrality gap that the transportation-only allocator does not exhibit. It does not establish superiority to Borg, scheduler throughput, end-to-end job latency, causal availability improvement or ROI. Solver wall times are machine-dependent, and a time-limited incumbent must not be called optimal. Retain source digests, input fingerprints, environment and model limitations when sharing results.

The [checked example summary](../examples/google_cluster_benchmark_summary.json) preserves the four-machine notebook results and a first-cutoff sensitivity grid over 1, 2, 4 and 8 selected machines. The grid includes cases where every method ties. It uses the same bounded task identities, and is a capacity sensitivity experiment rather than independent held-out evaluation.
