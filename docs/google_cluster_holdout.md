# Job-disjoint Google cluster holdout

This benchmark checks whether resource-placement improvements persist across separate submission windows and job identities in Google's public 2011 trace. It uses the same bounded, checksum-verified files as the [resource-placement benchmark](google_cluster_benchmark.md). No private data, account, API key or GPU is needed.

[Run in Colab](https://colab.research.google.com/github/crypticsymmetry/improved-giggle/blob/main/notebooks/google_cluster_holdout.ipynb), or install and run:

```bash
assumption-ops-cluster --holdout --machines-grid 1 2 4 8 \
  --tasks 64 --time-limit 5 --output-dir cluster_holdout_results
```

To reuse downloaded inputs, provide `--machine-events cluster_data/google_machine_events.csv.gz` and `--task-events cluster_data/google_task_events.csv.gz`. The notebook also supports `ASSUMPTION_OPS_CLUSTER_DIR` for offline validation while preserving checksum checks.

## Fixed protocol

Settings are recorded before optimization: 1, 2, 4 and 8 selected machines, at most 64 admitted task identities per window, and a default 5-second MILP limit per case. Held-out results do not calibrate weights, choose a baseline, change the grid or select a preferred result.

Default half-open submission intervals use source microseconds:

| Split | Interval | Minutes after trace start |
|---|---|---|
| Development | [600,000,000, 900,000,000) | [0, 5) |
| Validation | [1,200,000,000, 1,500,000,000) | [10, 15) |
| Holdout 1 | [1,800,000,000, 2,100,000,000) | [20, 25) |
| Holdout 2 | [2,400,000,000, 2,700,000,000) | [30, 35) |
| Holdout 3 | [3,000,000,000, 3,300,000,000) | [40, 45) |
| Holdout 4 | [3,600,000,000, 3,900,000,000) | [50, 55) |
| Holdout 5 | [4,200,000,000, 4,500,000,000) | [60, 65) |
| Holdout 6 | [4,800,000,000, 5,100,000,000) | [70, 75) |

Later windows exclude every job admitted into an earlier window, including jobs whose selected tasks have departed. Qualifying submissions must be inside the window; their event-time state is reconstructed through its evaluation cutoff. Departures do not refill the bounded cohort. Empty eligible cohorts are explicitly recorded.

Each capacity case receives the same cohort, deterministic machine ordering and objective. Snapshots are independent, with no previous placement and zero migration cost. Proposals do not change observed lifecycle events or future cohort selection. Counts across alternatives must not be summed as completed work.

## Methods and checks

MILP assigns each task wholly to one machine or leaves it pending, under simultaneous CPU and memory limits. Pending priority is minimized; trace priority plus one is an explicitly assumed admission weight. Priority-first fit, best fit and stability-aware best fit provide integer greedy baselines. Without a previous placement, stability-aware best fit equals ordinary best fit here. Their equality is not independent extra evidence.

Every proposal is independently evaluated for task coverage and both resource limits. `optimal` is a solver-certified optimum; `feasible_limit` is a verified incumbent without an optimality claim. An independently checked optimal fractional LP provides a lower bound when available. It is never rounded into a task plan.

Summaries retain strict wins, ties, losses, regret relative to the MILP proposal, optimality counts and LP gaps. A time-limited MILP can lose to a heuristic; that result remains visible. Regret against an uncertified incumbent is a comparison to that incumbent, not proven regret to the unknown optimum. Inspect case statuses and bounds alongside means. Development and validation labels reserve a future calibration workflow; this version performs no fitting or automatic selection.

## Outputs and interpretation

The CLI exports `cluster_holdout_report.json` and `cluster_holdout_comparison.csv`. The notebook exports `holdout.json` and `holdout.csv`. JSON preserves fixed settings, source hashes, cohort metadata, input fingerprints, proposals, evaluations, solver bounds, environment and limitations. CSV rows are alternatives, not an execution ledger.

This is a descriptive job-disjoint check within one approximately 93.5-minute shard. Windows share a trace, infrastructure and workload environment; grid cases reuse their window's tasks. No statistical independence, bootstrap confidence interval, p-value or cross-cluster generalization guarantee is claimed. The development region was previously explored; this extension is reproducible, but is not an untouched external test set or preregistered study.

Publisher CPU/memory values are normalized resource requests/limits, not measured consumption. Selected machines contribute full recorded capacity; existing production occupancy is not reconstructed. Requests round up and capacities down to integer ticks. The model excludes unsupported affinity, disk, networking, durations, general machine attributes and actual ingestion delay, and reserves resources without overcommit. Results measure feasible proposed scores on this projection, not superiority to Borg, actual latency or business ROI.

See the [source and download details](google_cluster_benchmark.md) and [Google publisher documentation](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md) for attribution and schema.
