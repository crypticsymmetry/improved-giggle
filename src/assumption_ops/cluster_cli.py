"""Download verified public Google traces and compare controlled task placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .cluster_benchmark import run_cluster_benchmark, write_cluster_csv
from .cluster_data import download_cluster
from .cluster_holdout import run_cluster_holdout, write_cluster_holdout_csv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-events", type=Path)
    parser.add_argument("--task-events", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("cluster_data"))
    parser.add_argument(
        "--holdout", action="store_true", help="Run the fixed job-disjoint window protocol"
    )
    parser.add_argument(
        "--machines-grid", type=int, nargs="+", help="Holdout machine-count grid (default: 1 2 4 8)"
    )
    parser.add_argument("--cutoffs-us", type=int, nargs="+")
    parser.add_argument("--machines", type=int)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--migration-penalty", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_results"))
    args = parser.parse_args(argv)
    if (args.machine_events is None) != (args.task_events is None):
        parser.error("Supply both --machine-events and --task-events, or neither")
    if args.holdout and any(
        value is not None for value in (args.machines, args.cutoffs_us, args.migration_penalty)
    ):
        parser.error(
            "Holdout uses fixed windows and independent cases; use --machines-grid instead of --machines/--cutoffs-us/--migration-penalty"
        )
    if not args.holdout and args.machines_grid is not None:
        parser.error("--machines-grid requires --holdout")
    if args.machine_events is None:
        paths = download_cluster(args.data_dir)
        args.machine_events, args.task_events = paths["machines"], paths["tasks"]
    if args.holdout:
        report = run_cluster_holdout(
            args.machine_events,
            args.task_events,
            machine_counts=tuple(args.machines_grid)
            if args.machines_grid is not None
            else (1, 2, 4, 8),
            max_tasks=args.tasks,
            time_limit=args.time_limit if args.time_limit is not None else 5,
        )
    else:
        report = run_cluster_benchmark(
            args.machine_events,
            args.task_events,
            cutoffs_us=tuple(args.cutoffs_us)
            if args.cutoffs_us is not None
            else (900_000_000, 1_200_000_000, 1_800_000_000),
            max_machines=args.machines if args.machines is not None else 8,
            max_tasks=args.tasks,
            migration_penalty=args.migration_penalty if args.migration_penalty is not None else 1,
            time_limit=args.time_limit if args.time_limit is not None else 30,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        "cluster_holdout_report.json" if args.holdout else "cluster_report.json"
    )
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.holdout:
        write_cluster_holdout_csv(report, args.output_dir / "cluster_holdout_comparison.csv")
    else:
        write_cluster_csv(report, args.output_dir / "cluster_comparison.csv")
    print(
        json.dumps(
            {
                "report": str(path),
                "controlled_model": report["controlled_model"],
                "summary": report["summary"],
                "limitations": report["limitations"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
