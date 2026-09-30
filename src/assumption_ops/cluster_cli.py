"""Download verified public Google traces and compare controlled task placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .cluster_benchmark import run_cluster_benchmark, write_cluster_csv
from .cluster_data import download_cluster


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-events", type=Path)
    parser.add_argument("--task-events", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("cluster_data"))
    parser.add_argument(
        "--cutoffs-us", type=int, nargs="+", default=[900_000_000, 1_200_000_000, 1_800_000_000]
    )
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--migration-penalty", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_results"))
    args = parser.parse_args(argv)
    if (args.machine_events is None) != (args.task_events is None):
        parser.error("Supply both --machine-events and --task-events, or neither")
    if args.machine_events is None:
        paths = download_cluster(args.data_dir)
        args.machine_events, args.task_events = paths["machines"], paths["tasks"]
    report = run_cluster_benchmark(
        args.machine_events,
        args.task_events,
        cutoffs_us=tuple(args.cutoffs_us),
        max_machines=args.machines,
        max_tasks=args.tasks,
        migration_penalty=args.migration_penalty,
        time_limit=args.time_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "cluster_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
