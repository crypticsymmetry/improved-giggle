"""Reproduce a provisional allocation projection of the public warehouse data."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import scipy

from .baselines import compare_allocators
from .warehouse_data import download_warehouse, read_warehouse, warehouse_scenario


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("warehouse_results/source.accdb"))
    parser.add_argument("--download", action="store_true", help="Download the checksum-pinned file")
    parser.add_argument(
        "--allow-inferred-demand",
        action="store_true",
        help="Acknowledge the documented, author-unconfirmed demand column mapping",
    )
    parser.add_argument("--stock-fractions", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--output-dir", type=Path, default=Path("warehouse_results"))
    args = parser.parse_args(argv)
    if args.download:
        download_warehouse(args.data)
    data = read_warehouse(args.data, allow_inferred_demand=args.allow_inferred_demand)
    runs, rows = [], []
    for index, fraction in enumerate(args.stock_fractions):
        scenario = warehouse_scenario(data, stock_fraction=fraction)
        comparison = compare_allocators(scenario.supplies, scenario.orders, scenario.policy)
        run = {
            "run": index,
            "stock_fraction": fraction,
            "modeled_supply_stress": fraction != 1.0,
            "orders": len(scenario.orders),
            "supply_lots": len(scenario.supplies),
            "requested_units": sum(order.quantity for order in scenario.orders),
            "available_units": sum(supply.quantity for supply in scenario.supplies),
            "policy": asdict(scenario.policy),
            **comparison,
        }
        runs.append(run)
        for result in comparison["results"]:
            rows.append(
                {
                    "run": index,
                    "stock_fraction": fraction,
                    "method": result["method"],
                    "status": result["status"],
                    "weighted_score": result["evaluation"]["costs"]["total"],
                    "gap_to_lp_bound": result["gap_to_lp_bound"],
                    "lp_status": comparison["lp_relaxation"]["status"],
                    "lp_lower_bound": comparison["lp_relaxation"]["lower_bound"],
                    "milp_proven_optimal": comparison["milp_proven_optimal"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    **result["evaluation"]["metrics"],
                }
            )
    report = {
        "provisional": True,
        "external_dispatch": False,
        "source": data.provenance,
        "environment": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Demand mapping is an explicitly acknowledged interpretation; not author-confirmed.",
            "Projected SKU allocation omits original warehouse geometry, lanes, batches and storage costs.",
            "All methods know the same full horizon; this is an offline planning comparison, not forecasting.",
            "Costs and priorities are illustrative; acquisition cost is zero because it is not supplied.",
            "Stock fractions below one are modeled capacity stress, not observed shortages or disruptions.",
            "LP bounds and MILP can tie on the transportation model; ties are not evidence of unique superiority.",
        ],
        "runs": runs,
    }
    # Solve before exporting; local filesystem report writes are not atomic as a suite.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "methods.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: value for key, value in report.items() if key != "runs"}
    summary["run_settings"] = [
        {
            key: run[key]
            for key in (
                "run",
                "stock_fraction",
                "modeled_supply_stress",
                "orders",
                "supply_lots",
                "requested_units",
                "available_units",
                "policy",
            )
        }
        for run in runs
    ]
    summary["results"] = rows
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
