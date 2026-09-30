"""Export paired LP/MILP measurements from an installed checkout."""

import argparse
import csv
import json
from pathlib import Path

from .scaling import benchmark_solvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[25, 100, 500])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("solver_results"))
    args = parser.parse_args()
    report = benchmark_solvers(args.orders, args.seeds, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scaling.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        {key: value for key, value in row.items() if key != "metrics"} for row in report["runs"]
    ]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
