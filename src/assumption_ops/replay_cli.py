"""Run offline event traces and export per-case results without external dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .replay import run_replay
from .replay_fixtures import synthetic_cases
from .replay_io import load_replay_case, save_replay_case, write_replay_report_csv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, action="append", help="Replay JSON; repeat for multiple cases"
    )
    parser.add_argument("--orders", type=int, default=20, help="Orders per synthetic case")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--output-dir", type=Path, default=Path("replay_results"))
    args = parser.parse_args(argv)
    if args.orders < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("orders must be positive and seeds must be nonnegative")
    if args.case:
        cases = tuple(load_replay_case(path) for path in args.case)
    else:
        cases = tuple(case for seed in args.seeds for case in synthetic_cases(args.orders, seed))
    # Complete all solves before writing reports, so invalid input or a failed
    # solve cannot masquerade as a successful complete suite.
    reports = [run_replay(case) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (case, report) in enumerate(zip(cases, reports), start=1):
        # User-provided case names never become paths.
        stem = f"case-{index:03d}"
        source_path = args.output_dir / f"{stem}.input.json"
        report_path = args.output_dir / f"{stem}.report.json"
        csv_path = args.output_dir / f"{stem}.snapshots.csv"
        save_replay_case(case, source_path)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_replay_report_csv(report, csv_path)
        entries.append(
            {
                "case_name": case.name,
                "synthetic": case.synthetic,
                "seed": case.seed,
                "input": source_path.name,
                "report": report_path.name,
                "snapshots_csv": csv_path.name,
                "summary": report["summary"],
            }
        )
    manifest = {
        "schema_version": 1,
        "external_dispatch": False,
        "notes": [
            "Final service and cost are reported per case; repeated snapshots are not summed.",
            "Cost scores use configured penalty units, not automatically currency or ROI.",
            "Decision counts are review proxies, not measured human review time.",
            "Input synthetic=false is a caller label, not independent source verification.",
            "This manifest lists this invocation only; older output files may remain.",
        ],
        "cases": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
