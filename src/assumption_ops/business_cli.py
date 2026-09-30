"""Validate or compare local timestamped business-pilot CSV bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .business import compare_business_pilot, compile_business_pilot
from .business_io import load_business_pilot, write_business_csv
from .business_template import create_business_template


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--init-dir", type=Path, help="Create a new unfilled export bundle")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.init_dir is not None:
        if args.validate_only or args.output_dir is not None:
            parser.error("--init-dir cannot be combined with evaluation options")
        manifest = create_business_template(args.init_dir)
        print(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "data_request": str(manifest.parent / "DATA_REQUEST.md"),
                    "ready_for_evaluation": False,
                    "next_step": "Fill source records, provenance and policy settings; then run --validate-only.",
                },
                indent=2,
            )
        )
        return
    args.output_dir = args.output_dir or Path("business_results")
    pilot = load_business_pilot(args.manifest)
    if args.validate_only:
        snapshots = compile_business_pilot(pilot)
        print(
            json.dumps(
                {
                    "name": pilot.name,
                    "group_id": pilot.group_id,
                    "synthetic": pilot.synthetic,
                    "external_dispatch": False,
                    "snapshots": [
                        {
                            "snapshot_id": s.snapshot_id,
                            "cutoff": s.cutoff,
                            "inventory_units": sum(v.quantity for v in s.supplies),
                            "open_order_units": sum(v.quantity for v in s.orders),
                            "metadata": s.metadata,
                        }
                        for s in snapshots
                    ],
                },
                indent=2,
                allow_nan=False,
            )
        )
        return
    report = compare_business_pilot(pilot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "business_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_business_csv(report, args.output_dir / "business_comparison.csv")
    print(
        json.dumps(
            {
                "report": str(path),
                "name": report["name"],
                "synthetic": report["synthetic"],
                "summary": report["summary"],
                "limitations": report["limitations"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
