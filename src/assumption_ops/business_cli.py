"""Validate or compare local timestamped business-pilot CSV bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .business import compare_business_pilot, compile_business_pilot
from .business_io import load_business_pilot, write_business_csv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("business_results"))
    args = parser.parse_args(argv)
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
