"""Validate and calibrate explicitly partitioned offline pilot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .calibration import calibrate
from .calibration_io import load_calibration_config, write_calibration_csv
from .pilot import dataset_summary, load_pilot_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("pilot_results"))
    args = parser.parse_args(argv)
    dataset = load_pilot_dataset(args.manifest)
    summary = dataset_summary(dataset)
    if args.validate_only:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    if args.config is None:
        parser.error("--config is required unless --validate-only is supplied")
    candidates, scoring_policy, constraints = load_calibration_config(args.config)
    report = calibrate(dataset, candidates, scoring_policy, **constraints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    path = args.output_dir / "calibration_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_calibration_csv(report, args.output_dir / "calibration_rankings.csv")
    print(json.dumps({"report": str(path), **report}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
