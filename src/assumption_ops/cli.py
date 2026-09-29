"""JSON and CSV scenario intake, reproducible replanning, and comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .export import write_comparison_csv
from .intake import load_csv_scenario, load_json_scenario, load_policy_json
from .pipeline import OperationsPipeline
from .scenarios import Scenario


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evidence-backed supplier allocation")
    parser.add_argument(
        "scenario", nargs="?", help="JSON initializer; existing databases keep accepted state"
    )
    parser.add_argument("--supplies-csv", help="Supply CSV initializer, paired with --orders-csv")
    parser.add_argument("--orders-csv", help="Order CSV initializer, paired with --supplies-csv")
    parser.add_argument("--policy-json", help="Policy object JSON for CSV intake")
    parser.add_argument(
        "--db", default=":memory:", help="SQLite file; defaults to disposable memory"
    )
    parser.add_argument("--delay-supply", help="Supply ID to delay in a demonstration")
    parser.add_argument("--available-day", type=int, help="Revised relative arrival day")
    parser.add_argument("--compare", help="JSON array of named intervention candidates")
    parser.add_argument("--report-csv", help="Write compact comparison rows to this CSV path")
    args = parser.parse_args(argv)
    if bool(args.delay_supply) != (args.available_day is not None):
        parser.error("--delay-supply and --available-day must be used together")
    if bool(args.supplies_csv) != bool(args.orders_csv):
        parser.error("--supplies-csv and --orders-csv must be used together")
    if bool(args.scenario) == bool(args.supplies_csv):
        parser.error("Supply either a JSON initializer or the two CSV initializers")
    if args.policy_json and not args.supplies_csv:
        parser.error("--policy-json is used with CSV intake")
    if args.report_csv and not args.compare:
        parser.error("--report-csv requires --compare")
    if args.scenario:
        data = load_json_scenario(args.scenario)
    else:
        policy = None
        if args.policy_json:
            policy = load_policy_json(args.policy_json)
        data = load_csv_scenario(args.supplies_csv, args.orders_csv, policy)
    candidates = None
    if args.compare:
        raw = json.loads(Path(args.compare).read_text())
        if not isinstance(raw, list):
            parser.error("Comparison input must be a JSON array")
        candidates = [Scenario(**row) for row in raw]
    with OperationsPipeline(args.db) as pipeline:
        if not pipeline.is_seeded:
            pipeline.seed(data.supplies, data.orders, data.policy)
        baseline = pipeline.plan()
        result = {"baseline": baseline.to_dict()}
        if args.delay_supply:
            entity = f"supply:{args.delay_supply}"
            evidence_id = pipeline.propose(
                entity, "available_day", args.available_day, "CLI confirmed update"
            )
            pipeline.accept(evidence_id)
            result["impact_before_resolution"] = pipeline.impact()
            pipeline.resolve(entity, "available_day", evidence_id)
            result["revised"] = pipeline.plan(previous=baseline.plan).to_dict()
        if candidates is not None:
            comparison = pipeline.compare_scenarios(candidates, previous=baseline.plan)
            result["comparison"] = comparison.to_dict()
            result["comparison_rows"] = comparison.rows()
            if args.report_csv:
                write_comparison_csv(comparison, args.report_csv)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
