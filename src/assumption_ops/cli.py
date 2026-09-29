"""Small JSON interface for reproducible local scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .optimizer import Order, Policy, Supply
from .pipeline import OperationsPipeline


def load_scenario(path: str) -> tuple[list[Supply], list[Order], Policy]:
    data = json.loads(Path(path).read_text())
    return (
        [Supply(**r) for r in data["supplies"]],
        [Order(**r) for r in data["orders"]],
        OperationsPipeline._policy(data.get("policy", {})),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evidence-backed supplier allocation")
    parser.add_argument(
        "scenario",
        help="JSON initializer for a new database; existing databases keep accepted state",
    )
    parser.add_argument(
        "--db", default=":memory:", help="SQLite file; defaults to disposable memory"
    )
    parser.add_argument("--delay-supply", help="Supply ID to delay in a what-if demonstration")
    parser.add_argument("--available-day", type=int, help="Revised relative arrival day")
    args = parser.parse_args(argv)
    if bool(args.delay_supply) != (args.available_day is not None):
        parser.error("--delay-supply and --available-day must be used together")
    supplies, orders, policy = load_scenario(args.scenario)
    with OperationsPipeline(args.db) as pipeline:
        if not pipeline.is_seeded:
            pipeline.seed(supplies, orders, policy)
        result = {"baseline": pipeline.plan().to_dict()}
        if args.delay_supply:
            entity = f"supply:{args.delay_supply}"
            evidence_id = pipeline.propose(
                entity, "available_day", args.available_day, "CLI what-if"
            )
            pipeline.accept(evidence_id)
            result["impact_before_resolution"] = pipeline.impact()
            pipeline.resolve(entity, "available_day", evidence_id)
            result["revised"] = pipeline.plan().to_dict()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
