"""Deterministic synthetic workload measurements; no claim of business ROI.

Run: python scripts/benchmark.py --orders 25 100 --repeats 3
The benchmark measures seed/plan/compare wall time on the current machine.
It keeps raw repetitions and reports medians; timings are not CI assertions.
"""

from __future__ import annotations

import argparse
import json
import platform

import scipy
from random import Random
from statistics import median
from time import perf_counter

from assumption_ops import OperationsPipeline, Order, Policy, Scenario, Supply


def workload(order_count: int, seed: int):
    rng = Random(seed)
    supply_count = max(8, order_count // 3)
    supplies = [
        Supply(
            f"lot-{i}", f"SKU-{i % 4}", rng.randint(8, 20), rng.choice([0, 2, 4]), rng.randint(1, 4)
        )
        for i in range(supply_count)
    ]
    orders = [
        Order(f"order-{i}", f"SKU-{i % 4}", rng.randint(3, 8), rng.randint(3, 7), rng.randint(1, 3))
        for i in range(order_count)
    ]
    policy = Policy(substitutions={"SKU-0": ("SKU-1",)})
    return supplies, orders, policy


def measure(order_count: int, seed: int):
    supplies, orders, policy = workload(order_count, seed)
    with OperationsPipeline() as ops:
        start = perf_counter()
        ops.seed(supplies, orders, policy)
        seeded = perf_counter()
        baseline = ops.plan()
        planned = perf_counter()
        comparison = ops.compare_scenarios(
            [
                Scenario("delay one lot", {"supply:lot-0": {"available_day": 12}}),
                Scenario(
                    "add ten units", {"supply:lot-0": {"quantity": supplies[0].quantity + 10}}, 15
                ),
                Scenario("permit late supply", {"policy": {"config": {"allow_late": True}}}),
            ],
            previous=baseline.plan,
        )
        compared = perf_counter()
        assert ops.store.revision == comparison.evidence_revision
        assert len(ops.decisions()) == len(baseline.decisions)
        return {
            "orders": order_count,
            "supplies": len(supplies),
            "seed": seed,
            "seed_seconds": seeded - start,
            "plan_seconds": planned - seeded,
            "compare_four_solves_seconds": compared - planned,
            "baseline_unfilled_units": sum(baseline.plan.unfilled.values()),
            "baseline_score": baseline.plan.objective,
            "ranked_scenarios": comparison.rows(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.orders):
        parser.error("Order counts and repeats must be positive")
    runs, summary = [], []
    for n in args.orders:
        batch = [measure(n, 1729 + i) for i in range(args.repeats)]
        runs.extend(batch)
        summary.append(
            {
                "orders": n,
                "repetitions": args.repeats,
                **{
                    field: median(r[field] for r in batch)
                    for field in ("seed_seconds", "plan_seconds", "compare_four_solves_seconds")
                },
            }
        )
    print(
        json.dumps(
            {
                "synthetic": True,
                "environment": {
                    "python": platform.python_version(),
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                },
                "summary": summary,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
