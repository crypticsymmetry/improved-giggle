"""Paired, warmed backend measurements on reproducible synthetic transport inputs."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import isclose
import platform
from random import Random
from statistics import median
from time import perf_counter
from typing import Sequence

import scipy

from .evaluation import evaluate_plan
from .optimizer import Order, Policy, Supply, optimize


def scaling_workload(
    order_count: int, seed: int
) -> tuple[tuple[Supply, ...], tuple[Order, ...], Policy]:
    if type(order_count) is not int or order_count < 1:
        raise ValueError("order_count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = Random(seed)
    skus = tuple(f"SKU-{index}" for index in range(16))
    supplies = tuple(
        Supply(f"lot-{i}", skus[i % 16], rng.randint(8, 24), rng.randint(0, 8), rng.randint(1, 9))
        for i in range(max(16, order_count // 3))
    )
    orders = tuple(
        Order(f"order-{i}", skus[i % 16], rng.randint(3, 9), rng.randint(2, 10), rng.randint(1, 3))
        for i in range(order_count)
    )
    policy = Policy(
        substitutions={sku: (skus[(index + 1) % 16],) for index, sku in enumerate(skus)},
        allow_late=True,
        late_penalty=7,
        substitution_penalty=3,
        unfilled_penalty=300,
    )
    return supplies, orders, policy


def _values(values: Sequence[int], name: str, minimum: int) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty list or tuple")
    if any(type(value) is not int or value < minimum for value in values):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(values)


def benchmark_solvers(
    order_counts: Sequence[int] = (25, 100, 500),
    seeds: Sequence[int] = (1729, 1730, 1731),
    repeats: int = 3,
) -> dict:
    """Keep raw paired observations; compare times only at proven equal quality.

    Timings cover the complete backend call. Independent external accounting is
    checked afterward. Both backends are warmed outside measurements, and their
    timed execution order alternates. Allocation ties need not use the same arcs.
    """
    counts, seeds = _values(order_counts, "order_counts", 1), _values(seeds, "seeds", 0)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    warm = scaling_workload(8, 0)
    for solver in ("milp", "lp"):
        optimize(*warm, solver=solver)
    runs = []
    pair_index = 0
    for count in counts:
        for seed in seeds:
            supplies, orders, policy = scaling_workload(count, seed)
            payload = {
                "supplies": [asdict(s) for s in supplies],
                "orders": [asdict(o) for o in orders],
                "policy": asdict(policy),
            }
            fingerprint = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            eligible_arcs = sum(
                1
                for order in orders
                for supply in supplies
                if supply.sku == order.sku or supply.sku in policy.substitutions[order.sku]
            )
            for repeat in range(repeats):
                execution_order = ("milp", "lp") if pair_index % 2 == 0 else ("lp", "milp")
                pair = []
                for position, solver in enumerate(execution_order):
                    started = perf_counter()
                    plan = optimize(supplies, orders, policy, solver=solver)
                    elapsed = perf_counter() - started
                    evaluated = evaluate_plan(supplies, orders, policy, plan)
                    pair.append(
                        {
                            "orders": count,
                            "supplies": len(supplies),
                            "eligible_arcs": eligible_arcs,
                            "seed": seed,
                            "repeat": repeat,
                            "pair": pair_index,
                            "position": position,
                            "solver": solver,
                            "input_fingerprint": fingerprint,
                            "elapsed_seconds": elapsed,
                            "status": plan.status,
                            "objective": plan.objective,
                            "metrics": evaluated.metrics,
                        }
                    )
                equivalent = all(row["status"] == "optimal" for row in pair) and isclose(
                    pair[0]["objective"], pair[1]["objective"], rel_tol=1e-9, abs_tol=1e-6
                )
                if all(row["status"] == "optimal" for row in pair) and not equivalent:
                    raise RuntimeError("Verified optimal backends disagree on a shared input")
                for row in pair:
                    row["proven_equal_quality"] = equivalent
                runs.extend(pair)
                pair_index += 1
    summary = []
    for count in counts:
        selected = [row for row in runs if row["orders"] == count]
        verified = [row for row in selected if row["proven_equal_quality"]]
        times = {
            solver: [row["elapsed_seconds"] for row in verified if row["solver"] == solver]
            for solver in ("milp", "lp")
        }
        milp_median = median(times["milp"]) if times["milp"] else None
        lp_median = median(times["lp"]) if times["lp"] else None
        summary.append(
            {
                "orders": count,
                "seeds": len(seeds),
                "repeats_per_seed": repeats,
                "verified_pairs": len(verified) // 2,
                "unverified_pairs": (len(selected) - len(verified)) // 2,
                "milp_median_seconds": milp_median,
                "lp_median_seconds": lp_median,
                "ratio_milp_to_lp_medians": milp_median / lp_median
                if milp_median is not None and lp_median
                else None,
            }
        )
    return {
        "synthetic": True,
        "external_dispatch": False,
        "environment": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "methodology": [
            "Both solvers warmed outside timing; execution order alternates between pairs.",
            "Identical inputs, eligibility and objective weights; no previous allocation.",
            "Parity means independently feasible plans with proven optimal equal scores, not equal arcs.",
            "Timing summaries exclude pairs without proven equal quality.",
            "Ratios of medians are descriptive for this machine and synthetic workload, not guarantees.",
            "Backend time includes its internal validation; external accounting is untimed.",
        ],
        "summary": summary,
        "runs": runs,
    }
