"""Deterministic synthetic business-shaped fixtures, never production evidence."""

from __future__ import annotations

from random import Random

from .events import EvidenceUpdate
from .optimizer import Order, Policy, Supply
from .replay import ReplayBatch, ReplayCase


def synthetic_cases(order_count: int = 20, seed: int = 1729) -> tuple[ReplayCase, ...]:
    """Create four controlled regimes with three absolute-update batches each.

    Quantities and priorities vary reproducibly. These are illustrative workloads
    with no commitments, calibrated costs, empirical service claims, or ERP writes.
    """
    if type(order_count) is not int or order_count < 1:
        raise ValueError("order_count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random = Random(seed)
    skus = ("A", "B", "C")
    orders = tuple(
        Order(
            f"order-{index:04d}",
            skus[index % len(skus)],
            random.randint(4, 12),
            random.randint(2, 6),
            random.randint(1, 4),
        )
        for index in range(order_count)
    )
    supplies = []
    for sku in skus:
        demand = sum(order.quantity for order in orders if order.sku == sku)
        cost = random.randint(2, 6)
        supplies.extend(
            (
                Supply(f"stock-{sku}", sku, max(1, demand // 4), 0, cost),
                Supply(f"shipment-{sku}", sku, max(1, demand * 4 // 5), 2, cost),
                Supply(f"backup-{sku}", f"{sku}-alt", max(1, demand // 3), 1, cost + 8),
            )
        )
    supplies = tuple(supplies)
    policy = Policy(
        substitutions={sku: (f"{sku}-alt",) for sku in skus},
        allow_late=True,
        late_penalty=20,
        substitution_penalty=8,
        disruption_penalty=3,
        unfilled_penalty=500,
    )
    cases = []
    supply_by_id = {supply.id: supply for supply in supplies}
    for regime in ("supplier_delay", "stock_loss", "demand_surge", "cost_shock"):
        batches = []
        for step in range(3):
            source = f"SYNTHETIC {regime} confirmation {step + 1}; seed={seed}"
            updates = []
            if regime == "supplier_delay":
                for offset, sku in enumerate(skus):
                    updates.append(
                        EvidenceUpdate(
                            f"supply:shipment-{sku}",
                            "available_day",
                            (4, 8, 3)[step] + offset,
                            source,
                        )
                    )
            elif regime == "stock_loss":
                for sku in skus:
                    original = supply_by_id[f"stock-{sku}"].quantity
                    updates.append(
                        EvidenceUpdate(
                            f"supply:stock-{sku}",
                            "quantity",
                            (0, original // 2, original * 3 // 4)[step],
                            source,
                        )
                    )
            elif regime == "demand_surge":
                for order in orders[: max(1, order_count // 3)]:
                    updated_quantity = order.quantity + (step + 1) * max(1, order.quantity // 2)
                    updates.append(
                        EvidenceUpdate(f"order:{order.id}", "quantity", updated_quantity, source)
                    )
            else:
                for sku in skus:
                    original = supply_by_id[f"shipment-{sku}"].unit_cost
                    updates.append(
                        EvidenceUpdate(
                            f"supply:shipment-{sku}",
                            "unit_cost",
                            original + (10, 30, 15)[step],
                            source,
                        )
                    )
            batches.append(ReplayBatch(f"{regime}-event-{step + 1}", tuple(updates)))
        cases.append(ReplayCase(regime, supplies, orders, policy, tuple(batches), True, seed))
    return tuple(cases)
