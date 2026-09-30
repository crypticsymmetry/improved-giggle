# Public data admission audit

This audit records what the available sources establish and what is still missing
for an observed inventory-allocation benchmark. Provider descriptions are source
claims, not independent verification that a dataset is complete or authentic.

| Source | Documented fields | Current admission status |
|---|---|---|
| [Industrial warehouse, DOI 10.5281/zenodo.18229759](https://zenodo.org/records/18229759) | Production, initial inventory and a demand table over a 130-day horizon | Existing adapter remains provisional: the published demand column roles require an author-unconfirmed interpretation. |
| [Retail Transactions and Stocks Data, DOI 10.17632/27x8mjm8k4.1](https://data.mendeley.com/datasets/27x8mjm8k4/1) | Provider describes sales and on-hand stock for 40 stores and 2,326 SKUs; CC BY 4.0 | Public file API returned HTTP 403 during this audit. CSV schema, snapshot timing, units and stock-movement semantics were not inspected; no adapter was added. |
| [FreshRetailNet-50K, official dataset card](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/blob/main/README.md) | Observed sales, hourly stock status and contextual variables | Useful for censored-demand research, but listed stock status is not an on-hand quantity or replenishment record. It is not admitted as observed supply input. |

The Mendeley public listing credits Jimmy Smith (2026). A working file download
and provider documentation would be required before mapping its columns. Listing
availability alone does not establish that stock is measured before or after the
day's transactions.

For an allocation pilot, source semantics must establish:

- A dated opening inventory quantity used once, rather than summing repeated
  on-hand snapshots as new supply.
- Dated receipt quantities and when they become allocatable, or a documented
  inventory movement ledger. Snapshot differences alone can also include returns,
  transfers, adjustments and unobserved sales; they are not confirmed receipts.
- Requested orders, due dates, cancellations and unmet orders. Observed sales can
  be censored by stockouts and must be labeled as fulfilled-sales proxies when
  unobserved demand is unavailable.
- SKU identity, quantity units, aggregation boundaries, and documented substitute
  relationships. Fractional sales cannot be silently rounded into integer demand.
- Independent episode/group splits and the units/rationale for cost weights.

The existing strict CSV scenario loader is ready for explicitly prepared,
documented order and supply records. It does not infer receipts, lost sales or
missing inventory from these public sources. No new dataset in this audit clears
the full observed-inventory-and-demand admission boundary.
