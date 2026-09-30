# Business pilot export request

Please supply a documented, internally consistent operational episode in these
files. Preserve stable SKU/order identifiers across all files; pseudonymous
identifiers are sufficient. No customer names, addresses or payment details are
needed. State which system each source label refers to and how its columns map
to these meanings. This package contains headers only, not business observations.

1. catalog.csv: all tracked SKUs, one consistent interchangeable integer unit,
   and integer modeled cost per unit. Document any chosen scaling. The catalog
   and costs must be known before the opening snapshot and remain fixed here.
2. stock_snapshots.csv: a complete physical on-hand count for every catalog SKU
   at opening and every evaluation cutoff, including explicit zero counts.
   All rows of a snapshot share its ID, actual UTC observation time and source.
3. movements.csv: every physical receipt, dispatch and recorded inward/outward
   adjustment after opening. Use positive quantities and receipt, dispatch,
   adjustment_in or adjustment_out kinds. Include event_time and observed_at;
   observation cannot precede the event. Do not infer movements by subtracting
   stock snapshots. An unexplained inventory difference blocks evaluation.
4. open_orders.csv: authoritative remaining requested backlog, first at opening
   and whenever it changes. Quantity replaces the previous quantity, rather than
   adding demand. Zero closes an order. Preserve the requested SKU and due date.
   Include event_time and observed_at; existing orders may predate opening.
   Historical fulfilled sales are not a substitute for requested open backlog.
5. manifest.json: replace the episode/group name and explicitly set synthetic to
   true or false after verifying provenance. Keep overlapping snapshots in one
   group if calibration and holdout splits are constructed later.
6. policy.json: replace every null. Supply explicit acceptable substitutes,
   allow_late, and nonnegative integer late/substitution/disruption/shortage
   weights. Weights are model assumptions, not verified financial costs.

Use UTC YYYY-MM-DDTHH:MM:SSZ timestamps and YYYY-MM-DD due dates. This solver uses
UTC calendar-day deadlines, not exact intraday service deadlines. Do not invent
observation timestamps from event timestamps: confirm the system actually knew
the record then, or document that visibility cannot be assessed. Confirm order
coverage, movement completeness, units, and snapshot timing with the data owner.
A complete snapshot is assumed synchronous as of its observation time.

Record mapping decisions and source limitations in a separate local note. This
request is a draft for the data owner; the framework does not send it or access
an ERP. Once filled, run:

    assumption-ops-business --manifest manifest.json --validate-only
    assumption-ops-business --manifest manifest.json --output-dir results

The unfilled template intentionally fails validation. Successful evaluation
compares plans under admitted facts; it does not execute allocations, measure
causal service improvements or establish ROI. No actual-data benchmark can be
reported until the filled source records are supplied and validated.
