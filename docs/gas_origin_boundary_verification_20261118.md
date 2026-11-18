# Gas origin-context as-of boundary verification (2026-11-18)

An audit pass on 2026-11-18 flagged `_trailing_gas_price`
(`forecasting/datasets.py`) as a possible as-of violation: the read was that
the 7-day trailing DWGM gas mean is indexed by a delivery hour up to 24 hours
after the origin, so the last included gas day's final schedule (stamped
22:00) could post after the origin. That reading is incorrect. This record
closes the question with a code-path analysis and a full empirical check.

## Code path

- `forecasting/datasets.py:346` computes one gas value per frame row from that
  row's own `delivery` hour: `current_gas_day = (delivery - 6h).normalize()`,
  value = `daily.rolling(7).mean().shift(1)` at `current_gas_day`, i.e. the
  mean of the seven gas days strictly before the gas day containing the row's
  delivery hour.
- `forecasting/datasets.py:571-573` sets the sample's origin to
  `delivery_unix_seconds[forecast_start]`, and the aligned contract has
  `target_time_unix[:, 0] == origin_unix` (enforced in
  `tests/test_aligned_inputs.py`), so the origin instant is exactly the
  delivery hour of its own row.
- `forecasting/datasets.py:578-581` reads
  `origin_context_values[forecast_start]` — the value indexed at the origin
  row itself, not at any later delivery in the 24-hour target window.

For an origin T in gas day G, the latest gas day entering the mean is G−1,
which ends 06:00 on G, and T ≥ 06:00 G by definition of the gas day
(06:00–06:00 AEST). The inclusion boundary is therefore "gas day ended at or
before the origin", and the final 22:00 schedule of G−1 precedes that end by
8 hours. The earlier report had assumed the feature was indexed around
origin + 24 h; it is not.

## Empirical check

`scripts/diagnostics/verify_gas_origin_boundary.py` replicates the exact
indexing over all hourly rows and the real gas archive, and fails loudly on
any violation, NaN, or non-positive value. Result on 2026-11-18, all three
regions, 2015-01-01 through 2024-12-31 (87,672 rows each):

- boundary violations (gas-day end after origin): **0**
- margin from origin back to the last gas day's end: 0.0–23.0 hours
- margin back to that day's final 22:00 schedule: at least 8.0 hours
- gas archive complete 2007-02-01 → 2026-08-31 (no missing days)

## Conclusion

No repair is needed and no baseline needs re-running on this account. The
baseline comparison and data pipeline audit conclusions of 2026-11-18 stand
with this item reclassified from "leak risk" to "verified as-of safe".

The residual, previously disclosed limitation is unchanged: the archive holds
final (possibly ex-post revised) gas prices rather than true publication
vintages, and gas publication times remain unverified
(`docs/baseline_maintenance_20261024.md`, item 1). The verified end-before-
origin boundary is the conservative envelope for that uncertainty; removing
it entirely would require vintage gas archives.

Reproduce: `.venv/bin/python scripts/diagnostics/verify_gas_origin_boundary.py`
