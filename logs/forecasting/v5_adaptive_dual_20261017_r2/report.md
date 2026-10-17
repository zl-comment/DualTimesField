# Adaptive historical dual fields: fixed contribution check

Seed2026, three regions. Both encoders fit July-August, September reconstruction diagnosis, October-November correction fitting, December2022 retrospective scoring.
No final test accessed. Historical reconstruction uses already-known prices; it is not future spike prediction. First-publication timestamps remain independently unaudited.

| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |
|---|---:|---:|---:|---:|---:|
| B | 25.710190 | 26.291141 | 27.044704 | 26.348678 | 15.219256 |
| C_roll | 31.216519 | 29.512742 | 27.226839 | 29.318700 | 17.428757 |
| H_recent | 25.850057 | 25.771022 | 27.284182 | 26.301754 | 15.480187 |
| H_raw | 34.245663 | 33.199710 | 28.009585 | 31.818319 | 19.991982 |
| H_smooth | 34.819984 | 33.641896 | 27.256311 | 31.906064 | 19.742986 |
| H_old_dual | 34.081454 | 34.305058 | 27.183577 | 31.856696 | 19.387707 |
| H_fixed_pulses | 34.445462 | 33.192032 | 27.873288 | 31.836927 | 19.961561 |
| H_adaptive_trend | 33.933217 | 32.912276 | 27.336758 | 31.394084 | 19.432391 |
| H_adaptive_full | 34.537993 | 33.261186 | 28.012560 | 31.937246 | 20.060419 |
| H_adaptive_event_off | 34.030337 | 33.422524 | 27.561346 | 31.671402 | 19.672162 |

Incremental value gate: **False**; all-region non-regression vs C_roll: **False**.

| September full-visible reconstruction | NSW1 MAE | QLD1 MAE | TAS1 MAE |
|---|---:|---:|---:|
| fixed_smooth | 39.714317 | 48.184418 | 29.255464 |
| old_dual | 45.895787 | 50.946730 | 34.638244 |
| fixed_pulses | 19.364577 | 22.544927 | 13.635024 |
| adaptive | 11.370437 | 16.456184 | 10.288880 |

| September hidden-only reconstruction (same 20% mask) | NSW1 MAE | QLD1 MAE | TAS1 MAE |
|---|---:|---:|---:|
| old_dual | 45.188830 | 50.747480 | 37.631726 |
| fixed_pulses | 43.393973 | 51.613598 | 31.963150 |
| adaptive | 29.123306 | 39.768467 | 23.922108 |

## NSW1
- Learned parameters: 962; fit rows: 1464; latest fit label: 1661950800.
- Adaptive correction weights: [0.5954622542714264, 0.541269773427389, 0.5231785124616442, 0.5004417367172954]
- Changed predictions vs C_roll: {'fraction': 1.0, 'mean_abs': 13.946607657669466}
- Unique December actual spike hours: 2; wall time: 66.9s.

## QLD1
- Learned parameters: 962; fit rows: 1464; latest fit label: 1661950800.
- Adaptive correction weights: [0.5022905562758551, 0.5171213529352964, 0.474303406351405, 0.42362392269465543]
- Changed predictions vs C_roll: {'fraction': 1.0, 'mean_abs': 11.71555036591881}
- Unique December actual spike hours: 10; wall time: 64.3s.

## TAS1
- Learned parameters: 962; fit rows: 1464; latest fit label: 1661950800.
- Adaptive correction weights: [0.2962877048655543, 0.3325188187386147, 0.318743956370703, 0.24549703906127004]
- Changed predictions vs C_roll: {'fraction': 1.0, 'mean_abs': 7.126424178067984}
- Unique December actual spike hours: 1; wall time: 63.9s.

All matched variants use identical historical neighbor IDs and four downstream weights. Event-off reuses the full head. Fixed pulses share the analytic solver and peak proposals, isolating geometry learning.
Regional JSON includes tail errors, event localization, hidden-only diagnostics and maturity audits. Bootstrap intervals are descriptive: December supplies few independent weekly blocks.
