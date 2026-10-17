# Dual-field historical features: contribution results

Three regions, seed2026. Historical encoder: Jul–Sep; correction weights: Oct–Nov; scores: December2022.
This is retrospective development. Actual first publication timestamps remain unaudited; causality is under archive proxies.

| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |
|---|---:|---:|---:|---:|---:|
| B | 25.710190 | 26.291141 | 27.044704 | 26.348678 | 15.219256 |
| C_roll | 31.216519 | 29.512742 | 27.226839 | 29.318700 | 17.428757 |
| H_recent | 25.850057 | 25.771022 | 27.284182 | 26.301754 | 15.480187 |
| H_raw | 34.245663 | 33.199710 | 28.009585 | 31.818319 | 19.991982 |
| H_smooth | 34.819984 | 33.641896 | 27.256311 | 31.906064 | 19.742986 |
| H_dual_trend | 31.505068 | 30.251012 | 27.207734 | 29.654605 | 17.675739 |
| H_dual_full | 31.818581 | 30.571454 | 27.159124 | 29.849720 | 17.808058 |
| H_dual_event_off | 31.538596 | 30.402065 | 27.175702 | 29.705454 | 17.707603 |

## Contribution checks

- field_activation: **True**
- forecast_activation: **True**
- incremental_value_pass: **False**
- per_region_non_regression: **False**
- event_increment_pass: **False**
- event_fixed_head_pass: **False**
- beats_original_baseline: **False**

## NSW1

- Encoder parameters: 7501; fit rows: 2184; latest training-label availability: 1664542800.
- Full history weights: [0.24839572136020646, 0.10638621310587744, 0.09458080403842034, 0.0]
- Trend-only weights: [0.21885934696949405, 0.049789048409787456, 0.09845568447261221, 0.0]
- Full forecast changes: {'fraction': 0.75, 'mean_abs': 2.8403206310769975, 'max_abs': 28.379768163137868}
- Historical reconstruction MAE: 50.999680; fixed-smoothing MAE: 29.707556.
- Wall time: 24.5s.

## QLD1

- Encoder parameters: 7501; fit rows: 2184; latest training-label availability: 1664542800.
- Full history weights: [0.12479039681912991, 0.08693755487757587, 0.1043244845362302, 0.06770650284520423]
- Trend-only weights: [0.09266808635462692, 0.04054316430270624, 0.10713427728379574, 0.06715218749712851]
- Full forecast changes: {'fraction': 1.0, 'mean_abs': 3.9543559267181703, 'max_abs': 18.718559522869498}
- Historical reconstruction MAE: 62.557497; fixed-smoothing MAE: 35.084275.
- Wall time: 22.8s.

## TAS1

- Encoder parameters: 7501; fit rows: 2184; latest training-label availability: 1664542800.
- Full history weights: [0.1320802470246231, 0.1325584979173111, 0.06205119884081847, 0.0]
- Trend-only weights: [0.15574016049171083, 0.1965567721251391, 0.059239551120817496, 0.0]
- Full forecast changes: {'fraction': 0.75, 'mean_abs': 2.324367347115883, 'max_abs': 13.966818783289071}
- Historical reconstruction MAE: 58.049482; fixed-smoothing MAE: 48.608796.
- Wall time: 22.9s.

All matched variants share candidate IDs and four downstream parameters. Nonzero fields or better reconstruction alone do not establish forecast improvement.
Bootstrap intervals and all price-band contributions are in bootstrap.json and regional JSON. No automatic architecture selection or final-test launch follows.
