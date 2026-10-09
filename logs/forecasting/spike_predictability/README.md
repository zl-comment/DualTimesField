# Spike predictability from slow precursors (static split, one seed)

Script: `forecasting/spike_predictability.py`. Train origins to 2022-09-30, validation 2022Q4
(early stopping only), test 2023-2024. Pooled XGBoost classifier over the 24 leads, label =
actual price above 300 or 1000 AUD/MWh. Feature sets: cal, P (price history), F (fundamentals
snapshot), S (F + predispatch snapshot), R (forecast revisions over 3/6/12 h), T (driver
trajectories and forward margin shape). Full numbers in `results.json`; predictions in `predictions_*.npz`.

Test PR-AUC, spike > 300 (hour-level; base rate 1.75% NSW1, 1.85% QLD1, 0.5% TAS1):

| set | NSW1 | QLD1 | TAS1 |
|---|---|---|---|
| S | 0.200 | 0.310 | 0.123 |
| S+R+T | 0.367 | 0.377 | 0.192 |
| S+P | 0.410 | 0.442 | 0.254 |
| S+P+R+T | 0.410 | 0.446 | 0.237 |

Caveats: one seed; validation quarter has few spikes, so early stopping and the episode alarm
thresholds are noisy (episode recall in `results.json` is at validation-set thresholds and the
achieved alarm rates differ across sets; compare equal-alarm-rate recall instead). Week-block
bootstrap intervals do not include model-fitting variance.
