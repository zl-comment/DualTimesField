# History-conditional information probe (2026-10-10)

`forecasting/history_probe.py`: the quarterly XGBoost-with-predispatch baseline (raw prices, seed 2026, same
features, hyperparameters, splits and device) refit with extra features known at the origin: the recent
predispatch error (realized price minus the lead-0 predispatch price of the same hour) averaged over the last 3,
6 and 24 hours with its 24-hour mean absolute and maximum, and, per forecast hour, the error of the same hour
yesterday, two days ago and a week ago. An upper bound for what a head that attends to the history could extract
from the persistence of predispatch error. 24 refits, 8 quarters x 3 regions, compared with the stored baseline
on the same 17,521 origins per region.

| | NSW1 | QLD1 | TAS1 | mean |
|---|---|---|---|---|
| baseline MAE (seed 2026) | 43.38 | 36.47 | 26.80 | 35.55 |
| with history features | 43.23 | 36.39 | 26.77 | 35.46 |

Pooled Diebold-Mariano, probe minus baseline (negative favours the probe): MAE -0.088 (p=0.006; NSW1 -0.15,
QLD1 -0.08, TAS1 -0.03), CRPS~ -0.02 (p=0.42), negative-hour CRPS~ -0.03 (p=0.35), spike-hour CRPS~ -0.9 (p=0.48).

Reading: the information exists but is worth about 0.09 MAE (0.25%), no more than a few seed standard deviations
of the neural models; the gap to the strongest baseline from history conditioning is small. Single seed (XGBoost
seed spread is about 0.02), one family of features; analogs of other conditions were not tested.
