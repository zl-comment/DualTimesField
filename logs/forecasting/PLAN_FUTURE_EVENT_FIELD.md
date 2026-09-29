# Plan: extending the DGF into a future event field

Status: design, not yet implemented. Branch: `feature/future-event-field`,
from `feature/ablation-no-reconstruction` (`ba37ad3`, trunk `b3fd5a9` plus
the ablation).

## Why

The ablation in `EXPERIMENT_RESULTS.md` shows that feeding the raw history to
both expert heads instead of the CTF and DGF fields leaves MAE (41.90 vs
41.88) and CRPS~ (28.40 vs 28.43) unchanged. The code explains why:

- The DGF (`src/dualfield/core.py`, `GaborAtom`) has 16 atoms whose centres
  `tau` are global parameters in window-relative time `[0, 1]`, shared by
  every sample. A sample only changes the atoms' amplitudes and gates, which
  come from a mean-pooled encoding of the window. The field cannot place an
  event at a sample-specific time, and it cannot represent anything after the
  origin.
- The CTF output at time `t` is a function of the Fourier time features of `t`
  and the low-pass-filtered history at `t`. It has no value at a future `t`.

So both fields only re-describe the past 72 hours, which linear heads can read
directly. The forecast needs an event field that lives in the forecast
horizon.

## Evidence for a two-sided event process

| Test split unless noted | NSW1 | QLD1 | TAS1 |
|---|---:|---:|---:|
| Spike hours (> 300 AUD/MWh), train -> test | 0.25% -> 1.75% | 0.68% -> 1.85% | 0.43% -> 0.50% |
| Negative-price hours, train -> test | 0.37% -> 6.19% | 1.63% -> 13.21% | 2.52% -> 7.37% |
| P(spike) if a spike in the last 24 h, else | 5.7%, 0.86% | 5.3%, 0.78% | 5.7%, 0.21% |
| P(negative) if negative in the last 24 h, else | 11.8%, 2.6% | 18.6%, 4.6% | 18.6%, 2.0% |
| P(spike) if spare capacity below its training 10% quantile, else | 8.6%, 0.39% | 7.9%, 0.46% | 3.1%, 0.40% |

Spikes and negative-price troughs both persist, spikes follow scarcity, and
troughs have become far more frequent than in training. Their probability must
therefore come from forecast drivers (spare capacity, net load) and from the
state of recent events, not from a learned base rate.

## Design

All quantities are per forecast hour `h = 1..24`, in the trunk's point target
space `z` (standardized asinh price; 300 AUD/MWh is `z` of about 2.5-2.8 and
0 AUD/MWh about -1.1 to -1.4).

**Three-component mixture.**

`z_h ~ (1 - p+_h - p-_h) N(m_h, s_h) + p+_h N(m_h + d+_h, s+_h) + p-_h N(m_h - d-_h, s-_h)`

- Level component `(m_h, s_h)`: the continuous level. `m_h` is the existing
  fused point path (CTF heads plus gated DGF heads); `s_h` is a softplus
  linear head on the CTF features.
- Spike component `(p+_h, d+_h, s+_h)`: driven by scarcity. Inputs: the
  spare-capacity shortfalls and spare capacity at hour `h`, the hour's
  calendar, and the past event state.
- Trough component `(p-_h, d-_h, s-_h)`: driven by renewable surplus. Inputs:
  the soft-saturated net load at hour `h`, the hour's calendar, and the past
  event state.
- Weights: `softmax([0, a+_h, a-_h])`. `d` and `s` use softplus with floors.
  The logit biases start at the training event rates.

**Past event state, the part that makes the DGF matter.** Per sample, the
DGF's price-channel atom amplitudes and gates (2 x 16) and the last 24 hours
of the DGF event signal are projected to a small vector (8 values) shared by
both event components. Two ablations test whether this beats the same
projection of the raw last 24 hours, and no state at all.

**Outputs.**

- Event probabilities `p+_h`, `p-_h`, the first interpretable forecast of
  spikes and troughs in this model.
- Quantiles: the 0.05/0.10/0.50/0.90/0.95 quantiles of the mixture, found by
  bisection on its CDF in `z`, then mapped to prices by the inverse asinh
  transform (monotone, so quantiles are preserved).
- Point: the mixture median (the MAE-optimal point).

**Loss.** Mixture negative log-likelihood of `z_h`, plus a weak binary
cross-entropy that anchors the components to their meaning: `p+_h` to
`price > 300` and `p-_h` to `price < 0`. The existing point, decomposition,
smoothness, and sparsity terms stay.

## Stages

| Stage | Change | Forecast used for scoring | Question |
|---|---|---|---|
| A | Add the mixture head in parallel; add its loss | Trunk point and quantiles | Are spike and trough probabilities calibrated and better than persistence? Does training stay stable? |
| B | Replace the price-space quantile heads with mixture quantiles | Trunk point, mixture quantiles | Does CRPS~ and 90% AIS improve? |
| C | Use the mixture median as the point forecast | Mixture point and quantiles | Does MAE hold or improve? |

Each stage runs seeds 2026-2028 on the three regions (about 20 minutes).

## Evaluation

- Standard metrics: MAE, window RMSE and SDE, 90% coverage, width and AIS,
  CRPS~, against the trunk and the re-implemented XGBoost.
- Event metrics per region: Brier score, ROC-AUC, and PR-AUC for `price > 300`
  and `price < 0`, with reliability curves. Reference forecasts: persistence
  (event in the last 24 hours) and a logistic regression on the same inputs.
- Mechanism: `p+` against spare-capacity shortfall, `p-` against net load,
  and case studies of the largest 2023-2024 events.

## Success criteria

| Criterion | Target |
|---|---|
| Stage A: event forecast | Spike and trough PR-AUC above persistence and logistic regression in NSW1 and QLD1 |
| Stage A: DGF relevance | DGF event state beats raw-history state on event metrics |
| Stage B: probabilistic | CRPS~ below XGBoost's 27.69 with 90% AIS no worse than the trunk's 412.60 |
| Stage C: point | MAE no worse than the trunk's 41.88 |
| Story | The redesigned model beats the raw-history ablation, which the transplanted fields did not |

## Code touch points

- `forecasting/models.py`: a `FutureEventField` module (event-state
  projection, spike and trough heads, level scale head), mixture quantiles and
  median, extra outputs (`event_probability_spike`, `event_probability_trough`,
  `mixture_*`).
- `forecasting/losses.py`: mixture negative log-likelihood and anchoring
  cross-entropy.
- `forecasting/train.py` and `forecasting/evaluate_paper_metrics.py`: a switch
  so that mixture quantiles, which are in the target space, are denormalized
  with `denormalize_target` rather than the price-space quantile standardizer.
- `configs/`: one configuration per stage.

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Mixture collapse (one component absorbs all mass) | Anchoring cross-entropy, softplus floors, base-rate bias initialization |
| Upper quantiles too wide after the inverse asinh transform, as in the asinh-quantile experiment | Keep price-space quantile heads and use the mixture only for event probabilities and the point (stop at stage A or C) |
| Too few TAS1 spikes to fit the spike component | Share spike-head weights across regions only as a later option; report TAS1 separately |
| Point accuracy drops when the median replaces the fused point | Keep the fused point (stop at stage B) |
