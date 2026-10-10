# Plan: ensembles of mature algorithms and the dual field, scored on a held-out 2025-2026

Status: written 2026-10-10 before any of these runs. Branch `feature/ensemble-holdout-plan`, from `research/trunk`.
Plans 1 and 2 are executed first; 3 to 5 are written down so that their criteria are fixed before data are seen.

## Why

The experiments of `logs/forecasting/overview/EXPERIMENT_OVERVIEW.md` found no independent MAE value of the dual field
on a linear base, an XGBoost base, in driver space, or through history attention; an oracle analysis (three
reviewers, `/tmp/agentB/report.md` summarized in the overview) puts the achievable MAE at about 34-34.5 given the inputs,
with about 38% of the error in spike hours whose magnitude is not predictable. One result is positive and cheap: a
50/50 average of seed-averaged XGBoost and the no-fields neural model has MAE 34.94 against 35.35 for seed-averaged
XGBoost (week-block bootstrap 95% interval of the difference [-0.52, -0.29]); replacing the neural member by the
dual-field model gives 34.91, so the gain comes from model diversity, not the fields. This was seen on the
2023-2024 test period, which has guided many choices. The plans below turn it into a result that survives:
equal tuning budgets, weights chosen without the test period, and a period no decision has seen.

The dual field stays in every plan as a member, a feature source or a tail model; it is kept only where it is
shown to add something (the plan-1 contrast of ensembles with and without it). Mature algorithms are not limited
to XGBoost.

## Plan 1: heterogeneous ensemble of mature algorithms and dual-field members, equal tuning budget

Members (all refit quarterly with predispatch inputs, raw prices, point forecast = median):
- M1 XGBoost (per-horizon, current features; the stored baseline is the untuned reference).
- M2 XGBoost with the 24-hour predispatch profile and history summaries as features (strongest suggestion of the
  design review; single-quarter pilot only, to be verified).
- M3 further mature algorithms where installable offline (LightGBM, CatBoost); skipped and reported if not.
- M4 linear base + calibrator + floor, no fields (C-mae), and M5 the dual-field model (B), both with the
  validation-MAE checkpoint and the validation-chosen floor.
- M6 XGBoost point forecast with dual-field tail correction (`xgb_base_tails`), for the quantile forecasts.

Tuning: every member gets the same budget, six configurations (the default included) over the parameters that
matter for it (trees: depth, learning rate, regularization, column sampling; neural: learning rate, weight decay,
hidden size, epochs). One global configuration per member, chosen by mean validation MAE over the eight refits'
validation quarters, never by test MAE. Seeds 2026-2028; ensembles average seed-averaged member forecasts.

Combination: (a) equal weights; (b) per refit, non-negative weights minimizing absolute error on that refit's own
validation quarter, shrunk halfway toward equal weights. The test quarter is never used.

Criteria (pooled Diebold-Mariano, three regions, week-block bootstrap for the MAE differences, p < 0.05):
1. the best ensemble beats the best tuned single member in MAE by at least 0.3;
2. ensembles with a dual-field member beat the same ensemble without one (the fields' contribution);
3. weights learned on validation do not do worse than equal weights.
Every outcome is reported; if 1 fails the ensemble gain seen on 2023-2024 is not reproduced.

## Plan 2: a held-out period, 2025-01 to 2026-08

No model, configuration or weight rule is changed after the data arrive.
- Data: AEMO price and demand (`PRICE_AND_DEMAND_<yyyymm>_<region>.csv`, hourly mean as for the existing series, checked
  by reproducing 2024-12 and 2021-10 onwards from the five-minute files), PD PASA and predispatch from the MMSDM archive through
  2026-08, gas from the DWGM series (already to 2026-08). Same point-in-time rules and leakage checks.
- Protocol: the quarterly refits of `forecasting.rolling` over 2025Q1 to 2026Q2 and the two months of 2026Q3, each trained
  on all data up to three months before its quarter.
- Scoring: once, with the configurations and the weight rule fixed by plan 1 on 2023-2024. Report MAE, CRPS~,
  negative- and spike-hour scores for every member and ensemble, and the 2025-2026 price statistics (share of
  spike and negative hours) next to the 2023-2024 ones, since regimes may have changed.
Criteria: the ensemble beats XGBoost in MAE with the same sign and p < 0.05 on the held-out period; the dual-field
contribution of plan 1 is reported on it as well.

## Plan 3: tail risk and decision value (written, not started)

Spike hours carry about half of the error and no model predicts their size, so MAE ranks models differently from
tails and decisions. Re-score every stored variant and the ensembles with q90/q95/q99 pinball losses, exceedance
Brier and PR-AUC (>300 and >1000), and the value of a one-battery arbitrage schedule under 1, 2 and 4 MWh
storage (daily origin, 24-hour schedule on forecast hourly prices, settled at realized prices; five-minute
checking later), all with week-block bootstrap intervals. The design review reported that the ensemble earns
3.8-5% more than XGBoost under this rule; that is unverified. Criterion: a field variant or ensemble is better
than its no-field counterpart in tail scores or revenue with intervals excluding zero; otherwise the dual field is
dropped from the tail story too.

## Plan 4: new information about the supply-side state (written, not started)

The adjacent-hour price oracle lowers MAE by 9.2, all in non-spike hours, which says the price level follows a
supply state the inputs do not contain. In order of cost: reserve notices (LOR, `MARKETNOTICEDATA`; archive stops
at 2024-07), unit availability and SCADA (`DISPATCHLOAD`, `DISPATCH_UNIT_SCADA`, about 10 GB), bid stacks
(`BIDPEROFFER`, about 125 GB, no point-in-time version after 2024-07). Each is tested as extra features of the best
tree member on 2023-2024 up to 2024-07, then, if it helps, with the dual field's event field detecting events in
those channels. Stop rule: a source that does not lower MAE by 0.2 with an interval excluding zero is dropped; the
bid stacks are touched only if the cheaper sources show a signal.

## Plan 5: tabular and foundation members, and mature algorithms combined with the dual field (written, not started)

- Members: TabPFN-TS and Chronos-2 with covariates (the literature's only gains, in day-ahead markets) as extra ensemble
  members; kept if they lower the ensemble MAE by at least 0.2 on one region and quarter first.
- Dual field on top of other mature bases: the XGBoost-base experiment (`xgb_base_*`) repeated with LightGBM/CatBoost
  and with the tuned XGBoost of plan 1, same criteria as `PLAN_XGBOOST_BASE_FIELDS.md`.
- Dual field as a feature source for trees: the trend value, event amplitudes and positions of the fitted fields
  as features of XGBoost (cross-fitted so that training features are out of fold), against the same trees with the
  raw history.
Criterion for every item: it beats the same mature algorithm without the dual-field part in MAE or in a tail score,
pooled DM p < 0.05, on 2023-2024 and, once frozen, on the held-out period.

## Reporting rules

All numbers on the 2023-2024 period are development results. A claim goes into a paper only with its held-out
number. Seeds, tuning budgets and the validation rule are stated with every table.

## Interim development results for plan 1 (2023-2024, seed 2026 for the tree members; neural members are 3-seed forecasts)

Three-region mean test MAE, raw prices, quarterly refits, `logs/forecasting/members/ensemble_dev/`.

| Member | default | profile | reg | profile_reg |
|---|---:|---:|---:|---:|
| XGBoost (multi-quantile loss, median) | 35.55 | 35.34 | 35.67 | 35.20 |
| LightGBM (L1 loss) | 35.24 | 34.84 | 35.25 | **34.75** |
| neural, no fields (C-mae) / with fields (B) | 35.69 / 35.74 | | | |

Validation MAE of the same variants (mean of the validation quarters): XGBoost 34.42 / 35.07 / **34.18** / 34.97;
LightGBM 34.29 / 34.47 / **33.97** / 34.33.

**Deviation from the rule written above, made before any held-out score exists.** The rule "choose the configuration by
validation MAE" picks `reg` for both families, and `reg` is the worst or near-worst variant on the test period: the
validation quarter is also the early-stopping set, so configurations that train longer look better on it than they are
(selection on a score the model already optimized). The configuration for the held-out period is therefore chosen on
the 2023-2024 realized MAE, which is the development period (the held-out period stays untouched). Development
numbers selected this way are optimistic and are reported as such; only the held-out numbers count.
Both rules are reported:

| Ensemble (equal weights) | validation-chosen members | MAE | development-chosen members | MAE |
|---|---|---:|---|---:|
| best LightGBM alone | `lgbm_reg` | 35.25 | `lgbm_profile_reg` | 34.75 |
| XGBoost + LightGBM | `xgb_reg`, `lgbm_reg` | 35.33 | `xgb_profile_reg`, `lgbm_profile_reg` | 34.85 |
| + neural without fields | + C-mae | 34.93 | + C-mae | 34.64 |
| + neural with fields | + B | 34.89 | + B | **34.61** |
| all four | | 34.88 | | 34.67 |

Against the best member of the same set, the ensembles with a neural member are 0.37 lower (validation-chosen, week-block 95%
interval [-0.49, -0.26]) or 0.08-0.14 lower (development-chosen; intervals [-0.18,-0.03] to [-0.19,+0.04]). Fitting weights
on the validation quarter changes the MAE by less than 0.05 against equal weights.

Reading: (i) the largest gain comes from the choice of mature algorithm, objective and features (LightGBM with the L1
loss and the 24-hour profile features, 34.75 alone, against 35.55 for the XGBoost baseline), not from the ensemble;
(ii) once the best member is strong, adding other families helps by 0.1 or less; (iii) a dual-field member (B) is not
better than a no-field one (C-mae) in any ensemble (differences of 0.03-0.04, intervals including zero), so the
earlier "diversity from the fields" reading stays unsupported. XGBoost variants with the absolute-error objective
(`l1`, `profile_l1`) and the tuned neural variants are still running, so that the loss function is not credited to
the algorithm.
