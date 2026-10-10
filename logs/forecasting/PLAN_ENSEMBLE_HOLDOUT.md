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

## Frozen before the held-out scoring (2026-10-10, no held-out score exists yet)

Development results of every member of plan 1 (three-region mean MAE on 2023-2024; tree members seed 2026, neural members the
average of three seeds). Validation MAE in brackets.

| Family | variants (test MAE) | chosen on 2023-2024 |
|---|---|---|
| XGBoost | default 35.55 (34.42), profile 35.34, reg 35.67 (34.18), profile_reg 35.20, l1 35.43, profile_l1 **35.04** | `profile_l1` |
| LightGBM | default 35.24, profile 34.84, reg 35.25 (33.97), profile_reg **34.75** | `profile_reg` |
| neural, no fields | default 35.69, v1 **35.61**, v2 35.94, v3 35.67 | `nn_c_mae_v1` |
| neural, with fields | default **35.74**, v1 35.81, v2 35.93, v3 35.75 | `nn_dual_b` (default) |

The absolute-error objective helps XGBoost too (35.55 to 35.43 on the default features, 35.34 to 35.04 with the profile features), so
part of LightGBM's advantage over the baseline is the loss, part the algorithm (34.75 against 35.04 with the same features and loss
family). Tuning the neural members changes the MAE by less than 0.35 either way.

Equal-weight ensembles on 2023-2024 (members chosen as above; development numbers, optimistic): LightGBM alone 34.75;
XGBoost + LightGBM 34.75; + C 34.55; + B 34.55; + C + B 34.61. Against LightGBM alone the three-member ensembles are 0.20 lower
(week-block 95% interval [-0.28, -0.13]); B and C are equal (34.552 and 34.548). Criterion 1 of plan 1 (at least 0.3 lower than
the best single member) is **not met on the development period**.

To be scored once on 2025-01 to 2025-07 (5,040 origins per region; quarterly refits; same code):
`xgb_default` (the baseline), `xgb_profile_l1`, `lgbm_profile_reg`, `nn_c_mae_v1`, `nn_dual_b`; equal-weight ensembles
{xgb_profile_l1, lgbm_profile_reg}, {+ nn_c_mae_v1}, {+ nn_dual_b}, {+ both}. Weights learned per refit on the validation quarter are
reported next to equal weights. Held-out criteria: the best ensemble beats `lgbm_profile_reg` and `xgb_default` in MAE with
p < 0.05 and the same sign as in 2023-2024; the field contrast is {+ nn_dual_b} against {+ nn_c_mae_v1}. All outcomes are reported.

## Held-out results, 2025-01-01 to 2025-07-29 (scored once, 2026-10-10)

5,040 origins per region, three refits (2025Q1, 2025Q2, July), raw prices, the members frozen above, neural members the
average of three seeds. `logs/forecasting/members_hold/ensemble/holdout_report.json`. Pooled Diebold-Mariano / week-block
bootstrap over the three-region mean loss per origin; differences are first minus second.

| | NSW1 | QLD1 | TAS1 | mean MAE |
|---|---:|---:|---:|---:|
| XGBoost, default (the baseline) | 49.28 | 35.62 | 33.57 | 39.49 |
| XGBoost, profile features, L1 loss | 48.72 | 35.24 | 32.57 | 38.84 |
| LightGBM, profile features, regularized, L1 | 48.47 | 35.00 | 32.21 | **38.56** |
| neural, no fields (`nn_c_mae_v1`) | 47.85 | 36.23 | 35.64 | 39.91 |
| neural, with fields (`nn_dual_b`) | 48.07 | 36.07 | 35.46 | 39.87 |
| ensemble XGBoost + LightGBM | 48.50 | 35.00 | 32.25 | 38.58 |
| ensemble + neural no fields | 47.78 | 34.59 | 32.18 | **38.18** |
| ensemble + neural with fields | 47.78 | 34.74 | 32.07 | 38.20 |
| ensemble + both neural | 47.59 | 34.73 | 32.58 | 38.30 |

| Ensemble (equal weights) against | `xgb_default` | `lgbm_profile_reg` | `xgb_profile_l1` |
|---|---|---|---|
| XGBoost + LightGBM | -0.91 (p<0.001) | +0.02 (p=0.17) | -0.26 (p<0.001) |
| + neural no fields | -1.31 (p<0.001; [-1.87,-0.89]) | -0.38 (p=0.023; [-0.75,-0.10]) | -0.66 (p<0.001) |
| + neural with fields | -1.29 (p<0.001) | -0.36 (p=0.010; [-0.68,-0.10]) | -0.65 (p<0.001) |
| + both neural | -1.20 (p<0.001) | -0.27 (p=0.24; [-0.76,+0.12]) | -0.55 (p=0.021) |

Field contrast (+ neural with fields minus + neural no fields): +0.016 (p=0.74, interval [-0.08, +0.11]).
Weights fitted on each refit's validation quarter give 38.12 to 38.17 for the three-neural-member ensembles, 0.03-0.07 below
equal weights (not tested).

Criteria registered above: (1) the best ensemble beats `lgbm_profile_reg` and `xgb_default` with p < 0.05 and the same sign as in
2023-2024: **met** (-0.38, p=0.023; -1.31, p<0.001). The 0.3 margin of plan 1 criterion 1, not met in development (0.20), is met here
(0.38, interval [-0.75,-0.10]). (2) The dual-field member adds nothing over the no-field member: +0.016, p=0.74. (3) Validation weights do
not do worse than equal weights: met.

Reading and limits. The held-out period is seven months (three refits), a single regime window; the intervals on the ensemble margin are
wide ([-0.75,-0.10]). The gain over the XGBoost baseline (-1.3, 3.3%) is mostly the choice of a tuned L1 gradient-boosting member
(-0.93 for LightGBM alone) plus about 0.4 from adding the neural family; the neural members alone are 0.4 worse than the
trees on the held-out period (better on NSW1, worse on QLD1 and TAS1). The dual field is not shown to matter. PD PASA after July
2025 is not comparable (see `logs/forecasting/holdout_data/README.md`), so the later months are untested.
