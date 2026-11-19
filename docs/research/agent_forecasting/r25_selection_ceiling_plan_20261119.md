# R25 selection-ceiling pre-screen: pre-registered plan (2026-11-19)

Status: pre-registered; not implemented, not run. Research branch
`research/r25-selection-ceiling`, baseline commit `a126e03` (parent:
`research/r24-iterative-loop` after the R24b close-out and the gas
origin-boundary verification). Run-time source and config fingerprints must
be recorded with the results before any number below is read.

## Motivation

The R24 programme closed with the conclusion that the remaining accuracy is
upstream of decision policy: the operation space (two demand-window mean
replacements) capped every selector, and R23's deterministic admission rule
remained the best measured configuration (region-equal QS 39.2503 on the
2022H2 development fold). The candidate direction is now model selection /
routing over the existing model zoo — hard routing (choose one model per
origin), soft combination (weight models), or a jointly trained MoE.

Before any router is built, the ceiling must be measured: how much would a
perfect selector gain over always-LightGBM on development folds? If the
ceiling is small, no routing mechanism — learned, agentic, or MoE — can pay,
and the direction should be abandoned in favour of candidate-model quality.
This step is a frozen-prediction diagnostic plus a bounded set of new
development-fold baseline runs; it trains no router and touches no test data.

## Development folds

Two chronological folds over the same regions (NSW1/QLD1/TAS1), protocol and
inputs identical to the production aligned configs (72h history, 24h
horizon, stride 1, common five quantiles, raw-RRP scoring):

- Fold A: train 2015-01-01 → 2021-01-01, selection 2021H1,
  evaluation 2021H2.
- Fold B: train 2015-01-01 → 2022-01-01, selection 2022H1,
  evaluation 2022H2.

Guards: no configuration in this step may contain a `test` split or load any
row at or after 2023-01-01 (`load_end_exclusive` caps at the fold's
evaluation end); the closed 2023–2024 test split is neither predicted on nor
scored. Fold B's evaluation period coincides with the production calibration
split; that is acceptable for a development diagnostic.

## Candidates

Development methods (Full_joint and variants) are excluded by decision; only
static baselines and foundation models participate. Candidate inclusion is
mechanical, fixed before any development-fold number exists:

1. Tier 1 (trained per fold; cheap trees): `baseline_lightgbm`,
   `baseline_xgboost`. Production hyperparameters and seed 2026, no re-tuning.
2. Tier 1b (no training; computed directly): persistence-24h (last 24
   observed hours as the forecast for all five quantiles) and weekly
   seasonal-naive (same hour one week earlier, quantiles from a trailing
   empirical window). These cost nothing and are legitimate oracle
   candidates in spike hours.
3. Tier 1c (no training; inference only): `chronos2_zeroshot`, the only
   foundation model that is Q-table comparable with published macro QS < 35.
   `tabpfn_v3` (39.48) is excluded by that threshold; `moirai2_zeroshot` and
   `timesfm25_xreg` are excluded as P-table point-only outputs.
4. Tier 2 (trained per fold; neural, subject to explicit compute budget,
   priority order fixed as patchtst → dlinear → gatets → timexer → linear →
   mlp). The minimum viable analysis is Tiers 1, 1b, 1c; whichever Tier 2
   models complete must be recorded, and ceilings must be reported both with
   and without Tier 2 so the candidate set's influence is visible.

Exclusion by published test-table QS is a compute-bound resource decision
made before any development-fold measurement; it cannot inflate the reported
ceilings.

## Metrics

Per (fold, region, origin): QS = 2 × mean pinball over the five common
quantiles and 24 horizon hours; MAE on the sorted q50. Aggregation mirrors
the main table: per-seed scoring, region-equal macro first, then the two
folds reported separately and as their mean. Primary endpoint: region-equal
QS. Paired per-origin deltas against always-LightGBM with bootstrap 95% CIs
over origins (cluster bootstrap by day).

## References (implementable policies)

- R0 always-LightGBM.
- R1 per-region static best: model chosen on fold A applied unchanged to
  fold B (and separately, chosen on each fold's selection split — both
  reported, labelled by where the choice was made).
- R2 simple ensembles: per-quantile mean and per-quantile median over all
  candidates; and over the top-3 by selection-split QS.
- R3 state-bucket routing ceiling: best fixed model per (region ×
  demand-regime state bucket), buckets defined exactly as in the R23/R24b
  ledger (as-of predispatch demand features only), chosen on the selection
  split, applied to the evaluation split. This is the strongest policy that
  a hard router could plausibly implement.

## Ceilings (hindsight; not implementable)

- C1 per-origin oracle: best model per origin.
- C2 per-(region × month) oracle.
- C3 per-origin oracle restricted to {LightGBM, persistence, chronos2}: how
  much of C1 survives with only one strong model plus free references.

## Pre-registered decision gates

All deltas are region-equal QS versus R0 on the same fold; a gate counts as
passed only if the delta is positive in BOTH folds A and B (point estimate;
CI overlap does not rescue a negative fold).

- G1 (stop/go for routing): C1 − R0 < 0.30 → stop; the direction is closed
  and the retrospective records it. C1 − R0 ≥ 0.30 → the router direction
  proceeds to G2.
- G2 (hard vs soft): R3 − R0 ≥ 0.15 → a hard-router prototype is justified
  as R25 proper. R3 − R0 < 0.15 while C1 − R0 ≥ 0.30 → the gap between
  implementable and perfect routing is too large for hard routing; only
  soft combination is carried forward.
- G3 (informational): if R2 (median ensemble) beats R0 in both folds,
  combination-first is flagged as the cheaper default regardless of G2.

Threshold rationale: LightGBM's published macro QS is ≈24.2 on test and
R23's entire contribution was 0.028 QS; a routing programme must show at
least an order of magnitude more headroom (0.30 ≈ 1.2% relative) to justify
new machinery, and the implementable variant must capture at least half of
whatever smaller margin survives.

## What this step does not do

No router, gate, or agent is trained; no prompt or threshold is tuned on
evaluation folds; the main comparison table and all frozen archives stay
untouched; the 2023–2024 test split stays closed. Results are development
diagnostics only and cannot be represented as method improvements.

## Planned artefacts

`outputs/forecasting/r25_ceiling_<fold>_<seed>_202611xx/` per fold with
per-origin prediction artifacts (candidate set as executed, fingerprints),
plus `scripts/analyze_r25_ceiling.py` producing: per-origin score table,
references R0–R3, ceilings C1–C3, paired deltas with CIs, gate outcomes, and
a result document `r25_selection_ceiling_result_<date>.md` in this
directory.
