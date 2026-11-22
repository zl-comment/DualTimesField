# R25 selection-ceiling result (2026-11-22)

Plan `r25_selection_ceiling_plan_20261119.md` (v4.1). Branch
`research/r25-selection-ceiling`. Candidate pool as executed:
lightgbm seeds 2026/2027/2028, xgboost 2026, chronos2_zeroshot,
persistence-24h, weekly seasonal naive — 63 cells over folds A/B/C,
selection+evaluation artifacts under
`outputs/forecasting/r25_ceiling_20261119/`; analysis
`scripts/analyze_r25_ceiling.py`, output `analysis_final.json`.
Neural baselines and tabpfn were not run (compute-bound optional tiers;
pool composition recorded per plan section 6). The 2023–2024 test split
was never loaded, predicted on, or scored.

## Verdict: G1 fails on all three conditions; direction closed as pre-registered

| Fold | R0 | S2 | ΔS2 | S3 | ΔS3 | C1−C1_null |
|---|---:|---:|---:|---:|---:|---:|
| A (2019H2) | 9.59 | 9.63 | −0.042 (−0.44%) | 9.41 | +0.173 (+1.81%) | +0.69 |
| B (2021H2) | 17.83 | 17.79 | +0.034 (+0.19%) | 17.54 | +0.286 (+1.60%) | +1.18 |
| C (2022H2) | 39.39 | 37.07 | +2.326 (+5.90%) | 37.18 | +2.213 (+5.62%) | +10.81 |

- (a) fails: S2 negative in fold A, +0.19% below bar in fold B.
- (b) fails: fold A NSW1 QS harm +0.111 > 0.05 (S2 switching to
  xgboost/seed2027 buckets in the calm fold slightly hurts); fold C QLD1
  MAE harm +3.953 >> tolerance 0.458 (routing 438 origins to chronos2
  helps NSW1 QS by −7.18 but degrades QLD1 MAE).
- (c) fails: bootstrap p = 1.000 / 0.099 / 0.000.
- Section 5.4 triggers: not met (net headroom 0.69 < 1.00 in fold A, and
  S2 failed on (b)/(c), not only (a)). No redesign round.
- S3 branch: (b) and (c) pass everywhere (p = 0.000 all folds, jackknife
  ≤ 9.3%); (a) fails only on fold A's absolute margin (+0.173 < 0.25;
  relative +1.81% passes). Under the pre-registered wording this is a
  fail — no post-hoc reinterpretation.

## Admission-band boundary effect (mandatory disclosure)

Per-bucket 1.25× admission: persistence was blocked in every bucket of
every fold (average QS far outside the band); seasonal naive was admitted
only in a few high-demand buckets (mostly QLD1, plus isolated NSW1 C-b3);
chronos2 was admitted in most fold-C buckets and scattered A/B buckets;
the three LightGBM seeds and xgboost passed nearly everywhere. The band
therefore excluded the worst average performers from S2/S3 while leaving
the documented fold-C naive oracle wins (argmin share up to 43% in the
6-candidate preview) mostly outside the implementable pool: with
admission+margin, snaive was never chosen by S2 in any fold. A Tier-0
full-pool oracle retains those wins in the C1 report only.

## Fragility and noise calibration

C1_null (3-seed same-model oracle): 0.22 / 0.52 / 0.94 QS region-equal in
A/B/C — the noise floor the v3 gate would have mistaken for headroom.
Net headroom (C1 − C1_null) is +0.69 / +1.18 / +10.81. S3's paired
deltas are significant in all folds (block-bootstrap p = 0.000,
jackknife max single-day contribution 3.6% / 9.3% / 2.6%). S2's fold-B
jackknife share (69.9%) is ill-defined (|Δ| ≈ 0.03). Fold-A S2 delta is
negative, so its fragility items are moot.

## Scope and limitations

Retrospective development-fold measurement, not a blind test; no claim
about 2023–24 performance or about routing versus the development-method
line. 2020H2 excluded by pre-registration (COVID regime). Four origins
per day (00/06/12/18), not hourly. Selection-period scores of
early-stopped trees are mildly optimistic (rounds chosen on the same
period); affects all tree candidates equally. Bucket feature and quartile
count were pre-registered before any S2 evaluation number was seen.

## What survives the negative verdict

1. Crisis-period complementarity is real: fold C carries +10.8 net
   headroom; routing to chronos2 improves NSW1 QS by 7.2 in isolation —
   the largest single-mechanism effect measured in this research line.
2. The zero-parameter median combination improves every fold
   significantly (+1.6% to +5.6%) but missed the pre-registered
   absolute bar in the calm fold by 0.077 QS.
3. The value is region- and regime-conditional: TAS1 never moves off R0;
   QLD1 trades QS against MAE — consistent with the R23/R24 heterogeneity
   record. Any future method should be shaped as per-region/per-regime
   switching or combination, not a uniform mechanism.
4. Per the pre-registered rules the selection/routing direction is
   closed here. Any follow-up (e.g. a combination-focused arm with
   fold-scale-appropriate bars) requires a new pre-registration on a new
   branch; it must not reuse this evaluation as its selection set.

Reproduce: `PYTHONPATH=. .venv/bin/python scripts/analyze_r25_ceiling.py
--label final` (reads frozen artifacts only).
