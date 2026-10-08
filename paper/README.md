# Paper materials

Figures and tables for the AEMO price-forecasting paper, generated from the
recorded results. Regenerate everything from the repository root:

```bash
PYTHONPATH=. .venv/bin/python paper/make_materials.py
```

The script reads `logs/forecasting/**` (metrics, tail scores, and
`significance/paper_final.json`) and, for the field-separation figure, the
saved checkpoints of `configs/aemo_forecast_scarcity_quantile_inputs.yaml` and
`configs/aemo_forecast_detected_dgf.yaml`. Figures are vector PDF and 300-dpi
PNG; tables are LaTeX fragments for `booktabs`.

## Manuscript

`manuscript/` holds the Applied Energy manuscript (`main.tex`, Elsevier
`elsarticle` class and `elsarticle-num-names` style from CTAN, audited
`references.bib`) with copies of the figures and tables it uses. Build from
`paper/manuscript/`:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Author details, the generative-AI declaration, and the code-availability
statement are placeholders to complete before submission. The current title is
**Dual-Field Electricity Price Forecasting with Predispatch**. Source-by-source
reference checks, corrections and remaining full-text limits are recorded in
[`REFERENCE_AUDIT.md`](REFERENCE_AUDIT.md).

The main framework figure is an editable vector schematic, generated without
checkpoints by `.venv/bin/python paper/draw_framework.py` (PDF, SVG and PNG).
After regenerating an included figure, copy its PDF into `manuscript/figures/`
before rebuilding the manuscript. The manuscript remains in the Elsevier
Applied Energy template; only the diagram adopts a compact ML-paper style.

## LaTeX components

`components/` wraps each figure and table in a float with a draft caption and
label, ready to `\input`. They need `graphicx` and `booktabs` and read files
through `\materialsdir` (default `.`, i.e. compile from `paper/`).

| Component | Float | Label | Role |
|---|---|---|---|
| `fig_architecture.tex` | `figure*` | `fig:architecture` | Main |
| `fig_settings.tex` | `figure*` | `fig:settings` | Main |
| `fig_comparison.tex` | `figure*` | `fig:comparison` | Main |
| `tab_main.tex` | `table*` | `tab:main` | Main |
| `tab_significance.tex` | `table*` | `tab:significance` | Main |
| `tab_ablations.tex` | `table*` | `tab:ablations` | Main |
| `tab_settings.tex` | `table*` | `tab:settings` | Main |
| `tab_tails.tex` | `table*` | `tab:tails` | Main |
| `fig_field_separation.tex` | `figure*` | `fig:field-separation` | Supplementary |
| `tab_field_separation.tex` | `table` | `tab:field-separation` | Supplementary |
| `fig_ablation_ladder.tex` | `figure*` | `fig:ablation-ladder` | Supplementary |
| `tab_ablations_static.tex` | `table*` | `tab:ablations-static` | Supplementary |

`preview.tex` inputs all of them in an ICML-sized two-column page
(`pdflatex preview.tex` from `paper/`).

## Story

Electricity prices combine a slowly varying level with rare scarcity spikes
and frequent negative-price troughs. DualTimesField (ICML 2026) describes time
series as a continuous trend field (CTF) plus a discrete event field (DGF) for
reconstruction. Transplanted into a forecaster, the DGF collapses and the CTF
absorbs the spikes; detecting events explicitly (largest departures from the
window median with non-maximum suppression) separates them again.

On a single fit to 2015-2021, adding the fields worsens the same base's MAE
but improves its CRPS. Quarterly refits and AEMO predispatch give pooled gains
in both metrics, although adding the fields still worsens raw-price MAE in
QLD1. The final model includes a per-hour calibrator and a validation-selected
lower bound, and beats PatchTST, DLinear and a linear baseline on MAE.
XGBoost retains the best overall MAE and CRPS. The final model has the lowest
negative-price scores among the main-table models, but the calibrator without
the lower bound has still better negative-price scores. The bound trades some
tail accuracy for better overall MAE. These are rolling hourly 24-hour forecasts,
not fixed-origin day-ahead auction forecasts.

## Key numbers (main setting, three seeds)

| | Raw MAE | Capped MAE | Raw CRPS |
|---|---:|---:|---:|
| Ours (calibrator + floor) | 36.64 | 25.29 | 25.07 |
| Linear base only | 37.60 | 26.72 | 26.04 |
| XGBoost | 35.52 | 24.54 | 23.11 |
| PatchTST | 38.85 | 28.10 | 24.33 |
| DLinear | 38.20 | 27.82 | 23.62 |
| Linear | 37.92 | 27.01 | 23.66 |

## Figures

| File | Content | Suggested caption |
|---|---|---|
| `figures/architecture.pdf` | Model structure | See `fig_architecture.tex` |
| `figures/settings.pdf` | Mean MAE of every model across static split, quarterly refit, and quarterly refit with predispatch | The dual fields fall behind a linear model on the static split and pull ahead of every model except XGBoost with recalibration and predispatch |
| `figures/framework.pdf` | Vector schematic of the additive model, forecast transformations and quarterly protocol | Main overview figure; editable source in `draw_framework.py` |
| `figures/cases.pdf` | Two test days chosen by fixed rules: deepest negative-price day (QLD1), predispatch phantom spike (NSW1) | Forecast stage by stage: base, + fields, + calibrator, + floor |
| `figures/same_data_comparison.pdf` | Per-region MAE in the main setting, raw and capped | Same splits, inputs, refits, and scoring; three seeds each |
| `figures/field_separation.pdf` | Transplanted Gabor DGF vs detected-event DGF (supplementary) | Event-to-price mean ratio in standardized asinh space on historical spike positions: 0.28-0.33 versus -0.04 to 0.00; not a monetary share |
| `figures/ablation_ladder.pdf` | Development on the static split (supplementary) | Each step changes one component |

## Tables

| File | Content | Source |
|---|---|---|
| `tables/same_data_comparison.tex` | Main comparison, raw and capped | `logs/forecasting/{price_floor,predispatch_rolling,baselines_rolling_pd}` |
| `tables/significance.tex` | Diebold-Mariano, final model against each comparator, MAE and CRPS | `logs/forecasting/significance/paper_final.json` |
| `tables/ablations.tex` | Fields against the linear base in each setting; hinge features, calibrator, floor, interconnector inputs | same |
| `tables/settings.tex` | Mean MAE across settings | `logs/forecasting/{linear_base,rolling,baselines,baselines_rolling,...}` |
| `tables/tails.tex` | Negative-price and spike-hour scores | `logs/forecasting/{predispatch_rolling,price_floor}/*_tail_metrics.json` |
| `tables/cost.tex` | Parameters and seconds per quarterly refit | `logs/forecasting/cost/seconds.tsv` (`scripts/benchmark_cost.sh`) |
| `tables/field_separation.tex`, `tables/ablations_static.tex` | Supplementary: field separation; structural and detector ablations on the static split | recomputed from checkpoints; `logs/forecasting/{detected_dgf,step5,...}` |

## Limitations to state

- XGBoost with the same inputs and refits has lower MAE (by 1.11 raw, 0.75
  capped) and CRPS; the gap is mostly QLD1 and the 100-300 AUD/MWh band.
- On raw prices, ours does not beat PatchTST, DLinear, or the linear model
  on CRPS (differences not significant); on capped prices it does.
- On the static split the fields cost 0.9 MAE against the same base but improve
  CRPS. With refits and predispatch, pooled MAE improves, while raw QLD1 MAE
  worsens by 0.42 before the calibrator and bound are added.
- The hard event-activity penalty has no training gradient; it contributes to
  the recorded objective and checkpoint selection, not parameter updates.
- Main-table scores average seed scores. CRPS tests and tail tables score
  seed-averaged quantiles; their differences are not directly interchangeable.
- The price floor is a post-hoc clip whose level is chosen per refit and
  region on validation data; the interconnector solution adds nothing beyond
  predispatch.
- October 2022 has no predispatch run history in AEMO's archive; its origins
  carry no predispatch information.
- On raw spike hours, ours is worse than PatchTST, DLinear, and the linear
  model (tail tests in `logs/forecasting/significance/paper_tails.json`).
- Tried and rejected, with records in `logs/forecasting/EXPERIMENT_RESULTS.md`:
  a mixture future event field, attention-located atoms, daily event echoes,
  a clock-anchored CTF extrapolation, window-normalized fields, and
  asinh-space field quantiles.
