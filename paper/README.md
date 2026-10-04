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
| `tab_re_price.tex` | `table*` | `tab:re-price` | Main (reference) |
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

Separating the fields is not enough to forecast better. On a single fit to
2015-2021, a linear regression on the same inputs beats the dual-field model:
reconstruction adds no information about the future, and the fields overfit
the training-period price level, which shifts in 2022-2024. Two changes,
standard in electricity-price forecasting practice, reverse this. Refitting
every test quarter removes most of the regime shift, and AEMO's predispatch
(the market operator's own dispatch simulation with current offers, the
primary source behind market news such as RE-Price's) adds information no
other input carries. In that setting the dual fields, added as a residual to a
linear base, improve both point and interval accuracy over the same base, and
the model beats PatchTST, DLinear, and a linear model on MAE. A diagnosis of
the remaining gap to XGBoost found over-extrapolated negative prices, which a
per-hour calibrator and a validation-chosen price floor reduce. XGBoost stays
best on mean MAE; the dual-field model is best in the negative-price lower
tail and, on capped prices, on spike hours.

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
| `figures/same_data_comparison.pdf` | Per-region MAE in the main setting, raw and capped | Same splits, inputs, refits, and scoring; three seeds each |
| `figures/field_separation.pdf` | Transplanted Gabor DGF vs detected-event DGF (supplementary) | With detected events, the event field carries 28-33% of spike-hour prices, against -0.04 to 0.00 before |
| `figures/ablation_ladder.pdf` | Development on the static split (supplementary) | Each step changes one component |

## Tables

| File | Content | Source |
|---|---|---|
| `tables/same_data_comparison.tex` | Main comparison, raw and capped | `logs/forecasting/{price_floor,predispatch_rolling,baselines_rolling_pd}` |
| `tables/significance.tex` | Diebold-Mariano, final model against each comparator, MAE and CRPS | `logs/forecasting/significance/paper_final.json` |
| `tables/ablations.tex` | Fields against the linear base in each setting; hinge features, calibrator, floor, interconnector inputs | same |
| `tables/settings.tex` | Mean MAE across settings | `logs/forecasting/{linear_base,rolling,baselines,baselines_rolling,...}` |
| `tables/tails.tex` | Negative-price and spike-hour scores | `logs/forecasting/{predispatch_rolling,price_floor}/*_tail_metrics.json` |
| `tables/re_price_reference.tex` | Ours (capped) beside RE-Price's reported numbers | RE-Price Tables 2-3 (reported, with news) |
| `tables/field_separation.tex`, `tables/ablations_static.tex` | Supplementary: field separation; structural and detector ablations on the static split | recomputed from checkpoints; `logs/forecasting/{detected_dgf,step5,...}` |

## Limitations to state

- XGBoost with the same inputs and refits has lower MAE (by 1.11 raw, 0.75
  capped) and CRPS; the gap is mostly QLD1 and the 100-300 AUD/MWh band.
- On raw prices, ours does not beat PatchTST, DLinear, or the linear model
  on CRPS (differences not significant); on capped prices it does.
- The fields help only with recalibration and predispatch; on the static
  split they cost 0.9 MAE against the same linear base.
- The price floor is a post-hoc clip whose level is chosen per refit and
  region on validation data; the interconnector solution adds nothing beyond
  predispatch.
- October 2022 has no predispatch run history in AEMO's archive; its origins
  carry no predispatch information.
- RE-Price's reported numbers use self-collected news, an undisclosed spike
  treatment, and a single fit; the comparison is a reference only
  (`logs/forecasting/PROTOCOL_ALIGNMENT.md`).
- Tried and rejected, with records in `logs/forecasting/EXPERIMENT_RESULTS.md`:
  a mixture future event field, attention-located atoms, daily event echoes,
  a clock-anchored CTF extrapolation, window-normalized fields, and
  asinh-space field quantiles.
