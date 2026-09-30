# Paper materials

Figures and tables for the AEMO price-forecasting paper, generated from the
recorded results. Regenerate everything from the repository root:

```bash
PYTHONPATH=. .venv/bin/python paper/make_materials.py
```

The script reads `logs/forecasting/**` and the saved checkpoints under
`outputs/forecasting/` (for the field-separation figure), so it needs the
trained models of `configs/aemo_forecast_scarcity_quantile_inputs.yaml` and
`configs/aemo_forecast_detected_dgf.yaml`. Figures are written as vector PDF
and 300-dpi PNG; tables are LaTeX fragments for `booktabs`.

## LaTeX components

`components/` wraps each figure and table in a float with a draft caption and
label, ready to `\input`. They need `graphicx` and `booktabs`, and read files
through `\materialsdir` (default `.`, i.e. compile from `paper/`); from another
directory set it first, e.g. `\newcommand{\materialsdir}{../paper}`.

| Component | Float | Label |
|---|---|---|
| `fig_architecture.tex` | `figure*`, 0.8 text width | `fig:architecture` |
| `fig_field_separation.tex` | `figure*` | `fig:field-separation` |
| `fig_ablation_ladder.tex` | `figure*` | `fig:ablation-ladder` |
| `fig_comparison.tex` | `figure*` | `fig:comparison` |
| `tab_main.tex` | `table*` | `tab:main` |
| `tab_significance.tex` | `table*` | `tab:significance` |
| `tab_ablations.tex` | `table*` | `tab:ablations` |
| `tab_field_separation.tex` | `table` (one column) | `tab:field-separation` |
| `tab_re_price.tex` | `table*` | `tab:re-price` |

`preview.tex` inputs all of them in an ICML-sized two-column page
(`pdflatex preview.tex` from `paper/`).

## Story in one paragraph

Electricity prices combine a slowly varying level with rare scarcity spikes
and surplus troughs. DualTimesField (ICML 2026) describes time series as a
continuous trend field (CTF) plus a discrete event field (DGF), but it is a
reconstruction model. Transplanted into a forecaster, the CTF absorbs the
spikes and the DGF collapses: its event field has a standard deviation of
about 0.05 and carries none of the spike hours. Linear heads reading the raw
history do exactly as well, so the decomposition adds nothing. We detect
sparse price events explicitly, as the largest departures from the window
median with non-maximum suppression. The event field is then fixed by the
data, and the decomposition loss forces the CTF to describe only what
remains. Each field's heads read the public AEMO forecasts that drive it:
gas price and net load for the level, spare capacity and its shortfalls for
events. A trigonometric gate, with per-quantile variants, fuses the two. On
identical data and inputs, the 184k-parameter model has the lowest MAE of
all re-implemented baselines in every region, and pooled Diebold-Mariano
tests favour it over each of them.

## Figures

| File | Content | Suggested caption |
|---|---|---|
| `figures/architecture.pdf` | Model structure | Detected-event dual-field forecaster. Events detected in the history form the DGF; the CTF fits the remainder. Field-specific linear heads read field-matched AEMO forecasts and are fused by a trigonometric gate. |
| `figures/field_separation.pdf` | Top: one NSW1 test window, transplanted Gabor DGF vs detected-event DGF. Bottom: share of past spike hours carried by the event field | Transplanted, the CTF absorbs the spikes and the DGF is flat. With detected events, the event field carries 28-33% of spike-hour prices in every region, against -0.04 to 0.00 before. |
| `figures/ablation_ladder.pdf` | Three-region mean test MAE and CRPS~ along the development chain | Each step changes one component. Steps 4-10 are means over seeds 2026-2028. |
| `figures/same_data_comparison.pdf` | Per-region MAE against re-implemented XGBoost, GRU, DeepAR on raw and 650-capped prices | Same splits, inputs, and scoring for all models; three seeds each. |

## Tables

| File | Content | Source |
|---|---|---|
| `tables/same_data_comparison.tex` | Main result: MAE per region, mean MAE ± seed std, window RMSE, 90% AIS, CRPS~ on raw and capped prices | `logs/forecasting/detected_dgf`, `step5`, `baselines` |
| `tables/significance.tex` | Diebold-Mariano tests, pooled and per region | `logs/forecasting/significance` |
| `tables/ablations.tex` | Structural ablations and detector settings with paired per-seed MAE differences | `logs/forecasting/{detected_dgf,adaptive_dgf,step5,...}` |
| `tables/field_separation.tex` | Event-field share of spike-hour prices, transplanted vs ours | recomputed from checkpoints |
| `tables/re_price_reference.tex` | Ours on 650-capped prices beside RE-Price's reported numbers | RE-Price Tables 2-3 (reported, with news) |

## Limitations to state

- TAS1 ties XGBoost and GBDT (not significant under either price treatment),
  and in QLD1 the gain over the transplanted model and the raw-history
  ablation is not significant on raw prices.
- XGBoost keeps the best CRPS~, and the best 90% AIS on capped prices.
- RE-Price's reported numbers use self-collected news and an undisclosed
  spike treatment; the 650 cap is inferred from its error ratios, and its
  TAS1 numbers are not reproduced by any local model
  (`logs/forecasting/PROTOCOL_ALIGNMENT.md`).
- Tried and rejected, with records in `logs/forecasting/EXPERIMENT_RESULTS.md`:
  a mixture future event field, attention-located atoms, daily event echoes,
  and a clock-anchored CTF extrapolation.
