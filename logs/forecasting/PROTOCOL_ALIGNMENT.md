# Aligning the comparison with RE-Price

This note checks how far the local results can be compared with the numbers
reported by RE-Price ([Chen et al., Applied Energy 426 (2026)
128712](https://doi.org/10.1016/j.apenergy.2026.128712)). No model was
retrained for it. It re-scores the current research trunk and three
same-data baselines under several price treatments.

- Model: soft-saturated PD PASA net-load trunk
  (`aemo-pdpasa-netload-softclip-ctf-30epoch`, `bfd5f55`), mean over seeds
  2026, 2027, and 2028.
- Test set: 17,521 rolling hourly origins per region, 2023-01-01 to
  2024-12-31, the same 70/10/20 chronological split that RE-Price states.
- Metrics: MAE and RMSE computed per 24-hour window and then averaged, which
  is the reading of RE-Price's "averaged across prediction horizons" that
  makes its RMSE almost equal to its per-window SDE (eq. 29).

## What RE-Price states and what its numbers imply

RE-Price cites AEMO's real-time regional reference price (RRP) as its source
and does not mention clipping, winsorizing, or removing spikes. Its baselines
(GPT4TS, Informer, DeepAR, GRU, XGBoost) receive the same inputs as RE-Price,
including the noisy temperature proxy and sentence-BERT embeddings of
WattClarity news, so they are not news-free baselines.

The test years contain spikes up to 16,157 (NSW1), 15,450 (QLD1), and 12,488
(TAS1) AUD/MWh. Hours above 300 AUD/MWh are 1.75%, 1.85%, and 0.50% of test
hours. On raw prices the trunk's window RMSE is 2.06, 2.05, and 1.45 times its
MAE, and the seasonal naive forecast's ratio is 2.17, 2.26, and 1.69.
RE-Price reports 1.45, 1.44, and 1.20, and its own XGBoost 1.35, 1.43, and
1.07. Such ratios are reachable only after spikes are removed or capped.

| Window RMSE / MAE of the trunk | NSW1 | QLD1 | TAS1 |
|---|---:|---:|---:|
| Raw prices | 2.06 | 2.05 | 1.45 |
| Prices capped at 1000 | 1.53 | 1.54 | 1.25 |
| Prices capped at 650 | 1.43 | 1.44 | 1.22 |
| Prices capped at 500 | 1.39 | 1.39 | 1.21 |
| Prices capped at 300 | 1.34 | 1.34 | 1.19 |
| RE-Price, Table 2 | 1.45 | 1.44 | 1.20 |

A single cap near 650 AUD/MWh reproduces RE-Price's ratio in all three
regions. The ratio-matched cap found by a 25 AUD/MWh grid search is 725 for
NSW1 and 650 for QLD1. TAS1's ratio is flat (1.19-1.22) for caps from 125 to
650, so its matched cap is not identified.

## Scores under each treatment

Each cell is MAE / window RMSE in AUD/MWh. `cap c` clips both actual and
forecast prices at `c`. `drop300` removes every 24-hour window with an actual
price above 300, which keeps 81.5% (NSW1), 76.4% (QLD1), and 94.9% (TAS1) of
windows, so it scores a different subset of the test set.

The two GBDT rows are per-horizon gradient-boosted trees
([`forecasting/gbdt_baseline.py`](../../forecasting/gbdt_baseline.py)) fitted on
the same splits. "Trunk inputs" uses the same history, gas-price context, PD
PASA spare capacity, and PD PASA net load as the trunk. "History and
calendar" uses only price and demand history and calendar features.

### NSW1

| Forecast | Raw | Cap 1000 | Cap 500 | Cap 300 | Drop300 |
|---|---:|---:|---:|---:|---:|
| Dual-field trunk (MAE ± seed std) | **48.53 ± 0.32** / **100.10** | **32.42 ± 0.33** / **49.74** | **29.48 ± 0.25** / **40.96** | **27.24 ± 0.14** / **36.61** | **24.83 ± 0.31** / 33.41 |
| GBDT, trunk inputs | 49.94 / 102.44 | 33.81 / 51.68 | 31.11 / 42.88 | 29.23 / 38.66 | 25.08 / **33.17** |
| GBDT, history and calendar | 51.55 / 104.84 | 35.42 / 54.10 | 32.72 / 45.29 | 30.83 / 40.98 | 27.09 / 35.90 |
| Seasonal naive | 72.51 / 157.01 | 42.91 / 68.31 | 38.02 / 53.34 | 35.04 / 47.33 | 42.55 / 76.87 |
| RE-Price, as reported | 23.48 / 34.15 | | | | |
| RE-Price's XGBoost, as reported | 38.63 / 51.99 | | | | |

### QLD1

| Forecast | Raw | Cap 1000 | Cap 500 | Cap 300 | Drop300 |
|---|---:|---:|---:|---:|---:|
| Dual-field trunk (MAE ± seed std) | **42.51 ± 0.49** / **87.26** | **32.75 ± 0.49** / **50.37** | **30.10 ± 0.46** / **41.83** | **28.03 ± 0.45** / **37.52** | 25.62 ± 0.49 / 34.66 |
| GBDT, trunk inputs | 42.58 / 87.93 | 32.79 / 50.47 | 30.23 / 41.86 | 28.34 / 37.73 | **24.79** / **32.88** |
| GBDT, history and calendar | 44.87 / 90.95 | 35.08 / 53.49 | 32.52 / 44.74 | 30.56 / 40.28 | 27.28 / 35.66 |
| Seasonal naive | 59.17 / 133.54 | 40.71 / 65.24 | 36.30 / 50.73 | 33.56 / 45.00 | 39.02 / 71.29 |
| RE-Price, as reported | 25.85 / 37.15 | | | | |
| RE-Price's XGBoost, as reported | 44.70 / 64.05 | | | | |

### TAS1

| Forecast | Raw | Cap 1000 | Cap 500 | Cap 300 | Drop300 |
|---|---:|---:|---:|---:|---:|
| Dual-field trunk (MAE ± seed std) | 34.55 ± 0.51 / 49.96 | 31.98 ± 0.51 / 40.17 | 31.38 ± 0.51 / 37.95 | 30.96 ± 0.51 / 36.89 | 29.39 ± 0.39 / 34.83 |
| GBDT, trunk inputs | **33.84** / **49.44** | **31.26** / **39.66** | **30.67** / **37.44** | **30.24** / **36.36** | **28.33** / **33.96** |
| GBDT, history and calendar | 35.12 / 50.87 | 32.55 / 41.09 | 31.95 / 38.88 | 31.53 / 37.81 | 29.58 / 35.39 |
| Seasonal naive | 39.83 / 67.37 | 34.68 / 48.27 | 33.53 / 44.24 | 32.84 / 42.55 | 34.99 / 52.60 |
| RE-Price, as reported | 18.96 / 22.81 | | | | |
| RE-Price's XGBoost, as reported | 37.56 / 40.33 | | | | |

## Findings

| Finding | Evidence |
|---|---|
| RE-Price's error ratios imply spike treatment | Its RMSE/MAE ratios of 1.45, 1.44, and 1.20 are reached by the trunk only when prices are capped near 650 AUD/MWh. On raw prices the ratio is 2.06, 2.05, and 1.45 |
| Spike treatment explains most of the NSW1 and QLD1 gap | With spike windows removed, the trunk scores 24.83 / 33.41 (NSW1) and 25.62 / 34.66 (QLD1) against RE-Price's 23.48 / 34.15 and 25.85 / 37.15; this subset keeps 81.5% and 76.4% of windows |
| The TAS1 gap is not explained | TAS1 has few spikes, and no treatment brings any local forecast below 28.3 MAE, against 18.96 reported. RE-Price's TAS1 XGBoost ratio (1.07) is below every local ratio (at least 1.18) |
| RE-Price's baselines are weak on this data | Under every treatment, both local GBDT baselines beat RE-Price's reported XGBoost. At the ratio-matched caps, the history-and-calendar GBDT scores 34.03 (NSW1) and 33.38 (QLD1) against 38.63 and 44.70 |
| The dual-field trunk leads the same-data baselines in NSW1 and QLD1 | It has the lowest MAE under raw and capped scoring in NSW1 and QLD1. In TAS1 the trunk-input GBDT is ahead by 0.7-1.1 MAE under every treatment |

## Recommended reporting

1. Report the headline comparison on raw RRP against baselines run on the same
   data and inputs: seasonal naive, the two GBDT variants, and the ablation
   chain of this repository.
2. Report RE-Price's published numbers as an external reference, noting that
   it uses self-collected news and that its error ratios imply an undisclosed
   spike treatment.
3. Add this capped and spike-window-removed sensitivity table, so that
   readers can compare magnitudes under the treatment RE-Price appears to use.
4. Treat TAS1 as unresolved until its data or scoring differences are known.

## Reproduction

```bash
python -m forecasting.protocol_alignment collect \
  --config configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml \
  --checkpoint-root outputs/forecasting/pdpasa_netload_softclip_ctf \
  --output-dir outputs/forecasting/protocol_alignment/trunk
python -m forecasting.gbdt_baseline \
  --config configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml \
  --region NSW1 --output outputs/forecasting/gbdt_baseline/trunk_inputs/NSW1.npz
python -m forecasting.gbdt_baseline \
  --config configs/aemo_forecast_asinh_price_space_quantiles.yaml \
  --region NSW1 --output outputs/forecasting/gbdt_baseline/history_calendar/NSW1.npz
python -m forecasting.protocol_alignment report \
  --model outputs/forecasting/protocol_alignment/trunk \
  --baseline gbdt_trunk_inputs=outputs/forecasting/gbdt_baseline/trunk_inputs \
  --baseline gbdt_history_calendar=outputs/forecasting/gbdt_baseline/history_calendar \
  --output logs/forecasting/protocol_alignment/report.json
```

Repeat the GBDT commands for QLD1 and TAS1. Each GBDT run took about six
minutes with `OMP_NUM_THREADS=20`. The report is stored in
[`protocol_alignment/report.json`](protocol_alignment/report.json), and the
GBDT summaries are copied to
[`protocol_alignment/gbdt/`](protocol_alignment/gbdt/).
