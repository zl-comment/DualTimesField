# Extension of the data to 2025-2026 (2026-10-10)

Built from the AEMO archives with the pipelines of the main period, checked against the stored series on the overlap:

| Series | Source | Coverage | Check |
|---|---|---|---|
| hourly price and demand | `PRICE_AND_DEMAND_<yyyymm>_<region>.csv` (`forecasting/build_price_demand.py`) | to 2026-08-31 | rebuilding 2024-11 and 2024-12 reproduces the stored hourly values exactly (price, demand, interval counts) |
| predispatch | MMSDM `PREDISP_ALL_DATA` (`forecasting/build_predispatch_exogenous.py`, rebuilt from 2024-11) | to 2026-08 | 1,463 overlapping origins per region equal the stored arrays exactly |
| PD PASA | MMSDM `PDPASA_REGIONSOLUTION` + `STPASA_REGIONSOLUTION` (`forecasting/build_pdpasa_exogenous.py`, rebuilt from 2024) | **to 2025-07-30 only** | 8,783 overlapping origins per region equal the stored arrays exactly |
| gas | DWGM series (already extended earlier) | to 2026-08-31 | - |

Merged arrays: `data/aemo_exogenous_holdout/merged/`; combined price series: `data/holdout/`
(`forecasting/merge_holdout_exogenous.py` refuses a merge whose overlap differs).

## PD PASA stops being comparable after July 2025

The pipeline keeps the PD PASA rows of run type `OUTAGE_LRC`. In the archive from 2025-08 on, only run type `LOR`
exists (0 `OUTAGE_LRC` rows). Both types are published side by side from at least 2025-01 to 2025-07, so they can be
compared: `MAXSPARECAPACITY`, `DEMAND10/50/90` and `AGGREGATECAPACITYAVAILABLE` are identical, but `UIGF` is not (the
`LOR` run is about 690 MW above the `OUTAGE_LRC` run on NSW1, against a mean of about 960, with no column of the
`LOR` run reproducing it; correlation of the nearest, the semi-scheduled capacity, is 0.86). The net load used by the
models (demand50 - UIGF) is therefore on a different scale from August 2025.

Decision: the held-out period is **2025-01-01 to 2025-07-29 (5,040 origins per region)**, where the pipeline is identical to
the main period. 2025-08 to 2026-08 would need either a calibration of the `LOR` UIGF to the old definition or models
without net load, which are different models; it is left for a separate, labelled experiment.

2025-2026 price regime (hourly, whole 2025-01..2026-08): share of hours above 300 AUD/MWh is 0.89% (NSW1), 0.41% (QLD1),
0.39% (TAS1) against 1.75%, 1.85%, 0.50% in 2023-2024; negative hours 8.6%, 14.5%, 3.0%.
