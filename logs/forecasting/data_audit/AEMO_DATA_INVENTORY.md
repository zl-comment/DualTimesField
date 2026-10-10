# AEMO data inventory for spike drivers (2026-10-10)

Purpose: which published data can stand in for the causes of price spikes (supply stack position, outages,
constraints, reserve notices, weather-driven demand and renewables), and whether they are point-in-time at a
forecast origin. Sizes are compressed MMSDM archive sizes summed over 2015-01..2024-12, from the directory
listings of `nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/<y>/MMSDM_<y>_<mm>/.../DATA/` (fetched
through the local proxy). Free disk on this machine: 180 GB, so large tables must be streamed month by month.

## Already used
PD PASA region solution (spare capacity, demand 10/50/90, UIGF, available capacity), predispatch (price, demand,
available generation, net interchange), predispatch interconnector flows and limits (no gain), DWGM gas price,
calendar. All are point-in-time through the latest run published by the origin.

## Candidates

| Data (table) | What it tells us | Point-in-time? | Size, coverage | Value / cost |
|---|---|---|---|---|
| **Supply stack: `BIDPEROFFER` + `BIDDAYOFFER`** | Per unit and 5-min interval, the MW offered in each of 10 price bands (`BANDAVAIL1..10`) and the band prices (`PRICEBAND1..10`); the stack position of demand is the direct cause of price jumps. `BIDDAYOFFER` also carries `REBIDEXPLANATION` text. | **Yes**: every version has `OFFERDATE` and `VERSIONNO` (checked on 2015-01), so the stack as of the origin is the latest version offered at or before it. Only through 2024-07 | `BIDPEROFFER` 22 GB (2015-01..2022-05) + `BIDPEROFFER1/2` 100 GB (2022-06..2024-07, 1-7 GB/month each after five-minute settlement); `BIDDAYOFFER` 4.7 GB | Highest value, highest cost: about 125 GB compressed, roughly 1 TB of CSV to stream. From 2024-08 only `BIDPEROFFER_D` (final, not point-in-time) is archived |
| Market notices (`MARKETNOTICEDATA`) | `RESERVE NOTICE` rows carry forecast lack-of-reserve (LOR1/2/3) conditions by region, interval and MW reserve requirement versus available (text is base64 in `REASON`); also `INTER-REGIONAL TRANSFER`, `RECLASSIFY CONTINGENCY`, `PRICES SUBJECT TO REVIEW` | Yes (`EFFECTIVEDATE` is the issue time); 345 notices in 2024-06 | 0.01 GB, 2015-01..2024-07; **not in the archive from 2024-08** | Very cheap, directly a scarcity forecast by the operator. Needs text parsing; last 5 months of the test period must come from another source |
| `DISPATCHLOAD`, `DISPATCH_UNIT_SCADA` | Per unit availability, initial MW and actual output every 5 min: outages (availability drops), generation mix, wind/solar output | Observed, usable only as history before the origin | 8.4 GB + 1.9 GB, full 2015-2024 | Medium. Gives the unobserved "persistent state" (outage, ramp) that price history now proxies. Needs `DUDETAILSUMMARY` (0.02 GB) for fuel and region |
| `DISPATCHCONSTRAINT` | Binding network constraints and marginal values per interval | Observed history | 15 GB, full | Medium; sparse and region-specific, mapping constraints to regions is work |
| `ROOFTOP_PV_FORECAST` / `_ACTUAL` | Rooftop solar forecast and estimate, which enters operational demand | Forecast has run time | 2.8 GB, 2016-08..2024-12 | Medium-low: PD PASA demand likely already nets it |
| `MTPASA_REGIONRESULT`, `MTPASA_LOLPRESULT` | Weeks-ahead reserve outlook and loss-of-load probability | Yes | 0.05 GB, 2018-05..2024-12 | Cheap, slow precursor; short history before 2018 |
| `P5MIN_REGIONSOLUTION` | Five-minute predispatch (one hour ahead) | Yes | 4.4 GB, full | Useful only for horizons under one hour; not for 24 h |
| `PDPASA_INTERCONNECTORSOLN`, `PDPASA_CONSTRAINTSOLUTION` | Forecast interconnector limits and constraints | Yes | 1.3 + 1.2 GB, **2021-02..** only | Cheap, short history; interconnector inputs gave no gain so far |
| `STPASA_REGIONSOLUTION` | 7-day reserve outlook | Yes | 8.9 GB, full | Already used as the tail of PD PASA |
| FCAS requirements (`DISPATCH_FCAS_REQ`) | Contingency reserve demands | Observed | about 27 MB/month | Low |
| **Weather (temperature, wind, irradiance)** | Not in AEMO MMSDM. Demand, wind and solar forecasts in PD PASA already embed the operator's weather forecasts | Observed reanalysis (ERA5, BoM) is not point-in-time as a forecast; archived NWP forecasts exist only for recent years | external | Expected added value is the forecast error of demand and renewables only; test last |

## Reading
1. The only source that observes the supply stack, the stated cause of price jumps, is the bid data; it is point
   in time but needs about 125 GB of downloads and heavy streaming, and has no point-in-time version for the last
   five months of 2024.
2. Market notices (reserve notices) are the cheapest direct precursor, but are absent from the archive after
   2024-07.
3. Unit availability and SCADA can replace what price history now proxies (outages, ramps) at modest cost.
4. A staged plan: (a) notices + MTPASA + SCADA/availability, about 12 GB, test with the existing spike classifier
   whether exogenous-only features reach the "exogenous + price history" level (PR-AUC 0.37-0.44 against
   0.15-0.27); (b) only if that falls short, build the supply-stack features.
