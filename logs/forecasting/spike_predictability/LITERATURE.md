# Literature review: electricity price spike prediction (2026-10-09)

Evidence level: **full** = I read the full text or the author's own preprint; **abstract** = abstract or
publisher summary only; **search** = only what a web-search summary reported. Nothing here has been
checked against the published versions of record unless stated. Citations must be verified before they
enter the manuscript (ScienceDirect, MDPI, ResearchGate and RBNZ blocked direct download).

## 1. Spike occurrence as a classification or hazard problem (the closest prior work)

| Paper | Market / data | Method | Inputs | Evidence |
|---|---|---|---|---|
| Lu, Dong, Li (2005), *Electric Power Systems Research* 73:19-29 | Queensland | Data mining spike classifier | RRP, demand, reserve | search |
| Zhao, Dong, Li, Wong (2007), *IEEE Trans. Power Syst.* 22:376-385 | NEM (Queensland / NSW) | SVM and probability classifier | demand, supply, spike history, season/time, net interchange, dispatchable load | abstract |
| Amjady, Keynia (2010), *Electr. Power Syst. Res.* 80(3):318-327 | Queensland, PJM | Hybrid data model (wavelet + time-domain + calendar), relevance/redundancy feature selection | large candidate set | abstract |
| Mount et al. (2006), regime switching with time-varying parameters | PJM | Transition probabilities logistic in load / reserve margin | reserve margin | search |
| Kanamura, Ohashi (2008) | - | Spike model with demand trend | demand trend as well as reserve margin | search |
| Christensen, Hurn, Lindsay (2012), *Int. J. Forecasting* 28(2):400-411 | NEM half-hourly 2001-2007 | Autoregressive conditional hazard (ACH), probit; one-step probability | own spike history, load | abstract |
| "Models for short-term forecasting of spike occurrences in Australian electricity markets: a comparative study" (authors and journal from memory: Eichler, Grothe, Manner, Tuerk 2012, *J. Energy Markets*; verify) | NEM | Dynamic binary response models; AEMO PASA tried as reserve-margin proxy | - | search |
| Manner, Tuerk, Eichler (2016), *Energy Economics* 60:255-265 | NEM, four regions, half-hourly | Dynamic multivariate binary model with copula innovations | spike indicators; spillovers between regions | full (author preprint) |
| Stathakis, Papadimitriou, Gogas (2021), *Rev. Econ. Analysis* 13:65-87 | German intraday | Multiclass SVM; spike = above GPD-estimated 95th quantile of AR-EGARCH innovations; compared with NN, GBM | - | full |
| Zamudio Lopez, Zareipour, Quashie (2024), *Forecasting* 6:115-137 | Alberta | Tree classifiers (one interpretable), spike thresholds; statistical and economic evaluation | - | abstract |
| Shwe (2025), Univ. Sydney thesis | NSW, Feb 2024 - Mar 2025; spike = RRP >= 300; rate 1.64% | GAM vs SVM with class weights | lagged price (most important, spikes cluster), generation reserve (next) | search |
| "Hybrid data-driven methodology for normal and extreme electricity prices" (2026, ScienceDirect S2666546826002429) | Germany, Finland 2023-2024 | Classify negative / normal / extreme days, then forecast; sparse filtering + PNN, ELM-bootstrap | forecast renewables and residual load | search |
| Peng et al. (2024), IFAC-PapersOnLine 58:899-904 | - | "Real-time extreme electricity price forecasting" | - | search (title only) |
| Studies tabulated in the Forecasting 8(3):52 review (authors not identified by me) | ERCOT, PJM | TFT quantiles + extreme-price classifier; two-stage DNN spike classifier plus regressor (F1 about 0.63) | - | search |
| *Learning Rare Events: Deep Learning Approaches to Extreme Price Prediction*, Forecasting 8(3):52 (2026) | review, 20 studies 2020-2026 | PRISMA review | - | search; its conclusion: standard metrics hide poor rare-event detection, future work should be spike-aware formulation, imbalance handling, probabilistic forecasts, rare-event metrics |

Take-away: "reserve margin / demand trend drive spikes" and "spike history is the strongest predictor"
are established for the NEM since 2005-2012. Both agree with our experiment (price history was the
strongest added group). Claiming the precursor idea itself as novel would not survive review.

## 2. Probabilistic NEM forecasting and AEMO predispatch as input or benchmark

| Paper | Finding relevant to us | Evidence |
|---|---|---|
| Cornell, Dinh, Pourmousavi (2023/24), *Int. J. Forecasting*, arXiv 2311.07289 | South Australia. **Filters extreme spikes out of the training data** to improve quantiles 0.10-0.95; QRA-style ensemble of different training lengths; smoothing and autoregressive shift post-processing. AEMO predispatch mean absolute error 223.82 versus median 18.17 (Q-QRA: 41.12 / 17.26): predispatch misses spikes badly. Uses one predispatch vintage per day. | full |
| Lu et al. (2026), arXiv 2604.23908, SA NEM | Tree models (GBRT R2 0.88) beat LSTM / SVR; high MAPE for all | full (not read in detail) |
| *Learning the Grid: Transformers for EPF in the NEM*, Appl. Sci. 16:75 (2026) | Predispatch covariates dominate importance (NSW1 PD price > 60%); models without them 20-50% worse; MAE-trained models regress toward the centre at extremes | search |
| Bhattacharjee, Saha (2025), arXiv 2511.14158 and arXiv 2510.03657 | Predispatch forecasts are deterministic, no uncertainty; economic evaluation by battery arbitrage; errors grow with horizon | full (first), search (second) |
| Neural networks with GPD tails on NEM 1998-2013, *Energy Conversion and Management* (authors not identified by me) | Neural net body plus Generalized Pareto tail for peaks; spikes handled separately | search |

## 3. Probabilistic / distributional deep models (not spike-specific)

Quantile Neural Basis Models (arXiv 2509.14113), uncertainty-quantification comparison (arXiv 2509.19417),
EPF reviews (arXiv 2008.08004, 2602.10071, 2601.02856). Not read.

## 4. What I could not find (not proof it does not exist)

* Forecast **revisions across successive predispatch / PD PASA runs** used as a spike predictor. The Q-QRA
  authors note vintages exist but use one per day.
* A neural model that outputs point, quantiles **and** a spike probability, compared with XGBoost under
  quarterly refits on 2023-2024 NEM data (negative prices rising to 13% in QLD1).
* A trend-field plus event-field decomposition applied to price forecasting.

## 5. Implications for the manuscript

1. Add a "price spike prediction" paragraph to Related work: Lu 2005, Zhao 2007, Amjady 2010, Mount 2006,
   Christensen 2012, Eichler 2012, Manner 2016, Zamudio Lopez 2024, the 2026 review.
2. Cornell et al. is a direct design alternative: filtering spikes from training helps quantiles. Our
   capped-at-650 treatment is close to it, but it is not cited or compared. A "spike-filtered training"
   baseline for XGBoost and ours is cheap and likely to be requested by reviewers.
3. Any novelty claim must be limited to what section 4 supports, and checked again before submission.
