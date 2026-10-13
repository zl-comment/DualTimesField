# TimeXer upstream source and AEMO adaptation

- Paper: [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072), NeurIPS 2024.
- Official repository: https://github.com/thuml/TimeXer
- Pinned commit: `76011909357972bd55a27adba2e1be994d81b327`; retrieved 2026-10-13.
- Exact upstream snapshots and SHA256 values: `source_manifest.json`.
- The pinned upstream repository has no license file. No license grant is inferred here.

`upstream_*.py` are archival originals. Runtime `model.py` changes only the two local dependency import paths. `layers.py` contains verbatim selected classes from the snapshots, with only the imports needed for those classes; unused Reformer/einops dependencies are omitted. The upstream endogeneous patch/global token, self-attention, exogenous cross-attention, feedforward layers, positional encoding and normalization are preserved.

`forecasting/timexer/model.py` adapts the official MS model:

1. Common history price is moved from the first to the final variable position expected by upstream. Historical demand and calendar use upstream inverted embeddings.
2. History is 72 hours and horizon 24 hours. Patch size 24, width 512, 8 attention heads, 3 layers, feedforward width 512, GELU, dropout 0.1 follow the official NP EPF preset and parser defaults; the history length is reduced to the common protocol.
3. The linear prediction head emits `24 * 5` values. All five levels use the unchanged shared pinball training, validation selection and raw-price scoring rules.
4. Future calendar, future exogenous drivers, CTF/quantile exogenous features and origin context become additional variate tokens via a separate horizon-length inverted embedding. They join the native exogenous cross-attention context. This is an explicit future-covariate adaptation, not the original historical-only experiment. No labels enter these tokens.
5. Internal instance normalization matches upstream; every output quantile is inverted using the target history scale. It does not replace or refit the shared training-set preprocessing.
6. Each region is trained independently from scratch, with no external pretraining or neighbor-region inputs.

Tests compare the historical pathway against the official model with identical weights and a widened head, and verify gradients reach all seven canonical input blocks. The result is an AEMO quantile adaptation, not a reproduction of the paper's original data scores.
