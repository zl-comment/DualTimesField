# Full V3: conditional historical dual-field priors

Local implementation: memory, fixed-basis decomposition, retrieval priors,
residual transformation, orchestration and reporting. This is not the retired
Full V2 or the original neural INR/Gabor Full.

Official tree backend: [microsoft/LightGBM](https://github.com/microsoft/LightGBM).
The existing native quantile horizon fitter is reused with identical budgets.
Retrieval-related prior art: [RAFT](https://arxiv.org/abs/2505.04163);
no source code is copied from that project.

See `docs/full_v3_design_20261013.md` for equations, five contrasts, information
boundaries, limitations and execution commands. Formal settings are centralized
in `configs/full_v3/experiment.yaml` to prevent contrast-specific budget drift.
