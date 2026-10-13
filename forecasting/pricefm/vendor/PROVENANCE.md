# PriceFM source and adaptations

- Repository: https://github.com/runyao-yu/PriceFM
- Paper: https://arxiv.org/abs/2508.04875
- Pinned commit: `c72d1228bde80417d5cc782521328e02ab5401c3`
- Official file: `PriceFM/model.py`; copied without edits as `model.py` here.
- SHA256: `c5ae12e0a9ae85535cad30e41f84c3b58fe2265c464249f2705802ece74f03c2`
- License: not declared upstream; see `LICENSE_STATUS.md`.
- Architecture: March 2026 MoE update, not an assertion of the 2025 paper's
  original architecture or reported results.

`forecasting/pricefm/model.py` ports the official TensorFlow Dense projections,
softmax gate, swish experts, graph-weighted average (epsilon 1e-6), and median
anchor with cumulative absolute quantile steps into PyTorch. Dense parameters
use Glorot-uniform initialization and zero bias like the original. Defaults are
embedding size 32 and four experts. Runtime training does not import TensorFlow.

The shared AEMO protocol supplies one region per model. Consequently the graph
has exactly one node with gate=1; spatial transfer and multi-country pretraining
are not part of this adaptation. Historical price/load and historical calendar
are concatenated in the lag branch. Future calendar, future exogenous variables,
CTF variables, quantile shortfalls and broadcast origin context are concatenated
in that order in the lead branch. The seven canonical blocks are read directly,
so quantile shortfalls are not appended twice through the legacy native wrapper.

Dimensions are changed to the shared 72-hour history, 24-hour forecast and five
quantiles (0.05, 0.10, 0.50, 0.90, 0.95). The shared preprocessed inputs and target
space are used unchanged; this adapter fits no scaler and accesses no labels.
The standard shared pinball training and validation procedure is applied by the
runner. All weights are trained from scratch on the permitted training split;
the upstream pretrained checkpoint is neither downloaded nor loaded, because its
training chronology is not compatible with claiming an unseen 2023–2024 test.

Tests cover each information block, gradients through all heads and parameters,
negative prices, noncrossing quantiles, source integrity and checkpoint reload.
An optional TensorFlow test loads the exact snapshot, transfers Dense weights,
and compares forward results and input/parameter gradients for both nontrivial
multi-node masks and the actual one-node canonical adaptation. It can run in an
isolated TensorFlow 2.18.0 environment without modifying the experiment venv.

Validation performed on 2026-10-13: the normal environment passed seven tests,
including independent NumPy forward equations with nonuniform multi-node graph
masks (atol=rtol=2e-5). The optional official TensorFlow parity test is present
but is **not validated on this host**: an isolated tensorflow-cpu 2.18.0 runtime
failed a standalone all-ones matrix multiplication sanity check when oneDNN was
disabled, and raised SIGILL in the official TimeDistributed transpose with
oneDNN enabled, including one-thread/AVX2-restricted attempts. This is recorded
as a runtime blocker, not a passing equivalence claim. Detailed results are in
`logs/forecasting/aligned_baselines_20261013_extension/pricefm_parity.json`.
