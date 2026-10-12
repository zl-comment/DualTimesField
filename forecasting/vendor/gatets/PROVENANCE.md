# GateTS upstream source and adaptation

Authors: Kyrylo Yemets, Mykola Lukashchuk, Ivan Izonin.
Paper: https://arxiv.org/abs/2508.17515
Official repository: https://github.com/ruiite/MOE_time_series
Pinned commit: `a3b821fff6fda3bde8124a79564a8d33df9732f1`.
Retrieved: 2026-10-11.

`new_moe.py` is copied byte-for-byte from that commit (SHA256
`0e7d9c23e55b3dcf9ee8984e1a21cb5e4b17353d9ee7cd4d5ffbf4c1b0d4bb54`).
`UPSTREAM_README.md` preserves the upstream attribution, defaults and license
statement, “CC-BY 4.0”. `LICENSE.txt` is the CC Attribution 4.0 International
legal text from https://creativecommons.org/licenses/by/4.0/legalcode.txt.
Only the PyTorch model module is imported; its example executable training
code is not run. No extra runtime dependency is needed beyond PyTorch.

The local wrapper `forecasting/gatets_baseline.py` is an AEMO adaptation,
not a reproduction of the paper's reported dataset scores. It uses the
README defaults: dimension 48, 4 heads, 1 layer, FFN 64, 16 decoder experts,
2 expert heads, top-k 2, dropout 0.1, and `gating="kronecker"`.
Changes outside the vendored source:

- A 72-hour price-only input and 24-hour forecast horizon.
- The output layer has 24 times 5 outputs for the local fixed quantiles.
- The local trainer optimizes standardized raw-price pinball with the common
  validation rule; historical prices retain the common asinh transform.
- `gatets_adapter` adds the same linear known-input adapter used for the
  local DLinear/PatchTST controls: the complete known future/calendar/context
  profile and non-price historical variables. `gatets` is price-history-only.
- The adapter includes quantile exogenous shortfalls through the trainer's
  `FairWindows` concatenation into `ctf_exogenous`.

Upstream implementation details are deliberately preserved: its top-k route
masks probabilities and divides by k (not the selected probability sum);
it evaluates all expert outputs before masking them. Thus top-k activation
must not be reported as demonstrated sparse computation or speedup. The
local wrapper does not add load-balancing losses or silently change routing.
