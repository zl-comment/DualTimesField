# R15: frozen Kev controller and frozen Chronos predictor

Branch: `research/kev-frozen-agent-20261102-r15`.
Base: `46e92fd9053a8c5e63619435e599496e023c669c`.

The user authorized deployment in tmux and subsequent Agent development on the
last two GPUs. GPU 6 hosts Kev; GPU 7 runs the frozen Chronos predictor. Existing
processes, environments, baseline predictions, and the original main table are
preserved. GPU 6 already had another process using approximately 2 GB.

## Deployment

The runtime is isolated under `/root/workspace/services/kev`. Source commit,
backbone revision, adapter revision, GPU UUIDs, endpoint, and request limits are
recorded in `configs/agent_forecasting/r15_kev_runtime.json`.

Kev uses its official `kev.serve` implementation, not an ordinary vLLM generation
endpoint. This first deployment uses BF16, the reference unfused path, and disables
CUDA graphs. Consequently the upstream fused-path latency figures do not describe
this deployment. The service binds only to `127.0.0.1:8009` and loads offline after
download verification. `scripts/check_kev_service.py` must pass before experiments.

The Qwen weight shards are also available from the official Qwen ModelScope repo
at revision `da6287668dd32f301f22a92f28e465b837653677`. Their SHA-256 values match
the pinned Hugging Face revision. Downloading from that source changes the transport,
not the model. All large model files are checked against the official HF hashes.

CUDA packages reused from existing environments are copied into the new environment
only after checking every packaged file against its RECORD hash. The original
environments are not modified. Runtime dependencies are frozen after installation.

Start: `bash scripts/deploy_kev_tmux.sh` after model and environment readiness.
Inspect: `tmux attach -t kev-4b`; logs: `/root/workspace/services/kev/logs/server.log`.
Startup alone is not a readiness or forecasting-quality claim.

## Agent v1

This implements a bounded controller with two versioned skills (`triage` and
`select_forecast`). It supersedes the retrieval-only R14 proposal for the first
executable prototype. Historical news retrieval is not implemented or claimed.

For each origin, the coordinator creates a separate immutable snapshot containing
the same 72-hour history, ten calendar features, and direct 24-hour target axis as
the existing baseline protocol. No outcome array is included. Kev receives only
the current snapshot's numerical summaries and approved tool results, never cache
paths, arbitrary shell access, internet search, future origins, or test labels.

Kev can request a recent-fit tool: frozen Chronos predicts the last 24 already
observed hours using the preceding 48 hours. The tool reports numerical error for
three registered covariate profiles. Kev then chooses a profile for the future
24 hours using all 72 history hours:

- BASE: original price, past demand, and calendar mapping.
- calendar: price and calendar, without past demand.
- price_only: price history alone.

There is no task training, residual model fitting, online weight update, numerical
price editing by the LLM, or new target construction. Profile choice changes actual
predictor inputs. The selected forecast is copied from the frozen predictor output.
For this small pilot the numerical candidates are eagerly cached; the tool trace
records which diagnostic the Agent actually requested. Thus it does not yet claim
adaptive GPU-compute savings.

FIXED sees the same available profiles and diagnostic and chooses lowest full-day
hindcast QS, changing from BASE only for a reduction greater than 2%. AGENT may
consider consistency across the two 12-hour halves. A change requires option
probability at least .55 and the same >2% numerical support; otherwise it uses BASE.
These constants are preregistered engineering guards, not calibrated probabilities
of improved forecasting accuracy. API errors fall back to FIXED and remain scored.

## Development experiment

The first pilot uses 24 uniformly spaced origins per region from the existing
2022 H1 `selection` cache (72 region-origins total). The 2023–2024 test split and
original main table are not used for development. At most two Kev calls are made
per origin. The three profiles share frozen weights and have no cross-learning.

All predictions and API decisions finish before any evaluation-label file is opened.
The scorer reuses `forecasting.aligned_scoring`. Artifacts separately identify the
common input and effective selected profile. The existing strict `assert_comparable`
function is unchanged. Scores are averaged with equal region weight. Source hashes,
model hashes, API identity, requests, responses, choices, fallback reasons, and timing
are saved. Sources must remain unchanged throughout a frozen run.

This pilot tests the integration and whether the controller shows useful development
gain. It does not establish a full-table rank or statistical superiority. If it fails
to beat BASE and FIXED, record that result and do not declare the forecasting objective
achieved. Historical evaluation still cannot rule out foundation-model pretraining
overlap, even with strict runtime data isolation.

Run on GPU 7 with the existing Chronos environment:

```bash
CUDA_VISIBLE_DEVICES=GPU-c931f386-8fcf-9f27-2526-47e12aac3145 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python \
  -m forecasting.api_agent.pilot --origins-per-region 24 --out <new-output-directory>
```

Predeployment verification: eight boundary/decision/fallback unit tests passed; a real
GPU 7 synthetic check produced all six profile/hindcast arrays of shape `[24,5]`
with finite values. Deployment and the R15 pilot subsequently completed; see
`logs/forecasting/kev_r15_20261102/service_smoke.json` and `pilot_report.json`.
The R16 follow-up records the paired workflow optimization and its limitations.

The adapter was downloaded from the publisher's ModelScope mirror at revision
`54ac5810dae3e2d8a7fac8c963a7cc9b475ccfac`; its SHA-256 matches the pinned HF
adapter (`90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2`).
