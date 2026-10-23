#!/usr/bin/env bash
# Unified entry for the 2026-10-23 foundation-baseline batch.
# Every subcommand maps 1:1 onto the CLIs committed in this repository; run
# from the worktree root: /root/workspace/Code/DualTimesField-foundation-baselines
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PY="${REPO_PY:-/root/workspace/Code/DualTimesField/.venv/bin/python}"
PARENT="/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/forecasting/foundation_baselines_20261023_r1}"
export HF_HOME="${HF_HOME:-$ROOT/weights/hf_home}"

case "${1:-help}" in
  prepare)   # freeze sources/config and create the 36-job matrix
    shift; cd "$ROOT" && exec "$REPO_PY" -m forecasting.foundation_experiment prepare \
      --config configs/experiments/aemo_forecast_foundation_20261023.yaml \
      --run-root "$RUN_ROOT" --extend-from "$PARENT" "$@" ;;
  run)       # controller loop; resumable, one worker per idle GPU
    shift; cd "$ROOT" && exec "$REPO_PY" -m forecasting.foundation_experiment run \
      --run-root "$RUN_ROOT" "$@" ;;
  worker)    # single cell inside the model's own interpreter (controller does this)
    shift; cd "$RUN_ROOT/source" && exec "$REPO_PY" -m forecasting.foundation_experiment worker \
      --run-root "$RUN_ROOT" "$@" ;;
  smoke)     # real-weight dev-data smoke for one model (its own venv)
    model="${2:?model id required}"; shift 2
    venv="$ROOT/.venvs/$(case $model in
      tabpfn_v3) echo tabpfn_v3;; chronos2_zeroshot) echo chronos2;;
      moirai2_zeroshot) echo moirai2;; timesfm25_xreg) echo timesfm25;; esac)"
    cd "$ROOT" && exec "$venv/bin/python" -m forecasting.foundation_smoke \
      --model "$model" --parent "$PARENT" "$@" ;;
  verify)    # stage-2 isolation/mapping/batch verification (its own venv)
    model="${2:?model id required}"; shift 2
    venv="$ROOT/.venvs/$(case $model in
      tabpfn_v3) echo tabpfn_v3;; chronos2_zeroshot) echo chronos2;;
      moirai2_zeroshot) echo moirai2;; timesfm25_xreg) echo timesfm25;; esac)"
    cd "$ROOT" && exec "$venv/bin/python" -m forecasting.foundation_verify \
      --model "$model" "$@" ;;
  audit)     # independent recomputation for one model (its own venv)
    model="${2:?model id required}"; shift 2
    venv="$ROOT/.venvs/$(case $model in
      tabpfn_v3) echo tabpfn_v3;; chronos2_zeroshot) echo chronos2;;
      moirai2_zeroshot) echo moirai2;; timesfm25_xreg) echo timesfm25;; esac)"
    cd "$RUN_ROOT/source" && exec "$venv/bin/python" -m forecasting.foundation_audit \
      --model "$model" --run-root "$RUN_ROOT" "$@" ;;
  audit-collect)  # merge per-model audits + cross-model alignment
    shift; cd "$RUN_ROOT/source" && exec "$REPO_PY" -m forecasting.foundation_audit \
      --collect --run-root "$RUN_ROOT" "$@" ;;
  report)    # candidate comparison vs the frozen main table (read-only)
    shift; out="${1:-$ROOT/logs/forecasting/foundation_baselines_20261023/candidate}"
    cd "$RUN_ROOT/source" && exec "$REPO_PY" -m forecasting.foundation_candidate \
      --run-root "$RUN_ROOT" --output "$out" "$@" ;;
  tests)     # framework tests incl. first-batch regression
    shift; cd "$ROOT" && exec "$REPO_PY" -m pytest -q \
      tests/test_foundation_registry.py tests/test_foundation_worker.py \
      tests/test_expansion_model_layout.py tests/test_aligned_inputs.py \
      tests/test_aligned_scoring.py tests/test_aligned_experiment.py "$@" ;;
  *) cat <<'EOF'
Usage: scripts/foundation_baselines.sh {prepare|run|worker|smoke|verify|audit|audit-collect|report|tests} [args]
  prepare        freeze sources/config; create the 36-cell matrix from the parent run
  run            start the resumable controller (GPU 6/7, one worker per idle card)
  worker         run one cell (used by the controller; needs --job-id/--device)
  smoke MODEL    real-weight dev-data smoke for one model in its own venv
  verify MODEL   stage-2 isolation/mapping/batch-companion verification
  audit MODEL    independent recomputation (re-score, chunk reassembly, replay sample)
  audit-collect  merge audits and cross-check alignment
  report         build the candidate comparison (never writes the main table)
  tests          framework + regression tests
EOF
  ;;
esac
