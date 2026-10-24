#!/usr/bin/env bash
# Finisher: wait for the foundation controller to complete, then run the
# per-model independent recomputation audits, the audit collection and the
# candidate report. Idempotent; safe to relaunch. Logs to finisher.log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PY=/root/workspace/Code/DualTimesField/.venv/bin/python
RUN=$ROOT/outputs/forecasting/foundation_baselines_20261023_r1
cd "$ROOT" || exit 1
export HF_HOME=$ROOT/weights/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$RUN/finisher.log"; }

while true; do
  if [ -f "$RUN/report.json" ]; then log "controller wrote report.json"; break; fi
  if ! pgrep -f "foundation_experiment run --run-root $RUN" > /dev/null 2>&1; then
    # controller gone without report -> crash; stop and let the operator retry
    log "controller not running and no report.json — stopping for operator"
    exit 2
  fi
  sleep 120
done

venv_for() { case $1 in
  tabpfn_v3) echo "$ROOT/.venvs/tabpfn_v3/bin/python";;
  chronos2_zeroshot) echo "$ROOT/.venvs/chronos2/bin/python";;
  moirai2_zeroshot) echo "$ROOT/.venvs/moirai2/bin/python";;
  timesfm25_xreg) echo "$ROOT/.venvs/timesfm25/bin/python";; esac; }

for model in chronos2_zeroshot moirai2_zeroshot timesfm25_xreg tabpfn_v3; do
  log "audit $model"
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1) \
    "$(venv_for "$model")" -m forecasting.foundation_audit --model "$model" --run-root "$RUN" \
    >> "$RUN/finisher.log" 2>&1 || { log "audit $model FAILED"; exit 1; }
done
log "audit-collect"
"$REPO_PY" -m forecasting.foundation_audit --collect --run-root "$RUN" >> "$RUN/finisher.log" 2>&1 || { log "collect FAILED"; exit 1; }
log "candidate report"
"$REPO_PY" -m forecasting.foundation_candidate --run-root "$RUN" \
  --output "$ROOT/logs/forecasting/foundation_baselines_20261023/candidate" >> "$RUN/finisher.log" 2>&1 || { log "report FAILED"; exit 1; }
log "finisher complete"
