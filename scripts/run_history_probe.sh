#!/usr/bin/env bash
# XGBoost with predispatch-error history features, all quarterly refits (seed 2026), for NSW1/QLD1/TAS1.
# Usage: scripts/run_history_probe.sh "<GPU list>"
set -u
GPUS=($1); cd "$(dirname "$0")/.."
LOG=logs/forecasting/history_probe/logs; mkdir -p "$LOG"
i=0
for q in 0 1 2 3 4 5 6 7; do for region in NSW1 QLD1 TAS1; do
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -u -m forecasting.history_probe \
    --config configs/aemo_forecast_rolling_pd_baseline_raw_q$q.yaml --region $region \
    --output outputs/forecasting/history_probe/q$q > "$LOG/q${q}_${region}.log" 2>&1 &
done; done
wait
