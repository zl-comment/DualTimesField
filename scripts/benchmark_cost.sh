#!/usr/bin/env bash
# Wall-clock cost of one quarterly refit (Q3 2023 test quarter, NSW1, seed 2026)
# per model, run one at a time on one GPU, for the paper's cost table.
# Usage: scripts/benchmark_cost.sh <GPU>
set -u
GPU=${1:-7}
cd "$(dirname "$0")/.."
OUT=logs/forecasting/cost
mkdir -p "$OUT" outputs/forecasting/cost
: > "$OUT/seconds.tsv"
time_run() {  # name, command...
  local name=$1; shift
  local start=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 PYTHONPATH=. "$@" > "$OUT/$name.log" 2>&1
  local status=$?
  printf "%s\t%.1f\t%s\n" "$name" "$(echo "$(date +%s.%N) - $start" | bc)" "$status" >> "$OUT/seconds.tsv"
}
for name in pd_calibrator pd_linear_base_only; do
  time_run "$name" .venv/bin/python -m forecasting.train --config "configs/aemo_forecast_rolling_${name}_q3.yaml" --region NSW1
done
for kind in xgboost patchtst dlinear linear_mse; do
  time_run "$kind" .venv/bin/python -m forecasting.baselines --baseline "$kind" \
    --config configs/aemo_forecast_rolling_pd_baseline_raw_q3.yaml --region NSW1 --seed 2026 \
    --output-dir "outputs/forecasting/cost/$kind" --device cuda
done
