#!/usr/bin/env bash
# Trains the quarterly refits of forecasting.rolling for seeds 2026-2028 and
# NSW1/QLD1/TAS1, then stitches their test forecasts.
# Usage: scripts/run_rolling.sh <log_root> "<GPUs>" <runs per GPU> <name>...
# (configs from: python -m forecasting.rolling make-configs --base-config ... --name <name>)
set -u
ROOT=$1; GPUS=($2); PER_GPU=$3; shift 3; NAMES=("$@")
cd "$(dirname "$0")/.."
mkdir -p "$ROOT/logs"
jobs_list=()
for seed in 2026 2027 2028; do for q in 0 1 2 3 4 5 6 7; do for name in "${NAMES[@]}"; do for region in NSW1 QLD1 TAS1; do
  jobs_list+=("$name $q $seed $region"); done; done; done; done
slots=$(( ${#GPUS[@]} * PER_GPU ))
worker() {
  local slot=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}
  for (( i = slot; i < ${#jobs_list[@]}; i += slots )); do
    read -r name q seed region <<< "${jobs_list[$i]}"
    seed_arg=(); [ "$seed" != 2026 ] && seed_arg=(--seed "$seed")
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. .venv/bin/python -u -m forecasting.train \
      --config "configs/aemo_forecast_rolling_${name}_q$q.yaml" --region "$region" "${seed_arg[@]}" \
      > "$ROOT/logs/${name}_q${q}_seed${seed}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
for name in "${NAMES[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[0]} PYTHONPATH=. .venv/bin/python -m forecasting.rolling collect --name "$name" \
    --static-npz-dir "outputs/forecasting/significance_inputs/$name" \
    --output-dir "outputs/forecasting/significance_inputs/rolling_$name" \
    --metrics-dir "$ROOT/paper_metrics/$name" --device cuda:0
done
