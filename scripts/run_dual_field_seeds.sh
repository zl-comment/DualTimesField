#!/usr/bin/env bash
# Trains configs/aemo_forecast_<name>.yaml for seeds 2026-2028 and NSW1/QLD1/TAS1,
# then writes RE-Price style test metrics and the seed-stacked test predictions.
# Usage: scripts/run_dual_field_seeds.sh <log_root> "<GPUs>" <runs per GPU> <name>...
set -u
ROOT=$1; GPUS=($2); PER_GPU=$3; shift 3; NAMES=("$@")
cd "$(dirname "$0")/.."
mkdir -p "$ROOT/logs"
jobs_list=()
for name in "${NAMES[@]}"; do for seed in 2026 2027 2028; do for region in NSW1 QLD1 TAS1; do
  jobs_list+=("$name $seed $region"); done; done; done
slots=$(( ${#GPUS[@]} * PER_GPU ))
worker() {
  local slot=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}
  for (( i = slot; i < ${#jobs_list[@]}; i += slots )); do
    read -r name seed region <<< "${jobs_list[$i]}"
    seed_arg=(); [ "$seed" != 2026 ] && seed_arg=(--seed "$seed")
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. .venv/bin/python -u -m forecasting.train \
      --config "configs/aemo_forecast_$name.yaml" --region "$region" "${seed_arg[@]}" \
      > "$ROOT/logs/${name}_seed${seed}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
for name in "${NAMES[@]}"; do
  for seed in 2026 2027 2028; do
    suffix=""; [ "$seed" != 2026 ] && suffix="_seed$seed"
    PYTHONPATH=. .venv/bin/python -m forecasting.evaluate_paper_metrics evaluate \
      --config "configs/aemo_forecast_$name.yaml" --checkpoint-dir "outputs/forecasting/$name$suffix" \
      --output-dir "$ROOT/paper_metrics/$name/seed$seed" --device cuda:0 > /dev/null 2>&1
  done
  PYTHONPATH=. .venv/bin/python -m forecasting.protocol_alignment collect \
    --config "configs/aemo_forecast_$name.yaml" --checkpoint-root "outputs/forecasting/$name" \
    --output-dir "outputs/forecasting/significance_inputs/$name" --device cuda:0
done
