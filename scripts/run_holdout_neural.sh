#!/usr/bin/env bash
# Neural members on the held-out period: 3 refits x 3 regions x 3 seeds per configuration, then the validation-chosen
# floor with validation forecasts (forecasting.price_floor, ROLLING_PERIOD=holdout) in outputs/forecasting/members_hold/<member>.
# Usage: scripts/run_holdout_neural.sh "<GPU list, one per slot>" <config name>:<member name> ...
set -u
GPUS=($1); shift; SPECS=("$@")
cd "$(dirname "$0")/.."
LOG=logs/forecasting/members_hold/logs; mkdir -p "$LOG"
jobs=()
for spec in "${SPECS[@]}"; do name=${spec%%:*}; for seed in 2026 2027 2028; do for q in 0 1 2; do for region in NSW1 QLD1 TAS1; do
  jobs+=("$name $q $seed $region"); done; done; done; done
slots=${#GPUS[@]}
worker() {
  local slot=$1 gpu=${GPUS[$1]}
  for (( i = slot; i < ${#jobs[@]}; i += slots )); do
    read -r name q seed region <<< "${jobs[$i]}"
    seed_arg=(); [ "$seed" != 2026 ] && seed_arg=(--seed "$seed")
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. .venv/bin/python -u -m forecasting.train \
      --config "configs/aemo_forecast_rolling_hold_${name}_q$q.yaml" --region "$region" "${seed_arg[@]}" \
      > "$LOG/train_${name}_q${q}_seed${seed}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
for spec in "${SPECS[@]}"; do
  name=${spec%%:*}; member=${spec##*:}
  for region in NSW1 QLD1 TAS1; do
    CUDA_VISIBLE_DEVICES=${GPUS[0]} ROLLING_PERIOD=holdout PYTHONPATH=. .venv/bin/python -W ignore -m forecasting.price_floor \
      --name "hold_$name" --region "$region" --output-dir "outputs/forecasting/members_hold/$member" \
      --metrics-dir "logs/forecasting/members_hold/metrics_$member" --checkpoint best_mae_model.pt --device cuda:0 \
      > "$LOG/floor_${member}_$region.log" 2>&1 &
  done; wait
done
