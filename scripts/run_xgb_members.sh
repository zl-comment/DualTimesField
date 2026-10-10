#!/usr/bin/env bash
# XGBoost member variants (forecasting.xgb_member) for every quarterly refit and region.
# Usage: scripts/run_xgb_members.sh "<GPU list, one entry per slot>" <variant>...
set -u
GPUS=($1); shift; VARIANTS=("$@")
cd "$(dirname "$0")/.."
LOG=logs/forecasting/members/logs; mkdir -p "$LOG"
jobs=()
for variant in "${VARIANTS[@]}"; do for q in 0 1 2 3 4 5 6 7; do for region in NSW1 QLD1 TAS1; do jobs+=("$variant $q $region"); done; done; done
slots=${#GPUS[@]}
worker() {
  local slot=$1 gpu=${GPUS[$1]}
  for (( i = slot; i < ${#jobs[@]}; i += slots )); do
    read -r variant q region <<< "${jobs[$i]}"
    out=outputs/forecasting/members/xgb_$variant/q$q
    [ -f "$out/$region.npz" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -u -m forecasting.xgb_member \
      --config configs/aemo_forecast_rolling_pd_baseline_raw_q$q.yaml --region "$region" --variant "$variant" \
      --output "$out" > "$LOG/xgb_${variant}_q${q}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
