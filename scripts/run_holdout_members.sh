#!/usr/bin/env bash
# Tree members on the held-out period (3 refits x 3 regions per variant), configurations frozen on 2023-2024.
# Usage: scripts/run_holdout_members.sh <xgb|lgbm> "<GPU list or slot count>" <variant>...
set -u
KIND=$1; SLOTS=$2; shift 2; VARIANTS=("$@")
cd "$(dirname "$0")/.."
LOG=logs/forecasting/members_hold/logs; mkdir -p "$LOG"
jobs=()
for variant in "${VARIANTS[@]}"; do for q in 0 1 2; do for region in NSW1 QLD1 TAS1; do jobs+=("$variant $q $region"); done; done; done
slots=$(echo $SLOTS | wc -w); [ "$slots" = 1 ] && slots=$SLOTS
GPUS=($SLOTS)
worker() {
  local slot=$1
  for (( i = slot; i < ${#jobs[@]}; i += slots )); do
    read -r variant q region <<< "${jobs[$i]}"
    out=outputs/forecasting/members_hold/${KIND}_$variant/q$q
    [ -f "$out/$region.npz" ] && continue
    if [ "$KIND" = xgb ]; then
      CUDA_VISIBLE_DEVICES=${GPUS[$slot]} OMP_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -u -m forecasting.xgb_member \
        --config configs/aemo_forecast_rolling_hold_pd_baseline_raw_q$q.yaml --region "$region" --variant "$variant" \
        --output "$out" > "$LOG/${KIND}_${variant}_q${q}_${region}.log" 2>&1
    else
      OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -u -m forecasting.lgbm_member \
        --config configs/aemo_forecast_rolling_hold_pd_baseline_raw_q$q.yaml --region "$region" --variant "$variant" --threads 4 \
        --output "$out" > "$LOG/${KIND}_${variant}_q${q}_${region}.log" 2>&1
    fi
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
