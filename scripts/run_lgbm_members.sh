#!/usr/bin/env bash
# LightGBM member variants (forecasting.lgbm_member) for every quarterly refit and region.
# Usage: scripts/run_lgbm_members.sh <number of slots> <variant>...
set -u
SLOTS=$1; shift; VARIANTS=("$@")
cd "$(dirname "$0")/.."
LOG=logs/forecasting/members/logs; mkdir -p "$LOG"
jobs=()
for variant in "${VARIANTS[@]}"; do for q in 0 1 2 3 4 5 6 7; do for region in NSW1 QLD1 TAS1; do jobs+=("$variant $q $region"); done; done; done
slots=$SLOTS
worker() {
  local slot=$1
  for (( i = slot; i < ${#jobs[@]}; i += slots )); do
    read -r variant q region <<< "${jobs[$i]}"
    out=outputs/forecasting/members/lgbm_$variant/q$q
    [ -f "$out/$region.npz" ] && continue
    OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -u -m forecasting.lgbm_member \
      --config configs/aemo_forecast_rolling_pd_baseline_raw_q$q.yaml --region "$region" --variant "$variant" --threads 4 \
      --output "$out" > "$LOG/lgbm_${variant}_q${q}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
