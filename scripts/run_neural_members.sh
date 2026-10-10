#!/usr/bin/env bash
# Trains the tuning variants of the neural members (72 refits each) and writes their validation-chosen floors with
# validation forecasts (forecasting.price_floor) to outputs/forecasting/members/<member>_<variant>.
# Usage: scripts/run_neural_members.sh "<GPU list, one per slot>" "<names...>"   e.g. pd_calibrator_nofields_v1 ...
set -u
GPUS="$1"; shift; NAMES=("$@")
cd "$(dirname "$0")/.."
STATIC_REF=rolling_pd_calibrator scripts/run_rolling.sh logs/forecasting/members_neural "$GPUS" 1 "${NAMES[@]}"
GPU0=$(echo "$GPUS" | awk '{print $1}')
for name in "${NAMES[@]}"; do
  case "$name" in pd_calibrator_nofields*) member=nn_c_mae${name#pd_calibrator_nofields};; *) member=nn_dual_b${name#pd_calibrator};; esac
  for region in NSW1 QLD1 TAS1; do
    CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. .venv/bin/python -W ignore -m forecasting.price_floor --name "$name" --region "$region" \
      --static-npz-dir outputs/forecasting/significance_inputs/rolling_pd_calibrator --output-dir "outputs/forecasting/members/$member" \
      --metrics-dir "logs/forecasting/members/metrics_$member" --checkpoint best_mae_model.pt --device cuda:0 \
      > "logs/forecasting/members/logs/floor_${member}_$region.log" 2>&1 &
  done; wait
done
