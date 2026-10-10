#!/usr/bin/env bash
# Driver-space dual field: quarterly refits of five variants, the validation-chosen price floor for two
# checkpoint rules, and the Diebold-Mariano tests. See logs/forecasting/PLAN_DRIVER_SPACE.md.
# Usage: scripts/run_driver_space.sh "<GPU list, one entry per slot>" 1
set -u
GPUS="$1"; PER_GPU=$2
cd "$(dirname "$0")/.."
LOG_ROOT=logs/forecasting/driver_space
VARIANTS=(ds_events ds_raw ps_drivers ds_nostate base_drivers)
INPUTS=outputs/forecasting/significance_inputs
mkdir -p "$LOG_ROOT/logs"
for name in "${VARIANTS[@]}"; do
  [ -f "configs/aemo_forecast_rolling_${name}_q7.yaml" ] || PYTHONPATH=. .venv/bin/python -m forecasting.rolling \
    make-configs --base-config "configs/aemo_forecast_${name}.yaml" --name "$name" > /dev/null
done
STATIC_REF=rolling_pd_calibrator scripts/run_rolling.sh "$LOG_ROOT" "$GPUS" "$PER_GPU" "${VARIANTS[@]}"

GPU0=$(echo "$GPUS" | awk '{print $1}')
for name in "${VARIANTS[@]}"; do for ckpt in best_model best_mae_model; do for region in NSW1 QLD1 TAS1; do
  CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. .venv/bin/python -W ignore -m forecasting.price_floor --name "$name" \
    --region "$region" --static-npz-dir "$INPUTS/rolling_pd_calibrator" --output-dir "outputs/forecasting/driver_space/${ckpt}_${name}" \
    --metrics-dir "$LOG_ROOT/${ckpt}_${name}" --checkpoint "$ckpt.pt" --device cuda:0 \
    > "$LOG_ROOT/logs/floor_${ckpt}_${name}_${region}.log" 2>&1 &
done; wait; done; done

O=outputs/forecasting/driver_space
mkdir -p logs/forecasting/driver_space/significance
for ckpt in best_mae_model best_model; do
  sources=()
  for name in "${VARIANTS[@]}"; do sources+=(--source "$name=$O/${ckpt}_${name}/{region}.npz"); done
  sources+=(--source "C_mae=outputs/forecasting/mae_alignment/nofields_floor_best_mae_model/{region}.npz")
  sources+=(--source "xgboost=outputs/forecasting/baselines_rolling_pd/raw/xgboost/seed{seed}/{region}.npz")
  for loss in mae crps crps_negative q05_negative crps_spike q95_spike; do
    PYTHONPATH=. .venv/bin/python -m forecasting.significance --reference ds_events "${sources[@]}" --loss "$loss" \
      --output "$LOG_ROOT/significance/${ckpt}_ds_events_${loss}.json" > /dev/null 2>&1 || echo "FAIL $ckpt $loss"
  done
done
