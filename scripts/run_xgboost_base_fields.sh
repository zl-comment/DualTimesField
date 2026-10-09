#!/usr/bin/env bash
# XGBoost base + dual-field residual, refit every test quarter with predispatch.
#  1. Builds the cross-fitted XGBoost base of every refit (forecasting.xgboost_base;
#     8 quarters, NSW1/QLD1/TAS1, raw and 650-capped prices, seed 2026). The
#     variants of one price treatment share it.
#  2. Trains the variants for seeds 2026-2028 and stitches their test forecasts
#     (scripts/run_rolling.sh).
#  3. Diebold-Mariano tests of the fields against the XGBoost base and the
#     raw-history controls: MAE, CRPS~, and the negative- and spike-hour tails.
# Usage: scripts/run_xgboost_base_fields.sh "<GPUs>" <runs per GPU>
# FOLDS (default 5) sets the cross-fitting blocks. XGB_ROOT holds the stitched
# quarterly XGBoost-with-predispatch forecasts (scripts/run_rolling_baselines.sh
# with CONFIG_NAME=pd_baseline OUT=outputs/forecasting/baselines_rolling_pd).
# STATIC_REF_RAW / STATIC_REF_CAPPED name static prediction sets used only to
# check the stitched origins.
set -u
GPUS=($1); PER_GPU=$2
cd "$(dirname "$0")/.."
FOLDS=${FOLDS:-5}
XGB_ROOT=${XGB_ROOT:-outputs/forecasting/baselines_rolling_pd}
LOG_ROOT=logs/forecasting/xgb_base
INPUTS=outputs/forecasting/significance_inputs
VARIANTS=(xgb_base_fields xgb_base_tails xgb_base_raw xgb_base_raw_tails)
mkdir -p "$LOG_ROOT/logs"

jobs_list=()
for q in 0 1 2 3 4 5 6 7; do for protocol in raw capped650; do for region in NSW1 QLD1 TAS1; do
  jobs_list+=("$protocol $q $region"); done; done; done
slots=$(( ${#GPUS[@]} * PER_GPU ))
worker() {
  local slot=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}
  for (( i = slot; i < ${#jobs_list[@]}; i += slots )); do
    read -r protocol q region <<< "${jobs_list[$i]}"
    prefix=""; [ "$protocol" = capped650 ] && prefix=capped650_
    config=configs/aemo_forecast_rolling_${prefix}xgb_base_tails_q$q.yaml
    base_dir=$(PYTHONPATH=. .venv/bin/python -c \
      "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['external_base']['directory'])" "$config")
    if [ -f "$base_dir/$region.npz" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -u -m forecasting.xgboost_base \
      --config "$config" --region "$region" --folds "$FOLDS" --device cuda \
      > "$LOG_ROOT/logs/base_${protocol}_q${q}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait

STATIC_REF=${STATIC_REF_RAW:-pd_linear_base} scripts/run_rolling.sh "$LOG_ROOT" "$1" "$PER_GPU" "${VARIANTS[@]}"
STATIC_REF=${STATIC_REF_CAPPED:-capped650_pd_linear_base} scripts/run_rolling.sh "$LOG_ROOT" "$1" "$PER_GPU" \
  "${VARIANTS[@]/#/capped650_}"

for protocol in raw capped650; do
  prefix=""; [ "$protocol" = capped650 ] && prefix=capped650_
  sources=()
  for name in "${VARIANTS[@]}"; do sources+=(--source "$name=$INPUTS/rolling_${prefix}$name/{region}.npz"); done
  sources+=(--source "xgboost=$XGB_ROOT/$protocol/xgboost/seed{seed}/{region}.npz")
  for loss in mae crps crps_negative q05_negative crps_spike q95_spike; do
    for reference in xgb_base_tails xgb_base_fields; do
      PYTHONPATH=. .venv/bin/python -m forecasting.significance --reference "$reference" "${sources[@]}" \
        --loss "$loss" --output "logs/forecasting/significance/${protocol}_${reference}_${loss}.json"
    done
  done
  PYTHONPATH=. .venv/bin/python -m forecasting.tail_metrics "${sources[@]}" \
    --output "$LOG_ROOT/${protocol}_tail_metrics.json"
done
