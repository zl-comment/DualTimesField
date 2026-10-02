#!/usr/bin/env bash
# Recalibrates the baselines of forecasting.baselines every test quarter, on raw
# and 650-capped prices, seeds 2026-2028, NSW1/QLD1/TAS1, then stitches the
# quarters (configs: forecasting.rolling make-configs --name baseline_raw / baseline_capped650).
# Usage: scripts/run_rolling_baselines.sh "<GPUs>" <runs per GPU> <kind>...
# CONFIG_NAME (default baseline) selects configs aemo_forecast_rolling_<CONFIG_NAME>_<protocol>_q<k>.yaml;
# OUT and LOG_ROOT override the output and log roots.
set -u
GPUS=($1); PER_GPU=$2; shift 2; KINDS=("$@")
cd "$(dirname "$0")/.."
CONFIG_NAME=${CONFIG_NAME:-baseline}
OUT=${OUT:-outputs/forecasting/baselines_rolling}
LOG_ROOT=${LOG_ROOT:-logs/forecasting/baselines_rolling}
LOGS=$LOG_ROOT/logs
mkdir -p "$LOGS"
jobs_list=()
for seed in 2026 2027 2028; do for q in 0 1 2 3 4 5 6 7; do for protocol in raw capped650; do
  for kind in "${KINDS[@]}"; do for region in NSW1 QLD1 TAS1; do
    jobs_list+=("$protocol $kind $seed $q $region"); done; done; done; done; done
slots=$(( ${#GPUS[@]} * PER_GPU ))
worker() {
  local slot=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}
  for (( i = slot; i < ${#jobs_list[@]}; i += slots )); do
    read -r protocol kind seed q region <<< "${jobs_list[$i]}"
    out=$OUT/$protocol/$kind/seed$seed/q$q
    if [ -f "$out/$region.json" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m forecasting.baselines \
      --baseline "$kind" --config "configs/aemo_forecast_rolling_${CONFIG_NAME}_${protocol}_q$q.yaml" \
      --region "$region" --seed "$seed" --output-dir "$out" --device cuda \
      > "$LOGS/${protocol}_${kind}_seed${seed}_q${q}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
for protocol in raw capped650; do
  static=outputs/forecasting/significance_inputs/linear_base_only
  [ "$protocol" = capped650 ] && static=outputs/forecasting/significance_inputs/capped650_linear_base_only
  for kind in "${KINDS[@]}"; do
    PYTHONPATH=. .venv/bin/python -m forecasting.rolling stitch-baselines --root "$OUT/$protocol/$kind" \
      --static-npz-dir "$static" --metrics-dir "$LOG_ROOT/$protocol/$kind"
  done
done
