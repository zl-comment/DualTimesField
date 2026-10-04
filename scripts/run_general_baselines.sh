#!/usr/bin/env bash
# Runs DLinear, PatchTST, iTransformer, and Informer on raw and 650-capped
# prices, seeds 2026-2028, regions NSW1/QLD1/TAS1 (72 runs).
# Usage: scripts/run_general_baselines.sh "6 7" 3   (GPUs, parallel runs per GPU)
# KINDS overrides the models, e.g. KINDS="known_linear linear" for attribution runs.
set -u
GPUS=(${1:-6 7})
PER_GPU=${2:-3}
cd "$(dirname "$0")/.."
LOGS=logs/forecasting/baselines/logs/general
mkdir -p "$LOGS"

jobs_list=()
for protocol in raw capped650; do
  if [ "$protocol" = raw ]; then config=configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml
  else config=configs/aemo_forecast_capped650_trunk.yaml; fi
  for kind in ${KINDS:-informer patchtst itransformer dlinear}; do
    for seed in 2026 2027 2028; do
      for region in NSW1 QLD1 TAS1; do
        jobs_list+=("$protocol $config $kind $seed $region")
      done
    done
  done
done

slots=$(( ${#GPUS[@]} * PER_GPU ))
worker() {
  local slot=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}
  for (( i = slot; i < ${#jobs_list[@]}; i += slots )); do
    read -r protocol config kind seed region <<< "${jobs_list[$i]}"
    out=outputs/forecasting/baselines/$protocol/$kind/seed$seed
    if [ -f "$out/$region.json" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m forecasting.baselines \
      --baseline "$kind" --config "$config" --region "$region" --seed "$seed" \
      --output-dir "$out" --device cuda > "$LOGS/${protocol}_${kind}_seed${seed}_${region}.log" 2>&1
  done
}
for (( s = 0; s < slots; s++ )); do worker "$s" & done
wait
