#!/usr/bin/env bash
# Holdout (2025-01 to 2025-07) configurations of the members: the paths of a main-period configuration point to the
# extended series, and forecasting.rolling writes the quarterly refits with ROLLING_PERIOD=holdout.
set -eu
cd "$(dirname "$0")/.."
for base in pd_baseline_raw pd_calibrator pd_calibrator_nofields pd_calibrator_nofields_v1; do
  sed -e 's#\.\./electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem/#data/holdout/#' \
      -e 's#data/aemo_exogenous/\([a-z0-9]*\)_pdpasa\.npz#data/aemo_exogenous_holdout/merged/\1_pdpasa.npz#' \
      -e 's#data/aemo_exogenous/\([a-z0-9]*\)_predispatch\.npz#data/aemo_exogenous_holdout/merged/\1_predispatch.npz#' \
      -e "s#outputs/forecasting/${base}\$#outputs/forecasting/hold_${base}#" \
      "configs/aemo_forecast_${base}.yaml" > "configs/aemo_forecast_hold_${base}.yaml"
  ROLLING_PERIOD=holdout PYTHONPATH=. .venv/bin/python -m forecasting.rolling make-configs \
    --base-config "configs/aemo_forecast_hold_${base}.yaml" --name "hold_${base}" > /dev/null
done
ls configs | grep -c hold_
