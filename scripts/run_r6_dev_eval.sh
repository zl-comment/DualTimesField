#!/usr/bin/env bash
# r6 minimal A/B/C/D 2022 development evaluation driver.
# Usage: bash scripts/run_r6_dev_eval.sh smoke|full
set -euo pipefail

MODE="${1:-smoke}"
REPO_PY="${REPO_PY:-/root/workspace/Code/DualTimesField/.venv/bin/python}"
CHRONOS_PY="${CHRONOS_PY:-/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python}"
OUT="${OUT:-outputs/forecasting/agentic_m2m3_dev_c2_20261030_r6}"
export HF_HOME="${HF_HOME:-/root/workspace/Code/DualTimesField/weights/hf_home}"

LIMIT_FLAG=""
REGIONS=(NSW1 QLD1 TAS1)
SCORE_REGIONS="NSW1,QLD1,TAS1"
if [ "$MODE" = "smoke" ]; then
  LIMIT_FLAG="--limit 2"
  REGIONS=(NSW1)
  SCORE_REGIONS="NSW1"
  OUT="${OUT}/smoke"
fi

for region in "${REGIONS[@]}"; do
  echo "== arm A base: $region =="
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$CHRONOS_PY" -m forecasting.agentic_dev_eval.run_base \
    --region "$region" --output-dir "$OUT" $LIMIT_FLAG
done

for region in "${REGIONS[@]}"; do
  for arm in B C D; do
    echo "== arm $arm: $region =="
    "$REPO_PY" -m forecasting.agentic_dev_eval.run_decide \
      --region "$region" --arm "$arm" --output-dir "$OUT" $LIMIT_FLAG
  done
done

echo "== score =="
"$REPO_PY" -m forecasting.agentic_dev_eval.run_score --output-dir "$OUT" --regions "$SCORE_REGIONS"
echo "done: $OUT/results"
