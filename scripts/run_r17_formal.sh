#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/Code/DualTimesField-worktrees/frozen-api-agent-alignment-20261102-r14
export CUDA_VISIBLE_DEVICES=GPU-c931f386-8fcf-9f27-2526-47e12aac3145
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p logs/forecasting/kev_r17_formal_20261103
/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python \
  -m forecasting.api_agent.formal_r17 \
  --out outputs/forecasting/kev_r17_formal_20261103 \
  >> logs/forecasting/kev_r17_formal_20261103/run.log 2>&1
for name in manifest.json report.json metrics.csv progress.json inference_complete.json; do
  cp "outputs/forecasting/kev_r17_formal_20261103/$name" "logs/forecasting/kev_r17_formal_20261103/$name"
done
