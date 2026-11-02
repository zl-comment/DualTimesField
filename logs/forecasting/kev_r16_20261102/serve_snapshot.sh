#!/bin/bash
set -euo pipefail
cd /root/workspace/services/kev/runtime-src
export CUDA_VISIBLE_DEVICES=GPU-d38f792e-b077-9b2f-c8c9-e2c692cc6001
export HF_HOME=/root/workspace/services/kev/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KEV_DTYPE=bf16 KEV_BACKEND=torch KEV_FUSED=0 KEV_CUDA_GRAPHS=0
export KEV_PREFIX_CACHE=2 KEV_PREFIX_MAX_TOKENS=16384
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec /root/workspace/services/kev/venv/bin/python -u -m kev.serve \
  --run /root/workspace/services/kev/models/kev-4b \
  --fallback /root/workspace/services/kev/models/kev-4b \
  --host 127.0.0.1 --port 8009 \
  >> /root/workspace/services/kev/logs/server.log 2>&1
