#!/bin/bash
# Start an installed, pinned Kev service; never stop another GPU process.
set -euo pipefail
service_root=/root/workspace/services/kev
session=kev-4b
if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session $session already exists. Inspect it with: tmux attach -t $session"
    exit 0
fi
test -f "$service_root/models.ready"
test -x "$service_root/venv/bin/python"
test "$(git -C "$service_root/runtime-src" rev-parse HEAD)" = 6b719c3c3f367295f6ef336f4f751cf5ff970abc
mkdir -p "$service_root/logs"
cat > "$service_root/serve.sh" <<'SH'
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
SH
chmod +x "$service_root/serve.sh"
tmux new-session -d -s "$session" "/bin/bash -c '$service_root/serve.sh; exec /bin/bash'"
tmux rename-window -t "$session:0" server
tmux new-window -d -t "$session" -n logs "tail -n 50 -F $service_root/logs/server.log"
tmux select-window -t "$session:logs"
echo "Started tmux session $session; readiness requires GET /v1/models and a decision smoke test."
