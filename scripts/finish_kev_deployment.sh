#!/bin/bash
# Autonomous continuation after downloads: verify, serve, smoke, then development.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
service_root=/root/workspace/services/kev
report_root="$repo_root/logs/forecasting/kev_r15_20261102"
mkdir -p "$report_root"
cd "$repo_root"
status_file="$service_root/deployment.status"
trap 'result=$?; if (( result != 0 )); then echo "FAILED: exit $result; see $service_root/logs/continuation.log" > "$status_file"; fi' EXIT
echo WAITING_FOR_VERIFIED_DOWNLOADS_AND_RUNTIME > "$status_file"
deadline=$((SECONDS + 21600))
while [[ ! -f "$service_root/models.ready" || ! -f "$service_root/runtime.ready" ]]; do
    if (( SECONDS >= deadline )); then
        echo 'Timed out waiting for verified model files or runtime' >&2
        exit 1
    fi
    sleep 10
done
echo VERIFYING_ALL_MODEL_FILES > "$status_file"
/usr/bin/python3.11 scripts/verify_kev_files.py
echo STARTING_KEV > "$status_file"
bash scripts/deploy_kev_tmux.sh
deadline=$((SECONDS + 900))
while ! /usr/bin/curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8009/v1/models > "$report_root/service_identity.json"; do
    if (( SECONDS >= deadline )); then
        echo 'Kev did not become ready within 15 minutes' >&2
        exit 1
    fi
    sleep 5
done
echo VERIFYING_KEV > "$status_file"
/usr/bin/python3.11 scripts/check_kev_service.py --out "$report_root/service_smoke.json"
echo RUNNING_DEVELOPMENT_PILOT > "$status_file"
export CUDA_VISIBLE_DEVICES=GPU-c931f386-8fcf-9f27-2526-47e12aac3145
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python \
    -u -m forecasting.api_agent.pilot --origins-per-region 24 \
    --out "$repo_root/outputs/forecasting/kev_r15_pilot_20261102" \
    > "$report_root/pilot.log" 2>&1
cp "$repo_root/outputs/forecasting/kev_r15_pilot_20261102/report.json" "$report_root/pilot_report.json"
cp "$repo_root/outputs/forecasting/kev_r15_pilot_20261102/manifest.json" "$report_root/pilot_manifest.json"
echo DEPLOYED_AND_PILOT_FINISHED_REVIEW_REPORT > "$status_file"
