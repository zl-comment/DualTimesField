"""One real development-origin integration check; never opens evaluation labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .pilot import ROOT, CHECKPOINT, CHECKPOINT_SHA, load_snapshots, sha_file
from .transport import KevClient
from .workflow import diagnostics, digest, forecast_profiles
from .workflow_r17 import decide, RELEASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-c931f386-8fcf-9f27-2526-47e12aac3145':
        raise ValueError('Use the reserved predictor GPU 7 UUID')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for filename, expected in CHECKPOINT_SHA.items():
        if sha_file(CHECKPOINT / filename) != expected:
            raise ValueError('Frozen predictor checksum mismatch')
    client = KevClient()
    identity = client.identity()
    paths = sorted((ROOT / 'forecasting/api_agent').glob('*.py')) + sorted(
        (ROOT / 'forecasting/api_agent/skills').glob('*.txt'))
    paths.append(ROOT / 'forecasting/chronos2_zeroshot/model.py')
    hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in paths}
    _, snapshots, _ = load_snapshots('NSW1', 1)
    snapshot = snapshots[0]
    manifest = {
        'release': RELEASE, 'scope': 'one development-origin integration check; no accuracy evaluation',
        'base_commit': 'caddb08ef9f749c6a2e029d83538d8b1a3015506',
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': hashes, 'source_fingerprint': digest(hashes),
        'snapshot_id': snapshot.identity(), 'kev_identity': identity,
        'predictor_weights': CHECKPOINT_SHA, 'training_calls': 0, 'evaluation_labels_opened': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    pipeline = Chronos2Pipeline.from_pretrained(str(CHECKPOINT), device_map='cuda', torch_dtype=torch.float32)
    with torch.inference_mode():
        forecasts, hindcasts = forecast_profiles(snapshot, pipeline)
    selected, trace = decide(snapshot, diagnostics(snapshot, hindcasts), client, forecasts=forecasts)
    np.savez_compressed(out / 'prediction.npz', quantiles=forecasts[selected],
                        target_time_unix=snapshot.target_time_unix)
    (out / 'trace.json').write_text(json.dumps(trace, indent=2, allow_nan=False) + '\n')
    for path, expected in hashes.items():
        if sha_file(ROOT / path) != expected:
            raise RuntimeError('Source changed during smoke check')
    report = {'status': 'PASS' if trace['fallback'] is None else 'FALLBACK',
              'selected_profile': selected, 'calls': len(trace['calls']),
              'tools': [t['tool'] for t in trace['tool_calls']],
              'stop_reason': trace['stop_reason'], 'prediction_shape': list(forecasts[selected].shape),
              'accuracy_evaluated': False, 'source_fingerprint': manifest['source_fingerprint']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if trace['fallback'] is not None:
        raise RuntimeError('Integration check used fallback; inspect trace')


if __name__ == '__main__':
    main()
