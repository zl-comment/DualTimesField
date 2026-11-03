"""Explicit live integration check; no fitting and no future labels read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-c931f386-8fcf-9f27-2526-47e12aac3145'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    from chronos import Chronos2Pipeline
    from forecasting.adaptive_agent.context import ContextRepository
    from forecasting.adaptive_agent.events import EventResolver
    from forecasting.adaptive_agent.predictors.chronos import FrozenChronosAdapter
    from forecasting.adaptive_agent.predictors.lightgbm import FrozenLightGBMAdapter
    from forecasting.api_agent.transport import KevClient
    base = Path('/root/workspace/Code/DualTimesField')
    checkpoint = base / 'weights/hf_home/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c'
    expected = {'model.safetensors': 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42',
                'config.json': 'ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031'}
    for file, value in expected.items():
        assert sha(checkpoint / file) == value, file
    pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map='cuda:0')
    chronos = FrozenChronosAdapter(pipeline, {'sha256': expected['model.safetensors']})
    client = KevClient(timeout=30)
    result = {'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
              'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_files': {str(f.relative_to(ROOT)): sha(f) for f in sorted((ROOT / 'forecasting/adaptive_agent').rglob('*'))
                               if f.is_file() and '__pycache__' not in f.parts},
              'chronos_files': expected, 'kev_identity': client.identity(), 'regions': {},
              'training_calls': 0, 'actual_labels_read': 0}
    start = time.monotonic()
    for region in ('NSW1', 'QLD1', 'TAS1'):
        data = base / 'outputs/forecasting/aligned_baselines_20261012'
        repository = ContextRepository(data / 'data_cache' / region / 'test')
        ctx = repository.context(0)
        modeldir = data / 'runs/lightgbm/seed2026' / region
        lightgbm = FrozenLightGBMAdapter(modeldir / 'models', seed=2026, region=region)
        first = lightgbm.predict(ctx)
        second = lightgbm.predict(ctx)
        with np.load(modeldir / 'test.npz', allow_pickle=False) as saved:
            # Access only the saved predictions, never saved actual labels.
            key = 'quantiles' if 'quantiles' in saved.files else 'quantile_values'
            reference = saved[key][0]
        native_difference = float(np.max(np.abs(first - reference)))
        # Existing common scoring sorts quantiles; compare that same convention.
        reference = np.sort(reference, axis=-1)
        difference = float(np.max(np.abs(first - reference)))
        assert difference < 1e-8
        assert np.array_equal(first, second)
        forecasts = {}
        for profile in ('base', 'calendar', 'price_only'):
            for window in (24, 48, 72):
                value = chronos.predict(ctx, profile, window)
                assert value.shape == (24, 5) and np.isfinite(value).all()
                forecasts[f'{profile}:{window}'] = {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                                                    'mean_median': float(value[:, 2].mean())}
        result['regions'][region] = {'lightgbm_max_abs_difference': difference,
                                    'reference_rule': 'sort existing saved quantiles exactly as shared scoring',
                                    'native_unsorted_max_abs_difference': native_difference,
                                    'lightgbm_manifest': lightgbm.identity,
                                    'chronos_profiles': forecasts}
        print(region + ': LightGBM reproduction and nine Chronos inputs PASS', flush=True)
    record = {'evidence_id': 'synthetic-live-api-only', 'availability_basis': 'verified',
              'region': 'NSW1', 'payload': {'text': 'A planned outage affects 300 MW.'}}
    resolver = EventResolver()
    candidates = resolver.resolve(record)
    choice, trace = client.choose({'fixture': 'synthetic; not historical coverage', 'quoted_candidates': candidates['facts']},
                                 'Select the interpretation directly supported by the quotation. Do not infer price changes.',
                                 candidates['choices'])
    fact = resolver.resolve(record, choice)
    assert choice in candidates['choices']
    result['event_live_api'] = {'synthetic_evidence': True, 'result': fact, 'trace': trace}
    result['elapsed_seconds'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
