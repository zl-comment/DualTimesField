"""Predeclared 2022 development pilot; never opens the 2023–2024 test split.

Prediction completes for every region before any evaluation label file is read.
The existing main table is read-only and is not ranked using this small pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .workflow import (PROFILES, Snapshot, decide, diagnostics, digest, fixed_choice,
                       forecast_profiles)
from .transport import KevClient

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path('/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012/data_cache')
CHECKPOINT = Path('/root/workspace/Code/DualTimesField/weights/hf_home/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c')
CHECKPOINT_SHA = {'config.json': 'ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031',
                  'model.safetensors': 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_snapshots(region, count):
    directory = CACHE / region / 'selection'
    meta = json.loads((directory / 'metadata.json').read_text())
    origins = np.load(directory / 'origin_unix.npy', allow_pickle=False)
    targets = np.load(directory / 'target_time_unix.npy', allow_pickle=False)
    # Unix bounds for 2022-01-01 and 2022-07-01 in fixed AEST.
    if np.any(origins < 1640959200) or np.any(targets >= 1656597600):
        raise ValueError('Development data outside the frozen 2022 H1 boundary')
    if not 1 <= count <= len(origins):
        raise ValueError('Invalid origin count')
    indices = np.unique(np.linspace(0, len(origins) - 1, count, dtype=int))
    arrays = {name: np.load(directory / f'input__{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in ('history_values', 'history_calendar', 'future_calendar')}
    snapshots = []
    for index in indices:
        history = np.asarray(arrays['history_values'][index], dtype=np.float64)
        snapshots.append(Snapshot(region, int(origins[index]), targets[index],
                                  history[:, 0] * meta['target_std'] + meta['target_mean'],
                                  history[:, 1], arrays['history_calendar'][index],
                                  arrays['future_calendar'][index], meta['preprocessing_hash']))
    return indices, snapshots, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origins-per-region', type=int, default=24)
    parser.add_argument('--workflow', choices=('r15', 'r16', 'r17'), default='r15')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    decision_function = decide
    if args.workflow == 'r16':
        from .workflow_r16 import decide as decision_function
    elif args.workflow == 'r17':
        from .workflow_r17 import decide as decision_function
    if not 1 <= args.origins_per_region <= 24:
        raise ValueError('This preregistered pilot is capped at 24 origins per region')
    out = Path(args.out).resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-c931f386-8fcf-9f27-2526-47e12aac3145':
        raise ValueError('Set CUDA_VISIBLE_DEVICES to the reserved predictor GPU UUID')
    out.mkdir(parents=True, exist_ok=False)
    for filename, expected in CHECKPOINT_SHA.items():
        if sha_file(CHECKPOINT / filename) != expected:
            raise ValueError('Frozen predictor checksum mismatch: ' + filename)
    runtime_root = Path('/root/workspace/services/kev')
    if not (runtime_root / 'models.ready').exists() or not (runtime_root / 'runtime.ready').exists():
        raise RuntimeError('Kev deployment is not yet verified')
    model_files = json.loads((runtime_root / 'verified_weight_manifest.json').read_text())
    for filename, expected in model_files.items():
        if sha_file(filename) != expected['sha256']:
            raise ValueError('Kev weight checksum mismatch: ' + filename)
    client = KevClient()
    identity = client.identity()
    from ..chronos2_zeroshot.model import package_versions
    source_paths = sorted((ROOT / 'forecasting/api_agent').glob('*.py')) + sorted(
        (ROOT / 'forecasting/api_agent/skills').glob('*.txt'))
    source_paths += [ROOT / 'forecasting/aligned_scoring.py',
                     ROOT / 'forecasting/chronos2_zeroshot/model.py',
                     ROOT / 'configs/agent_forecasting/r15_kev_runtime.json']
    source_hashes = {str(p.relative_to(ROOT)): sha_file(p) for p in source_paths}
    input_sources = {}
    for region in ('NSW1', 'QLD1', 'TAS1'):
        for filename in ('metadata.json', 'origin_unix.npy', 'target_time_unix.npy',
                         'input__history_values.npy', 'input__history_calendar.npy',
                         'input__future_calendar.npy'):
            path = CACHE / region / 'selection' / filename
            input_sources[str(path)] = sha_file(path)
    manifest = {
        'release': args.workflow + '-pilot-v1',
        'base_commit': {
            'r15': '46e92fd9053a8c5e63619435e599496e023c669c',
            'r16': '0ae728e7e8eb478a7dc463c8b87dfbb0890660ad',
            'r17': 'caddb08ef9f749c6a2e029d83538d8b1a3015506',
        }[args.workflow],
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'source_hashes': source_hashes, 'source_fingerprint': digest(source_hashes),
        'input_source_hashes': input_sources,
        'kev_identity': identity, 'kev_weights': model_files, 'predictor_weights': CHECKPOINT_SHA,
        'predictor_packages': package_versions(),
        'kev_runtime_packages': (runtime_root / 'requirements.freeze.txt').read_text(),
        'kev_source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=runtime_root / 'runtime-src', text=True).strip(),
        'development_only': True, 'regions': ['NSW1', 'QLD1', 'TAS1'],
        'origins_per_region': args.origins_per_region, 'sampling': 'uniform linspace over selection cache',
        'training_calls': 0, 'max_kev_calls_per_origin': {'r15': 2, 'r16': 1, 'r17': 4}[args.workflow],
        'profiles': list(PROFILES), 'fixed_policy': 'lowest visible-day QS; change only for >2% reduction',
        'agent_guard': {
            'r15': 'choice probability >=.55 and >2% visible-day QS reduction; otherwise BASE',
            'r16': 'route conflicting eligible profiles only; probability <.55 falls back to FIXED',
            'r17': 'inspect fit before alternative commit; >2% visible-day QS support; probability <.55 or API failure uses FIXED; at most four decisions',
        }[args.workflow],
        'interpretation': 'development subset, not main-table rank; no pretraining-contamination exclusion',
        'input_scope': 'same 72-hour common input pool; effective covariate profiles may differ',
        'historical_fit': '48-to-24 hindcast wholly inside each origin history; no parameter fitting'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('The pilot requires the assigned prediction GPU')
    torch.manual_seed(2026)
    pipeline = Chronos2Pipeline.from_pretrained(str(CHECKPOINT), device_map='cuda', torch_dtype=torch.float32)
    started = time.monotonic()
    pending_scores = []
    for region in manifest['regions']:
        indices, snapshots, meta = load_snapshots(region, args.origins_per_region)
        predictions = {arm: [] for arm in ('BASE', 'FIXED', 'AGENT')}
        chosen = {arm: [] for arm in predictions}
        traces = []
        for ordinal, snapshot in enumerate(snapshots):
            # Exactly the same frozen weights for every option; no cross-origin state.
            with torch.inference_mode():
                forecasts, hindcasts = forecast_profiles(snapshot, pipeline)
            evidence = diagnostics(snapshot, hindcasts)
            fixed = fixed_choice(evidence)
            kwargs = {'forecasts': forecasts} if args.workflow == 'r17' else {}
            agent, trace = decision_function(snapshot, evidence, client, **kwargs)
            trace.update(region=region, origin_unix=snapshot.origin_unix,
                         recent_fit=evidence, ordinal=ordinal)
            for arm, profile in [('BASE', 'base'), ('FIXED', fixed), ('AGENT', agent)]:
                predictions[arm].append(forecasts[profile])
                chosen[arm].append(profile)
            traces.append(trace)
            with (out / 'decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(trace, allow_nan=False) + '\n')
            print(json.dumps({'region': region, 'done': ordinal + 1,
                              'fixed': fixed, 'agent': agent, 'fallback': trace['fallback']}), flush=True)
        array_path = out / f'{region}_predictions.npz'
        np.savez_compressed(array_path, indices=indices,
                            origin_unix=np.array([s.origin_unix for s in snapshots]),
                            target_time_unix=np.stack([s.target_time_unix for s in snapshots]),
                            **{arm: np.stack(values) for arm, values in predictions.items()})
        pending_scores.append((region, indices, snapshots, meta, predictions, chosen, traces))
    del pipeline
    torch.cuda.empty_cache()
    # No label file is opened before every forecast and API decision has finished.
    from ..aligned_scoring import make_prediction_artifact, score_predictions, QUANTILE_LEVELS
    scores = {}
    for region, indices, snapshots, meta, predictions, chosen, traces in pending_scores:
        actual = np.load(CACHE / region / 'selection' / 'actual.npy', allow_pickle=False)[indices]
        origins = np.array([s.origin_unix for s in snapshots])
        targets = np.stack([s.target_time_unix for s in snapshots])
        common_hash = digest([s.identity() for s in snapshots])
        scores[region] = {}
        for arm in predictions:
            effective_hash = digest({'common_input_hash': common_hash, 'profiles': chosen[arm]})
            metadata = dict(meta, input_hash=effective_hash, base_input_hash=common_hash,
                            sample_hash=digest([origins.tolist(), targets.tolist()]),
                            seed=2026, source=args.workflow + '_development:' + arm,
                            output_type='quantile', prediction_space='raw_rrp')
            artifact = make_prediction_artifact(actual=actual, origin_unix=origins,
                target_time_unix=targets, metadata=metadata, quantiles=np.stack(predictions[arm]),
                quantile_levels=QUANTILE_LEVELS)
            scores[region][arm] = score_predictions(artifact)
            scores[region][arm]['profile_counts'] = dict(Counter(chosen[arm]))
        scores[region]['agent_fallbacks'] = sum(t['fallback'] is not None for t in traces)
    metrics = ('quantile_score_proxy', 'mae', 'rmse_pooled', 'coverage_90', 'interval_width_90')
    summary = {arm: {m: float(np.mean([scores[r][arm][m] for r in scores])) for m in metrics}
               for arm in ('BASE', 'FIXED', 'AGENT')}
    report = {'scope': '2022 selection pilot only; no main-table rank', 'scores': scores,
              'region_equal_weight': summary, 'elapsed_seconds': time.monotonic() - started,
              'agent_beats_base_and_fixed_on_qs': summary['AGENT']['quantile_score_proxy'] < min(
                  summary['BASE']['quantile_score_proxy'], summary['FIXED']['quantile_score_proxy'])}
    for relative_path, expected in source_hashes.items():
        if sha_file(ROOT / relative_path) != expected:
            raise RuntimeError('Source changed during frozen pilot: ' + relative_path)
    for filename, expected in input_sources.items():
        if sha_file(filename) != expected:
            raise RuntimeError('Input cache changed during frozen pilot: ' + filename)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
