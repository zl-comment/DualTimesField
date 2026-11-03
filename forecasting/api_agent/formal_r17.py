"""Resumable full-grid R17 inference, followed by scoring after all nine cells.

No evaluation labels are extracted during inference. A saved BASE NPZ contains
labels, but the inference loader only extracts its predictions, axes and metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from .pilot import ROOT, CACHE, CHECKPOINT, CHECKPOINT_SHA, sha_file
from .transport import KevClient
from .workflow import PROFILES, Snapshot, diagnostics, digest, fixed_choice, predictor_input
from .workflow_r17 import decide

CONFIG = ROOT / 'configs/agent_forecasting/r17_formal_20261103.json'
MAIN = ROOT / 'logs/forecasting/main_comparison_20261023'
ARMS = ('BASE', 'FIXED', 'AGENT')
METRICS = ('quantile_score_proxy', 'mae', 'rmse_pooled', 'coverage_90', 'interval_width_90')


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / 'forecasting/api_agent').glob('*.py'))
    paths += sorted((ROOT / 'forecasting/api_agent/skills').glob('*.txt'))
    paths += [CONFIG, ROOT / 'forecasting/aligned_scoring.py', ROOT / 'forecasting/chronos2_zeroshot/model.py',
              ROOT / 'scripts/run_r17_formal.sh']
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths}


def verify_hashes(hashes, root=Path('/')):
    for filename, expected in hashes.items():
        if sha_file(root / filename) != expected:
            raise ValueError('Frozen file changed: ' + str(filename))


def load_region(region, cfg):
    directory = CACHE / region / cfg['split']
    meta = json.loads((directory / 'metadata.json').read_text())
    names = ('origin_unix', 'target_time_unix', 'input__history_values',
             'input__history_calendar', 'input__future_calendar')
    arrays = {k: np.load(directory / (k + '.npy'), mmap_mode='r', allow_pickle=False) for k in names}
    origins, targets = arrays['origin_unix'], arrays['target_time_unix']
    expected = cfg['start_unix'] + 3600 * np.arange(cfg['origins_per_region'])
    if not np.array_equal(origins, expected):
        raise ValueError('Formal origin grid mismatch')
    if not np.array_equal(targets, origins[:, None] + 3600 * np.arange(24)):
        raise ValueError('Formal target grid mismatch')
    if targets.max() >= cfg['target_end_exclusive_unix']:
        raise ValueError('Target outside formal period')
    return meta, arrays


def snapshot_at(region, meta, arrays, i):
    history = np.asarray(arrays['input__history_values'][i], dtype=np.float64)
    return Snapshot(region, int(arrays['origin_unix'][i]), arrays['target_time_unix'][i],
                    history[:, 0] * meta['target_std'] + meta['target_mean'], history[:, 1],
                    arrays['input__history_calendar'][i], arrays['input__future_calendar'][i],
                    meta['preprocessing_hash'])


def load_base(record, meta, arrays):
    with np.load(record['path'], allow_pickle=False) as z:
        saved_meta = json.loads(str(z['metadata_json'].item()))
        for key in ('protocol_id', 'split', 'region', 'input_hash', 'sample_hash', 'preprocessing_hash'):
            if saved_meta[key] != meta[key]:
                raise ValueError('Saved BASE metadata mismatch: ' + key)
        for key in ('origin_unix', 'target_time_unix'):
            if not np.array_equal(z[key], arrays[key]):
                raise ValueError('Saved BASE axis mismatch: ' + key)
        # Deliberately do not extract z['actual'] here.
        return z['quantiles']


def candidate_batch(snapshots, base, pipeline, batch_size):
    from ..chronos2_zeroshot.model import predict_common_quantiles
    forecasts, hindcasts = {'base': base}, {}
    for profile in PROFILES:
        rows = []
        for snapshot in snapshots:
            if profile != 'base':
                rows.append(predictor_input(snapshot, profile))
            rows.append(predictor_input(snapshot, profile, hindcast=True))
        raw = predict_common_quantiles(pipeline, rows, batch_size=batch_size, context_length=72)
        if profile == 'base':
            hindcasts[profile] = raw
        else:
            forecasts[profile], hindcasts[profile] = raw[::2], raw[1::2]
    return forecasts, hindcasts


def decide_one(item):
    snapshot, raw, hindcast = item
    evidence = diagnostics(snapshot, hindcast)
    canonical = {p: np.sort(raw[p], axis=-1) for p in PROFILES}
    selected, trace = decide(snapshot, evidence, KevClient(), forecasts=canonical)
    trace['raw_crossed_quantile_hours'] = {
        p: int(np.any(np.diff(raw[p], axis=-1) < 0, axis=-1).sum()) for p in PROFILES}
    return selected, fixed_choice(evidence), trace


def checked_shard(directory, fingerprint):
    marker = directory / 'complete.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record['source_fingerprint'] != fingerprint:
        raise ValueError('Cannot resume a different release')
    verify_hashes(record['files'], directory)
    return True


def freeze(out, cfg):
    verify_hashes({MAIN / 'main.csv': cfg['main_table_sha256'],
                   MAIN / 'verification.json': cfg['verification_sha256']})
    records = [r for r in json.loads((MAIN / 'verification.json').read_text())['artifacts']
               if r['model'] == 'chronos2_zeroshot']
    if len(records) != 9:
        raise ValueError('Need all nine original BASE artifacts')
    inputs = {}
    for region in cfg['regions']:
        for name in ('metadata.json', 'origin_unix.npy', 'target_time_unix.npy',
                     'input__history_values.npy', 'input__history_calendar.npy', 'input__future_calendar.npy'):
            path = CACHE / region / cfg['split'] / name
            inputs[str(path)] = sha_file(path)
    hashes = source_hashes()
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['source_hashes'] != hashes or manifest['config'] != cfg or manifest['input_hashes'] != inputs:
            raise ValueError('Resume source/config/input mismatch')
        verify_hashes(manifest['weight_hashes'])
        verify_hashes({r['path']: r['sha256'] for r in records})
        return manifest, records
    runtime = Path('/root/workspace/services/kev')
    weights = {str(CHECKPOINT / f): sha for f, sha in CHECKPOINT_SHA.items()}
    weights.update({k: v['sha256'] for k, v in json.loads((runtime / 'verified_weight_manifest.json').read_text()).items()})
    verify_hashes(weights)
    verify_hashes({r['path']: r['sha256'] for r in records})
    from ..chronos2_zeroshot.model import package_versions
    manifest = {'config': cfg, 'source_hashes': hashes, 'source_fingerprint': digest(hashes),
                'input_hashes': inputs, 'weight_hashes': weights, 'original_baselines': records,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'kev_identity': KevClient().identity(), 'predictor_packages': package_versions(),
                'kev_runtime_packages': (runtime / 'requirements.freeze.txt').read_text(),
                'created_unix': time.time(), 'training_calls': 0}
    if manifest['branch'] != cfg['branch']:
        raise ValueError('Wrong experiment branch')
    write_json(out / 'manifest.json', manifest)
    return manifest, records


def run_inference(out, cfg, manifest, records):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Reserved prediction GPU is unavailable')
    pipeline = Chronos2Pipeline.from_pretrained(str(CHECKPOINT), device_map='cuda', torch_dtype=torch.float32)
    started, completed = time.monotonic(), 0
    size, total = cfg['shard_size'], cfg['origins_per_region'] * 9
    with ThreadPoolExecutor(max_workers=cfg['api_workers']) as pool:
        for seed in cfg['seeds']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for region in cfg['regions']:
                meta, arrays = load_region(region, cfg)
                baseline = next(r for r in records if r['region'] == region and r['seed'] == seed)
                base = load_base(baseline, meta, arrays)
                for start in range(0, len(base), size):
                    stop = min(start + size, len(base))
                    directory = out / f'seed{seed}' / region / f'{start:06d}'
                    directory.mkdir(parents=True, exist_ok=True)
                    if checked_shard(directory, manifest['source_fingerprint']):
                        completed += stop - start
                        continue
                    verify_hashes(manifest['source_hashes'], ROOT)
                    snapshots = [snapshot_at(region, meta, arrays, i) for i in range(start, stop)]
                    with torch.inference_mode():
                        forecasts, hindcasts = candidate_batch(snapshots, base[start:stop], pipeline,
                                                               cfg['predictor_batch_size'])
                    jobs = [(s, {p: forecasts[p][i] for p in PROFILES},
                             {p: hindcasts[p][i] for p in PROFILES}) for i, s in enumerate(snapshots)]
                    results = list(pool.map(decide_one, jobs))
                    chosen = {'BASE': ['base'] * len(results),
                              'FIXED': [r[1] for r in results], 'AGENT': [r[0] for r in results]}
                    predictions = {arm: np.stack([forecasts[p][i] for i, p in enumerate(profiles)])
                                   for arm, profiles in chosen.items()}
                    with (directory / 'predictions.npz.tmp').open('wb') as stream:
                        np.savez_compressed(stream, **predictions, origin_unix=arrays['origin_unix'][start:stop],
                                            target_time_unix=arrays['target_time_unix'][start:stop],
                                            **{arm + '_profiles': np.array(ps) for arm, ps in chosen.items()})
                    (directory / 'predictions.npz.tmp').replace(directory / 'predictions.npz')
                    with gzip.open(directory / 'decisions.jsonl.gz.tmp', 'wt') as stream:
                        for i, (_, _, trace) in enumerate(results):
                            trace.update(region=region, seed=seed, ordinal=start + i,
                                         origin_unix=snapshots[i].origin_unix)
                            stream.write(json.dumps(trace, allow_nan=False) + '\n')
                    (directory / 'decisions.jsonl.gz.tmp').replace(directory / 'decisions.jsonl.gz')
                    write_json(directory / 'complete.json', {
                        'source_fingerprint': manifest['source_fingerprint'], 'start': start, 'stop': stop,
                        'files': {name: sha_file(directory / name) for name in ('predictions.npz', 'decisions.jsonl.gz')}})
                    completed += len(results)
                    progress = {'status': 'RUNNING', 'completed': completed, 'total': total,
                                'seed': seed, 'region': region, 'next_origin': stop,
                                'seconds_this_process': time.monotonic() - started,
                                'updated_unix': time.time()}
                    write_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
    del pipeline
    torch.cuda.empty_cache()
    write_json(out / 'inference_complete.json', {'total': completed, 'completed_unix': time.time(),
                                               'source_fingerprint': manifest['source_fingerprint']})


def require_complete(out, cfg, fingerprint):
    directories = {}
    for seed in cfg['seeds']:
        for region in cfg['regions']:
            shards = [out / f'seed{seed}' / region / f'{i:06d}'
                      for i in range(0, cfg['origins_per_region'], cfg['shard_size'])]
            for shard in shards:
                if not checked_shard(shard, fingerprint):
                    raise ValueError('Scoring blocked: inference is incomplete')
            directories[(seed, region)] = shards
    return directories


def per_origin_qs(actual, quantiles):
    levels = np.array([.05, .1, .5, .9, .95])
    error = actual[..., None] - np.sort(quantiles, axis=-1)
    return 2 * np.maximum(levels * error, (levels - 1) * error).mean(axis=(1, 2))


def paired_block_interval(delta, block_hours=168, draws=2000):
    """Same time blocks across regions; seed repeats are averaged, not resampled."""
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(2026)
    count = int(np.ceil(len(delta) / block_hours))
    estimates = []
    for start in range(0, draws, 50):
        starts = rng.integers(0, len(delta), size=(min(50, draws - start), count))
        indices = ((starts[..., None] + np.arange(block_hours)) % len(delta)).reshape(len(starts), -1)
        estimates.extend(delta[indices[:, :len(delta)]].mean(axis=1).tolist())
    return {'mean_agent_minus_fixed_qs': float(delta.mean()),
            'percentile_95_interval': np.quantile(estimates, [.025, .975]).tolist(),
            'block_hours': block_hours, 'draws': draws,
            'interpretation': 'Paired circular weekly-block bootstrap; negative favors Agent. Descriptive given prior test exposure.'}


def score(out, cfg, manifest):
    from ..aligned_scoring import make_prediction_artifact, score_predictions, QUANTILE_LEVELS
    directories = require_complete(out, cfg, manifest['source_fingerprint'])
    verify_hashes(manifest['source_hashes'], ROOT)
    verify_hashes(manifest['input_hashes'])
    verify_hashes({MAIN / 'main.csv': cfg['main_table_sha256'],
                   MAIN / 'verification.json': cfg['verification_sha256']})
    rows, behavior, seed_arrays, paired_deltas = [], [], {}, []
    for (seed, region), shards in directories.items():
        meta, arrays = load_region(region, cfg)
        # First extraction of future outcomes: all nine cells are complete.
        label_file = CACHE / region / cfg['split'] / 'actual.npy'
        actual = np.load(label_file, allow_pickle=False)
        parts = {k: [] for k in (*ARMS, 'origin_unix', 'target_time_unix',
                                 *(a + '_profiles' for a in ARMS))}
        counts, call_count, fallback_count, changed = Counter(), 0, 0, 0
        latency, tokens = 0.0, 0
        for shard in shards:
            with np.load(shard / 'predictions.npz', allow_pickle=False) as z:
                for key in parts:
                    parts[key].append(z[key])
            with gzip.open(shard / 'decisions.jsonl.gz', 'rt') as stream:
                for line in stream:
                    t = json.loads(line)
                    counts.update(x['tool'] for x in t['tool_calls'])
                    counts['stop:' + t['stop_reason']] += 1
                    fallback_count += t['fallback'] is not None
                    changed += t['selected_profile'] != t['fixed_choice']
                    call_count += len(t['calls'])
                    for call in t['calls']:
                        latency += call.get('elapsed_seconds', 0)
                        tokens += call.get('response', {}).get('usage', {}).get('input_tokens', 0)
        data = {k: np.concatenate(v) for k, v in parts.items()}
        for key in ('origin_unix', 'target_time_unix'):
            if not np.array_equal(data[key], arrays[key]):
                raise ValueError('Incomplete or misaligned saved predictions')
        for arm in ARMS:
            effective_hash = digest({'common_input_hash': meta['input_hash'],
                                     'profiles': data[arm + '_profiles'].tolist()})
            metadata = dict(meta, input_hash=effective_hash, base_input_hash=meta['input_hash'], seed=seed,
                            source='r17_formal:' + arm, output_type='quantile', prediction_space='raw_rrp')
            artifact = make_prediction_artifact(actual=actual, quantiles=data[arm],
                origin_unix=data['origin_unix'], target_time_unix=data['target_time_unix'],
                quantile_levels=QUANTILE_LEVELS, metadata=metadata)
            metrics = score_predictions(artifact)
            rows.append({'seed': seed, 'region': region, 'arm': arm,
                         **{m: metrics[m] for m in METRICS}})
        paired_deltas.append(per_origin_qs(actual, data['AGENT']) - per_origin_qs(actual, data['FIXED']))
        seed_arrays[(seed, region)] = {a: digest(data[a].tolist()) for a in ARMS}
        behavior.append({'seed': seed, 'region': region, 'calls': call_count,
                         'fallbacks': fallback_count, 'changed_from_fixed': changed,
                         'input_tokens': tokens, 'summed_api_latency_seconds': latency,
                         'tool_and_stop_counts': dict(counts), 'label_sha256': sha_file(label_file)})
    summary = {}
    for arm in ARMS:
        per_seed = {seed: {m: float(np.mean([r[m] for r in rows if r['seed'] == seed and r['arm'] == arm]))
                           for m in METRICS} for seed in cfg['seeds']}
        summary[arm] = {'per_seed_region_mean': per_seed,
                        'mean': {m: float(np.mean([v[m] for v in per_seed.values()])) for m in METRICS},
                        'std': {m: float(np.std([v[m] for v in per_seed.values()])) for m in METRICS}}
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (MAIN / 'main.csv').open() as stream:
        originals = list(csv.DictReader(stream))
    original_base = next(r for r in originals if r['model'] == 'chronos2_zeroshot')
    for metric in METRICS:
        if not np.isclose(summary['BASE']['mean'][metric], float(original_base[metric + '_mean']), rtol=1e-9, atol=1e-9):
            raise ValueError('Reused BASE does not reproduce main-table metric: ' + metric)
    rankings = []
    for metric in ('quantile_score_proxy', 'mae'):
        comparable = [r for r in originals if r[metric + '_mean'] != '']
        for arm in ('FIXED', 'AGENT'):
            value = summary[arm]['mean'][metric]
            rankings.append({'arm': arm, 'metric': metric, 'value': value,
                             'rank_against_original_rows': 1 + sum(float(r[metric + '_mean']) < value for r in comparable),
                             'original_rows': len(comparable),
                             'scope': 'same targets and public input pool; effective profile and model training costs differ'})
    base_qs = summary['BASE']['mean']['quantile_score_proxy']
    fixed_qs = summary['FIXED']['mean']['quantile_score_proxy']
    agent_qs = summary['AGENT']['mean']['quantile_score_proxy']
    report = {'status': 'COMPLETE', 'config': cfg, 'summary': summary, 'behavior': behavior,
              'qs_gain_vs_base_percent': 100 * (base_qs - agent_qs) / base_qs,
              'qs_gain_vs_fixed_percent': 100 * (fixed_qs - agent_qs) / fixed_qs,
              'paired_uncertainty': paired_block_interval(np.mean(paired_deltas, axis=0)),
              'main_table_reference_ranks': rankings,
              'seed_identical_predictions': {region: {a: len({seed_arrays[(s, region)][a]
                  for s in cfg['seeds']}) == 1 for a in ARMS} for region in cfg['regions']},
              'source_fingerprint': manifest['source_fingerprint'],
              'limitations': ['No independent training seeds; Kev argmax has no sampling seed.',
                              'Earlier project versions have used this historical test period.',
                              'Pretraining overlap cannot be excluded.',
                              'Mean improvement alone does not establish statistical significance.']}
    write_json(out / 'report.json', report)
    write_json(out / 'progress.json', {'status': 'COMPLETE', 'total': cfg['origins_per_region'] * 9,
                                     'completed': cfg['origins_per_region'] * 9, 'updated_unix': time.time()})
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-c931f386-8fcf-9f27-2526-47e12aac3145':
        raise ValueError('Use reserved predictor GPU 7')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    try:
        manifest, records = freeze(out, cfg)
        run_inference(out, cfg, manifest, records)
        score(out, cfg, manifest)
    except Exception as exc:
        write_json(out / 'progress.json', {'status': 'FAILED', 'error': type(exc).__name__ + ': ' + str(exc),
                                         'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
