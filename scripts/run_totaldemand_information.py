#!/usr/bin/env python3
"""Frozen S1 historical experiment; preparation, prediction and scoring are separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.agent_forecasting.historical_replay import AEST, QUANTILES, build_records
from forecasting.agent_forecasting.historical_comparison import compare_s1
from forecasting.agent_forecasting.totaldemand_information import read_month
from forecasting.agent_forecasting.totaldemand_proxy import model_input, selected_proxy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def usage(start):
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_seconds=time.monotonic() - start, cpu_user_seconds=r.ru_utime,
                cpu_system_seconds=r.ru_stime, peak_rss_bytes=r.ru_maxrss * 1024)


def identity(cfg):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != cfg['branch']:
        raise ValueError('wrong experiment branch')
    hashes = {}
    for name in cfg['source_files']:
        raw = (ROOT / name).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{head}:{name}'], cwd=ROOT)
        if raw != committed:
            raise ValueError(f'uncommitted experiment source: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'execution_commit': head, 'branch': branch, 'source_sha256': hashes}


def load_price_window(path, first, end):
    """Parse numeric columns only for the declared historical input/label window."""
    columns = ('delivery_start_aest', 'available_at_aest', 'region',
               'rrp_aud_per_mwh', 'total_demand_mw')
    records = []
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if first <= row['delivery_start_aest'][:10] < end:
                records.append({key: row[key] for key in columns})
    frame = pd.DataFrame(records, columns=columns)
    for name in ('rrp_aud_per_mwh', 'total_demand_mw'):
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    return frame


def prepare(cfg):
    start = time.monotonic()
    ident = identity(cfg)
    root = ROOT / cfg['output_root']
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    (root / 'labels').mkdir()
    coverage_path = ROOT / cfg['coverage_file']
    if sha256(coverage_path) != cfg['coverage_sha256']:
        raise ValueError('source coverage snapshot changed')
    audited = pd.read_csv(coverage_path)
    required_seqnos = set(audited.loc[audited['covered'], 'selected_seqno'].astype(int))
    old_manifest = {row['month']: row for row in read_json(ROOT / cfg['source_manifest'])}
    frames, sources = [], []
    for month in cfg['months']:
        record = old_manifest[month]
        path = Path(record['path'])
        digest = sha256(path)
        if digest != record['sha256']:
            raise ValueError(f'archive changed: {month}')
        rows = read_month(path, cfg['regions'])
        frames.append(rows.loc[rows['seqno'].isin(required_seqnos)])
        sources.append({'month': month, 'path': str(path), 'sha256': digest,
                        'bytes': path.stat().st_size})
        print(f'prepared source {month}', flush=True)
    rows = pd.concat(frames, ignore_index=True)
    groups = rows.groupby(['region', 'seqno', 'runno'], sort=False)
    proxies = {}
    for record in audited.to_dict('records'):
        key = f"{record['region']}:{record['origin_unix']}"
        group = (groups.get_group((record['region'], int(record['selected_seqno']),
                                  int(record['selected_runno']))) if record['covered'] else rows.iloc[:0])
        proxies[key] = selected_proxy(group, record)
    write_json(root / 'inputs' / 'proxies.json', proxies)
    dates = pd.date_range(cfg['first_date'], cfg['last_date'], freq='D', tz=AEST)
    origins = np.array([int(day.timestamp()) for day in dates], dtype=np.int64)
    issues, input_counts, history_sources = {}, {}, {}
    for region in cfg['regions']:
        path = Path(cfg['data_root']) / f'{region.lower()}_hourly.csv'
        digest = sha256(path)
        if digest != cfg['data_sha256'][region]:
            raise ValueError(f'historical input snapshot changed: {region}')
        frame = load_price_window(path, cfg['history_first_date'], cfg['label_end_exclusive'])
        records, labels, missing = build_records(frame, region, dates)
        write_json(root / 'inputs' / f'{region}.json', records)
        save_npz(root / 'labels' / f'{region}.npz', actual=labels,
                 origin_unix=np.array([r['origin_unix'] for r in records], dtype=np.int64))
        issues[region], input_counts[region] = missing, len(records)
        history_sources[region] = {'path': str(path), 'sha256': digest,
            'numeric_window': [cfg['history_first_date'], cfg['label_end_exclusive']],
            'byte_hash_scope': 'whole CSV; numeric arrays restricted to declared window'}
    manifest = {**ident, 'dates': [str(day.date()) for day in dates], 'origins': origins.tolist(),
        'regions': cfg['regions'], 'input_counts': input_counts, 'issues': issues,
        'source_complete': [[proxies[f'{region}:{int(origin)}'] is not None
                             for region in cfg['regions']] for origin in origins],
        'sources': sources, 'historical_sources': history_sources,
        'publication_basis': 'archive_proxy', 'strict_asof_verified': False,
        'candidate': cfg['candidate_id'], 'model_calls': 0, 'price_scores': 0,
        'files': {str(path.relative_to(root)): sha256(path)
                  for folder in ('inputs', 'labels') for path in sorted((root / folder).iterdir())},
        'resources': usage(start)}
    write_json(root / 'prepared.json', manifest)
    print(json.dumps({'prepared': input_counts, 'resources': manifest['resources']}), flush=True)


def prepared(cfg, include_labels=False):
    root = ROOT / cfg['output_root']
    manifest = read_json(root / 'prepared.json')
    if identity(cfg) != {key: manifest[key] for key in ('execution_commit', 'branch', 'source_sha256')}:
        raise ValueError('execution source differs from prepared input')
    for name, digest in manifest['files'].items():
        if name.startswith('labels/') and not include_labels:
            continue
        if sha256(root / name) != digest:
            raise ValueError(f'prepared file changed: {name}')
    return root, manifest


def input_hash(task):
    digest = hashlib.sha256()
    for category in ('target', 'past_covariates', 'future_covariates'):
        values = {'target': task[category]} if category == 'target' else task[category]
        for name in sorted(values):
            array = np.asarray(values[name], dtype='<f8')
            digest.update(f'{category}/{name}/{array.shape}'.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def scoring_horizon(native, count):
    values = np.asarray(native, dtype=np.float64)
    if values.shape != (count, 1, 25, 5):
        raise ValueError(f'invalid native forecast shape: {values.shape}')
    return values[:, 0, 1:, :]


def predict(cfg):
    start = time.monotonic()
    root, manifest = prepared(cfg)  # No label files opened in this process.
    output = root / 'predictions'
    output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
    import torch
    from chronos import Chronos2Pipeline
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    checkpoint = cfg['checkpoint']
    for name, key in (('config.json', 'config_sha256'), ('model.safetensors', 'weights_sha256')):
        if sha256(Path(checkpoint['directory']) / name) != checkpoint[key]:
            raise ValueError(f'checkpoint changed: {name}')
    pipeline = Chronos2Pipeline.from_pretrained(
        checkpoint['directory'], device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
    pipeline.model.eval()
    for parameter in pipeline.model.parameters():
        parameter.requires_grad_(False)
    proxies = read_json(root / 'inputs' / 'proxies.json')
    attempts = {'A': 0, 'B': 0}
    calls = {'A': 0, 'B': 0}
    failures = []

    def infer(tasks, arm):
        calls[arm] += 1
        attempts[arm] += len(tasks)
        with torch.inference_mode():
            q, _ = pipeline.predict_quantiles(tasks, prediction_length=25,
                quantile_levels=QUANTILES.tolist(), batch_size=cfg['series_batch_size'],
                context_length=72, cross_learning=False)
        return scoring_horizon(q, len(tasks))

    with (output / 'ledger.jsonl').open('x') as ledger:
        for region in cfg['regions']:
            records = read_json(root / 'inputs' / f'{region}.json')
            a = np.full((len(records), 24, 5), np.nan)
            b = np.full_like(a, np.nan)
            complete = np.zeros(len(records), dtype=bool)
            # The first declared origin is the interface smoke and is reused, not rerun.
            slices = [(0, min(1, len(records)))] + [
                (i, min(i + cfg['origins_per_batch'], len(records))) for i in range(1, len(records), cfg['origins_per_batch'])]
            for first, end in slices:
                if first == end:
                    continue
                batch = records[first:end]
                tasks_a = [model_input(record) for record in batch]
                states = [{'region': region, 'origin_unix': record['origin_unix'],
                           'input_A_sha256': input_hash(task), 'input_B_sha256': None,
                           'A_status': 'unavailable', 'B_status': 'unavailable'}
                          for record, task in zip(batch, tasks_a)]
                try:
                    qa = infer(tasks_a, 'A')
                except (RuntimeError, ValueError) as exc:
                    failures.append({'region': region, 'first': first, 'arm': 'A', 'error': str(exc)})
                    qa = np.full((len(batch), 24, 5), np.nan)
                a[first:end] = qa
                b[first:end] = qa
                tasks_b, indices_b = [], []
                for j, (record, state) in enumerate(zip(batch, states)):
                    proxy = proxies[f"{region}:{record['origin_unix']}"]
                    complete[first + j] = proxy is not None
                    if not np.isfinite(qa[j]).all():
                        continue
                    state['A_status'] = 'predicted'
                    if proxy is None:
                        state['B_status'] = 'fallback_source_missing'
                        state['input_B_sha256'] = state['input_A_sha256']
                        continue
                    task_b = model_input(record, proxy['hourly_values'])
                    state['input_B_sha256'] = input_hash(task_b)
                    tasks_b.append(task_b)
                    indices_b.append(j)
                if tasks_b:
                    try:
                        qb = infer(tasks_b, 'B')
                        for j, values in zip(indices_b, qb):
                            if np.isfinite(values).all():
                                b[first + j] = values
                                states[j]['B_status'] = 'predicted'
                            else:
                                states[j]['B_status'] = 'fallback_nonfinite_forecast'
                    except (RuntimeError, ValueError) as exc:
                        failures.append({'region': region, 'first': first, 'arm': 'B', 'error': str(exc)})
                        for j in indices_b:
                            states[j]['B_status'] = 'fallback_model_error'
                for state in states:
                    ledger.write(json.dumps(state) + '\n')
                ledger.flush()
                print(f'{region}: {end}/{len(records)} paired origins', flush=True)
            save_npz(output / f'{region}.npz', A=a, B=b, source_complete=complete,
                     origin_unix=np.array([r['origin_unix'] for r in records], dtype=np.int64),
                     target_time_unix=np.array([r['target_time_unix'] for r in records], dtype=np.int64))
    report = {'execution_commit': manifest['execution_commit'], 'seed': cfg['seed'],
        'unit_attempts': attempts, 'pipeline_calls': calls, 'failures': failures,
        'device': 'cpu', 'dtype': 'float32', 'host': platform.node(), 'cpu_threads': torch.get_num_threads(),
        'affinity': sorted(os.sched_getaffinity(0)), 'cross_learning': False,
        'training_calls': 0, 'llm_calls': 0, 'checkpoint': checkpoint,
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'torch', 'chronos-forecasting', 'transformers')},
        'resources': usage(start),
        'files': {path.name: sha256(path) for path in sorted(output.iterdir())}}
    write_json(root / 'predicted.json', report)  # Completion marker written last.
    print(json.dumps({'prediction_complete': attempts, 'resources': report['resources']}), flush=True)


def score(cfg):
    start = time.monotonic()
    root, manifest = prepared(cfg, include_labels=True)
    run = read_json(root / 'predicted.json')
    for name, digest in run['files'].items():
        if sha256(root / 'predictions' / name) != digest:
            raise ValueError(f'prediction artifact changed: {name}')
    index = {int(origin): i for i, origin in enumerate(manifest['origins'])}
    days, regions = len(index), cfg['regions']
    a = np.full((days, len(regions), 24, 5), np.nan)
    b, y = np.full_like(a, np.nan), np.full((days, len(regions), 24), np.nan)
    source = np.asarray(manifest['source_complete'], dtype=bool)
    for r, region in enumerate(regions):
        with np.load(root / 'predictions' / f'{region}.npz', allow_pickle=False) as p:
            for j, origin in enumerate(p['origin_unix']):
                i = index[int(origin)]
                expected = int(origin) + np.arange(24) * 3600
                if not np.array_equal(p['target_time_unix'][j], expected):
                    raise ValueError('forecast target alignment changed')
                if bool(source[i, r]) != bool(p['source_complete'][j]):
                    raise ValueError('prediction source status differs from prepared source')
                a[i, r], b[i, r] = p['A'][j], p['B'][j]
        with np.load(root / 'labels' / f'{region}.npz', allow_pickle=False) as labels:
            for j, origin in enumerate(labels['origin_unix']):
                y[index[int(origin)], r] = labels['actual'][j]
    result = compare_s1(a, b, y, source, manifest['dates'], regions)
    result.update(execution_commit=manifest['execution_commit'], candidate=cfg['candidate_id'],
                  prediction_run=run, preparation_resources=manifest['resources'], scoring_resources=usage(start),
                  seeds_executed=[cfg['seed']], agent_experiments_executed=False)
    archive = ROOT / cfg['archive_root']
    archive.mkdir(parents=True, exist_ok=False)
    write_json(archive / 'metrics.json', result)
    save_npz(archive / 'paired_predictions.npz', A=a, B=b, actual=y, source_complete=source,
             origin_unix=np.array(manifest['origins'], dtype=np.int64))
    rows = []
    for region, row in result['regional'].items():
        for arm in ('A', 'B'):
            if row[arm] is not None:
                rows.append({'region': region, 'arm': arm, **{k: v for k, v in row[arm].items()
                                                           if k != 'pinball_by_quantile'}})
    pd.DataFrame(rows).to_csv(archive / 'regional_metrics.csv', index=False)
    write_json(archive / 'prepared_manifest.json', manifest)
    print(json.dumps({'decision': result['decision'], 'macro': result['macro'],
                      'failed_criteria': result['failed_criteria']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'predict', 'score'))
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage](read_json(args.config))
