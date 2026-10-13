"""Frozen, resumable full-data matrix for the shared-boundary experiment.

prepare snapshots code/configuration and builds shared caches once per region;
run schedules isolated workers, and can be restarted without repeating completed
cells. A failed cell remains failed until an explicit --retry-failed restart.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from .aligned_inputs import AlignedSplit, INPUT_KEYS, build_aligned_splits
from .aligned_runner import NEURAL_MODELS, fit_neural
from .aligned_scoring import (QUANTILE_LEVELS, make_prediction_artifact,
                              save_prediction_artifact, score_predictions, load_prediction_artifact)

REGIONS = ('NSW1', 'QLD1', 'TAS1')
SEEDS = (2026, 2027, 2028)
SPLITS = ('train', 'selection', 'calibration', 'test')
BASE_MODELS = ('full', 'no_fields', 'mlp', 'linear', 'dlinear', 'patchtst',
               'gatets_adapter', 'xgboost', 'lightgbm', 'lasso_arx72_cp')
EXTENSION_MODELS = ('timexer', 'pricefm')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config, *, extension=False):
    exp = config['experiment']
    if config.get('aligned_protocol', {}).get('stage') != 'full_fixed_configuration_rerun':
        raise ValueError('Expected a full-data fixed-configuration protocol')
    if tuple(exp['regions']) != REGIONS or tuple(exp['seeds']) != SEEDS:
        raise ValueError('This matrix requires all three regions and all three seeds')
    required = EXTENSION_MODELS if extension else BASE_MODELS
    if set(exp['models']) != set(required) or len(exp['models']) != len(required):
        raise ValueError(f'All {len(required)} required models must be included exactly once')
    if config['data'].get('price_transform') != 'identity' or config['data'].get('quantile_target') != 'transformed':
        raise ValueError('Use the shared affine price space')
    if tuple(config['data']['split_names']) != SPLITS:
        raise ValueError('Independent selection/calibration/test splits are required')
    if not np.array_equal(config['model']['quantiles'], QUANTILE_LEVELS):
        raise ValueError('Model quantiles must match shared scoring levels')
    if exp.get('max_origins') is not None:
        raise ValueError('Formal matrix must not truncate origins')
    if config['training']['epochs'] < 1 or config['training']['patience'] < 1:
        raise ValueError('Positive epoch and patience budgets required')
    boundaries = config['data']['split_boundaries']
    for left, right in zip(SPLITS, SPLITS[1:]):
        if boundaries[left]['end_exclusive'] != boundaries[right]['start']:
            raise ValueError('Experiment periods must be contiguous and ordered')


def freeze_sources(project, destination):
    for name in ('forecasting', 'src'):
        shutil.copytree(project / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, digest


def save_cache(root, splits):
    files = {}
    for name, split in splits.items():
        directory = root / name
        directory.mkdir(parents=True)
        arrays = {**{f'input__{k}': v for k, v in split.inputs.items()},
                  'actual': split.actual, 'target': split.target,
                  'origin_unix': split.origin_unix, 'target_time_unix': split.target_time_unix}
        for key, values in arrays.items():
            path = directory / f'{key}.npy'
            np.save(path, values, allow_pickle=False)
            files[str(path.relative_to(root))] = sha(path)
        atomic_json(directory / 'metadata.json', split.metadata)
        files[str((directory / 'metadata.json').relative_to(root))] = sha(directory / 'metadata.json')
    atomic_json(root / 'hashes.json', files)
    return {name: {'origins': len(split.origin_unix), 'metadata': split.metadata} for name, split in splits.items()}


def load_cache(root, expected_hashes_hash):
    if sha(root / 'hashes.json') != expected_hashes_hash:
        raise ValueError('Cache hash manifest changed')
    for name, digest in read_json(root / 'hashes.json').items():
        if sha(root / name) != digest:
            raise ValueError(f'Cached data changed: {name}')
    result = {}
    for name in SPLITS:
        folder = root / name
        def array(key):
            return np.load(folder / f'{key}.npy', mmap_mode='r', allow_pickle=False)
        result[name] = AlignedSplit(
            {k: array(f'input__{k}') for k in INPUT_KEYS}, array('actual'), array('origin_unix'),
            array('target_time_unix'), read_json(folder / 'metadata.json'), array('target'))
    return result


def prepare(config_path, run_root, extend_from=None):
    project = Path.cwd().resolve()
    config_path, run_root = Path(config_path).resolve(), Path(run_root).resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, extension=extend_from is not None)
    if run_root.exists():
        raise FileExistsError('Use a new dated run directory')
    parent = None
    if extend_from is not None:
        from .aligned_extension import audit_parent, assert_shared_protocol
        parent_root = Path(extend_from).resolve()
        parent = audit_parent(parent_root)
        assert_shared_protocol(yaml.safe_load((parent_root / 'config.yaml').read_text()), config)
    run_root.mkdir(parents=True)
    shutil.copyfile(config_path, run_root / 'config.yaml')
    source_files, source_hash = freeze_sources(project, run_root / 'source')
    config['project_root'] = str(project)
    paths = set()
    for values in config.values():
        if isinstance(values, dict):
            paths.update(values.get('region_files', {}).values())
            if isinstance(values.get('path'), str):
                paths.add(values['path'])
    data_hashes = (parent['data_hashes'] if parent is not None else
                   {str((project / path).resolve()): sha(project / path) for path in sorted(paths)})
    cache_info, cache_hashes = {}, {}
    if parent is not None:
        (run_root / 'data_cache').symlink_to(parent_root / 'data_cache', target_is_directory=True)
        cache_info, cache_hashes = parent['cache_info'], parent['cache_hashes']
    for region in (() if parent is not None else REGIONS):
        print(f'Preparing complete shared data cache: {region}', flush=True)
        splits = build_aligned_splits(config, region, split_names=SPLITS)
        cache_info[region] = save_cache(run_root / 'data_cache' / region, splits)
        cache_hashes[region] = sha(run_root / 'data_cache' / region / 'hashes.json')
        print(f'{region} origins: ' + str({k: len(v.origin_unix) for k, v in splits.items()}), flush=True)
        del splits
        gc.collect()
    jobs = []
    for seed in SEEDS:
        for region in REGIONS:
            for model in config['experiment']['models']:
                identifier = f'{model}/{seed}/{region}'
                jobs.append({'id': identifier, 'model': model, 'region': region, 'seed': seed,
                             'output': f'runs/{model}/seed{seed}/{region}',
                             'resource': 'gpu' if model in NEURAL_MODELS or model == 'xgboost' else 'cpu'})
    manifest = {'run_id': run_root.name, 'created_at_utc': timestamp(), 'project_root': str(project),
                'source_hash': source_hash, 'source_files': source_files,
                'config_hash': sha(run_root / 'config.yaml'), 'data_hashes': data_hashes,
                'cache_hashes': cache_hashes, 'cache_info': cache_info, 'jobs': jobs,
                'models': config['experiment']['models'], 'regions': list(REGIONS), 'seeds': list(SEEDS),
                'environment': {'python': platform.python_version(), 'executable': sys.executable,
                    **{name: version(name) for name in ('numpy', 'torch', 'xgboost', 'lightgbm', 'scikit-learn')}},
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'scope': 'aligned_full_data_fixed_configuration_not_hpo_not_blind'}
    if parent is not None:
        manifest['extension_from'] = {
            'run_root': str(parent_root), 'run_id': parent['run_id'],
            'jobs_hash': sha(parent_root / 'jobs.json'), 'source_hash': parent['source_hash'],
            'config_hash': parent['config_hash'], 'cache_hashes': parent['cache_hashes'],
            'cache_reuse': 'read_only_parent_cache_via_symlink',
        }
    atomic_json(run_root / 'jobs.json', manifest)
    print(f'Prepared {len(jobs)} jobs in {run_root}', flush=True)
    return manifest


def verify_snapshot(root, manifest):
    if sha(root / 'config.yaml') != manifest['config_hash']:
        raise ValueError('Frozen configuration changed')
    for name, digest in manifest['source_files'].items():
        if sha(root / 'source' / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    digest = hashlib.sha256(json.dumps(manifest['source_files'], sort_keys=True).encode()).hexdigest()
    if digest != manifest['source_hash']:
        raise ValueError('Source manifest fingerprint mismatch')
    if 'extension_from' in manifest:
        from .aligned_extension import verify_extension
        verify_extension(root, manifest)


def completed(root, job, manifest):
    directory = root / job['output']
    path = directory / 'status.json'
    if not path.exists():
        return False
    status = read_json(path)
    if status.get('state') != 'completed':
        return False
    if status.get('source_hash') != manifest['source_hash'] or status.get('config_hash') != manifest['config_hash']:
        raise ValueError('Completed job identity mismatch')
    if status.get('job_id') != job['id']:
        raise ValueError('Completed job id mismatch')
    hashes = status.get('artifact_hashes', {})
    if not {'test.npz', 'metrics.json'} <= hashes.keys():
        raise ValueError('Completed job lacks artifact hashes')
    for name in hashes:
        if not (directory / name).is_file() or sha(directory / name) != status.get('artifact_hashes', {}).get(name):
            raise ValueError(f'Completed artifact corrupt: {directory / name}')
    artifact = load_prediction_artifact(directory / 'test.npz')
    for key in ('model', 'region', 'seed'):
        if artifact.metadata.get(key) != job[key]:
            raise ValueError(f'Completed artifact identity mismatch: {key}')
    return True


def archive_incomplete(directory):
    old = [p for p in directory.iterdir() if p.name not in ('worker.lock', 'worker.log', 'attempts')]
    if old:
        archive = directory / 'attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.rename(archive / path.name)


def held_worker(directory):
    lock_path = directory / 'worker.lock'
    if not lock_path.exists():
        return False
    with lock_path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def worker(root, job_id, device):
    from .aligned_baselines import fit_predict
    import torch

    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    directory = root / job['output']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if completed(root, job, manifest):
        return
    archive_incomplete(directory)
    status = {'state': 'running', 'job_id': job_id, 'pid': os.getpid(), 'started_at': timestamp(),
              'source_hash': manifest['source_hash'], 'config_hash': manifest['config_hash'],
              'device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    atomic_json(directory / 'status.json', status)
    start = time.monotonic()
    try:
        config = yaml.safe_load((root / 'config.yaml').read_text())
        config['project_root'] = manifest['project_root']
        torch.set_num_threads(config['experiment']['cpu_threads_per_job'])
        splits = load_cache(root / 'data_cache' / job['region'], manifest['cache_hashes'][job['region']])
        train, selection, calibration, evaluation = (splits[k] for k in SPLITS)
        kind = job['model']
        if kind in NEURAL_MODELS:
            settings = config['experiment']['neural']
            quantiles, point, meta = fit_neural(
                kind, train, selection, evaluation, config, seed=job['seed'], device=device,
                epochs=config['training']['epochs'], batch_size=config['training']['batch_size'],
                learning_rate=settings['learning_rates'][kind], patience=config['training']['patience'],
                selection_metric='pinball', checkpoint_path=directory / 'best_model.pt',
                progress_path=directory / 'progress.jsonl')
        else:
            params = dict(config['experiment']['baseline_parameters'][kind])
            if kind == 'xgboost':
                params['device'] = device
            result = fit_predict(
                kind, train, selection, evaluation,
                calibration=calibration if kind == 'lasso_arx72_cp' else None,
                seed=job['seed'], params=params, progress_path=directory / 'progress.json',
                model_directory=directory / 'models')
            quantiles, point, meta = result.quantiles, result.point, result.metadata
        metadata = dict(evaluation.metadata, seed=job['seed'], model=kind,
                        source=f"{kind}:{manifest['source_hash']}", prediction_space='raw_rrp',
                        output_type='quantile', smoke=False, config_sha256=manifest['config_hash'],
                        run_id=manifest['run_id'], model_metadata=meta)
        artifact = make_prediction_artifact(
            actual=evaluation.actual, origin_unix=evaluation.origin_unix,
            target_time_unix=evaluation.target_time_unix, metadata=metadata,
            quantiles=quantiles, point=point, quantile_levels=QUANTILE_LEVELS)
        save_prediction_artifact(directory / 'test.npz', artifact)
        atomic_json(directory / 'metrics.json', score_predictions(artifact))
        verify_snapshot(root, manifest)
        status.update(state='completed', ended_at=timestamp(), seconds=time.monotonic() - start,
                      artifact_hashes={str(path.relative_to(directory)): sha(path)
                          for path in [directory / 'test.npz', directory / 'metrics.json',
                                       *([directory / 'best_model.pt'] if kind in NEURAL_MODELS else
                                         [p for p in (directory / 'models').rglob('*') if p.is_file()])]})
        atomic_json(directory / 'status.json', status)
    except BaseException as exc:
        status.update(state='failed', ended_at=timestamp(), seconds=time.monotonic() - start,
                      error=repr(exc), traceback=traceback.format_exc())
        atomic_json(directory / 'status.json', status)
        raise


def gpu_is_free(index, max_used_mib):
    try:
        out = subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        return int(out.strip()) < max_used_mib
    except (subprocess.SubprocessError, ValueError):
        return False


def run(root, retry_failed=False):
    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = yaml.safe_load((root / 'config.yaml').read_text())
    resources = config['experiment']['resources']
    slots = [('gpu', str(gpu)) for gpu in resources['gpu_ids']]
    slots += [('cpu', str(i)) for i in range(resources['cpu_workers'])]
    pending, done, failed, active = [], [], [], {}
    allowed_retry = set()
    for job in manifest['jobs']:
        if completed(root, job, manifest):
            done.append(job['id'])
            continue
        path = root / job['output'] / 'status.json'
        old = read_json(path) if path.exists() else {}
        if old.get('state') == 'failed' and not retry_failed:
            failed.append(job['id'])
        else:
            pending.append(job)
            if old.get('state') == 'failed':
                allowed_retry.add(job['id'])
    while pending or active:
        for slot, entry in list(active.items()):
            process, job, stream = entry
            if process.poll() is None:
                continue
            stream.close()
            if process.returncode == 0 and completed(root, job, manifest):
                done.append(job['id'])
            else:
                failed.append(job['id'])
                path = root / job['output'] / 'status.json'
                state = read_json(path) if path.exists() else {}
                if state.get('state') != 'failed':
                    state.update(state='failed', error=f'Worker exited {process.returncode}',
                                 source_hash=manifest['source_hash'], config_hash=manifest['config_hash'])
                    atomic_json(path, state)
            print(f'Finished {job["id"]}: exit={process.returncode}', flush=True)
            del active[slot]
        reserved, orphans = set(), []
        for job in list(pending):
            directory = root / job['output']
            if held_worker(directory):
                state = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
                physical_gpu = state.get('cuda_visible_devices', '')
                if job['resource'] == 'gpu' and physical_gpu:
                    reserved.add(('gpu', physical_gpu))
                elif job['resource'] == 'cpu':
                    free_cpu = [slot for slot in slots if slot[0] == 'cpu' and slot not in reserved]
                    reserved.update(free_cpu[:1])
                else:
                    reserved.update(slots)
                orphans.append({'id': job['id'], 'pid': state.get('pid'), 'adopted_existing_worker': True})
            elif completed(root, job, manifest):
                done.append(job['id'])
                pending.remove(job)
            elif (directory / 'status.json').exists() and read_json(directory / 'status.json').get('state') == 'failed' and job['id'] not in allowed_retry:
                failed.append(job['id'])
                pending.remove(job)
        orphan_ids = {j['id'] for j in orphans}
        for slot in slots:
            if slot in active or slot in reserved:
                continue
            kind, index = slot
            matches = [job for job in pending if job['resource'] == kind and job['id'] not in orphan_ids]
            if not matches:
                continue
            if kind == 'gpu' and not gpu_is_free(index, resources['gpu_max_initial_used_mib']):
                continue
            job = matches[0]
            directory = root / job['output']
            directory.mkdir(parents=True, exist_ok=True)
            # A controller restart must not duplicate an orphaned active worker.
            with (directory / 'worker.lock').open('a') as job_lock:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=index if kind == 'gpu' else '',
                       OMP_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       MKL_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       OPENBLAS_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       PYTHONUNBUFFERED='1')
            stream = (directory / 'worker.log').open('a')
            command = [sys.executable, '-m', 'forecasting.aligned_experiment', 'worker',
                       '--run-root', str(root), '--job-id', job['id'],
                       '--device', 'cuda:0' if kind == 'gpu' else 'cpu']
            process = subprocess.Popen(command, cwd=root / 'source', env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            active[slot] = process, job, stream
            pending.remove(job)
            print(f'Started {job["id"]} on {slot}, pid={process.pid}', flush=True)
        atomic_json(root / 'progress.json', {
            'run_id': manifest['run_id'], 'updated_at': timestamp(), 'controller_pid': os.getpid(),
            'total': len(manifest['jobs']), 'completed': done, 'failed': failed,
            'running': [{'id': job['id'], 'pid': process.pid, 'slot': list(slot)}
                        for slot, (process, job, _) in active.items()] + orphans,
            'pending': [job['id'] for job in pending if job['id'] not in orphan_ids],
        })
        if pending or active:
            time.sleep(10)
    from .aligned_report import build_aligned_report
    atomic_json(root / 'report.json', build_aligned_report(root))
    if 'extension_from' in manifest:
        from .aligned_extension import build_extension_report, write_extension_report
        write_extension_report(root / 'combined_report.json', build_extension_report(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--extend-from', type=Path,
                   help='Reuse a completed ten-model frozen run; prepare only TimeXer and PriceFM')
    p = sub.add_parser('run')
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--retry-failed', action='store_true')
    p = sub.add_parser('worker')
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--device', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config, args.run_root, args.extend_from)
    elif args.command == 'run':
        run(args.run_root, args.retry_failed)
    else:
        worker(args.run_root, args.job_id, args.device)


if __name__ == '__main__':
    main()
