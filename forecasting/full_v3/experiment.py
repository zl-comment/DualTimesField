"""Frozen, resumable 5 x 3 x 1 V3 experiment, separate from the baseline board.

prepare -> launch -> worker. Workers execute the frozen source, verify shared
cache and derived-feature hashes, and persist predictions in the common schema.
No test metrics choose parameters, weights, rounds, or variants automatically.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from ..aligned_experiment import (atomic_json, freeze_sources, read_json, sha,
                                  timestamp, verify_snapshot)
from ..common.cache import load_split
from ..aligned_inputs import AlignedSplit
from ..aligned_scoring import (QUANTILE_LEVELS, make_prediction_artifact,
                               save_prediction_artifact, score_predictions)
from .fields import decompose_curves
from .memory import HistoricalMemory, load_availability
from .model import fit_residual_model, predict_residual_model, residual_input_provenance
from .priors import VARIANTS, build_priors, variant_features

SPLITS = ('train', 'selection', 'calibration', 'test')


def validate_config(config):
    if tuple(config['variants']) != VARIANTS or config['regions'] != ['NSW1', 'QLD1', 'TAS1'] or config['seeds'] != [2026]:
        raise ValueError('Formal V3 requires the predeclared five contrasts x three regions x seed 2026')
    if config['lightgbm']['n_estimators'] != 2000 or config['lightgbm']['early_stopping_rounds'] != 50:
        raise ValueError('Formal V3 must retain the baseline training budget')


def validate_matrix(config, manifest):
    validate_config(config)
    expected = [dict(variant=v, region=r, seed=s) for v in VARIANTS
                for s in config['seeds'] for r in config['regions']]
    if manifest['jobs'] != expected:
        raise ValueError('Manifest jobs differ from the exact ordered 15-cell matrix')


def completed(directory, job, manifest):
    path = directory / 'state.json'
    if not path.exists() or read_json(path).get('status') != 'completed':
        return False
    state = read_json(path)
    for key, value in {**job, 'source_hash': manifest['source_hash'], 'config_hash': manifest['config_hash']}.items():
        if state.get(key) != value:
            raise ValueError(f'Completed job identity changed: {key}')
    hashes = read_json(directory / 'output_hashes.json')
    if set(hashes) != {'metrics.json', *(f'{s}_predictions.npz' for s in SPLITS[1:])}:
        raise ValueError('Incomplete completed output manifest')
    for name, digest in hashes.items():
        if sha(directory / name) != digest:
            raise ValueError(f'Completed output changed: {name}')
    from .model import _load_manifest
    _, checkpoint = _load_manifest(directory / 'models')
    if (checkpoint['model_metadata']['seed'] != job['seed']
            or checkpoint['train_metadata']['region'] != job['region']
            or sha(directory / 'models' / 'manifest.json') != state.get('model_manifest_sha256')):
        raise ValueError('Completed model identity or hash changed')
    return True


def _save(folder, name, array, hashes):
    path = folder / f'{name}.npy'
    np.save(path, array, allow_pickle=False)
    hashes[path.name] = sha(path)


def prepare(config_path, root):
    project, root = Path.cwd().resolve(), Path(root).resolve()
    config = yaml.safe_load(Path(config_path).read_text())
    validate_config(config)
    if root.exists():
        raise FileExistsError('Choose a fresh experiment directory')
    baseline = (project / config['baseline_root']).resolve()
    base_manifest = read_json(baseline / 'jobs.json')
    verify_snapshot(baseline, base_manifest)
    base_config = yaml.safe_load((project / config['baseline_config']).read_text())
    if base_config['data'] != yaml.safe_load((baseline / 'config.yaml').read_text())['data']:
        raise ValueError('Availability source configuration differs from frozen baseline')
    root.mkdir(parents=True)
    shutil.copyfile(config_path, root / 'config.yaml')
    files, source_hash = freeze_sources(project, root / 'source')
    # Copy the plan as immutable experimental provenance, without executing it.
    plan = project / 'docs/full_v3_design_20261013.md'
    if plan.exists():
        shutil.copyfile(plan, root / 'design.md')
    manifest = dict(experiment_id=config['experiment_id'], created_at=timestamp(),
                    project_root=str(project), baseline_root=str(baseline),
                    baseline_manifest_hash=sha(baseline / 'jobs.json'),
                    baseline_config_hash=sha(project / config['baseline_config']),
                    config_hash=sha(root / 'config.yaml'), source_files=files, source_hash=source_hash,
                    cache_hashes=base_manifest['cache_hashes'], feature_hashes={}, availability_hashes={},
                    jobs=[dict(variant=v, region=r, seed=s) for v in VARIANTS
                          for s in config['seeds'] for r in config['regions']])
    atomic_json(root / 'preparation.json', dict(status='preparing', started_at=timestamp()))
    for region in config['regions']:
        print(f'Preparing {region}: verify canonical data and as-of memory', flush=True)
        splits = {name: load_split(baseline, base_manifest, region, name) for name in SPLITS}
        raw_path = (project / base_config['data']['region_files'][region]).resolve()
        manifest['availability_hashes'][str(raw_path)] = sha(raw_path)
        if manifest['availability_hashes'][str(raw_path)] != base_manifest['data_hashes'][str(raw_path)]:
            raise ValueError('Raw availability/price CSV differs from frozen baseline source')
        availability = load_availability(raw_path, region)
        memory = HistoricalMemory(splits, availability, config=config['memory'])
        fields = decompose_curves(memory.curves, config=config['fields'])
        directory = root / 'features' / region
        directory.mkdir(parents=True)
        library_hashes = {}
        for name, array in dict(origins=memory.origins, curves=memory.curves,
                                levels=memory.levels, background=fields['background'],
                                event=fields['event'], residual=fields['residual'],
                                complete=memory.complete, max_availability=memory.max_availability).items():
            _save(directory, name, array, library_hashes)
        atomic_json(directory / 'library_hashes.json', library_hashes)
        manifest['feature_hashes'][f'{region}/library_hashes.json'] = sha(directory / 'library_hashes.json')
        for name, split in splits.items():
            print(f'  {region}/{name}: {len(split)} origins', flush=True)
            features = build_priors(memory, split, fields, config=config['retrieval'])
            folder = directory / name
            folder.mkdir()
            hashes = {}
            for key in ('raw_features', 'field_features'):
                _save(folder, key, features[key], hashes)
            for variant, prior in features['priors'].items():
                _save(folder, variant, prior, hashes)
            for key, values in features['audit'].items():
                _save(folder, f'audit_{key}', values, hashes)
            valid = features['audit']['valid']
            atomic_json(folder / 'audit.json', dict(n_origins=len(split),
                        base_metadata=split.metadata, candidates_per_origin_mean=float(valid.sum(1).mean()),
                        no_candidate_origins=int((~valid.any(1)).sum()),
                        future_publication_violations=0,
                        library_is_dynamic_asof=True, query_labels_used=False,
                        raw_feature_count=features['raw_features'].shape[1], field_feature_count=96,
                        library_reconstruction_mae_all_periods=float(np.abs(fields['residual']).mean()),
                        library_reconstruction_diagnostic_used_for_selection=False))
            hashes['audit.json'] = sha(folder / 'audit.json')
            atomic_json(folder / 'hashes.json', hashes)
            manifest['feature_hashes'][f'{region}/{name}/hashes.json'] = sha(folder / 'hashes.json')
            del features
        del memory, fields, splits
    if plan.exists():
        manifest['design_hash'] = sha(root / 'design.md')
    atomic_json(root / 'jobs.json', manifest)
    atomic_json(root / 'preparation.json', dict(status='completed', completed_at=timestamp(), jobs=15))
    print(f'Prepared 15 jobs: {root}', flush=True)
    return manifest


def _features(root, manifest, region, split, variant):
    directory = root / 'features' / region / split
    if sha(directory / 'hashes.json') != manifest['feature_hashes'][f'{region}/{split}/hashes.json']:
        raise ValueError('Derived feature manifest changed')
    # Audit arrays and all priors are also integrity protected, not only used columns.
    for name, digest in read_json(directory / 'hashes.json').items():
        if sha(directory / name) != digest:
            raise ValueError(f'Derived feature changed: {directory / name}')
    def load(name):
        return np.load(directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
    return variant_features(variant, load('raw_features'), load('field_features')), load(variant)


def _job_dir(root, job):
    return root / 'runs' / job['variant'] / f"seed{job['seed']}" / job['region']


def worker(root, variant, region, seed):
    root = Path(root).resolve()
    if Path(__file__).resolve() != root / 'source' / 'forecasting' / 'full_v3' / 'experiment.py':
        raise ValueError('Workers must execute frozen source; use the launch command')
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    config = yaml.safe_load((root / 'config.yaml').read_text())
    validate_matrix(config, manifest)
    job = dict(variant=variant, region=region, seed=seed)
    if job not in manifest['jobs']:
        raise ValueError('Job is not part of the frozen matrix')
    directory = _job_dir(root, job)
    directory.mkdir(parents=True, exist_ok=True)
    if completed(directory, job, manifest):
        return
    started = time.monotonic()
    state = dict(**job, status='running', pid=os.getpid(), started_at=timestamp(),
                 source_hash=manifest['source_hash'], config_hash=manifest['config_hash'])
    atomic_json(directory / 'state.json', state)
    try:
        baseline = Path(manifest['baseline_root'])
        if sha(baseline / 'jobs.json') != manifest['baseline_manifest_hash']:
            raise ValueError('Frozen baseline manifest changed')
        base_manifest = read_json(baseline / 'jobs.json')
        train, selection = [load_split(baseline, base_manifest, region, n) for n in SPLITS[:2]]
        xtr, ptr = _features(root, manifest, region, 'train', variant)
        xsel, psel = _features(root, manifest, region, 'selection', variant)
        model_dir = directory / 'models'
        if not (model_dir / 'manifest.sha256').exists():
            if model_dir.exists() and any(model_dir.iterdir()):
                # Preserve interrupted partial checkpoints; retrain this cell from scratch.
                model_dir.rename(directory / f'interrupted_models_{time.time_ns()}')
            fit_residual_model(train, selection, xtr, xsel, ptr, psel, model_directory=model_dir,
                               progress_path=directory / 'progress.json', seed=seed, parameters=config['lightgbm'])
        else:
            from .model import _load_manifest
            _, checkpoint = _load_manifest(model_dir)
            info = checkpoint['model_metadata']
            if info['seed'] != seed:
                raise ValueError('Resume checkpoint seed differs')
            for key, value in config['lightgbm'].items():
                if info['requested_params'].get(key) != value:
                    raise ValueError(f'Resume checkpoint parameter differs: {key}')
            for name, split, extra, prior in [('train', train, xtr, ptr), ('selection', selection, xsel, psel)]:
                if info[f'{name}_input_provenance'] != residual_input_provenance(split, extra, prior):
                    raise ValueError('Resume checkpoint effective inputs differ')
        del train, selection, xtr, ptr, xsel, psel
        metrics = {}
        for split_name in SPLITS[1:]:
            split = load_split(baseline, base_manifest, region, split_name)
            extra, prior = _features(root, manifest, region, split_name, variant)
            q = predict_residual_model(model_dir, split, extra, prior, threads=config['lightgbm']['n_jobs'])
            provenance = residual_input_provenance(split, extra, prior)
            meta = dict(split.metadata, seed=seed, source=variant, model=variant, output_type='quantile',
                        prediction_space='raw_rrp', base_input_hash=split.metadata['input_hash'],
                        input_provenance=provenance, source_hash=manifest['source_hash'],
                        config_hash=manifest['config_hash'], historical_memory='growing_asof_static_tree_weights')
            meta['input_hash'] = provenance['effective_input_hash']
            artifact = make_prediction_artifact(actual=split.actual, origin_unix=split.origin_unix,
                        target_time_unix=split.target_time_unix, metadata=meta, quantiles=q,
                        quantile_levels=QUANTILE_LEVELS)
            save_prediction_artifact(directory / f'{split_name}_predictions.npz', artifact)
            metrics[split_name] = score_predictions(artifact)
            del split, extra, prior, q, artifact
        atomic_json(directory / 'metrics.json', metrics)
        checksums = {p.name: sha(p) for p in directory.glob('*_predictions.npz')}
        checksums['metrics.json'] = sha(directory / 'metrics.json')
        atomic_json(directory / 'output_hashes.json', checksums)
        state.update(status='completed', completed_at=timestamp(), seconds=time.monotonic() - started,
                     model_manifest_sha256=sha(model_dir / 'manifest.json'))
        atomic_json(directory / 'state.json', state)
    except Exception:
        state.update(status='failed', error=traceback.format_exc(), failed_at=timestamp())
        atomic_json(directory / 'state.json', state)
        raise


def launch(root):
    import fcntl
    root = Path(root).resolve()
    with (root / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(root / 'jobs.json')
        verify_snapshot(root, manifest)
        config = yaml.safe_load((root / 'config.yaml').read_text())
        validate_matrix(config, manifest)
        env = dict(os.environ, PYTHONPATH=str(root / 'source'),
                   OMP_NUM_THREADS=str(config['lightgbm']['n_jobs']),
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        atomic_json(root / 'launcher.json', dict(status='running', pid=os.getpid(), started_at=timestamp()))
        def run(job):
            directory = _job_dir(root, job)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'worker.log').open('a') as log:
                result = subprocess.run([sys.executable, '-m', 'forecasting.full_v3.experiment',
                    'worker', '--run-root', str(root), '--variant', job['variant'],
                    '--region', job['region'], '--seed', str(job['seed'])],
                    cwd=root / 'source', env=env, stdout=log, stderr=subprocess.STDOUT)
            return dict(**job, returncode=result.returncode)
        from .report import write_report
        write_report(root)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['cpu_workers']) as pool:
            futures = [pool.submit(run, j) for j in manifest['jobs']]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
                write_report(root)
        atomic_json(root / 'launcher.json', dict(status='completed' if all(r['returncode'] == 0 for r in results)
                    else 'failed', completed_at=timestamp(), jobs=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'worker', 'report'])
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--config', default='configs/full_v3/experiment.yaml')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--region', choices=['NSW1', 'QLD1', 'TAS1'])
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.config, args.run_root)
    elif args.action == 'launch':
        launch(args.run_root)
    elif args.action == 'worker':
        worker(args.run_root, args.variant, args.region, args.seed)
    else:
        from .report import write_report
        write_report(Path(args.run_root))


if __name__ == '__main__':
    main()
