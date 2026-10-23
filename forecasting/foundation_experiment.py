"""Frozen, resumable matrix runner for the 2026-10-23 foundation batch.

Reuses the audited parent 90-cell run read-only through the same expansion
verification chain (pinned jobs/config/source/cache hashes). Differences from
the first-batch expansion runner are structural, not cosmetic:

- The prediction adapter never receives test labels. It gets a label-free
  ``PredictionView``; the worker loads actual values itself, after prediction,
  to build and score the artifact.
- Stage and information scope come from the foundation registry declaration
  (train-label context fit, selection/calibration use, weight training,
  input/seed/pretraining scope) and are written into every artifact.
- Long predictions are chunked by origin and persisted; restarts resume from
  verified chunks without refitting or dropping rows.
- Deterministic frozen models may reuse a first-seed cell for later seeds only
  after a bitwise verification subset reproduces it; the copy records
  ``reuse_of`` and the verification evidence.
- Every cell records load/fit/predict wall time, peak memory, hardware,
  throughput per thousand origins and adapter-declared weight identity.

The first-batch registry, schema, matrix validation and frozen runs are not
modified; ``validate_foundation_config`` refuses to touch their semantics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from .aligned_experiment import (
    REGIONS, SEEDS, atomic_json, completed, freeze_sources,
    gpu_is_free, held_worker, read_json, sha, verify_snapshot,
    _validate_common_protocol,
)
from .aligned_extension import assert_shared_protocol, audit_parent
from .aligned_inputs import INPUT_KEYS, AlignedSplit
from .aligned_report import build_aligned_report
from .aligned_scoring import QUANTILE_LEVELS, make_prediction_artifact, save_prediction_artifact, score_predictions
from .common.cache import load_split
from .common.foundation_api import PredictionView
from .common.foundation_registry import (
    FOUNDATION_BATCH, FOUNDATION_REGISTRY, FOUNDATION_SCHEMA, FoundationSpec,
    foundation_model_ids,
)

SPLITS = ('train', 'selection', 'calibration', 'test')

# Crash-safe resume: chunk ledgers and fit state must survive worker restarts;
# only transient outputs of an unfinished attempt are archived away.
_RESUME_PRESERVED = {'worker.lock', 'worker.log', 'attempts', 'predictions', 'models'}


def _archive_transient(directory: Path) -> None:
    old = [p for p in directory.iterdir() if p.name not in _RESUME_PRESERVED]
    if old:
        archive = directory / 'attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.rename(archive / path.name)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class FoundationContext:
    """Everything a foundation adapter may see; the test side is label-free."""

    run_root: Path
    config: dict
    manifest: dict
    job: dict
    region: str
    seed: int
    device: str
    test_view: PredictionView
    parent_run: Path
    progress_path: Path
    model_directory: Path
    prediction_directory: Path
    train: AlignedSplit | None = None
    selection: AlignedSplit | None = None
    calibration: AlignedSplit | None = None


def _foundation_section(config: dict) -> dict:
    section = config.get('foundation')
    if not isinstance(section, dict) or section.get('schema') != FOUNDATION_SCHEMA:
        raise ValueError(f"Foundation configs require foundation.schema == {FOUNDATION_SCHEMA}")
    if section.get('batch') != FOUNDATION_BATCH:
        raise ValueError(f"Only the declared {FOUNDATION_BATCH} foundation batch is runnable")
    return section


def validate_foundation_config(config: dict) -> list[str]:
    """Validate a foundation config; return the declared model ids."""
    _foundation_section(config)
    _validate_common_protocol(config)
    models = list(config['experiment']['models'])
    unknown = [model for model in models if model not in FOUNDATION_REGISTRY]
    if unknown:
        raise ValueError(f"Models outside the foundation registry: {unknown}")
    if len(set(models)) != len(models):
        raise ValueError("Foundation models must be unique")
    scope = config['experiment'].get('execution_scope')
    if scope == 'single_model':
        if len(models) != 1:
            raise ValueError("Single-model scope declares exactly one foundation model")
    elif scope == 'matrix':
        if set(models) != set(foundation_model_ids()):
            raise ValueError("Foundation matrix scope must declare every registered model exactly once")
    else:
        raise ValueError("Foundation configs require execution_scope matrix or single_model")
    interpreters = config['experiment'].get('interpreters') or {}
    missing = [model for model in models if not isinstance(interpreters.get(model), str)
               or not interpreters[model].strip()]
    if missing:
        raise ValueError(f"Every foundation model needs an interpreter path: {missing}")
    for model in models:
        section = config.get('foundation_models', {}).get(model)
        if not isinstance(section, dict):
            raise ValueError(f"Missing foundation_models.{model} configuration")
    if config.get('aligned_protocol', {}).get('id') is None:
        raise ValueError("Foundation must inherit the parent protocol id")
    reuse = config['experiment'].get('seed_reuse', {'mode': 'off'})
    if reuse.get('mode') not in ('off', 'verify_and_copy'):
        raise ValueError("seed_reuse mode must be off or verify_and_copy")
    if reuse.get('mode') == 'verify_and_copy':
        count = reuse.get('verification_origins')
        if (isinstance(count, bool) or not isinstance(count, int) or count < 8):
            raise ValueError("seed_reuse.verification_origins must be an integer >= 8")
    return models


def verify_foundation(root: Path, manifest: dict) -> Path:
    """Pin the parent identity, shared protocol and cache before any work."""
    provenance = manifest['foundation_from']
    parent_root = Path(provenance['run_root']).resolve()
    if parent_root == root:
        raise ValueError('Foundation run cannot refer to itself')
    if sha(parent_root / 'jobs.json') != provenance['jobs_hash']:
        raise ValueError('Parent jobs manifest changed')
    parent = read_json(parent_root / 'jobs.json')
    verify_snapshot(parent_root, parent)
    if any(provenance[key] != parent[key] for key in ('run_id', 'source_hash', 'config_hash', 'cache_hashes')):
        raise ValueError('Parent provenance mismatch')
    child_config = yaml.safe_load((root / 'config.yaml').read_text())
    parent_config = yaml.safe_load((parent_root / 'config.yaml').read_text())
    validate_foundation_config(child_config)
    assert_shared_protocol(parent_config, child_config)
    for key in ('cache_hashes', 'cache_info', 'data_hashes'):
        if manifest.get(key) != parent.get(key):
            raise ValueError(f'Foundation changes frozen data provenance: {key}')
    for region in REGIONS:
        if sha(root / 'data_cache' / region / 'hashes.json') != parent['cache_hashes'][region]:
            raise ValueError(f'Foundation cache manifest changed: {region}')
    return parent_root


def _model_env_locks(project: Path, models: list[str]) -> dict:
    locks = {}
    for model in models:
        path = project / 'forecasting' / model / 'requirements.lock.txt'
        if not path.is_file():
            raise ValueError(f"Missing environment lock for {model}: {path}")
        locks[model] = {'lock_file': str(path.relative_to(project)), 'sha256': sha(path)}
    return locks


def prepare(config_path, run_root, extend_from):
    project = Path.cwd().resolve()
    config_path, run_root = Path(config_path).resolve(), Path(run_root).resolve()
    if extend_from is None:
        raise ValueError('Foundation runs require --extend-from pointing at the 90-cell parent run')
    config = yaml.safe_load(config_path.read_text())
    models = validate_foundation_config(config)
    if run_root.exists():
        raise FileExistsError('Use a new dated run directory')
    parent_root = Path(extend_from).resolve()
    parent = audit_parent(parent_root)
    assert_shared_protocol(yaml.safe_load((parent_root / 'config.yaml').read_text()), config)
    interpreters = {}
    for model in models:
        relative = config['experiment']['interpreters'][model]
        candidate = (project / relative).resolve()
        if not candidate.is_file():
            raise ValueError(f"Interpreter for {model} does not exist: {candidate}")
        interpreters[model] = str(candidate)
    env_locks = _model_env_locks(project, models)
    run_root.mkdir(parents=True)
    shutil.copyfile(config_path, run_root / 'config.yaml')
    source_files, source_hash = freeze_sources(project, run_root / 'source')
    config['project_root'] = str(project)
    (run_root / 'data_cache').symlink_to(parent_root / 'data_cache', target_is_directory=True)
    jobs = []
    for seed in SEEDS:
        for region in REGIONS:
            for model in models:
                jobs.append({'id': f'{model}/{seed}/{region}', 'model': model, 'region': region,
                             'seed': seed, 'output': f'runs/{model}/seed{seed}/{region}',
                             'resource': FOUNDATION_REGISTRY[model].resource})
    manifest = {'run_id': run_root.name, 'created_at_utc': _now(),
                'project_root': str(project), 'source_hash': source_hash, 'source_files': source_files,
                'config_hash': sha(run_root / 'config.yaml'), 'data_hashes': parent['data_hashes'],
                'cache_hashes': parent['cache_hashes'], 'cache_info': parent['cache_info'], 'jobs': jobs,
                'models': models, 'regions': list(REGIONS), 'seeds': list(SEEDS),
                'scope': 'registered_foundation_fixed_configuration_not_hpo_not_blind',
                'foundation': {'schema': FOUNDATION_SCHEMA, 'batch': FOUNDATION_BATCH,
                               'declared_scopes': {m: FOUNDATION_REGISTRY[m].scope_declaration() for m in models}},
                'interpreters': interpreters, 'model_env_locks': env_locks,
                'environment': dict(parent.get('environment', {}), executable=sys.executable),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'foundation_from': {
                    'run_root': str(parent_root), 'run_id': parent['run_id'],
                    'jobs_hash': sha(parent_root / 'jobs.json'), 'source_hash': parent['source_hash'],
                    'config_hash': parent['config_hash'], 'cache_hashes': parent['cache_hashes'],
                    'cache_reuse': 'read_only_parent_cache_via_symlink',
                }}
    atomic_json(run_root / 'jobs.json', manifest)
    print(f'Prepared {len(jobs)} foundation jobs in {run_root}', flush=True)
    return manifest


def _load_test_view(root: Path, manifest: dict, region: str):
    """Load the test split; return (label-free view, actuals held by the runner)."""
    split = load_split(root, manifest, region, 'test')
    view = PredictionView(
        {key: np.asarray(split.inputs[key]) for key in INPUT_KEYS},
        np.asarray(split.origin_unix), np.asarray(split.target_time_unix),
        dict(split.metadata))
    return view, np.asarray(split.actual), split


def _chunk_meta(directory: Path, index: int) -> dict | None:
    path = directory / 'chunks' / f'chunk_{index:05d}.npz'
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(archive['meta_json'].item())
    except Exception:
        return None


def _save_chunk(directory: Path, index: int, meta: dict, payload: dict) -> None:
    chunks = directory / 'chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    arrays = {'meta_json': np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False))}
    arrays.update({key: value for key, value in payload.items() if value is not None})
    path = chunks / f'chunk_{index:05d}.npz'
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    # Append-only SHA ledger: a tampered-but-wellformed chunk file can never
    # pass resume verification just by keeping its metadata intact.
    ledger_path = chunks / 'chunks_manifest.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger[f'chunk_{index:05d}.npz'] = {'sha256': sha(path), 'written_at': _now()}
    atomic_json(ledger_path, ledger)


def _verify_chunk(path: Path, identity_hash: str, indices: np.ndarray) -> dict | None:
    """Return stored payload if the chunk matches identity, ledger and rows."""
    try:
        ledger_path = path.parent / 'chunks_manifest.json'
        ledger = json.loads(ledger_path.read_text())
        recorded = ledger.get(path.name, {}).get('sha256')
        if recorded is None or sha(path) != recorded:
            return None
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive['meta_json'].item())
            if meta.get('identity_hash') != identity_hash:
                return None
            if not np.array_equal(archive['indices'], indices):
                return None
            payload = {'indices': archive['indices']}
            for key in ('quantiles', 'point'):
                if key in archive:
                    payload[key] = archive[key]
            expected = {'quantiles', 'point'} & set(meta.get('outputs', []))
            if expected != set(k for k in ('quantiles', 'point') if k in payload):
                return None
            if not all(np.isfinite(payload[key]).all() for key in expected):
                return None
            return payload
    except Exception:
        return None


def _peak_gpu_memory(device: str) -> dict:
    try:
        import torch

        if device.startswith('cuda') and torch.cuda.is_available():
            return {'peak_gpu_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    'peak_gpu_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
                    'gpu_name': torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {}


def _hardware_note() -> dict:
    note = {'cpu_model': '', 'gpu_model': ''}
    try:
        note['cpu_model'] = subprocess.check_output(
            ['bash', '-c', "lscpu | grep 'Model name' | head -1 | sed 's/.*: *//'"],
            text=True).strip()
    except subprocess.SubprocessError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            note['gpu_model'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return note


def worker(root, job_id, device, *, allow_download=False):
    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    parent_root = verify_foundation(root, manifest)
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    spec: FoundationSpec = FOUNDATION_REGISTRY[job['model']]
    directory = root / job['output']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if completed(root, job, manifest):
        return
    _archive_transient(directory)
    status = {'state': 'running', 'job_id': job_id, 'pid': os.getpid(), 'started_at': _now(),
              'source_hash': manifest['source_hash'], 'config_hash': manifest['config_hash'],
              'device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    atomic_json(directory / 'status.json', status)
    started = time.monotonic()
    lock_sha = manifest['model_env_locks'][job['model']]['sha256']
    if sha(Path(manifest['project_root']) / manifest['model_env_locks'][job['model']]['lock_file']) != lock_sha:
        raise ValueError('Model environment lock changed after prepare')
    try:
        from .common.foundation_registry import load_foundation_adapter

        config = yaml.safe_load((root / 'config.yaml').read_text())
        config['project_root'] = manifest['project_root']
        import torch

        torch.set_num_threads(config['experiment']['cpu_threads_per_job'])
        load_started = time.monotonic()
        view, actual, test_split = _load_test_view(root, manifest, job['region'])
        stage_splits: dict[str, AlignedSplit] = {}
        if spec.fit_requires_train_labels:
            stage_splits['train'] = load_split(root, manifest, job['region'], 'train')
        if spec.selection_use != 'none':
            stage_splits['selection'] = load_split(root, manifest, job['region'], 'selection')
        if spec.calibration_use:
            stage_splits['calibration'] = load_split(root, manifest, job['region'], 'calibration')
        context = FoundationContext(
            run_root=root, config=config, manifest=manifest, job=job, region=job['region'],
            seed=job['seed'], device=device, test_view=view, parent_run=parent_root,
            progress_path=directory / 'progress.json', model_directory=directory / 'models',
            prediction_directory=directory / 'predictions', **stage_splits)
        adapter_cls = load_foundation_adapter(job['model'])
        adapter = adapter_cls(context)
        output_type = 'quantile' if spec.output_table == 'Q' else 'point_only'
        if getattr(adapter, 'output_type', output_type) != output_type:
            raise ValueError(f"{job['model']} adapter output_type disagrees with the registry")
        identity = adapter.identity()
        identity_hash = _identity_hash(identity)
        fit_started = time.monotonic()
        adapter.fit()
        fit_seconds = time.monotonic() - fit_started

        n = len(view)
        chunk_sizes = config['experiment'].get('chunk_size', {})
        chunk_size = int(chunk_sizes.get(job['model'], 512)
                         if isinstance(chunk_sizes, dict) else chunk_sizes)
        boundaries = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
        chunks_dir = directory / 'predictions' / 'chunks'
        chunks_dir.mkdir(parents=True, exist_ok=True)
        progress_events: list[dict] = []

        def progress(event: dict) -> None:
            progress_events.append(event)
            context.progress_path.parent.mkdir(parents=True, exist_ok=True)
            context.progress_path.write_text(
                '\n'.join(json.dumps(e, allow_nan=False) for e in progress_events) + '\n')

        progress({'event': 'worker_start', 'job': job_id, 'origins': n, 'chunks': len(boundaries),
                  'chunk_size': chunk_size, 'identity_hash': identity_hash, 'seed': job['seed']})

        # Deterministic-seed reuse: verify a fixed subset bitwise against the
        # first-seed cell, then copy its chunks instead of re-inference.
        reuse = config['experiment'].get('seed_reuse', {'mode': 'off'})
        reuse_evidence = None
        first_seed = min(SEEDS)
        reference_dir = root / 'runs' / job['model'] / f'seed{first_seed}' / job['region']
        copied = False
        if (reuse.get('mode') == 'verify_and_copy' and spec.deterministic_inference
                and job['seed'] != first_seed and completed(root, {**job, 'seed': first_seed,
                'id': f"{job['model']}/{first_seed}/{job['region']}",
                'output': f"runs/{job['model']}/seed{first_seed}/{job['region']}"}, manifest)):
            verification = int(reuse['verification_origins'])
            check = np.unique(np.linspace(0, n - 1, verification).astype(np.int64))
            reference = _load_completed_predictions(reference_dir, check)
            subset = None
            if reference is not None and reference.get('identity_hash') == identity_hash:
                check_started = time.monotonic()
                subset = _predict_indices(adapter, view, check, f'determinism check ({len(check)} origins)')
                equal, worst = _reuse_compare(subset, reference, spec.reuse_tolerance)
                reuse_evidence = {'reference_seed': first_seed, 'verified_origins': len(check),
                                  'comparison': ('bitwise_equal' if worst == 0.0 else
                                                 'max_abs_diff_within_tolerance'),
                                  'max_abs_diff': worst, 'tolerance': spec.reuse_tolerance,
                                  'check_seconds': round(time.monotonic() - check_started, 3)}
            if reuse_evidence and (reuse_evidence['comparison'] == 'bitwise_equal'
                                   or reuse_evidence['max_abs_diff'] <= spec.reuse_tolerance):
                copied = True
                for index, (start, stop) in enumerate(boundaries):
                    source = reference_dir / 'predictions' / 'chunks' / f'chunk_{index:05d}.npz'
                    payload = _verify_chunk(source, identity_hash, np.arange(start, stop))
                    if payload is None:
                        copied = False
                        break
                    meta = {'job_id': job_id, 'seed': job['seed'], 'chunk': index,
                            'start': start, 'stop': stop, 'identity_hash': identity_hash,
                            'outputs': sorted(k for k in ('quantiles', 'point') if payload.get(k) is not None),
                            'reuse_of_seed': first_seed,
                            'input_slice_hash': view.input_slice_hash(np.arange(start, stop))}
                    _save_chunk(directory / 'predictions', index, meta, payload)
                if copied:
                    progress({'event': 'seed_reuse', **reuse_evidence, 'chunks_copied': len(boundaries)})
                else:
                    progress({'event': 'seed_reuse_aborted', 'reference_chunk_invalid': True})
                    reuse_evidence = None
            else:
                progress({'event': 'seed_reuse_rejected',
                          'reason': 'verification subset differs or reference identity mismatch'})
                reuse_evidence = None

        predict_seconds = 0.0
        if not copied:
            for index, (start, stop) in enumerate(boundaries):
                indices = np.arange(start, stop)
                path = chunks_dir / f'chunk_{index:05d}.npz'
                payload = _verify_chunk(path, identity_hash, indices)
                if payload is None:
                    if path.exists():
                        archive = directory / 'attempts' / str(time.time_ns())
                        archive.mkdir(parents=True, exist_ok=True)
                        path.rename(archive / path.name)
                    chunk_started = time.monotonic()
                    payload = _predict_indices(adapter, view, indices, f'chunk {index + 1}/{len(boundaries)}')
                    payload['indices'] = indices
                    predict_seconds += time.monotonic() - chunk_started
                    meta = {'job_id': job_id, 'seed': job['seed'], 'chunk': index,
                            'start': start, 'stop': stop, 'identity_hash': identity_hash,
                            'outputs': sorted(k for k in ('quantiles', 'point') if payload.get(k) is not None),
                            'input_slice_hash': view.input_slice_hash(indices)}
                    _save_chunk(directory / 'predictions', index, meta, payload)
                    progress({'event': 'chunk_done', 'chunk': index, 'start': start, 'stop': stop,
                              'seconds': time.monotonic() - chunk_started,
                              'origins_per_second': (stop - start) / max(time.monotonic() - chunk_started, 1e-9)})

        quantiles = point = None
        payloads = []
        for index, (start, stop) in enumerate(boundaries):
            payload = _verify_chunk(chunks_dir / f'chunk_{index:05d}.npz', identity_hash,
                                    np.arange(start, stop))
            if payload is None:
                raise ValueError(f'Chunk {index} missing or invalid after prediction pass')
            payloads.append(payload)
        if output_type == 'quantile':
            quantiles = np.concatenate([p['quantiles'] for p in payloads], axis=0)
            if quantiles.shape != (n, view.target_time_unix.shape[1], len(QUANTILE_LEVELS)):
                raise ValueError('Assembled quantiles have the wrong shape')
            point = None
        else:
            point = np.concatenate([p['point'] for p in payloads], axis=0)
            if point.shape != (n, view.target_time_unix.shape[1]):
                raise ValueError('Assembled point predictions have the wrong shape')
            quantiles = None
        if not np.array_equal(np.concatenate([p['indices'] for p in payloads]), np.arange(n)):
            raise ValueError('Chunk assembly does not cover every origin exactly once')
        if quantiles is not None and not np.isfinite(quantiles).all():
            raise FloatingPointError('Non-finite quantile predictions')
        if point is not None and not np.isfinite(point).all():
            raise FloatingPointError('Non-finite point predictions')

        load_seconds = fit_started - load_started
        total_seconds = time.monotonic() - started
        budget = {
            'wall_seconds': {'load_split': round(load_seconds, 3), 'context_fit': round(fit_seconds, 3),
                             'predict': round(predict_seconds, 3), 'total': round(total_seconds, 3)},
            'peak_cpu_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            **_peak_gpu_memory(device), **_hardware_note(),
            'n_origins': int(n), 'n_rows_actual': int(actual.size),
            'features_flattened': int(sum(np.prod(view.inputs[key].shape[1:]) for key in INPUT_KEYS)),
            'chunk_size': chunk_size, 'n_chunks': len(boundaries),
            'origins_per_second': round(n / max(predict_seconds, 1e-9), 4),
            'seconds_per_1000_origins': round(1000.0 * predict_seconds / max(n, 1), 3),
            'seed_reuse': reuse_evidence or 'independent_inference',
            'pretrained_cost_note': 'foundation model training cost is external to this run and '
                                    'not comparable to from-scratch local training budgets',
        }
        metadata = dict(view.metadata, seed=job['seed'], model=job['model'],
                        source=f"{job['model']}:{manifest['source_hash']}",
                        prediction_space='raw_rrp', output_type=output_type, smoke=False,
                        config_sha256=manifest['config_hash'], run_id=manifest['run_id'],
                        model_metadata={'adapter_identity': identity, 'budget': budget,
                                        'foundation': dict(manifest['foundation'],
                                                           declared_scope=spec.scope_declaration())})
        artifact = make_prediction_artifact(
            actual=actual, origin_unix=view.origin_unix, target_time_unix=view.target_time_unix,
            metadata=metadata, quantiles=quantiles, point=point,
            quantile_levels=QUANTILE_LEVELS if quantiles is not None else None)
        save_prediction_artifact(directory / 'test.npz', artifact)
        atomic_json(directory / 'metrics.json', score_predictions(artifact))
        verify_snapshot(root, manifest)
        chunk_files = sorted(chunks_dir.glob('chunk_*.npz'))
        status.update(state='completed', ended_at=_now(), seconds=total_seconds,
                      artifact_hashes={str(path.relative_to(directory)): sha(path)
                                       for path in [directory / 'test.npz', directory / 'metrics.json',
                                                    *[f for f in chunk_files],
                                                    *([directory / 'models' / 'fit_state.json']
                                                      if (directory / 'models' / 'fit_state.json').exists() else [])]
                                       if path.is_file()})
        atomic_json(directory / 'status.json', status)
    except BaseException as exc:
        status.update(state='failed', ended_at=_now(), seconds=time.monotonic() - started,
                      error=repr(exc), traceback=traceback.format_exc())
        atomic_json(directory / 'status.json', status)
        raise


def _predict_indices(adapter, view: PredictionView, indices: np.ndarray, note: str) -> dict:
    payload = adapter.predict_chunk(indices)
    expected = {'quantiles', 'point'}
    if not isinstance(payload, dict) or not expected & set(payload):
        raise ValueError(f"Adapter returned no predictions while running {note}")
    result = {}
    for key in ('quantiles', 'point'):
        if payload.get(key) is None:
            continue
        array = np.asarray(payload[key], dtype=np.float64)
        if not np.isfinite(array).all():
            raise FloatingPointError(f"Non-finite {key} from adapter ({note})")
        result[key] = array
    return result


def _load_completed_predictions(reference_dir: Path, indices: np.ndarray) -> dict | None:
    """Load a completed cell's stored quantiles/point at given origin indices."""
    try:
        artifact = __import__('forecasting.aligned_scoring', fromlist=['load_prediction_artifact']) \
            .load_prediction_artifact(reference_dir / 'test.npz')
        result = {'identity_hash': None}
        # Identity comes from the cell's chunk metadata; recover from any chunk.
        chunks = sorted((reference_dir / 'predictions' / 'chunks').glob('chunk_*.npz'))
        if not chunks:
            return None
        with np.load(chunks[0], allow_pickle=False) as archive:
            result['identity_hash'] = json.loads(archive['meta_json'].item()).get('identity_hash')
        if artifact.quantiles is not None:
            result['quantiles'] = artifact.quantiles[indices]
        if artifact.native_point is not None:
            result['point'] = artifact.native_point[indices]
        return result
    except Exception:
        return None


def _reuse_compare(subset: dict, reference: dict, tolerance: float) -> tuple[bool, float]:
    """Compare a verification subset against the reference cell.

    tolerance=0.0 demands bitwise identity. A small nonzero tolerance documents
    information-free GPU kernel jitter (measured value always recorded); the
    adapter identity must still match exactly.
    """
    worst = 0.0
    for key in ('quantiles', 'point'):
        left, right = subset.get(key), reference.get(key)
        if (left is None) != (right is None):
            return False, float('inf')
        if left is None:
            continue
        diff = float(np.abs(np.asarray(left) - np.asarray(right)).max())
        worst = max(worst, diff)
    return bool(worst <= tolerance), worst


def run(root, retry_failed=False):
    """Controller loop; each job runs in its own model interpreter."""
    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = yaml.safe_load((root / 'config.yaml').read_text())
    resources = config['experiment']['resources']
    slots = [('gpu', str(gpu)) for gpu in resources['gpu_ids']]
    slots += [('cpu', str(i)) for i in range(resources.get('cpu_workers', 0))]
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
                else:
                    reserved.update(slots)
                orphans.append({'id': job['id'], 'pid': state.get('pid'), 'adopted_existing_worker': True})
            elif completed(root, job, manifest):
                done.append(job['id'])
                pending.remove(job)
            elif (directory / 'status.json').exists() and read_json(directory / 'status.json').get('state') == 'failed' and job['id'] not in allowed_retry:
                failed.append(job['id'])
                pending.remove(job)
        orphan_ids = {job['id'] for job in orphans}
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
            with (directory / 'worker.lock').open('a') as job_lock:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=index if kind == 'gpu' else '',
                       OMP_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       MKL_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       OPENBLAS_NUM_THREADS=str(config['experiment']['cpu_threads_per_job']),
                       PYTHONUNBUFFERED='1',
                       HF_HOME=str(Path(manifest['project_root']) / 'weights' / 'hf_home'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            stream = (directory / 'worker.log').open('a')
            command = [manifest['interpreters'][job['model']], '-m', 'forecasting.foundation_experiment',
                       'worker', '--run-root', str(root), '--job-id', job['id'],
                       '--device', 'cuda:0' if kind == 'gpu' else 'cpu']
            process = subprocess.Popen(command, cwd=root / 'source', env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            active[slot] = process, job, stream
            pending.remove(job)
            print(f'Started {job["id"]} on {slot}, pid={process.pid}', flush=True)
        atomic_json(root / 'progress.json', {
            'run_id': manifest['run_id'], 'updated_at': _now(), 'controller_pid': os.getpid(),
            'total': len(manifest['jobs']), 'completed': done, 'failed': failed,
            'running': [{'id': job['id'], 'pid': process.pid, 'slot': list(slot)}
                        for slot, (process, job, _) in active.items()] + orphans,
            'pending': [job['id'] for job in pending if job['id'] not in orphan_ids],
        })
        if pending or active:
            time.sleep(10)
    atomic_json(root / 'report.json', build_aligned_report(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--extend-from', type=Path, required=True,
                   help='Completed frozen ten-model parent run (read-only)')
    p = sub.add_parser('run')
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--retry-failed', action='store_true')
    p = sub.add_parser('worker')
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--allow-download', action='store_true',
                   help='Unset HF offline mode for first-time weight download (smoke only)')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config, args.run_root, args.extend_from)
    elif args.command == 'run':
        run(args.run_root, args.retry_failed)
    else:
        if args.allow_download:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
        worker(args.run_root, args.job_id, args.device, allow_download=args.allow_download)


if __name__ == '__main__':
    main()
