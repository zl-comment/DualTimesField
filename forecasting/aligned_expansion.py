"""Expansion matrix for the registered 2026-10-22 fair baselines.

Reuses the frozen 90-cell parent run read-only: its audited data cache via a
symlink, and — for the tree-calibration adapters — its frozen LightGBM/XGBoost
boosters under the reproduction gate. This module owns prepare/worker/run for
the expansion scope only; BASE_MODELS/EXTENSION_MODELS semantics, the parent
runs and their reports are never modified.

Config requirements: shared protocol sections identical to the parent
(``assert_shared_protocol``), models restricted to the registry, and a
top-level ``expansion.schema`` matching the registry schema version.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import yaml

from .aligned_experiment import (
    REGIONS, SEEDS, atomic_json, archive_incomplete, completed, freeze_sources,
    gpu_is_free, held_worker, read_json, sha, validate_config, verify_snapshot,
    _validate_common_protocol,
)
from .aligned_extension import assert_shared_protocol, audit_parent
from .aligned_inputs import AlignedSplit
from .aligned_report import build_aligned_report
from .aligned_scoring import QUANTILE_LEVELS, make_prediction_artifact, save_prediction_artifact, score_predictions
from .common.baseline_registry import EXPANSION_SCHEMA, REGISTRY, expansion_model_ids
from .common.cache import load_split
from .common.frozen_base import SUPPORTED_BASES

SPLITS = ('train', 'selection', 'calibration', 'test')


@dataclass
class ExpansionContext:
    """Everything a registered adapter may see; splits are stage-limited."""

    run_root: Path
    config: dict
    manifest: dict
    job: dict
    region: str
    seed: int
    device: str
    splits: dict[str, AlignedSplit]
    parent_run: Path
    progress_path: Path
    model_directory: Path


def _expansion_section(config: dict) -> dict:
    section = config.get('expansion')
    if not isinstance(section, dict) or section.get('schema') != EXPANSION_SCHEMA:
        raise ValueError(f"Expansion configs require expansion.schema == {EXPANSION_SCHEMA}")
    return section


def validate_expansion_config(config: dict) -> list[str]:
    """Validate an expansion config; return the declared model ids."""
    section = _expansion_section(config)
    _validate_common_protocol(config)
    models = list(config['experiment']['models'])
    unknown = [model for model in models if model not in REGISTRY]
    if unknown:
        raise ValueError(f"Models outside the expansion registry: {unknown}")
    if len(set(models)) != len(models):
        raise ValueError("Expansion models must be unique")
    scope = config['experiment'].get('execution_scope')
    if scope == 'single_model':
        validate_config(config)  # exercises the shared single-model whitelist
        if len(models) != 1:
            raise ValueError("Single-model scope declares exactly one expansion model")
    elif scope == 'matrix':
        if set(models) != set(expansion_model_ids()):
            raise ValueError("Expansion matrix scope must declare every registered model exactly once")
    else:
        raise ValueError("Expansion configs require execution_scope matrix or single_model")
    if section.get('batch') != 'first_20261022':
        raise ValueError("Only the declared first expansion batch is runnable")
    if config.get('aligned_protocol', {}).get('id') is None:
        raise ValueError("Expansion must inherit the parent protocol id")
    return models


def _required_splits(model_ids: list[str]) -> tuple[str, ...]:
    needs_calibration = any(REGISTRY[m].requires_calibration for m in model_ids)
    needs_training = any(REGISTRY[m].native_training for m in model_ids)
    names = ['test']
    if needs_calibration:
        names.insert(0, 'calibration')
    if needs_training:
        names = ['train', 'selection', *names]
    return tuple(dict.fromkeys(names))


def verify_expansion(root: Path, manifest: dict) -> Path:
    """Pin the parent identity, shared protocol and cache before any work."""
    provenance = manifest['expansion_from']
    parent_root = Path(provenance['run_root']).resolve()
    if parent_root == root:
        raise ValueError('Expansion cannot refer to itself')
    if sha(parent_root / 'jobs.json') != provenance['jobs_hash']:
        raise ValueError('Parent jobs manifest changed')
    parent = read_json(parent_root / 'jobs.json')
    verify_snapshot(parent_root, parent)
    if any(provenance[key] != parent[key] for key in ('run_id', 'source_hash', 'config_hash', 'cache_hashes')):
        raise ValueError('Parent provenance mismatch')
    child_config = yaml.safe_load((root / 'config.yaml').read_text())
    parent_config = yaml.safe_load((parent_root / 'config.yaml').read_text())
    validate_expansion_config(child_config)
    assert_shared_protocol(parent_config, child_config)
    for key in ('cache_hashes', 'cache_info', 'data_hashes'):
        if manifest.get(key) != parent.get(key):
            raise ValueError(f'Expansion changes frozen data provenance: {key}')
    for region in REGIONS:
        if sha(root / 'data_cache' / region / 'hashes.json') != parent['cache_hashes'][region]:
            raise ValueError(f'Expansion cache manifest changed: {region}')
    return parent_root


def _parent_bases_available(parent_root: Path, model_ids: list[str]) -> None:
    bases = sorted({REGISTRY[m].base_model for m in model_ids if REGISTRY[m].base_model})
    for base in bases:
        if base not in SUPPORTED_BASES:
            raise ValueError(f"Registry base {base} has no frozen loader")
        for seed in SEEDS:
            for region in REGIONS:
                status_path = parent_root / 'runs' / base / f'seed{seed}' / region / 'status.json'
                state = read_json(status_path).get('state') if status_path.exists() else None
                if state != 'completed':
                    raise ValueError(f"Frozen base cell incomplete: {base}/{seed}/{region}")


def prepare(config_path, run_root, extend_from):
    project = Path.cwd().resolve()
    config_path, run_root = Path(config_path).resolve(), Path(run_root).resolve()
    if extend_from is None:
        raise ValueError('Expansion runs require --extend-from pointing at the 90-cell parent run')
    config = yaml.safe_load(config_path.read_text())
    models = validate_expansion_config(config)
    if run_root.exists():
        raise FileExistsError('Use a new dated run directory')
    parent_root = Path(extend_from).resolve()
    parent = audit_parent(parent_root)
    assert_shared_protocol(yaml.safe_load((parent_root / 'config.yaml').read_text()), config)
    _parent_bases_available(parent_root, models)
    run_root.mkdir(parents=True)
    shutil.copyfile(config_path, run_root / 'config.yaml')
    source_files, source_hash = freeze_sources(project, run_root / 'source')
    config['project_root'] = str(project)
    (run_root / 'data_cache').symlink_to(parent_root / 'data_cache', target_is_directory=True)
    jobs = []
    for seed in SEEDS:
        for region in REGIONS:
            for model in models:
                identifier = f'{model}/{seed}/{region}'
                jobs.append({'id': identifier, 'model': model, 'region': region, 'seed': seed,
                             'output': f'runs/{model}/seed{seed}/{region}',
                             'resource': REGISTRY[model].resource})
    manifest = {'run_id': run_root.name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'project_root': str(project), 'source_hash': source_hash, 'source_files': source_files,
                'config_hash': sha(run_root / 'config.yaml'), 'data_hashes': parent['data_hashes'],
                'cache_hashes': parent['cache_hashes'], 'cache_info': parent['cache_info'], 'jobs': jobs,
                'models': models, 'regions': list(REGIONS), 'seeds': list(SEEDS),
                'scope': 'registered_expansion_fixed_configuration_not_hpo_not_blind',
                'expansion': {'schema': EXPANSION_SCHEMA, 'batch': 'first_20261022'},
                'environment': dict(parent.get('environment', {}),
                                    executable=sys.executable),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'expansion_from': {
                    'run_root': str(parent_root), 'run_id': parent['run_id'],
                    'jobs_hash': sha(parent_root / 'jobs.json'), 'source_hash': parent['source_hash'],
                    'config_hash': parent['config_hash'], 'cache_hashes': parent['cache_hashes'],
                    'cache_reuse': 'read_only_parent_cache_via_symlink',
                }}
    atomic_json(run_root / 'jobs.json', manifest)
    print(f'Prepared {len(jobs)} expansion jobs in {run_root}', flush=True)
    return manifest


def worker(root, job_id, device):
    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    parent_root = verify_expansion(root, manifest)
    job = next(j for j in manifest['jobs'] if j['id'] == job_id)
    directory = root / job['output']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if completed(root, job, manifest):
        return
    archive_incomplete(directory)
    status = {'state': 'running', 'job_id': job_id, 'pid': os.getpid(), 'started_at': _now(),
              'source_hash': manifest['source_hash'], 'config_hash': manifest['config_hash'],
              'device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    atomic_json(directory / 'status.json', status)
    start = time.monotonic()
    try:
        from .common.baseline_registry import run_expansion_model

        config = yaml.safe_load((root / 'config.yaml').read_text())
        config['project_root'] = manifest['project_root']
        import torch

        torch.set_num_threads(config['experiment']['cpu_threads_per_job'])
        spec = REGISTRY[job['model']]
        wanted = {'test'}
        if spec.requires_calibration:
            wanted.add('calibration')
        if spec.native_training:
            wanted.update(('train', 'selection'))
        splits = {name: load_split(root, manifest, job['region'], name)
                  for name in SPLITS if name in wanted}
        context = ExpansionContext(
            run_root=root, config=config, manifest=manifest, job=job,
            region=job['region'], seed=job['seed'], device=device, splits=splits,
            parent_run=parent_root, progress_path=directory / 'progress.json',
            model_directory=directory / 'models')
        result = run_expansion_model(job['model'], context)
        metadata = dict(splits['test'].metadata, seed=job['seed'], model=job['model'],
                        source=f"{job['model']}:{manifest['source_hash']}", prediction_space='raw_rrp',
                        output_type='quantile', smoke=False, config_sha256=manifest['config_hash'],
                        run_id=manifest['run_id'], model_metadata=result.metadata)
        artifact = make_prediction_artifact(
            actual=splits['test'].actual, origin_unix=splits['test'].origin_unix,
            target_time_unix=splits['test'].target_time_unix, metadata=metadata,
            quantiles=result.quantiles, point=result.point, quantile_levels=QUANTILE_LEVELS)
        save_prediction_artifact(directory / 'test.npz', artifact)
        atomic_json(directory / 'metrics.json', score_predictions(artifact))
        verify_snapshot(root, manifest)
        model_files = [p for p in (directory / 'models').rglob('*') if p.is_file()] \
            if (directory / 'models').exists() else []
        status.update(state='completed', ended_at=_now(), seconds=time.monotonic() - start,
                      artifact_hashes={str(path.relative_to(directory)): sha(path)
                                       for path in [directory / 'test.npz', directory / 'metrics.json',
                                                    directory / 'best_model.pt', *model_files]
                                       if path.is_file()})
        atomic_json(directory / 'status.json', status)
    except BaseException as exc:
        status.update(state='failed', ended_at=_now(), seconds=time.monotonic() - start,
                      error=repr(exc), traceback=traceback.format_exc())
        atomic_json(directory / 'status.json', status)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(root, retry_failed=False):
    """Controller loop mirroring aligned_experiment.run for expansion jobs."""
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
                       PYTHONUNBUFFERED='1')
            stream = (directory / 'worker.log').open('a')
            command = [sys.executable, '-m', 'forecasting.aligned_expansion', 'worker',
                       '--run-root', str(root), '--job-id', job['id'],
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
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config, args.run_root, args.extend_from)
    elif args.command == 'run':
        run(args.run_root, args.retry_failed)
    else:
        worker(args.run_root, args.job_id, args.device)


if __name__ == '__main__':
    main()
