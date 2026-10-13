"""Extend a frozen 90-cell run without rewriting its results or provenance.

The extension has its own source/configuration snapshot and 18-cell matrix.
Only identical cached inputs, labels, sampling and scoring protocols may be
combined. Partial extensions publish progress, never a partial leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

from .aligned_experiment import (
    BASE_MODELS, EXTENSION_MODELS, REGIONS, SEEDS, atomic_json, load_cache,
    read_json, sha, validate_config, verify_snapshot,
)
from .aligned_report import build_aligned_report
from .aligned_scoring import assert_comparable, load_prediction_artifact

SHARED_SECTIONS = ('data', 'forecast_protocol', 'aligned_protocol', 'origin_context',
                   'future_exogenous', 'ctf_exogenous', 'quantile_exogenous',
                   'predispatch_exogenous', 'interconnector_exogenous')
SHARED_TRAINING = ('epochs', 'batch_size', 'patience', 'weight_decay',
                   'primary_selection', 'checkpoint_metrics')


def assert_shared_protocol(parent_config, child_config):
    """Model internals/rates/resources may differ; data and budgets may not."""
    for key in SHARED_SECTIONS:
        if parent_config.get(key) != child_config.get(key):
            raise ValueError(f'Extension changes shared protocol section: {key}')
    for key in SHARED_TRAINING:
        if parent_config['training'].get(key) != child_config['training'].get(key):
            raise ValueError(f'Extension changes shared training budget: {key}')
    if parent_config['model']['quantiles'] != child_config['model']['quantiles']:
        raise ValueError('Extension changes quantile levels')


def _verify_parent_grid(manifest):
    if 'extension_from' in manifest:
        raise ValueError('Extension parent must be the original ten-model run')
    if (len(manifest.get('models', [])) != len(BASE_MODELS)
            or set(manifest.get('models', [])) != set(BASE_MODELS)):
        raise ValueError('Parent must declare all ten original models')
    cells = [(j['model'], j['region'], j['seed']) for j in manifest['jobs']]
    expected = {(m, r, s) for m in BASE_MODELS for r in REGIONS for s in SEEDS}
    if len(cells) != len(expected) or set(cells) != expected:
        raise ValueError('Parent must contain the complete 90-cell grid')


def _verify_caches(root, manifest):
    if set(manifest.get('cache_hashes', {})) != set(REGIONS):
        raise ValueError('All three parent caches must be declared')
    for region in REGIONS:
        # Also checks every .npy and metadata file against the frozen manifest.
        load_cache(root / 'data_cache' / region, manifest['cache_hashes'][region])


def audit_parent(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'jobs.json')
    _verify_parent_grid(manifest)
    verify_snapshot(root, manifest)
    validate_config(yaml.safe_load((root / 'config.yaml').read_text()))
    _verify_caches(root, manifest)
    report = build_aligned_report(root)
    if not report['comparison_available']:
        raise ValueError('Parent run must have 90 verified completed cells before extension')
    return manifest


def verify_extension(root, manifest, *, verify_cache=False):
    """Check pinned parent identity and shared protocol before work/reporting."""
    root = Path(root).resolve()
    provenance = manifest['extension_from']
    parent_root = Path(provenance['run_root']).resolve()
    if parent_root == root:
        raise ValueError('Extension cannot refer to itself')
    if sha(parent_root / 'jobs.json') != provenance['jobs_hash']:
        raise ValueError('Parent jobs manifest changed')
    parent = read_json(parent_root / 'jobs.json')
    _verify_parent_grid(parent)
    verify_snapshot(parent_root, parent)
    for key in ('run_id', 'source_hash', 'config_hash', 'cache_hashes'):
        if provenance[key] != parent[key]:
            raise ValueError(f'Parent provenance mismatch: {key}')
    parent_config = yaml.safe_load((parent_root / 'config.yaml').read_text())
    child_config = yaml.safe_load((root / 'config.yaml').read_text())
    validate_config(parent_config)
    validate_config(child_config, extension=True)
    assert_shared_protocol(parent_config, child_config)
    if (len(manifest.get('models', [])) != len(EXTENSION_MODELS)
            or set(manifest.get('models', [])) != set(EXTENSION_MODELS)):
        raise ValueError('Extension must declare TimeXer and PriceFM exactly once')
    for key in ('cache_hashes', 'cache_info', 'data_hashes'):
        if manifest.get(key) != parent.get(key):
            raise ValueError(f'Extension changes frozen data provenance: {key}')
    for region in REGIONS:
        if sha(root / 'data_cache' / region / 'hashes.json') != parent['cache_hashes'][region]:
            raise ValueError(f'Extension cache manifest changed: {region}')
    if verify_cache:
        _verify_caches(parent_root, parent)
        # Avoid hashing the same large data twice for the prepared symlink case.
        if (root / 'data_cache').resolve() != (parent_root / 'data_cache').resolve():
            _verify_caches(root, manifest)
    return parent_root, parent


def build_extension_report(run_root):
    root = Path(run_root).resolve()
    manifest = read_json(root / 'jobs.json')
    parent_root, parent = verify_extension(root, manifest, verify_cache=True)
    # Both independently validate source/config snapshots, all completed artifact
    # hashes, original scoring, and the complete region x seed matrix.
    parent_report = build_aligned_report(parent_root)
    child_report = build_aligned_report(root)
    complete = parent_report['comparison_available'] and child_report['comparison_available']
    sources = {}
    for location, source in ((parent_root, parent), (root, manifest)):
        sources[str(location)] = {key: source[key] for key in
                                 ('run_id', 'source_hash', 'config_hash', 'cache_hashes', 'models')}
        sources[str(location)]['jobs_hash'] = sha(location / 'jobs.json')
    report = {
        'run_id': manifest['run_id'] + '+parent',
        'state': 'complete' if complete else 'incomplete',
        'stage': 'fixed_configuration_comparison_not_tuned_sota',
        'comparison_available': complete,
        'sources': sources,
        'progress': {
            'expected_cells': 108,
            'completed_cells': sum(r['progress']['completed_cells'] for r in (parent_report, child_report)),
            'parent': parent_report['progress'], 'extension': child_report['progress'],
        },
        'completed_artifact_validation': 'passed',
        'frozen_snapshot_validation': {
            'parent': parent_report['frozen_snapshot_validation'],
            'extension': child_report['frozen_snapshot_validation'],
        },
    }
    if not complete:
        return report
    references = {}
    for location, source in ((parent_root, parent), (root, manifest)):
        for job in source['jobs']:
            artifact = load_prediction_artifact(location / job['output'] / 'test.npz')
            if job['region'] in references:
                assert_comparable(references[job['region']], artifact, allow_different_seed=True)
            else:
                references[job['region']] = artifact
    report['cross_run_alignment_validation'] = 'passed_all_108_artifacts'
    report['tables'] = {'P': {}, 'Q': {}}
    for location, source_report in ((parent_root, parent_report), (root, child_report)):
        for table, models in source_report['tables'].items():
            for model, row in models.items():
                if model in report['tables'][table]:
                    raise ValueError(f'Duplicate model across source runs: {model}')
                report['tables'][table][model] = dict(row, provenance=sources[str(location)])
    return report


def write_extension_report(output, report):
    """Write audited JSON and complete-only CSVs with per-row source identity."""
    output = Path(output)
    atomic_json(output, report)
    directory = output.parent / (output.stem + '_tables')
    filenames = ('main.csv', 'regional.csv', 'per_seed.csv')
    if not report['comparison_available']:
        # A refreshed incomplete report must not leave a stale leaderboard.
        for name in filenames:
            (directory / name).unlink(missing_ok=True)
        return

    def flatten(values, prefix=''):
        result = {}
        for key, value in values.items():
            name = prefix + key
            if isinstance(value, dict):
                result.update(flatten(value, name + '.'))
            else:
                result[name] = value
        return result

    main, regional, per_seed = [], [], []
    for table, models in report['tables'].items():
        for model, row in models.items():
            identity = dict(table=table, model=model, **{
                key: row['provenance'][key] for key in ('run_id', 'source_hash', 'config_hash')})
            main.append(dict(identity, **flatten(row['macro_regions'])))
            for region, region_row in row['regions'].items():
                regional.append(dict(identity, region=region,
                                     **flatten(region_row['scores']['seed_summary'])))
                for seed, scores in region_row['scores']['per_seed'].items():
                    per_seed.append(dict(identity, region=region, seed=seed, **flatten(scores)))
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(filenames, (main, regional, per_seed)):
        path = directory / name
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    write_extension_report(args.output, build_extension_report(args.run_root))


if __name__ == '__main__':
    main()
