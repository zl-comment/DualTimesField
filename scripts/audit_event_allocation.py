"""Frozen event allocation, with branch and provenance barriers; no training."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_correction_value import (
    configuration as base_configuration, digest, git, load_completed,
    save_completed, source_files as base_source_files, stage_artifacts,
    verify_frozen, write_json,
)
from scripts.audit_upper_scale import protected_files as preceding_protected_files
from scripts.audit_upper_scale import verify_snapshots


def configuration(path):
    cfg = base_configuration(path)
    for key in ('prior_audit_root', 'prior_audit_archive', 'upper_audit_root', 'upper_audit_archive'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if any(cfg[key] for key in ('checkpoint_reselection_allowed', 'alpha_reselection_allowed', 'outcome_routing_allowed')):
        raise ValueError('Frozen event allocation forbids reselection and outcome routing')
    if cfg['interventions'] != ['original', 'event_off', 'upper_only']:
        raise ValueError('Only the three prespecified event allocations are allowed')
    if cfg['expected_cases'] != 9 or cfg['expected_primary_cases'] != 9:
        raise ValueError('This protocol requires the full nine-case formal matrix')
    return cfg


def protected_files(cfg):
    files = set(preceding_protected_files(cfg))
    archive = Path(cfg['upper_audit_archive'])
    old = json.loads((archive / 'archive_manifest.json').read_text())
    for name, item in old['files'].items():
        path = archive / name
        if digest(path) != item['sha256']:
            raise ValueError(f'Previous upper-scale archive changed: {path}')
    files.update(p for p in archive.rglob('*') if p.is_file())
    prior = Path(cfg['upper_audit_root'])
    manifest = json.loads((prior / 'manifest.json').read_text())
    verify_frozen(manifest)
    verify_snapshots(prior, manifest)
    load_completed(prior, 'audit', manifest['run_id'])
    files.update(p for p in prior.rglob('*') if p.is_file())
    return sorted(files)


def sources(config_path):
    return sorted(set(base_source_files(config_path)) | {
        Path(__file__).resolve(), ROOT / 'scripts/audit_upper_scale.py',
        ROOT / 'scripts/summarize_event_allocation_episodes.py',
        ROOT / 'scripts/verify_event_allocation.py',
        ROOT / 'tests/test_event_allocation.py',
        ROOT / 'tests/test_full_v5_anchor_architecture_experiment.py',
        ROOT / 'docs/event_allocation_protocol_20261027.txt',
    })


def freeze(output, config_path, cfg):
    if git('status', '--porcelain'):
        raise ValueError('Commit the protocol and implementation before a new batch')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for path in sources(config_path):
        destination = output / 'source' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(path)] = digest(path)
    protected = {str(p): digest(p) for p in protected_files(cfg)}
    environment = dict(python=sys.version, platform=platform.platform(), packages={
        name: importlib.metadata.version(name) for name in
        ('numpy', 'pandas', 'scipy', 'torch', 'scikit-learn', 'lightgbm')})
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_commit=cfg['base_commit'], head=git('rev-parse', 'HEAD'), branch=cfg['branch'],
        working_tree_status=git('status', '--short'), configuration=cfg,
        environment=environment, sources=source_hashes, protected_inputs=protected,
        experiment_type='Frozen event quantile allocation; no fitting or selection')
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / 'manifest.json', manifest)
    return manifest


def run(output, config_path, cfg, resume=False):
    if resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['configuration'] != cfg:
            raise ValueError('Cannot resume with a changed protocol')
        verify_frozen(manifest)
        verify_snapshots(output, manifest)
    else:
        manifest = freeze(output, config_path, cfg)
    if not (output / 'audit_complete.json').exists():
        from forecasting.full_v5_event_allocation_audit import audit
        write_json(output / 'progress.json', dict(stage='audit', run_id=manifest['run_id']))
        print('[audit] starting frozen event allocation', flush=True)
        result = audit(output, cfg)
        save_completed(output, 'audit', manifest['run_id'], result)
    load_completed(output, 'audit', manifest['run_id'])
    if not (output / 'episodes_complete.json').exists():
        from scripts.summarize_event_allocation_episodes import summarize
        write_json(output / 'progress.json', dict(stage='episodes', run_id=manifest['run_id']))
        result = summarize(output, cfg)
        save_completed(output, 'episodes', manifest['run_id'], result)
    verify_frozen(manifest)
    verify_snapshots(output, manifest)
    artifacts = {}
    for stage in ('audit', 'episodes'):
        load_completed(output, stage, manifest['run_id'])
        artifacts.update(stage_artifacts(output, stage))
        artifacts[f'{stage}_complete.json'] = digest(output / f'{stage}_complete.json')
    verification = dict(run_id=manifest['run_id'], completed=['audit', 'episodes'],
        protected_files_unchanged=len(manifest['protected_inputs']),
        source_files_unchanged=len(manifest['sources']), source_snapshots_match=True,
        artifact_sha256=artifacts, trained_models=0, checkpoints_reselected=0,
        alphas_reselected=0, formal_tables_changed=False,
        scientific_confirmation=False, scope=cfg['scope'])
    write_json(output / 'verification.json', verification)
    write_json(output / 'progress.json', dict(stage='complete', run_id=manifest['run_id']))
    print('[audit] complete', flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/event_allocation_audit.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = configuration(args.config.resolve())
    output = args.output.resolve()
    if output.parent != ROOT / 'outputs/forecasting' or not output.name.startswith('event_allocation_audit'):
        raise ValueError('Use a fresh outputs/forecasting/event_allocation_audit* directory')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(output, args.config.resolve(), cfg, args.resume)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
