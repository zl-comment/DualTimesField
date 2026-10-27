"""Run upper-scale deletion diagnostics on the required independent branch.

The fixed transform consumes only B and Full. Outcomes are used for scoring,
never to choose a forecast, checkpoint, alpha, region, or scale multiplier.
"""
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
    protected_files as base_protected_files, save_completed,
    source_files as base_source_files, stage_artifacts, verify_frozen, write_json,
)


def configuration(path):
    cfg = base_configuration(path)
    for key in ('prior_audit_root', 'prior_audit_archive'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if any(cfg[key] for key in ('checkpoint_reselection_allowed',
                                'alpha_reselection_allowed', 'outcome_routing_allowed')):
        raise ValueError('This fixed intervention forbids reselection or outcome routing')
    if cfg['interventions'] != ['original', 'upper_off']:
        raise ValueError('Only the prespecified upper-scale removal is allowed')
    return cfg


def protected_files(cfg):
    files = set(base_protected_files(cfg))
    archive = Path(cfg['prior_audit_archive'])
    old_manifest = json.loads((archive / 'archive_manifest.json').read_text())
    for name, expected in old_manifest['files'].items():
        path = archive / name
        if digest(path) != expected:
            raise ValueError(f'Prior audit archive changed: {path}')
    files.update(p for p in archive.rglob('*') if p.is_file())
    prior = Path(cfg['prior_audit_root'])
    record = json.loads((prior / 'manifest.json').read_text())
    # This check also protects the original experiment's source identity.
    verify_frozen(record)
    for stage in ('saved', 'replay'):
        load_completed(prior, stage, record['run_id'])
    files.update(p for p in prior.rglob('*') if p.is_file())
    return sorted(files)


def sources(config_path):
    return sorted(set(base_source_files(config_path)) | {
        Path(__file__).resolve(), ROOT / 'tests/test_upper_scale_audit.py',
        ROOT / 'tests/test_full_v5_anchor_architecture_experiment.py',
    })


def freeze(output, config_path, cfg):
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for path in sources(config_path):
        destination = output / 'source' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(path)] = digest(path)
    protected = {str(p): digest(p) for p in protected_files(cfg)}
    environment = dict(python=sys.version, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in
                  ('numpy', 'pandas', 'scipy', 'torch', 'scikit-learn', 'lightgbm')})
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_commit=cfg['base_commit'], head=git('rev-parse', 'HEAD'), branch=cfg['branch'],
        working_tree_status=git('status', '--short'), configuration=cfg,
        environment=environment, sources=source_hashes, protected_inputs=protected,
        experiment_type='Frozen upper-scale deletion; no fitting or selection')
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / 'manifest.json', manifest)
    return manifest


def verify_snapshots(output, manifest):
    for source, expected in manifest['sources'].items():
        copied = output / 'source' / Path(source).relative_to(ROOT)
        if digest(copied) != expected:
            raise ValueError(f'Source snapshot changed: {copied}')


def run(output, config_path, cfg, resume=False):
    if resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['configuration'] != cfg:
            raise ValueError('Cannot resume with a changed configuration')
        verify_frozen(manifest)
        verify_snapshots(output, manifest)
    else:
        manifest = freeze(output, config_path, cfg)
    if (output / 'audit_complete.json').exists():
        result = load_completed(output, 'audit', manifest['run_id'])
    else:
        from forecasting.full_v5_upper_scale_audit.audit import audit
        write_json(output / 'progress.json', dict(stage='audit', run_id=manifest['run_id']))
        print('[audit] starting frozen upper-scale removal', flush=True)
        result = audit(output, cfg)
        save_completed(output, 'audit', manifest['run_id'], result)
    verify_frozen(manifest)
    verify_snapshots(output, manifest)
    load_completed(output, 'audit', manifest['run_id'])
    artifacts = stage_artifacts(output, 'audit')
    artifacts['audit_complete.json'] = digest(output / 'audit_complete.json')
    verification = dict(run_id=manifest['run_id'], completed=['audit'],
        protected_files_unchanged=len(manifest['protected_inputs']),
        source_files_unchanged=len(manifest['sources']),
        source_snapshots_match=True, artifact_sha256=artifacts,
        trained_models=0, checkpoints_reselected=0, alphas_reselected=0,
        formal_tables_changed=False, scientific_confirmation=False, scope=cfg['scope'])
    write_json(output / 'verification.json', verification)
    write_json(output / 'progress.json', dict(stage='complete', run_id=manifest['run_id']))
    print('[audit] complete', flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/upper_scale_audit.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = configuration(config_path)
    output = args.output.resolve()
    if output.parent != ROOT / 'outputs/forecasting' or not output.name.startswith('upper_scale_audit'):
        raise ValueError('Use a fresh outputs/forecasting/upper_scale_audit* directory')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(output, config_path, cfg, args.resume)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
