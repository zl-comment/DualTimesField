"""Run the frozen L0 correction-value audit on its dedicated research branch.

This entry point does not fit models or update published tables. Every run has
its own source snapshot, configuration, environment and immutable input hashes.
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
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def configuration(path, device=None):
    cfg = json.loads(path.read_text())
    cfg['root'] = str(ROOT)
    for key in ('formal_root', 'robustness_root', 'mechanism_diagnosis_root', 'bundle_root'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if device is not None:
        cfg['device'] = device
    if cfg['training_allowed'] or cfg['update_main_table']:
        raise ValueError('This audit forbids training and publication changes')
    if git('branch', '--show-current') != cfg['branch']:
        raise ValueError('New batch experiments require their dedicated configured branch')
    subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['base_commit'], 'HEAD'],
                   cwd=ROOT, check=True)
    return cfg


def protected_files(cfg):
    files = []
    for key in ('formal_root', 'robustness_root', 'mechanism_diagnosis_root'):
        directory = Path(cfg[key])
        files.extend(p for p in directory.rglob('*') if p.is_file()
                     and p.suffix in ('.json', '.npz', '.pt'))
        saved_manifest = json.loads((directory / 'manifest.json').read_text())
        input_keys = ('inputs_sha256',) if key == 'formal_root' else ('inputs',)
        for input_key in input_keys:
            for path, expected in saved_manifest[input_key].items():
                source = Path(path)
                if not source.is_absolute():
                    source = ROOT / source
                if digest(source) != expected:
                    raise ValueError(f'Archived input hash mismatch: {source}')
                files.append(source)
    for name in ('main_comparison_20261023', 'v5_anchor_confirmation_20261022',
                 'v5_calibration_robustness_20261023', 'v5_mechanism_state_20261023'):
        files.extend(p for p in (ROOT / 'logs' / 'forecasting' / name).rglob('*') if p.is_file())
    return sorted(set(files))


def source_files(config_path):
    paths = [*ROOT.joinpath('forecasting').rglob('*.py'),
             *ROOT.joinpath('src/dualfield').rglob('*.py'),
             Path(__file__).resolve(), config_path, ROOT / 'AGENTS.md']
    paths.extend(ROOT.joinpath('tests').glob('test_correction_value_audit.py'))
    return sorted(set(paths))


def freeze(output, config_path, cfg):
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for path in source_files(config_path):
        destination = output / 'source' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(path)] = digest(path)
    inputs = {str(path): digest(path) for path in protected_files(cfg)}
    environment = dict(python=sys.version, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in
                  ('numpy', 'pandas', 'scipy', 'torch', 'scikit-learn', 'lightgbm')})
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_commit=cfg['base_commit'], head=git('rev-parse', 'HEAD'), branch=cfg['branch'],
        working_tree_status=git('status', '--short'), configuration=cfg,
        environment=environment, sources=source_hashes, protected_inputs=inputs,
        experiment_type='L0: no fitting, no checkpoint selection, no publication')
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / 'manifest.json', manifest)
    return manifest


def verify_frozen(manifest):
    identity = {key: value for key, value in manifest.items() if key != 'run_id'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != manifest['run_id']:
        raise ValueError('Audit manifest fingerprint changed')
    for kind in ('sources', 'protected_inputs'):
        for path, expected in manifest[kind].items():
            if digest(path) != expected:
                raise ValueError(f'Frozen {kind} changed: {path}')


def stage_artifacts(output, name):
    return {str(p.relative_to(output)): digest(p)
            for p in sorted((output / name).rglob('*')) if p.is_file()}


def load_completed(output, name, run_id):
    record = json.loads((output / f'{name}_complete.json').read_text())
    if record['run_id'] != run_id or record['stage'] != name:
        raise ValueError('Stage completion identity changed')
    identity = {key: value for key, value in record.items() if key != 'completion_id'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != record['completion_id']:
        raise ValueError('Stage completion fingerprint changed')
    if stage_artifacts(output, name) != record['artifact_sha256']:
        raise ValueError(f'Completed {name} artifacts changed; refusing to re-certify them')
    return record['result']


def save_completed(output, name, run_id, result):
    record = dict(stage=name, run_id=run_id, result=result,
                  artifact_sha256=stage_artifacts(output, name))
    record['completion_id'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    write_json(output / f'{name}_complete.json', record)


def run(output, config_path, cfg, stage, resume):
    if resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['configuration'] != cfg:
            raise ValueError('Cannot resume with a changed configuration')
        verify_frozen(manifest)
    else:
        manifest = freeze(output, config_path, cfg)
    from forecasting.full_v5_correction_audit.saved import audit_saved
    from forecasting.full_v5_correction_audit.replay import audit_replay
    results = {}
    for name, action in (('saved', audit_saved), ('replay', audit_replay)):
        if stage not in ('all', name):
            continue
        completion = output / f'{name}_complete.json'
        if completion.exists():
            results[name] = load_completed(output, name, manifest['run_id'])
            continue
        write_json(output / 'progress.json', dict(stage=name, run_id=manifest['run_id']))
        print(f'[{name}] starting frozen audit', flush=True)
        results[name] = action(output, cfg)
        save_completed(output, name, manifest['run_id'], results[name])
        print(f'[{name}] complete', flush=True)
    verify_frozen(manifest)
    completed = [name for name in ('saved', 'replay') if (output / f'{name}_complete.json').exists()]
    artifacts = {}
    for name in completed:
        load_completed(output, name, manifest['run_id'])
        artifacts.update(stage_artifacts(output, name))
        artifacts[f'{name}_complete.json'] = digest(output / f'{name}_complete.json')
    verification = dict(run_id=manifest['run_id'], completed=completed,
        protected_files_unchanged=len(manifest['protected_inputs']), source_files_unchanged=len(manifest['sources']),
        artifact_sha256=artifacts, trained_models=0, formal_tables_changed=False,
        scientific_confirmation=False, scope=cfg['scope'])
    write_json(output / 'verification.json', verification)
    write_json(output / 'progress.json', dict(stage='complete' if len(completed) == 2 else 'partial',
        completed=completed, run_id=manifest['run_id']))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/correction_value_audit.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('all', 'saved', 'replay'), default='all')
    parser.add_argument('--device')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = configuration(config_path, args.device)
    output = args.output.resolve()
    if output.parent != ROOT / 'outputs' / 'forecasting' or not output.name.startswith('correction_value_audit'):
        raise ValueError('Use a new outputs/forecasting/correction_value_audit* directory')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(output, config_path, cfg, args.stage, args.resume)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
