"""Development-only mechanism constraints; full matrix or labelled integration smoke."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo
import numpy as np

from ..full_v5_anchor_balance import experiment as base, confirmation_experiment as c
from ..full_v5_anchor_balance.selection import blend
from .data import load_inputs, MechanismStore
from .model import VARIANTS, build_model
from .policy import fit_policy, classify
from .training import fit
from .diagnostics import diagnose

PROTOCOL = Path('docs/v5_mechanism_state_20261023.md')
OLD = Path('outputs/forecasting/v5_calibration_robustness_20261023_r2').resolve()


def configuration(devices, smoke=False):
    cfg = c.configuration(devices)
    cfg.update(experiment='mechanism_state_v1', variants=list(VARIANTS),
        scope='integration_smoke_not_evidence' if smoke else 'retrospective_development',
        objective='unchanged balance_loss', policy='four training blocks, support days 30, fixed state thresholds',
        old_reference=str(OLD))
    if smoke:
        cfg.update(epochs=2, regions=['NSW1'], folds=['2021'], devices=devices[:1],
                   sampling='96 train / 48 selection / 48 evaluation, chronological evenly spaced',
                   stopping='two epochs, integration verification only')
    return cfg


def split_rows(store, fold, cfg):
    rows = base.split_rows(store.origins, fold, 'full')
    if cfg['scope'] == 'integration_smoke_not_evidence':
        rows = {role: ids[np.linspace(0, len(ids)-1, min(len(ids), 96 if role == 'train' else 48), dtype=int)]
                for role, ids in rows.items()}
    return rows


def arm_path(output, region, fold, seed, arm):
    return Path(output) / region / fold / f'seed_{seed}' / arm


def verify_manifest(manifest):
    if c.fingerprint({k: v for k, v in manifest.items() if k != 'run_id'}) != manifest['run_id']:
        raise ValueError('Manifest identity changed')
    for kind in ('sources', 'inputs', 'policies'):
        if any(base.digest(p) != h for p, h in manifest[kind].items()):
            raise ValueError(f'Frozen {kind} changed')


def prepare(output, cfg, bundles, resume=False):
    if resume:
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['config'] != cfg or manifest['bundle_root'] != str(bundles):
            raise ValueError('Resume configuration changed')
        verify_manifest(manifest)
        return manifest
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config=cfg, bundle_root=str(bundles), sources={}, inputs={}, policies={}, samples={},
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    # Hash and snapshot all executable forecasting dependencies, not just this package.
    for path in [*sorted(Path('forecasting').rglob('*.py')), *sorted(Path('src/dualfield').rglob('*.py')),
                 Path('scripts/diagnose_mechanism_state.py'), Path('scripts/verify_mechanism_state.py'), PROTOCOL]:
        dest = output/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        manifest['sources'][str(path.resolve())] = base.digest(path)
    for region in cfg['regions']:
        store, paths = base.load_store(bundles, region, cfg['tree_seed'])
        inputs, audit = load_inputs(store, paths)
        directory = output/region
        directory.mkdir()
        base.write_json(directory/'input_audit.json', audit)
        manifest['inputs'].update(audit['sources'])
        for p in paths:
            meta = json.loads(p.with_suffix('.json').read_text())
            for f in [p, p.with_suffix('.json'), p.with_name('config.json'), *[p.parent/name for name in meta['model_files']]]:
                manifest['inputs'][str(f.resolve())] = base.digest(f)
        for fold in cfg['folds']:
            rows = split_rows(store, fold, cfg)
            ids = rows['train']
            policy = fit_policy(inputs[ids], store.a['actual'][ids], store.a['baseline'][ids], store.origins[ids])
            path = directory/fold/'policy.json'
            path.parent.mkdir()
            base.write_json(path, policy)
            manifest['policies'][str(path)] = base.digest(path)
            manifest['samples'][f'{region}/{fold}'] = {k: base.sample_manifest(store.origins, v) for k, v in rows.items()}
            if cfg['scope'] != 'integration_smoke_not_evidence':
                old_balanced = Path('outputs/forecasting/v5_anchor_confirmation_20261022_r1').resolve()
                for seed in cfg['seeds']:
                    for old_arm, old_root in [('bounded_calibration', OLD), ('reference', OLD), ('balanced', old_balanced)]:
                        old_dir = arm_path(old_root, region, fold, seed, old_arm)
                        for file in ('metrics.json', 'training.json', 'raw.pt', 'guarded.pt', 'last.pt'):
                            p = old_dir/file
                            manifest['inputs'][str(p)] = base.digest(p)
        del store, inputs
    manifest['run_id'] = c.fingerprint(manifest)
    base.write_json(output/'manifest.json', manifest)
    return manifest


def expected_keys(cfg):
    return {f'{r}/{f}/{s}/{v}' for r in cfg['regions'] for f in cfg['folds'] for s in cfg['seeds'] for v in cfg['variants']}


def verify_barrier(records, cfg, output):
    if set(records) != expected_keys(cfg):
        raise ValueError('All selections must freeze before evaluation')
    for key, record in records.items():
        region, fold, seed, variant = key.split('/')
        directory = arm_path(output, region, fold, int(seed), variant)
        if set(record['checkpoints']) != {str(directory/f'{m}.pt') for m in ('raw', 'guarded')}:
            raise ValueError('Checkpoint does not match region/fold/seed/variant')
        if record['epochs_completed'] != cfg['epochs']:
            raise ValueError('Incomplete training budget')
        c.verify_checkpoints(record)


def adapted_store(store, inputs, splits, roles, device):
    rows = np.unique(np.concatenate([part[role] for part in splits.values() for role in roles]))
    adapted = c.DeviceHistoryStore(store, rows, device)
    audit = adapted.verify(store, rows[np.linspace(0, len(rows)-1, min(64, len(rows)), dtype=int)])
    return MechanismStore(adapted, inputs), audit


def train_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    store, paths = base.load_store(bundles, region, cfg['tree_seed'])
    inputs, _ = load_inputs(store, paths)
    splits = {f: split_rows(store, f, cfg) for f in cfg['folds']}
    adapted, audit = adapted_store(store, inputs, splits, ('train', 'selection'), device)
    base.write_json(Path(output)/region/'training_input_audit.json', audit)
    records = {}
    for fold, rows in splits.items():
        policy = json.loads((Path(output)/region/fold/'policy.json').read_text())
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                path = arm_path(output, region, fold, seed, variant)
                records[f'{region}/{fold}/{seed}/{variant}'] = fit(adapted, rows, variant, seed, cfg, path, run_id, policy)
                if variant == 'bounded_reference' and cfg['scope'] != 'integration_smoke_not_evidence':
                    from ..full_v5_calibration_robustness.experiment import verify_reference
                    verify_reference(path, arm_path(OLD, region, fold, seed, 'bounded_calibration'))
    return records


def evaluate_fit(store, rows, cfg, path, record, policy):
    c.verify_checkpoints(record)
    ids = rows['evaluation']
    origins, y, b = store.origins[ids], store.a['actual'][ids], store.a['baseline'][ids]
    inputs = store.inputs[ids].cpu().numpy()
    groups = base.evaluation_groups(origins, b)
    counts = {g: dict(n_origins=int(m.sum()), n_cells=int(m.sum()*24)) for g, m in groups.items()}
    result = dict(groups=counts, B={g: dict(**counts[g], **c.score(y[m], b[m])) for g, m in groups.items() if m.any()})
    for mode in ('raw', 'guarded'):
        model, selected = base.restore_checkpoint(path/f'{mode}.pt', model_builder=build_model)
        if selected != record['selection'][mode]:
            raise ValueError('Frozen selection changed')
        model.to(store.device)
        raw = base.predict(model, store, ids, cfg['prediction_batch_size'])
        chosen = {k: blend(b, q, selected['alpha']) for k, q in raw.items()}
        result[mode] = dict(selection=selected,
            metrics={g: base.diagnostics(y[m], b[m], {k: q[m] for k, q in chosen.items()}) for g, m in groups.items() if m.any()},
            mechanism=diagnose(y, b, chosen['anchor'], chosen['full'], inputs, policy),
            shape=c.shape_diagnostics(y, b, chosen['anchor'], chosen['full']))
        temporary = path/f'{mode}.tmp.npz'
        states = classify(inputs, policy['thresholds'])
        np.savez_compressed(temporary, origin_unix=origins, target_time_unix=origins[:, None]+np.arange(24)[None]*base.HOUR,
            actual=y, baseline=b, anchor=chosen['anchor'], quantiles=chosen['full'],
            raw_anchor=raw['anchor'], raw_quantiles=raw['full'], quantile_levels=base.LEVELS,
            mechanism_state=states, policy_gates=np.asarray(policy['gates'])[states],
            alpha=selected['alpha'], epoch=selected['epoch'])
        temporary.replace(path/f'{mode}.npz')
    base.write_json(path/'metrics.json', result)
    return result


def evaluate_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    output = Path(output)
    records = json.loads((output/'selection_complete.json').read_text())
    verify_barrier(records, cfg, output)
    store, paths = base.load_store(bundles, region, cfg['tree_seed'])
    inputs, _ = load_inputs(store, paths)
    splits = {f: split_rows(store, f, cfg) for f in cfg['folds']}
    adapted, audit = adapted_store(store, inputs, splits, ('evaluation',), device)
    base.write_json(output/region/'evaluation_input_audit.json', audit)
    results = {}
    for fold, rows in splits.items():
        policy = json.loads((output/region/fold/'policy.json').read_text())
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                key = f'{region}/{fold}/{seed}/{variant}'
                results[key] = evaluate_fit(adapted, rows, cfg, arm_path(output, region, fold, seed, variant), records[key], policy)
    return results


def run(cfg, bundles, output, resume=False):
    manifest = prepare(output, cfg, bundles, resume)
    verify_manifest(manifest)
    specs = [(r, cfg, str(bundles), str(output), manifest['run_id'], d) for r, d in zip(cfg['regions'], cfg['devices'])]
    base.write_json(output/'progress.json', dict(stage='training_selection', fits=len(expected_keys(cfg))))
    records = c.run_pool(train_region, specs)
    verify_manifest(manifest)
    verify_barrier(records, cfg, output)
    barrier = output/'selection_complete.json'
    if barrier.exists() and json.loads(barrier.read_text()) != records:
        raise ValueError('Frozen selection barrier changed')
    base.write_json(barrier, records)
    base.write_json(output/'progress.json', dict(stage='evaluation', fits=len(records)))
    results = c.run_pool(evaluate_region, specs)
    verify_manifest(manifest)
    verify_barrier(records, cfg, output)
    if set(results) != expected_keys(cfg):
        raise ValueError('Incomplete evaluation matrix')
    base.write_json(output/'report.json', results)
    from .report import write_report
    write_report(output, results, cfg)
    base.write_json(output/'progress.json', dict(stage='complete', fits=len(results), scope=cfg['scope'],
        selection_sha256=base.digest(barrier), report_sha256=base.digest(output/'report.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bundle-root', default=base.BUNDLE_ROOT)
    parser.add_argument('--devices', nargs=3, default=['cuda:3', 'cuda:6', 'cuda:7'])
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not args.development:
        raise ValueError('--development required; final-test unsupported')
    output, bundles = Path(args.output).resolve(), Path(args.bundle_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Output already has an active runner') from None
        if output.exists() and not args.resume:
            raise FileExistsError('Output exists; use --resume')
        try:
            run(configuration(args.devices, args.smoke), bundles, output, args.resume)
        except Exception as error:
            if output.exists():
                base.write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
