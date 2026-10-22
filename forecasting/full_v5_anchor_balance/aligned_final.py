"""Latest asymmetric Full_joint, three neural seeds, frozen aligned test protocol."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ..aligned_inputs import AlignedSplit, INPUT_KEYS
from ..aligned_scoring import (assert_comparable, load_prediction_artifact,
    make_prediction_artifact, save_prediction_artifact, score_predictions)
from ..full_v5.data import atomic_json, load_cache_split, set_stage
from ..full_v5_joint.data import HistoryStore, bundle_from_aligned
from ..full_v5_joint.final_test import combine_memory
from ..full_v5_joint.model import LEVELS
from ..full_v5_joint.protocol import DAY, segment_rows, timestamp
from ..lightgbm.predictor import FrozenLightGBMPredictor
from . import confirmation_experiment as c
from .confirmation_data import DeviceHistoryStore
from .confirmation_model import build_model
from .selection import blend

PROTOCOL = Path('docs/v5_latest_aligned_20261023.md')
PARENT = Path('outputs/forecasting/aligned_baselines_20261012')
OUTPUTS = {'Full_joint': {}, 'Anchor_joint': {'history_off': True},
           'event_off_same_head': {'event_off': True},
           'time_shift_off_same_head': {'time_shift_off': True}}
ALL_OUTPUTS = ('B_retrained', *OUTPUTS, 'Full_joint_raw')


def configuration(devices):
    cfg = c.configuration(devices)
    cfg.update(experiment='latest_asymmetric_aligned_v1', variants=['asymmetric_scale'],
               folds=['aligned'], train_start='2018-01-01', train_end='2022-01-01',
               selection_end='2022-07-01', evaluation='exact canonical 2023-2024',
               main_output='guarded', tree_seed=2026,
               seed_scope='neural initialization/order only; fixed tree seed 2026')
    return cfg


def training_rows(origins, cfg):
    origins = np.asarray(origins)
    if np.any(origins >= timestamp('2023-01-01')):
        raise ValueError('Final-test origins are forbidden during fitting')
    rows = dict(train=segment_rows(origins, cfg['train_start'], cfg['train_end']),
                selection=segment_rows(origins, cfg['train_end'], cfg['selection_end']))
    if any(not len(r) for r in rows.values()):
        raise ValueError('Empty fitting/selection period')
    if origins[rows['train']].max() + DAY >= origins[rows['selection']].min():
        raise ValueError('Training labels overlap selection')
    return rows


def fit_path(output, region, seed):
    return Path(output) / 'fits' / region / f'seed{seed}'


def expected_keys(cfg):
    return {f'{r}/{s}' for r in cfg['regions'] for s in cfg['seeds']}


def verify_barrier(completed, cfg):
    if set(completed) != expected_keys(cfg):
        raise ValueError('All nine fits must finish before final-test access')
    for record in completed.values():
        if record['epochs_completed'] != cfg['epochs']:
            raise ValueError('Incomplete epoch budget')
        c.verify_checkpoints(record)


def verify_manifest(manifest):
    for key in ('source_sha256', 'inputs_sha256'):
        if any(c.base.digest(path) != expected for path, expected in manifest[key].items()):
            raise ValueError(f'Frozen {key} changed')
    if c.base.digest(PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')


def prepare(output, cfg, bundles, resume):
    manifest_path = output / 'manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest['config'] != cfg or manifest['bundle_root'] != str(bundles)
                or c.fingerprint({k: v for k, v in manifest.items() if k != 'run_id'}) != manifest['run_id']):
            raise ValueError('Resume identity changed')
        verify_manifest(manifest)
        return manifest
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted([*Path('forecasting').rglob('*.py'), *Path('src/dualfield').rglob('*.py')])
    manifest = dict(config=cfg, bundle_root=str(bundles),
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protocol_sha256=c.base.digest(PROTOCOL), source_sha256={}, inputs_sha256={}, samples={},
        torch_version=str(torch.__version__), numpy_version=str(np.__version__))
    for source in sources:
        destination = output / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest['source_sha256'][str(source.resolve())] = c.base.digest(source)
    shutil.copyfile(PROTOCOL, output / 'protocol.md')
    for region in cfg['regions']:
        store, paths = c.base.load_store(bundles, region, cfg['tree_seed'])
        rows = training_rows(store.origins, cfg)
        manifest['samples'][region] = {role: c.base.sample_manifest(store.origins, ids) for role, ids in rows.items()}
        for bundle_path in paths:
            meta = json.loads(bundle_path.with_suffix('.json').read_text())
            evidence = [bundle_path, bundle_path.with_suffix('.json'), bundle_path.parent / 'config.json']
            evidence += [bundle_path.parent / name for name in meta['model_files']]
            manifest['inputs_sha256'].update({str(p.resolve()): c.base.digest(p) for p in evidence})
        cache_manifest = PARENT / 'data_cache' / region / 'hashes.json'
        manifest['inputs_sha256'][str(cache_manifest.resolve())] = c.base.digest(cache_manifest)
    manifest['run_id'] = c.fingerprint(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def train_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    set_stage('development')
    directory = Path(output) / 'fits' / region
    directory.mkdir(parents=True, exist_ok=True)
    store, _ = c.base.load_store(bundles, region, cfg['tree_seed'])
    rows = training_rows(store.origins, cfg)
    ids = np.unique(np.concatenate(list(rows.values())))
    def progress(count):
        atomic_json(directory / 'progress.json', dict(stage='preparing_causal_training', prepared=count, count=len(ids)))
    adapted = DeviceHistoryStore(store, ids, device, progress)
    audit = adapted.verify(store, ids[np.linspace(0, len(ids)-1, min(64, len(ids)), dtype=int)])
    atomic_json(directory / 'device_audit.json', audit)
    return {f'{region}/{seed}': c.fit(adapted, rows, 'asymmetric_scale', seed, cfg,
                                    fit_path(output, region, seed), run_id) for seed in cfg['seeds']}


@torch.no_grad()
def predict_outputs(model, store, ids, batch_size, controls):
    model.eval()
    chunks = {name: [] for name in controls}
    for offset in range(0, len(ids), batch_size):
        batch = store.batch(ids[offset:offset + batch_size])
        for name, flags in controls.items():
            chunks[name].append(model(batch, **flags)['prediction'].cpu().numpy())
    return {name: np.concatenate(values) for name, values in chunks.items()}


def evaluate_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    root = Path(output)
    completed = json.loads((root / 'selection_complete.json').read_text())
    verify_barrier(completed, cfg)  # Before opening any test artifacts/cache.
    set_stage('final-test')
    directory = root / 'evaluation' / region
    directory.mkdir(parents=True, exist_ok=True)
    arrays, meta = load_cache_split(PARENT / 'data_cache', region, 'test')
    split = AlignedSplit({k: arrays[k] for k in INPUT_KEYS}, arrays['actual'], arrays['origin_unix'],
                         arrays['target_time_unix'], meta, arrays['target'])
    reference = load_prediction_artifact(PARENT / 'runs/lightgbm/seed2026' / region / 'test.npz')
    for key in ('actual', 'origin_unix', 'target_time_unix'):
        if not np.array_equal(getattr(reference, key), getattr(split, key)):
            raise ValueError(f'Canonical test mismatch: {key}')
    tree = FrozenLightGBMPredictor(bundles / 'oof' / region / '2022H2/models', seed=cfg['tree_seed'], region=region)
    baseline = tree.predict(split)
    history, _ = c.base.load_store(bundles, region, cfg['tree_seed'])
    test = bundle_from_aligned(split, baseline, timestamp('2022-01-01'), timestamp('2022-07-01'))
    ids = np.arange(len(history.origins), len(history.origins) + len(split))
    store = HistoryStore(combine_memory(history.a, test), candidates=cfg['candidates'], history_days=cfg['history_days'])
    def progress(count):
        atomic_json(directory / 'progress.json', dict(stage='preparing_causal_test', prepared=count, count=len(ids)))
    adapted = DeviceHistoryStore(store, ids, device, progress)
    audit = adapted.verify(store, ids[np.linspace(0, len(ids)-1, 64, dtype=int)])
    atomic_json(directory / 'manifest.json', dict(tree=tree.provenance, device_audit=audit,
        input_metadata=meta, canonical_test='2023-2024', n_origins=len(split),
        source_run_id=run_id, neural_seeds=cfg['seeds'], tree_seed=cfg['tree_seed']))
    result = {}
    for seed in cfg['seeds']:
        fit = fit_path(output, region, seed)
        record = completed[f'{region}/{seed}']
        for mode, controls in (('guarded', OUTPUTS), ('raw', {'Full_joint_raw': {}})):
            path = fit / f'{mode}.pt'
            model, selection = c.base.restore_checkpoint(path, model_builder=build_model)
            if selection != record['selection'][mode]:
                raise ValueError('Selection record changed')
            model.to(device)
            prediction = predict_outputs(model, adapted, ids, cfg['prediction_batch_size'], controls)
            alpha = selection['alpha']
            if mode == 'guarded':
                prediction = {k: blend(baseline, q, alpha) for k, q in prediction.items()}
                prediction['B_retrained'] = baseline
            for name, values in prediction.items():
                output_alpha = 0.0 if name == 'B_retrained' else alpha
                output_fallback = False if name == 'B_retrained' else selection['fallback']
                meta_out = dict(meta, seed=seed, tree_seed=cfg['tree_seed'], source=run_id,
                    model=name, model_version='asymmetric_balance_20261023', output_mode=mode,
                    output_type='quantile', prediction_space='raw_rrp',
                    checkpoint_sha256=c.base.digest(path), alpha=output_alpha,
                    input_scope='public_inputs' if name == 'B_retrained' else 'public_inputs_plus_mature_history_and_errors',
                    seed_scope='fixed_tree_repeated_reference' if name == 'B_retrained' else cfg['seed_scope'])
                artifact = make_prediction_artifact(actual=split.actual, origin_unix=split.origin_unix,
                    target_time_unix=split.target_time_unix, metadata=meta_out, quantiles=values, quantile_levels=LEVELS)
                assert_comparable(reference, artifact, allow_different_seed=True)
                target = root / 'runs' / name / f'seed{seed}' / region / 'test.npz'
                target.parent.mkdir(parents=True, exist_ok=True)
                save_prediction_artifact(target, artifact)
                scores = score_predictions(artifact)
                result[f'{name}/{seed}/{region}'] = dict(model=name, seed=seed, region=region,
                    output_mode=mode, alpha=output_alpha, fallback=output_fallback, epoch=selection['epoch'],
                    checkpoint_sha256=c.base.digest(path), path=str(target), sha256=c.base.digest(target), scores=scores)
            atomic_json(directory / 'progress.json', dict(stage='evaluation', seed=seed, mode=mode))
    atomic_json(directory / 'metrics.json', result)
    atomic_json(directory / 'progress.json', dict(stage='complete', outputs=len(result)))
    return result


def run(cfg, bundles, output, resume=False):
    manifest = prepare(output, cfg, bundles, resume)
    specs = [(region, cfg, bundles, str(output), manifest['run_id'], device)
             for region, device in zip(cfg['regions'], cfg['devices'])]
    atomic_json(output / 'progress.json', dict(stage='training_selection', total_fits=9))
    completed = c.run_pool(train_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    barrier = output / 'selection_complete.json'
    if barrier.exists() and json.loads(barrier.read_text()) != completed:
        raise ValueError('Frozen selection barrier changed')
    atomic_json(barrier, completed)
    atomic_json(output / 'progress.json', dict(stage='evaluation', completed_fits=9,
                                              selection_sha256=c.base.digest(barrier)))
    results = c.run_pool(evaluate_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    expected = {f'{m}/{s}/{r}' for m in ALL_OUTPUTS for s in cfg['seeds'] for r in cfg['regions']}
    if set(results) != expected:
        raise ValueError('Incomplete final test matrix')
    atomic_json(output / 'report.json', results)
    atomic_json(output / 'progress.json', dict(stage='complete', completed_fits=9, completed_artifacts=54,
        selection_sha256=c.base.digest(barrier), report_sha256=c.base.digest(output / 'report.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--devices', nargs=3, default=['cuda:3', 'cuda:6', 'cuda:7'])
    parser.add_argument('--bundle-root', default=c.base.BUNDLE_ROOT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output, bundles = Path(args.output).resolve(), Path(args.bundle_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This aligned final run already has an active runner') from None
        if output.exists() and not args.resume:
            raise FileExistsError('Output exists; use --resume')
        try:
            run(configuration(args.devices), bundles, output, args.resume)
        except Exception as error:
            if output.exists():
                atomic_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
