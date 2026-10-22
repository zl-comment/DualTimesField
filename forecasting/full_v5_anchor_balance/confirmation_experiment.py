"""Full-hourly, three-seed Anchor confirmation with recoverable epoch commits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from . import experiment as base
from .architecture_experiment import parameter_counts
from .confirmation_data import DeviceHistoryStore
from .confirmation_model import VARIANTS, build_model
from .objective import BalanceWeights, balance_loss
from .selection import blend, score, select_candidate, selection_key

PROTOCOL = Path('docs/v5_anchor_confirmation_20261022.md')
REGIONS = ('NSW1', 'QLD1', 'TAS1')
SEEDS = (2026, 2027, 2028)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configure_device(device):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)


def configuration(devices):
    if len(devices) != 3:
        raise ValueError('Supply one device per region (three total)')
    return dict(experiment='anchor_confirmation_v1', profile='full', epochs=20,
        batch_size=128, prediction_batch_size=256, cpu_threads=2, learning_rate=.001,
        weight_decay=1e-4, grad_clip=5., weights=asdict(BalanceWeights()),
        reconstruction_weight=.05, gate_penalty=.001, alphas=[0, .25, .5, 1],
        variants=list(VARIANTS), seeds=list(SEEDS), tree_seed=2026,
        candidates=16, history_days=730, regions=list(REGIONS), folds=list(base.FOLDS),
        devices=list(devices), sampling='all hourly origins',
        stopping='fixed 20 epochs for every arm; no early stopping',
        precision='float32 network, original float64 historical solves; TF32 disabled',
        objective='same balance_loss for all four arms',
        selection='raw: minimum selection Full QS; guarded: original four guards and shared alpha',
        acceptance='original macro/region/spring gates; seed-mean and every seed reported separately')


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(store, rows, variant, seed, cfg, output, run_id):
    """A last.pt is the epoch commit. Partial subsequent epochs are replayed.

    Selected model states live in that same atomic commit, avoiding a mismatch
    between raw/guarded checkpoints and optimizer state after interruption.
    Evaluation labels are never read here.
    """
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(dict(run=run_id, variant=variant, seed=seed,
                                train=rows['train'].tolist(), selection=rows['selection'].tolist()))
    complete = output / 'selection.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['fit_id'] != identity:
            raise ValueError('Completed fit identity changed')
        verify_checkpoints(saved)
        return saved
    model = build_model(variant, seed).to(store.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    best_raw = best_guard = None
    best_states, log, start = {}, [], 1
    resume_path = output / 'last.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['fit_id'] != identity:
            raise ValueError('Resume fit identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if store.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], store.device)
        best_raw, best_guard = saved['raw'], saved['guarded']
        best_states, log, start = saved['best_states'], saved['log'], saved['epoch'] + 1
    ids = rows['selection']
    actual, baseline = (store.a[k][ids] for k in ('actual', 'baseline'))
    started = time.monotonic()
    for epoch in range(start, cfg['epochs'] + 1):
        model.train()
        order, total, terms_total = rng.permutation(rows['train']), 0., {}
        epoch_started = time.monotonic()
        for offset in range(0, len(order), cfg['batch_size']):
            batch_ids = order[offset:offset + cfg['batch_size']]
            batch = store.batch(batch_ids)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss, terms = balance_loss(result, store.targets(batch_ids), batch['baseline'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite confirmation objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch_ids)
            for name, value in terms.items():
                terms_total[name] = terms_total.get(name, 0.) + float(value.detach()) * len(batch_ids)
            if offset == 0 or (offset // cfg['batch_size'] + 1) % 50 == 0:
                base.write_json(output / 'progress.json', dict(stage='training_selection', epoch=epoch,
                    epochs=cfg['epochs'], processed=offset + len(batch_ids), count=len(order),
                    elapsed_seconds=time.monotonic() - started))
        prediction = base.predict(model, store, ids, cfg['prediction_batch_size'])
        raw = dict(epoch=epoch, alpha=1., fallback=False, **base.diagnostics(actual, baseline, prediction))
        guarded = select_candidate(actual, baseline, prediction['anchor'], prediction['full'],
                                   alphas=cfg['alphas'], epoch=epoch)
        if best_raw is None or raw['full']['qs'] < best_raw['full']['qs']:
            best_raw, best_states['raw'] = raw, cpu_state(model)
        if best_guard is None or selection_key(guarded) < selection_key(best_guard):
            best_guard, best_states['guarded'] = guarded, cpu_state(model)
        log.append(dict(epoch=epoch, loss=total / len(order),
            terms={k: v / len(order) for k, v in terms_total.items()}, raw=raw, guarded=guarded,
            order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            elapsed_seconds=time.monotonic() - epoch_started))
        save_torch(resume_path, dict(fit_id=identity, model=cpu_state(model), optimizer=optimizer.state_dict(),
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(store.device) if store.device.type == 'cuda' else None,
            epoch=epoch, raw=best_raw, guarded=best_guard, best_states=best_states, log=log))
        base.write_json(output / 'training.json', log)
        print(f'{output}: epoch={epoch}/{cfg["epochs"]} loss={log[-1]["loss"]:.6f} '
              f'selection_QS={raw["full"]["qs"]:.6f} alpha={guarded["alpha"]}', flush=True)
    # Recover training.json too if interruption followed last.pt's atomic rename.
    base.write_json(output / 'training.json', log)
    for mode, selection in (('raw', best_raw), ('guarded', best_guard)):
        save_torch(output / f'{mode}.pt', dict(state_dict=best_states[mode], variant=variant,
                                              seed=seed, selection=selection))
    record = dict(fit_id=identity, selection=dict(raw=best_raw, guarded=best_guard),
                  parameters=parameter_counts(model), epochs_completed=len(log),
                  checkpoints={str(output / f'{mode}.pt'): base.digest(output / f'{mode}.pt')
                               for mode in ('raw', 'guarded')})
    base.write_json(complete, record)
    base.write_json(output / 'progress.json', dict(stage='selection_complete', epochs_completed=len(log)))
    return record


def verify_checkpoints(record):
    if any(base.digest(p) != expected for p, expected in record['checkpoints'].items()):
        raise ValueError('Selected checkpoint changed')


def arm_path(output, region, fold, seed, variant):
    return Path(output) / region / fold / f'seed_{seed}' / variant


def device_store(store, folds, device, progress_path, roles):
    splits = {fold: base.split_rows(store.origins, fold, 'full') for fold in folds}
    rows = np.unique(np.concatenate([parts[role] for parts in splits.values() for role in roles]))
    def progress(count):
        base.write_json(progress_path, dict(stage='preparing_causal_records', prepared=count, count=len(rows)))
    adapted = DeviceHistoryStore(store, rows, device, progress)
    audit_rows = rows[np.linspace(0, len(rows) - 1, min(64, len(rows)), dtype=int)]
    audit = adapted.verify(store, audit_rows)
    return adapted, splits, audit


def train_region(spec):
    region, cfg, bundle_root, output, run_id, device = spec
    configure_device(device)
    directory = Path(output) / region
    directory.mkdir(exist_ok=True)
    store, _ = base.load_store(bundle_root, region, cfg['tree_seed'])
    adapted, splits, audit = device_store(store, cfg['folds'], device, directory / 'progress.json',
                                          ('train', 'selection'))
    base.write_json(directory / 'device_training_audit.json', audit)
    completed = {}
    for fold, rows in splits.items():
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                key = f'{region}/{fold}/{seed}/{variant}'
                base.write_json(directory / 'progress.json', dict(stage='training_selection', fit=key))
                completed[key] = fit(adapted, rows, variant, seed, cfg,
                    arm_path(output, region, fold, seed, variant), run_id)
    base.write_json(directory / 'progress.json', dict(stage='selection_complete', fits=len(completed)))
    return completed


def shape_diagnostics(actual, baseline, anchor, full):
    """Physical-unit diagnostics; no division by possibly zero tree gaps."""
    b, a, f, y = (np.asarray(x, dtype=np.float64) for x in (baseline, anchor, full, actual))
    delta, correction = a[..., 2] - b[..., 2], f[..., 2] - a[..., 2]
    denom = np.abs(delta) + np.abs(correction)
    result = dict(mean_abs_delta=float(np.abs(delta).mean()), mean_abs_correction=float(np.abs(correction).mean()),
        opposing_fraction=float(((delta * correction) < 0).mean()),
        cancelled_magnitude=float((denom - np.abs(delta + correction)).mean()),
        cancellation_fraction=float((denom.sum() - np.abs(delta + correction).sum()) / denom.sum()) if denom.sum() else 0.)
    for name, q in (('baseline', b), ('anchor', a), ('full', f)):
        error = y[..., None] - q
        pb = np.maximum(np.asarray(base.LEVELS) * error, (np.asarray(base.LEVELS) - 1) * error).mean(axis=(0, 1))
        result[name] = dict(pinball_by_quantile=pb.tolist(),
            lower90_width=float((q[..., 2] - q[..., 0]).mean()),
            upper90_width=float((q[..., 4] - q[..., 2]).mean()),
            width80=float((q[..., 3] - q[..., 1]).mean()), width90=float((q[..., 4] - q[..., 0]).mean()),
            under_q05=float((y < q[..., 0]).mean()), over_q95=float((y > q[..., 4]).mean()))
    return result


def evaluate_fit(store, rows, variant, cfg, output, record):
    verify_checkpoints(record)
    ids = rows['evaluation']
    origins, actual, baseline = store.origins[ids], store.a['actual'][ids], store.a['baseline'][ids]
    groups = base.evaluation_groups(origins, baseline)
    counts = {g: dict(n_origins=int(m.sum()), n_cells=int(m.sum() * 24)) for g, m in groups.items()}
    result = dict(groups=counts, B={g: dict(**counts[g], **score(actual[m], baseline[m]))
                                  for g, m in groups.items() if m.any()})
    for mode in ('raw', 'guarded'):
        model, selection = base.restore_checkpoint(output / f'{mode}.pt', model_builder=build_model)
        model.to(store.device)
        raw = base.predict(model, store, ids, cfg['prediction_batch_size'])
        chosen = {k: blend(baseline, q, selection['alpha']) for k, q in raw.items()}
        result[mode] = dict(selection=selection,
            metrics={g: base.diagnostics(actual[m], baseline[m], {k: q[m] for k, q in chosen.items()})
                     for g, m in groups.items() if m.any()},
            checkpoint_unblended=base.diagnostics(actual, baseline, raw),
            shape=shape_diagnostics(actual, baseline, chosen['anchor'], chosen['full']),
            shape_unblended=shape_diagnostics(actual, baseline, raw['anchor'], raw['full']))
        path = output / f'{variant}_{mode}.npz'
        temporary = output / f'{variant}_{mode}.tmp.npz'
        np.savez_compressed(temporary, origin_unix=origins,
            target_time_unix=origins[:, None] + np.arange(24)[None] * base.HOUR,
            actual=actual, baseline=baseline, anchor=chosen['anchor'], quantiles=chosen['full'],
            raw_anchor=raw['anchor'], raw_quantiles=raw['full'], quantile_levels=base.LEVELS,
            alpha=selection['alpha'], epoch=selection['epoch'])
        temporary.replace(path)
    base.write_json(output / 'metrics.json', result)
    base.write_json(output / 'progress.json', dict(stage='complete', evaluated_origins=len(ids)))
    return result


def expected_keys(cfg):
    return {f'{r}/{f}/{s}/{v}' for r in cfg['regions'] for f in cfg['folds']
            for s in cfg['seeds'] for v in cfg['variants']}


def verify_barrier(completed, cfg):
    if set(completed) != expected_keys(cfg):
        raise ValueError('All confirmation fits must finish before evaluation')
    for record in completed.values():
        if record['epochs_completed'] != cfg['epochs']:
            raise ValueError('Incomplete epoch budget')
        verify_checkpoints(record)


def evaluate_region(spec):
    region, cfg, bundle_root, output, run_id, device = spec
    configure_device(device)
    completed = json.loads((Path(output) / 'selection_complete.json').read_text())
    verify_barrier(completed, cfg)
    directory = Path(output) / region
    store, _ = base.load_store(bundle_root, region, cfg['tree_seed'])
    adapted, splits, audit = device_store(store, cfg['folds'], device, directory / 'progress.json', ('evaluation',))
    base.write_json(directory / 'device_evaluation_audit.json', audit)
    result = {}
    for fold, rows in splits.items():
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                key = f'{region}/{fold}/{seed}/{variant}'
                base.write_json(directory / 'progress.json', dict(stage='evaluation', fit=key))
                result[key] = evaluate_fit(adapted, rows, variant, cfg,
                    arm_path(output, region, fold, seed, variant), completed[key])
    base.write_json(directory / 'progress.json', dict(stage='complete', evaluated_fits=len(result)))
    return result


def verify_manifest(manifest):
    for key in ('source_sha256', 'bundles', 'bundle_provenance_sha256'):
        if any(base.digest(p) != sha for p, sha in manifest[key].items()):
            raise ValueError(f'Frozen {key} changed; run cannot continue')
    if base.digest(PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')


def prepare(output, cfg, bundle_root, resume=False):
    manifest_path = output / 'manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if fingerprint({k: v for k, v in manifest.items() if k != 'run_id'}) != manifest['run_id']:
            raise ValueError('Resume manifest identity changed')
        if manifest['config'] != cfg or manifest['bundle_root'] != str(Path(bundle_root).resolve()):
            raise ValueError('Resume configuration changed')
        verify_manifest(manifest)
        return manifest
    output.mkdir(parents=True, exist_ok=False)
    source = sorted(p for package in ('full_v5_anchor_balance', 'full_v5_joint', 'full_v5')
                    for p in (Path('forecasting') / package).glob('*.py'))
    for path in source:
        target = output / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(PROTOCOL, output / 'protocol.md')
    (output / 'source.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], text=True))
    manifest = dict(scope='retrospective_development_not_final_test', config=cfg,
        bundle_root=str(Path(bundle_root).resolve()), folds=base.FOLDS,
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): base.digest(p) for p in source}, protocol_sha256=base.digest(PROTOCOL),
        diff_sha256=base.digest(output / 'source.diff'), bundles={}, bundle_provenance_sha256={},
        samples={}, oof_stage_mapping={},
        availability_audit='archive_timestamp_proxies_only',
        evaluation_policy='all 72 training/selection fits finish and hash-lock before any evaluation',
        runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda))
    base.write_json(manifest_path, manifest)
    for region in cfg['regions']:
        store, paths = base.load_store(bundle_root, region, cfg['tree_seed'])
        manifest['bundles'].update({str(p): base.digest(p) for p in paths})
        for path in paths:
            metadata = json.loads(path.with_suffix('.json').read_text())
            evidence = [path.with_suffix('.json'), path.parent / 'config.json']
            evidence += [path.parent / name for name in metadata['model_files']]
            manifest['bundle_provenance_sha256'].update({str(p): base.digest(p) for p in evidence})
        for fold in cfg['folds']:
            rows = base.split_rows(store.origins, fold, 'full')
            key = f'{region}/{fold}'
            manifest['samples'][key] = {role: base.sample_manifest(store.origins, ids) for role, ids in rows.items()}
            manifest['oof_stage_mapping'][key] = {role: base.provenance_manifest(store, ids, paths) for role, ids in rows.items()}
        del store
    manifest['run_id'] = fingerprint(manifest)
    base.write_json(manifest_path, manifest)
    return manifest


def run_pool(worker, specs):
    result = {}
    with ProcessPoolExecutor(max_workers=len(specs), mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker, spec) for spec in specs]):
            result.update(future.result())
    return result


def run(cfg, bundle_root, output, resume=False):
    output = Path(output).resolve()
    manifest = prepare(output, cfg, bundle_root, resume)
    specs = [(region, cfg, bundle_root, str(output), manifest['run_id'], device)
             for region, device in zip(cfg['regions'], cfg['devices'])]
    base.write_json(output / 'progress.json', dict(stage='training_selection', total_fits=len(expected_keys(cfg))))
    completed = run_pool(train_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    barrier = output / 'selection_complete.json'
    if barrier.exists() and json.loads(barrier.read_text()) != completed:
        raise ValueError('Frozen selection barrier changed')
    base.write_json(barrier, completed)
    base.write_json(output / 'progress.json', dict(stage='evaluation', completed_fits=len(completed)))
    results = run_pool(evaluate_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    if set(results) != expected_keys(cfg):
        raise ValueError('Missing evaluation fits')
    base.write_json(output / 'report.json', results)
    from .confirmation_report import write_report
    write_report(output, results, cfg)
    base.write_json(output / 'progress.json', dict(stage='complete', completed_fits=len(completed),
        selection_barrier_sha256=base.digest(barrier), report_sha256=base.digest(output / 'report.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bundle-root', default=base.BUNDLE_ROOT)
    parser.add_argument('--devices', nargs=3, default=['cuda:3', 'cuda:6', 'cuda:7'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args(argv)
    if not args.development:
        raise ValueError('--development required; no final-test support')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This confirmation output already has an active runner') from None
        if output.exists() and not args.resume:
            raise FileExistsError('Output exists; use --resume with the frozen configuration')
        try:
            run(configuration(args.devices), args.bundle_root, output, args.resume)
        except Exception as error:
            if output.exists():
                base.write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
