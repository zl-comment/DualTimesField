"""Matched development trials: selection-only and bounded calibration, separately."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ..full_v5_anchor_balance import confirmation_experiment as c
from ..full_v5_anchor_balance import experiment as base
from ..full_v5_anchor_balance.objective import balance_loss
from ..full_v5_anchor_balance.selection import blend, select_candidate, selection_key
from .model import build_model
from .selection import quarter_windows, select_windows, window_key

PROTOCOL = Path('docs/v5_calibration_robustness_20261023.md')
HISTORICAL = Path('outputs/forecasting/v5_anchor_confirmation_20261022_r1').resolve()
ARMS = ('reference', 'multiwindow', 'bounded_calibration')
TRAJECTORIES = ('reference', 'bounded_calibration')


def configuration(devices):
    cfg = c.configuration(devices)
    cfg.update(experiment='calibration_robustness_v1', variants=list(ARMS),
        trajectories=list(TRAJECTORIES), location_limit=30., scale_limit=.1,
        selection='reference/bounded: original; multiwindow: four-quarter guards and worst relative QS',
        old_reference=str(HISTORICAL), logical_fits=54, physical_fits=36)
    return cfg


def arm_path(output, region, fold, seed, arm):
    return Path(output) / region / fold / f'seed_{seed}' / arm


def fit(store, rows, trajectory, seed, cfg, output, run_id):
    output.mkdir(parents=True, exist_ok=True)
    identity = c.fingerprint(dict(run=run_id, trajectory=trajectory, seed=seed, config=cfg,
        train=rows['train'].tolist(), selection=rows['selection'].tolist()))
    complete = output / 'selection.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if record['fit_id'] != identity:
            raise ValueError('Completed fit identity changed')
        c.verify_checkpoints(record)
        return record
    variant = 'asymmetric_scale' if trajectory == 'reference' else 'bounded_calibration'
    model = build_model(variant, seed).to(store.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    best, states, log, start = {}, {}, [], 1
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
        best, states, log, start = saved['best'], saved['best_states'], saved['log'], saved['epoch']+1
    ids = rows['selection']
    actual, baseline = (store.a[k][ids] for k in ('actual', 'baseline'))
    windows = quarter_windows(store.origins[ids]) if trajectory == 'reference' else None
    started = time.monotonic()
    for epoch in range(start, cfg['epochs']+1):
        model.train()
        order, total, terms_total = rng.permutation(rows['train']), 0., {}
        epoch_started = time.monotonic()
        for offset in range(0, len(order), cfg['batch_size']):
            batch_ids = order[offset:offset+cfg['batch_size']]
            batch = store.batch(batch_ids)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss, terms = balance_loss(result, store.targets(batch_ids), batch['baseline'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite robustness objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch_ids)
            for name, value in terms.items():
                terms_total[name] = terms_total.get(name, 0.) + float(value.detach()) * len(batch_ids)
            if offset == 0 or (offset // cfg['batch_size']+1) % 50 == 0:
                base.write_json(output / 'progress.json', dict(stage='training_selection', epoch=epoch,
                    epochs=cfg['epochs'], processed=offset+len(batch_ids), count=len(order),
                    elapsed_seconds=time.monotonic()-started))
        prediction = base.predict(model, store, ids, cfg['prediction_batch_size'])
        candidates = dict(raw=dict(epoch=epoch, alpha=1., fallback=False,
            **base.diagnostics(actual, baseline, prediction)),
            guarded=select_candidate(actual, baseline, prediction['anchor'], prediction['full'],
                                     alphas=cfg['alphas'], epoch=epoch))
        if trajectory == 'reference':
            candidates['multiwindow'] = select_windows(actual, baseline, prediction['anchor'],
                prediction['full'], windows, cfg['alphas'], epoch=epoch)
        for mode, candidate in candidates.items():
            key = (lambda row: row['full']['qs']) if mode == 'raw' else (
                window_key if mode == 'multiwindow' else selection_key)
            if mode not in best or key(candidate) < key(best[mode]):
                best[mode], states[mode] = candidate, c.cpu_state(model)
        log.append(dict(epoch=epoch, loss=total/len(order),
            terms={k: v/len(order) for k, v in terms_total.items()}, **candidates,
            order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-epoch_started))
        c.save_torch(resume_path, dict(fit_id=identity, model=c.cpu_state(model), optimizer=optimizer.state_dict(),
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(store.device) if store.device.type == 'cuda' else None,
            epoch=epoch, best=best, best_states=states, log=log))
        base.write_json(output / 'training.json', log)
        print(f'{output}: epoch={epoch}/{cfg["epochs"]} loss={log[-1]["loss"]:.6f} '
              f'alpha={candidates["guarded"]["alpha"]}' +
              (f' window_alpha={candidates["multiwindow"]["alpha"]}' if trajectory == 'reference' else ''), flush=True)
    base.write_json(output / 'training.json', log)
    for mode, selected in best.items():
        c.save_torch(output / f'{mode}.pt', dict(state_dict=states[mode], variant=variant,
                                              seed=seed, selection=selected))
    record = dict(fit_id=identity, trajectory=trajectory, model_variant=variant, selection=best,
        epochs_completed=len(log), parameters=c.parameter_counts(model),
        checkpoints={str(output / f'{mode}.pt'): base.digest(output / f'{mode}.pt') for mode in best})
    base.write_json(complete, record)
    base.write_json(output / 'progress.json', dict(stage='selection_complete', epochs_completed=len(log)))
    return record


def verify_reference(directory, old_directory):
    """Old predictions are never used for selection: this only audits replay."""
    new, old = [json.loads((p / 'training.json').read_text()) for p in (directory, old_directory)]
    if len(new) != len(old):
        raise ValueError('Reference replay budget mismatch')
    for a, b in zip(new, old):
        for key in ('epoch', 'order_sha256', 'loss', 'terms', 'raw', 'guarded'):
            if a[key] != b[key]:
                raise ValueError(f'Reference replay mismatch: {directory}, epoch={a["epoch"]}, {key}')
    for mode in ('raw', 'guarded', 'last'):
        a, b = [torch.load(p / f'{mode}.pt', map_location='cpu', weights_only=True)
                for p in (directory, old_directory)]
        key = 'model' if mode == 'last' else 'state_dict'
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    result = dict(exact=True, epochs=len(new), all_losses_terms_orders_and_old_selections_exact=True,
                  raw_guarded_and_final_parameters_bitwise_exact=True, old_directory=str(old_directory))
    base.write_json(directory / 'reference_replay.json', result)
    return result


def train_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    directory = Path(output) / region
    directory.mkdir(exist_ok=True)
    store, _ = base.load_store(bundles, region, cfg['tree_seed'])
    adapted, splits, audit = c.device_store(store, cfg['folds'], device, directory / 'progress.json',
                                           ('train', 'selection'))
    base.write_json(directory / 'device_training_audit.json', audit)
    result = {}
    for fold, rows in splits.items():
        for seed in cfg['seeds']:
            for trajectory in TRAJECTORIES:
                path = arm_path(output, region, fold, seed, trajectory)
                base.write_json(directory / 'progress.json', dict(stage='training_selection', fit=str(path)))
                record = fit(adapted, rows, trajectory, seed, cfg, path, run_id)
                if trajectory == 'reference':
                    verify_reference(path, arm_path(cfg['old_reference'], region, fold, seed, 'asymmetric_scale'))
                for arm in (('reference', 'multiwindow') if trajectory == 'reference' else (trajectory,)):
                    mode = 'multiwindow' if arm == 'multiwindow' else 'guarded'
                    result[f'{region}/{fold}/{seed}/{arm}'] = dict(
                        fit_id=record['fit_id'], trajectory=trajectory, epochs_completed=record['epochs_completed'],
                        selection=dict(raw=record['selection']['raw'], guarded=record['selection'][mode]),
                        checkpoint_paths=dict(raw=str(path / 'raw.pt'), guarded=str(path / f'{mode}.pt')),
                        checkpoints=record['checkpoints'])
    base.write_json(directory / 'progress.json', dict(stage='selection_complete', logical_fits=len(result)))
    return result


def expected_keys(cfg):
    return {f'{r}/{f}/{s}/{v}' for r in cfg['regions'] for f in cfg['folds']
            for s in cfg['seeds'] for v in ARMS}


def verify_barrier(records, cfg):
    if set(records) != expected_keys(cfg):
        raise ValueError('All robustness fits must finish before evaluation')
    if len({r['fit_id'] for r in records.values()}) != 2*len(records)//3:
        raise ValueError('Expected two physical trajectories per three logical arms')
    for record in records.values():
        if record['epochs_completed'] != cfg['epochs']:
            raise ValueError('Incomplete epoch budget')
        c.verify_checkpoints(record)
        for path in record['checkpoint_paths'].values():
            if path not in record['checkpoints']:
                raise ValueError('Selected checkpoint missing from frozen hashes')
    for r in cfg['regions']:
        for f in cfg['folds']:
            for s in cfg['seeds']:
                a, b = (records[f'{r}/{f}/{s}/{arm}'] for arm in ('reference', 'multiwindow'))
                if a['fit_id'] != b['fit_id'] or a['checkpoint_paths']['raw'] != b['checkpoint_paths']['raw']:
                    raise ValueError('Selection-only arm must share the exact reference trajectory')


def evaluate_fit(store, rows, arm, cfg, path, record):
    c.verify_checkpoints(record)
    path.mkdir(parents=True, exist_ok=True)
    ids = rows['evaluation']
    origins, actual, baseline = store.origins[ids], store.a['actual'][ids], store.a['baseline'][ids]
    groups = base.evaluation_groups(origins, baseline)
    counts = {g: dict(n_origins=int(m.sum()), n_cells=int(m.sum()*24)) for g, m in groups.items()}
    result = dict(groups=counts, B={g: dict(**counts[g], **c.score(actual[m], baseline[m]))
                                   for g, m in groups.items() if m.any()})
    for mode in ('raw', 'guarded'):
        model, selected = base.restore_checkpoint(Path(record['checkpoint_paths'][mode]), model_builder=build_model)
        if selected != record['selection'][mode]:
            raise ValueError('Frozen checkpoint selection changed')
        model.to(store.device)
        raw = base.predict(model, store, ids, cfg['prediction_batch_size'])
        chosen = {k: blend(baseline, q, selected['alpha']) for k, q in raw.items()}
        result[mode] = dict(selection=selected,
            metrics={g: base.diagnostics(actual[m], baseline[m], {k: q[m] for k, q in chosen.items()})
                     for g, m in groups.items() if m.any()},
            checkpoint_unblended=base.diagnostics(actual, baseline, raw),
            shape=c.shape_diagnostics(actual, baseline, chosen['anchor'], chosen['full']))
        temporary = path / f'{arm}_{mode}.tmp.npz'
        np.savez_compressed(temporary, origin_unix=origins,
            target_time_unix=origins[:, None]+np.arange(24)[None]*base.HOUR,
            actual=actual, baseline=baseline, anchor=chosen['anchor'], quantiles=chosen['full'],
            raw_anchor=raw['anchor'], raw_quantiles=raw['full'], quantile_levels=base.LEVELS,
            alpha=selected['alpha'], epoch=selected['epoch'])
        temporary.replace(path / f'{arm}_{mode}.npz')
    base.write_json(path / 'metrics.json', result)
    return result


def evaluate_region(spec):
    region, cfg, bundles, output, run_id, device = spec
    c.configure_device(device)
    records = json.loads((Path(output) / 'selection_complete.json').read_text())
    verify_barrier(records, cfg)
    directory = Path(output) / region
    store, _ = base.load_store(bundles, region, cfg['tree_seed'])
    adapted, splits, audit = c.device_store(store, cfg['folds'], device, directory / 'progress.json', ('evaluation',))
    base.write_json(directory / 'device_evaluation_audit.json', audit)
    results = {}
    for fold, rows in splits.items():
        for seed in cfg['seeds']:
            for arm in ARMS:
                key = f'{region}/{fold}/{seed}/{arm}'
                base.write_json(directory / 'progress.json', dict(stage='evaluation', fit=key))
                results[key] = evaluate_fit(adapted, rows, arm, cfg,
                    arm_path(output, region, fold, seed, arm), records[key])
    base.write_json(directory / 'progress.json', dict(stage='complete', evaluated_fits=len(results)))
    return results


def verify_manifest(manifest):
    if c.fingerprint({k: v for k, v in manifest.items() if k != 'run_id'}) != manifest['run_id']:
        raise ValueError('Manifest identity changed')
    for kind in ('sources', 'inputs', 'historical'):
        if any(base.digest(p) != value for p, value in manifest[kind].items()):
            raise ValueError(f'Frozen {kind} changed')
    if base.digest(PROTOCOL) != manifest['protocol_sha256']:
        raise ValueError('Frozen protocol changed')


def prepare(output, cfg, bundles, resume):
    manifest_path = output / 'manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != cfg or manifest['bundle_root'] != str(bundles):
            raise ValueError('Resume configuration changed')
        verify_manifest(manifest)
        return manifest
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(p for package in ('full_v5_calibration_robustness', 'full_v5_anchor_balance', 'full_v5_joint', 'full_v5')
                     for p in (Path('forecasting') / package).glob('*.py'))
    sources.append(Path('scripts/diagnose_anchor_robustness.py'))
    manifest = dict(config=cfg, bundle_root=str(bundles), scope='retrospective_development_only',
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        protocol_sha256=base.digest(PROTOCOL), sources={}, inputs={}, historical={}, samples={})
    for source in sources:
        target = output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest['sources'][str(source.resolve())] = base.digest(source)
    shutil.copyfile(PROTOCOL, output / 'protocol.md')
    for region in cfg['regions']:
        store, paths = base.load_store(bundles, region, cfg['tree_seed'])
        for bundle_path in paths:
            meta = json.loads(bundle_path.with_suffix('.json').read_text())
            evidence = [bundle_path, bundle_path.with_suffix('.json'), bundle_path.parent / 'config.json']
            evidence.extend(bundle_path.parent / name for name in meta['model_files'])
            manifest['inputs'].update({str(p.resolve()):base.digest(p) for p in evidence})
        for fold in cfg['folds']:
            rows = base.split_rows(store.origins, fold, 'full')
            manifest['samples'][f'{region}/{fold}'] = {role:base.sample_manifest(store.origins, ids)
                                                      for role, ids in rows.items()}
            quarter_windows(store.origins[rows['selection']])
            for seed in cfg['seeds']:
                for variant in ('asymmetric_scale', 'balanced'):
                    old = arm_path(cfg['old_reference'], region, fold, seed, variant)
                    evidence = [old/name for name in ('training.json','selection.json','raw.pt','guarded.pt','last.pt','metrics.json')]
                    manifest['historical'].update({str(p.resolve()):base.digest(p) for p in evidence})
        del store
    # Preserve the entire tracked 72-fit failure archive as well as replay sources.
    archive = Path('logs/forecasting/v5_anchor_confirmation_20261022')
    manifest['historical'].update({str(p.resolve()):base.digest(p) for p in archive.rglob('*') if p.is_file()})
    manifest['run_id'] = c.fingerprint(manifest)
    base.write_json(manifest_path, manifest)
    return manifest


def run(cfg, bundles, output, resume=False):
    manifest = prepare(output, cfg, bundles, resume)
    specs = [(region, cfg, str(bundles), str(output), manifest['run_id'], device)
             for region, device in zip(cfg['regions'], cfg['devices'])]
    logical_count = len(expected_keys(cfg))
    physical_count = 2*logical_count//3
    base.write_json(output / 'progress.json', dict(stage='training_selection',
        physical_fits=physical_count, logical_fits=logical_count))
    completed = c.run_pool(train_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    barrier = output / 'selection_complete.json'
    if barrier.exists() and json.loads(barrier.read_text()) != completed:
        raise ValueError('Frozen selection barrier changed')
    base.write_json(barrier, completed)
    base.write_json(output / 'progress.json', dict(stage='evaluation', logical_fits=len(completed),
        physical_fits=len({r['fit_id'] for r in completed.values()}), selection_sha256=base.digest(barrier)))
    results = c.run_pool(evaluate_region, specs)
    verify_manifest(manifest)
    verify_barrier(completed, cfg)
    if set(results) != expected_keys(cfg):
        raise ValueError('Incomplete evaluation matrix')
    base.write_json(output / 'report.json', results)
    from .report import write_report
    write_report(output, results, cfg)
    base.write_json(output / 'progress.json', dict(stage='complete',
        physical_fits=physical_count, logical_fits=logical_count,
        selection_sha256=base.digest(barrier), report_sha256=base.digest(output / 'report.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bundle-root', default=base.BUNDLE_ROOT)
    parser.add_argument('--devices', nargs=3, default=['cuda:3','cuda:6','cuda:7'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args(argv)
    if not args.development:
        raise ValueError('--development required; final-test training/evaluation unsupported')
    output, bundles = Path(args.output).resolve(), Path(args.bundle_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This robustness output already has an active runner') from None
        if output.exists() and not args.resume:
            raise FileExistsError('Output exists; use --resume')
        try:
            run(configuration(args.devices), bundles, output, args.resume)
        except Exception as error:
            if output.exists():
                base.write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
