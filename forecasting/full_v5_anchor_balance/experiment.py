"""Predeclared, purged development comparisons; never loads final-test data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import torch

from ..full_v5_joint.data import HistoryStore, digest
from ..full_v5_joint.model import LEVELS, training_loss
from ..full_v5_joint.protocol import DAY, HOUR, assert_forecast_provenance, segment_rows, timestamp
from .objective import BalanceWeights, balance_loss, build_model
from .selection import blend, score, select_candidate, selection_key

VARIANTS = ('legacy', 'balanced', 'identity_anchor')
FOLDS = {'2021': ('2018-01-01', '2020-01-01', '2021-01-01', '2022-01-01'),
         '2022': ('2018-01-01', '2021-01-01', '2022-01-01', '2023-01-01')}
BUNDLE_ROOT = 'outputs/forecasting/v5_joint_history_20261019_full_r1'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def local_dates(origins):
    return (np.asarray(origins) + 10 * HOUR).astype('datetime64[s]')


def sample_rows(origins, rows, profile, role):
    if profile not in ('screen', 'full') or role not in ('train', 'selection', 'evaluation'):
        raise ValueError('Unknown profile or sample role')
    rows = np.asarray(rows, dtype=np.int64)
    if profile == 'full':
        return rows.copy()
    local = np.asarray(origins)[rows] + 10 * HOUR
    days, hours = local // DAY, (local % DAY) // HOUR
    keep = hours == days % 24 if role == 'train' else hours % 6 == days % 6
    return rows[keep]


def split_rows(origins, fold, profile):
    origins = np.asarray(origins)
    if np.any(origins >= timestamp('2023-01-01')):
        raise ValueError('Origins >=2023 are forbidden in development')
    edges = FOLDS[fold]
    rows = {role: sample_rows(origins, segment_rows(origins, start, end), profile, role)
            for role, start, end in zip(('train', 'selection', 'evaluation'), edges, edges[1:])}
    if any(not len(r) for r in rows.values()):
        raise ValueError('Empty training/selection/evaluation segment')
    for left, right in (('train', 'selection'), ('selection', 'evaluation')):
        if origins[rows[left]].max() + DAY >= origins[rows[right]].min():
            raise ValueError('Labels overlap next segment origins')
    return rows


def sample_manifest(origins, rows):
    selected = np.asarray(origins)[rows]
    dates = local_dates(selected)
    months, counts = np.unique(dates.astype('datetime64[M]').astype(str), return_counts=True)
    hours, hour_counts = np.unique(((selected + 10 * HOUR) % DAY) // HOUR, return_counts=True)
    return dict(count=len(rows), origin_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                first_local=str(dates.min()), last_local=str(dates.max()),
                months=dict(zip(months.tolist(), counts.tolist())),
                hours=dict(zip(hours.astype(str).tolist(), hour_counts.tolist())))


def provenance_manifest(store, rows, paths):
    """Map sampled stage rows back to archived OOF cutoffs, independently of labels."""
    origins = store.origins[rows]
    trained, selected = (store.a[k][rows] for k in ('trained_before', 'selected_before'))
    assert_forecast_provenance(origins, trained, selected)
    catalog = []
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        catalog.append(dict(fold=path.parent.name, bundle=str(path),
                            trained_before=meta['trained_through'], selected_before=meta['selected_through']))
    groups = []
    pairs, indices = np.unique(np.stack((trained, selected), axis=-1), axis=0, return_inverse=True)
    for i, (train_cutoff, select_cutoff) in enumerate(pairs):
        times = origins[indices == i]
        matches = [c['fold'] for c in catalog if (c['trained_before'], c['selected_before']) ==
                   (train_cutoff, select_cutoff)]
        if paths and not matches:
            raise ValueError('Used row cutoff pair is absent from the seven OOF manifests')
        groups.append(dict(oof_folds=matches, trained_before=int(train_cutoff), selected_before=int(select_cutoff),
                           n_origins=len(times), first_origin=int(times.min()), last_origin=int(times.max()),
                           first_local=str(local_dates(times).min()), last_local=str(local_dates(times).max())))
    return dict(n_origins=len(rows), every_row_cutoffs_strictly_before_origin=True,
                original_oof_catalog=catalog, cutoff_groups=groups)


def load_store(root, region, seed):
    labels = [str(y) for y in range(2017, 2023)] + ['2022H2']
    paths = [Path(root) / 'oof' / region / label / 'bundle.npz' for label in labels]
    metadata = [json.loads(path.with_suffix('.json').read_text()) for path in paths]
    expected = [(y, False) for y in range(2017, 2023)] + [(2022, True)]
    if [(m.get('year'), m.get('final_half')) for m in metadata] != expected:
        raise ValueError('Exactly the ordered seven 2017..2022H2 OOF folds are required')
    if any(m.get('region') != region or m.get('seed') != seed for m in metadata):
        raise ValueError('OOF region/seed provenance mismatch')
    store = HistoryStore.from_paths(paths, candidates=16, history_days=730)
    if np.any(store.origins >= timestamp('2023-01-01')):
        raise ValueError('Origins >=2023 are forbidden in development')
    if np.any(store.origins < timestamp('2017-01-01')):
        raise ValueError('OOF origins before 2017 are forbidden')
    return store, paths


@torch.no_grad()
def predict(model, store, rows, batch_size):
    model.eval()
    chunks = {'anchor': [], 'full': []}
    for offset in range(0, len(rows), batch_size):
        result = model(store.batch(rows[offset:offset + batch_size]))
        for name, key in (('anchor', 'anchor'), ('full', 'prediction')):
            chunks[name].append(result[key].cpu().numpy())
    return {name: np.concatenate(values) for name, values in chunks.items()}


def save_checkpoint(path, model, variant, seed, selection):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(state_dict=model.state_dict(), variant=variant, seed=seed,
                    selection=selection), temporary)
    temporary.replace(path)


def restore_checkpoint(path, model_builder=build_model):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = model_builder(saved['variant'], saved['seed'])
    model.load_state_dict(saved['state_dict'])
    return model, saved['selection']


def diagnostics(actual, baseline, predictions):
    base = score(actual, baseline)
    anchor, full = (score(actual, predictions[k]) for k in ('anchor', 'full'))
    gap = {m: dict(anchor_minus_B=anchor[m] - base[m],
                   full_minus_anchor=full[m] - anchor[m], full_minus_B=full[m] - base[m])
           for m in ('mae', 'qs')}
    return dict(n_origins=len(actual), n_cells=int(actual.size), baseline=base, anchor=anchor, full=full, gap=gap)


def fit(store, rows, variant, cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    model = build_model(variant, cfg['seed'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=.001, weight_decay=1e-4)
    rng = np.random.default_rng(cfg['seed'])
    selection_rows = rows['selection']
    actual, baseline = (store.a[k][selection_rows] for k in ('actual', 'baseline'))
    best_raw, best_guard, log = None, None, []
    started = time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order, total, terms_total = rng.permutation(rows['train']), 0., {}
        for offset in range(0, len(order), cfg['batch_size']):
            ids = order[offset:offset + cfg['batch_size']]
            batch = store.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            result, target = model(batch), store.targets(ids)
            if variant == 'balanced':
                loss, terms = balance_loss(result, target, batch['baseline'])
            else:
                loss = training_loss(result, target)
                terms = {'legacy_total': loss}
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            for name, value in terms.items():
                terms_total[name] = terms_total.get(name, 0.) + float(value.detach()) * len(ids)
            if offset == 0 or (offset // cfg['batch_size'] + 1) % 100 == 0:
                write_json(output / 'progress.json', dict(stage='training', epoch=epoch,
                           processed=min(offset + len(ids), len(order)), count=len(order),
                           elapsed_seconds=time.monotonic() - started))
        prediction = predict(model, store, selection_rows, cfg['batch_size'])
        raw = dict(epoch=epoch, alpha=1., fallback=False,
                   **diagnostics(actual, baseline, prediction))
        guarded = select_candidate(actual, baseline, prediction['anchor'], prediction['full'], epoch=epoch)
        if best_raw is None or raw['full']['qs'] < best_raw['full']['qs']:
            best_raw = raw
            save_checkpoint(output / 'raw.pt', model, variant, cfg['seed'], raw)
        if best_guard is None or selection_key(guarded) < selection_key(best_guard):
            best_guard = guarded
            save_checkpoint(output / 'guarded.pt', model, variant, cfg['seed'], guarded)
        log.append(dict(epoch=epoch, loss=total / len(order),
                        terms={k: v / len(order) for k, v in terms_total.items()}, raw=raw, guarded=guarded))
        write_json(output / 'training.json', log)
        print(json.dumps(dict(variant=variant, epoch=epoch, loss=log[-1]['loss'],
                              raw_qs=raw['full']['qs'], guard_alpha=guarded['alpha'])), flush=True)
    write_json(output / 'progress.json', dict(stage='selection_complete', raw=best_raw, guarded=best_guard))
    return dict(raw=best_raw, guarded=best_guard)


def evaluation_groups(origins, baseline):
    """Predeclared diagnostic strata; never consumed by checkpoint selection."""
    months = local_dates(origins).astype('datetime64[M]').astype(str)
    numbers = np.array([int(month[-2:]) for month in months])
    groups = {'all': np.ones(len(origins), dtype=bool)}
    for month in np.unique(months):
        year, number = month.split('-')
        key = f'{year}Q{(int(number) - 1) // 3 + 1}'
        groups[key] = groups.get(key, np.zeros(len(origins), dtype=bool)) | (months == month)
    if np.any(np.char.startswith(months, '2022')):
        groups['2022H1'] = (months >= '2022-01') & (months < '2022-07')
        groups['2022H2'] = (months >= '2022-07') & (months < '2023-01')
    seasons = {f'season/{name}': np.isin(numbers, values) for name, values in
               (('summer', (12, 1, 2)), ('autumn', (3, 4, 5)),
                ('winter', (6, 7, 8)), ('spring', (9, 10, 11)))}
    edges = (-np.inf, 0, 50, 100, 150, 300, np.inf)
    forecast_mean = np.asarray(baseline)[..., 2].mean(axis=1)
    bands = {f'B_q50_mean/[{lo},{hi})': (forecast_mean >= lo) & (forecast_mean < hi)
             for lo, hi in zip(edges, edges[1:])}
    groups.update(seasons)
    groups.update(bands)
    groups.update({f'{season}|{band}': s & b for season, s in seasons.items() for band, b in bands.items()})
    return groups


def evaluate(store, rows, variant, cfg, output, model_builder=build_model):
    ids = rows['evaluation']
    origins, actual, baseline = store.origins[ids], store.a['actual'][ids], store.a['baseline'][ids]
    groups = evaluation_groups(origins, baseline)
    counts = {g: dict(n_origins=int(m.sum()), n_cells=int(m.sum() * actual.shape[1]))
              for g, m in groups.items()}
    result = {'groups': counts,
              'B': {g: dict(**counts[g], **score(actual[m], baseline[m]))
                    for g, m in groups.items() if m.any()}}
    for mode in ('raw', 'guarded'):
        checkpoint = output / f'{mode}.pt'
        model, selection = (restore_checkpoint(checkpoint) if model_builder is build_model else
                            restore_checkpoint(checkpoint, model_builder=model_builder))
        raw = predict(model, store, ids, cfg['batch_size'])
        chosen = {k: blend(baseline, q, selection['alpha']) for k, q in raw.items()}
        result[mode] = dict(selection=selection,
            metrics={g: diagnostics(actual[m], baseline[m], {k: q[m] for k, q in chosen.items()})
                     for g, m in groups.items() if m.any()},
            checkpoint_unblended=diagnostics(actual, baseline, raw))
        np.savez_compressed(output / f'{variant}_{mode}.npz', origin_unix=origins,
            target_time_unix=origins[:, None] + np.arange(24)[None] * HOUR,
            actual=actual, baseline=baseline, anchor=chosen['anchor'], quantiles=chosen['full'],
            raw_anchor=raw['anchor'], raw_quantiles=raw['full'], quantile_levels=LEVELS,
            alpha=selection['alpha'], epoch=selection['epoch'])
    write_json(output / 'metrics.json', result)
    write_json(output / 'progress.json', dict(stage='complete', evaluated_origins=len(ids)))
    return result


def report_markdown(results, cfg):
    lines = ['# Anchor balance development comparison', '',
             f"Profile: {cfg['profile']}; epochs: {cfg['epochs']}; seed: {cfg['seed']}.", '',
             'Selection guards are not guarantees on evaluation periods. Screen results are preliminary.',
             'Raw compares Full-QS checkpoints; guarded compares independently selected shared-alpha checkpoints.',
             'Bias = prediction minus actual. Metrics aggregate forecast origins and all 24 horizons.', '',
             '| Region/fold | Variant | Mode | Epoch | Alpha | Fallback | Anchor MAE | Full MAE | Full QS | Bias | C80 | C90 |',
             '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for key, variants in results.items():
        b = next(iter(variants.values()))['B']['all']
        lines.append(f"| {key} | B | baseline | — | 0 | — | {b['mae']:.4f} | {b['mae']:.4f} | {b['qs']:.4f} | {b['bias']:.4f} | {b['coverage_80']:.4f} | {b['coverage_90']:.4f} |")
        for variant, result in variants.items():
            for mode in ('raw', 'guarded'):
                selected, metrics = result[mode]['selection'], result[mode]['metrics']['all']
                a, f = metrics['anchor'], metrics['full']
                lines.append(f"| {key} | {variant} | {mode} | {selected['epoch']} | {selected['alpha']} | {selected['fallback']} | {a['mae']:.4f} | {f['mae']:.4f} | {f['qs']:.4f} | {f['bias']:.4f} | {f['coverage_80']:.4f} | {f['coverage_90']:.4f} |")
    lines += ['', 'JSON metrics include matched raw/guarded gaps, quarters, 2022H1/H2, Australian seasons, B forecast price bands and their crosses, with group counts.',
              'All checkpoint and alpha decisions precede evaluation. Full history remains available causally.']
    return '\n'.join(lines) + '\n'


def configuration(args):
    epochs = args.epochs if args.epochs is not None else (3 if args.profile == 'screen' else 20)
    if epochs < 1 or (args.profile == 'full' and epochs != 20):
        raise ValueError('Positive epochs required; full profile requires exactly 20 epochs')
    if not args.development:
        raise ValueError('--development is required; this runner never supports final-test')
    return dict(profile=args.profile, epochs=epochs, seed=args.seed, batch_size=16, cpu_threads=2,
                learning_rate=.001, weight_decay=1e-4, reconstruction_weight=.05, gate_penalty=.001,
                weights=asdict(BalanceWeights()), alphas=[0, .25, .5, 1], device='cpu', tree_seed=args.tree_seed,
                candidates=16, history_days=730, variants=args.variants,
                diagnostic_price_edges=['-inf', 0, 50, 100, 150, 300, 'inf'],
                diagnostic_grouping='origin AEST season crossed with mean of B q50 across 24h; left-closed bands; reporting only',
                sampling='train daily rotating24h; selection/eval daily4 rotating6h' if args.profile == 'screen' else 'all hourly origins')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=['NSW1', 'QLD1', 'TAS1', 'all'], default='all')
    parser.add_argument('--fold', choices=[*FOLDS, 'all'], default='all')
    parser.add_argument('--profile', choices=['screen', 'full'], default='screen')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int, default=2026, help='Neural initialization and training order')
    parser.add_argument('--tree-seed', type=int, default=2026, help='Archived OOF tree provenance seed')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--bundle-root', default=BUNDLE_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args(argv)
    cfg = configuration(args)
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate variants')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg['cpu_threads'])
    regions = ['NSW1', 'QLD1', 'TAS1'] if args.region == 'all' else [args.region]
    folds = list(FOLDS) if args.fold == 'all' else [args.fold]
    source = sorted(p for package in ('full_v5_anchor_balance', 'full_v5_joint', 'full_v5')
                    for p in (Path('forecasting') / package).glob('*.py'))
    for path in source:
        snapshot = output / 'source' / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], text=True)
    (output / 'source.diff').write_text(diff)
    manifest = dict(scope='retrospective_development_not_final_test', config=cfg,
        folds={f: FOLDS[f] for f in folds}, regions=regions,
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): digest(p) for p in source}, diff_sha256=digest(output / 'source.diff'),
        bundles={}, samples={}, oof_stage_mapping={}, availability_audit='archive_timestamp_proxies_only',
        evaluation_policy='all training and selection complete before any evaluation scoring')
    write_json(output / 'manifest.json', manifest)
    stores, tasks = {}, []
    for region in regions:
        store, paths = load_store(args.bundle_root, region, cfg['tree_seed'])
        stores[region] = store
        manifest['bundles'].update({str(p): digest(p) for p in paths})
        for fold in folds:
            rows = split_rows(store.origins, fold, cfg['profile'])
            key = f'{region}/{fold}'
            manifest['samples'][key] = {role: sample_manifest(store.origins, ids) for role, ids in rows.items()}
            manifest['oof_stage_mapping'][key] = {role: provenance_manifest(store, ids, paths)
                                                   for role, ids in rows.items()}
            tasks.append((region, fold, rows))
    write_json(output / 'manifest.json', manifest)
    completed = {}
    for region, fold, rows in tasks:
        for variant in args.variants:
            write_json(output / 'progress.json', dict(stage='training_selection', region=region, fold=fold, variant=variant))
            directory = output / region / fold / variant
            selected = fit(stores[region], rows, variant, cfg, directory)
            completed[f'{region}/{fold}/{variant}'] = dict(selection=selected,
                checkpoints={str(directory / f'{mode}.pt'): digest(directory / f'{mode}.pt')
                             for mode in ('raw', 'guarded')})
    if any(digest(path) != expected for path, expected in manifest['source_sha256'].items()):
        raise ValueError('Source changed during training; evaluation aborted')
    write_json(output / 'selection_complete.json', completed)
    results = {}
    for region, fold, rows in tasks:
        key = f'{region}/{fold}'
        results[key] = {}
        for variant in args.variants:
            write_json(output / 'progress.json', dict(stage='evaluation', region=region, fold=fold, variant=variant))
            checkpoint_hashes = completed[f'{key}/{variant}']['checkpoints']
            if any(digest(path) != expected for path, expected in checkpoint_hashes.items()):
                raise ValueError('Selected checkpoint changed before evaluation')
            results[key][variant] = evaluate(stores[region], rows, variant, cfg, output / region / fold / variant)
    write_json(output / 'report.json', results)
    (output / 'report.md').write_text(report_markdown(results, cfg))
    write_json(output / 'progress.json', dict(stage='complete', comparisons=len(tasks) * len(args.variants)))


if __name__ == '__main__':
    main()
