"""Six-epoch matched Anchor architecture screen with a global evaluation barrier."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from . import experiment as base
from .architecture import VARIANTS, anchor_pretrain_loss, build_model, configure_stage
from .objective import balance_loss
from .selection import select_candidate, selection_key


def stage_for(variant, epoch):
    if variant not in VARIANTS or epoch not in range(1, 7):
        raise ValueError('Architecture screen requires a known variant and epochs 1..6')
    return ('anchor' if epoch <= 3 else 'history') if variant == 'staged' else 'joint'


def checkpoint_model(variant, seed):
    """All selected staged checkpoints come from the frozen-Anchor history stage."""
    return configure_stage(build_model(variant, seed), variant, 'history' if variant == 'staged' else 'joint')


def restore_checkpoint(path):
    model, selection = base.restore_checkpoint(path, model_builder=build_model)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    configure_stage(model, saved['variant'], selection['stage'])
    return model, selection


def parameter_counts(model):
    return {group: dict(total=sum(p.numel() for n, p in model.named_parameters() if n.split('.')[0] == group),
                        trainable=sum(p.numel() for n, p in model.named_parameters()
                                      if n.split('.')[0] == group and p.requires_grad))
            for group in sorted({n.split('.')[0] for n, _ in model.named_parameters()})}


def anchor_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name.startswith(('state.', 'calibration.'))}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def anchor_probe(model, batch):
    model.eval()
    return model(batch, history_off=True)['anchor'].cpu().clone()


def verify_frozen_anchor(model, reference, probe_batch, probe_reference):
    current = anchor_state(model)
    difference = max(float((current[k] - reference[k]).abs().max()) for k in reference)
    output_difference = float((anchor_probe(model, probe_batch) - probe_reference).abs().max())
    before, after = state_hash(reference), state_hash(current)
    if before != after or difference != 0 or output_difference != 0:
        raise ValueError('Frozen Anchor changed during history stage')
    return dict(before_sha256=before, after_sha256=after, max_parameter_difference=difference,
                max_probe_prediction_difference=output_difference, identical=True)


def fit(store, rows, variant, cfg, output):
    if cfg['epochs'] != 6 or cfg['profile'] != 'screen':
        raise ValueError('This architecture comparison is fixed to six screen epochs')
    output.mkdir(parents=True, exist_ok=False)
    model = build_model(variant, cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    select_ids = rows['selection']
    actual, baseline = (store.a[k][select_ids] for k in ('actual', 'baseline'))
    best_raw, best_guard, log, previous_stage = None, None, [], None
    reference = probe_batch = probe_reference = None
    started = time.monotonic()
    for epoch in range(1, 7):
        stage = stage_for(variant, epoch)
        if stage != previous_stage:
            configure_stage(model, variant, stage)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                          lr=.001, weight_decay=1e-4)
            previous_stage = stage
        model.train()
        order, total, totals = rng.permutation(rows['train']), 0., {}
        for offset in range(0, len(order), cfg['batch_size']):
            ids = order[offset:offset + cfg['batch_size']]
            batch = store.batch(ids)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, history_off=stage == 'anchor')
            loss_function = anchor_pretrain_loss if stage == 'anchor' else balance_loss
            loss, terms = loss_function(result, store.targets(ids), batch['baseline'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite architecture training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) * len(ids)
            if offset == 0 or (offset // cfg['batch_size'] + 1) % 100 == 0:
                base.write_json(output / 'progress.json', dict(stage=stage, epoch=epoch, epochs=6,
                                processed=min(offset + len(ids), len(order)), count=len(order),
                                elapsed_seconds=time.monotonic() - started))
        entry = dict(epoch=epoch, stage=stage, selection_eligible=stage != 'anchor',
                     loss=total / len(order), terms={k: v / len(order) for k, v in totals.items()},
                     parameters=parameter_counts(model),
                     order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        if stage == 'anchor' and epoch == 3:
            reference = anchor_state(model)
            probe_batch = store.batch(rows['train'][:cfg['batch_size']])
            probe_reference = anchor_probe(model, probe_batch)
            entry['fixed_anchor_sha256'] = state_hash(reference)
            base.save_checkpoint(output / 'anchor_pretrained.pt', model, variant, cfg['seed'],
                                 dict(epoch=3, stage='anchor', selection_eligible=False))
        if stage == 'history':
            entry['frozen_anchor'] = verify_frozen_anchor(model, reference, probe_batch, probe_reference)
        if stage != 'anchor':
            prediction = base.predict(model, store, select_ids, cfg['batch_size'])
            raw = dict(epoch=epoch, stage=stage, alpha=1., fallback=False,
                       **base.diagnostics(actual, baseline, prediction))
            guarded = select_candidate(actual, baseline, prediction['anchor'], prediction['full'], epoch=epoch)
            guarded['stage'] = stage
            if best_raw is None or raw['full']['qs'] < best_raw['full']['qs']:
                best_raw = raw
                base.save_checkpoint(output / 'raw.pt', model, variant, cfg['seed'], raw)
            if best_guard is None or selection_key(guarded) < selection_key(best_guard):
                best_guard = guarded
                base.save_checkpoint(output / 'guarded.pt', model, variant, cfg['seed'], guarded)
            entry.update(raw=raw, guarded=guarded)
        log.append(entry)
        base.write_json(output / 'training.json', log)
        print(f'{output}: epoch={epoch} stage={stage} loss={entry["loss"]:.6f}', flush=True)
    base.write_json(output / 'progress.json', dict(stage='selection_complete', raw=best_raw, guarded=best_guard))
    return dict(raw=best_raw, guarded=best_guard)


def train_region(spec):
    """One process owns one complete history store and all variants for its region."""
    region, folds, cfg, bundle_root, output = spec
    torch.set_num_threads(2)
    store, _ = base.load_store(bundle_root, region, cfg['tree_seed'])
    completed = {}
    for fold in folds:
        rows = base.split_rows(store.origins, fold, 'screen')
        for variant in cfg['variants']:
            directory = Path(output) / region / fold / variant
            selection = fit(store, rows, variant, cfg, directory)
            completed[f'{region}/{fold}/{variant}'] = dict(selection=selection,
                checkpoints={str(directory / f'{mode}.pt'): base.digest(directory / f'{mode}.pt')
                             for mode in ('raw', 'guarded')})
    return completed


def configuration(args):
    if args.profile != 'screen' or args.epochs != 6:
        raise ValueError('Architecture screen requires exactly six epochs and screen profile')
    cfg = base.configuration(args)
    cfg.update(experiment='anchor_architectures_v1', workers=args.workers,
               all_variants_objective='balance_loss; staged epochs1..3 anchor_pretrain_loss',
               staged_schedule=dict(anchor_epochs=[1, 2, 3], history_epochs=[4, 5, 6],
                                    anchor_end='fixed epoch3 without selection', optimizer_reset=True),
               baseline_schedule='original seven chronological OOF tree folds; neural checkpoints and alpha freeze before evaluation')
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=['NSW1', 'QLD1', 'TAS1', 'all'], default='all')
    parser.add_argument('--fold', choices=[*base.FOLDS, 'all'], default='all')
    parser.add_argument('--profile', choices=['screen'], default='screen')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--tree-seed', type=int, default=2026)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--workers', type=int, choices=[1, 3], default=3)
    parser.add_argument('--bundle-root', default=base.BUNDLE_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args(argv)
    cfg = configuration(args)
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate variants')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    regions = ['NSW1', 'QLD1', 'TAS1'] if args.region == 'all' else [args.region]
    folds = list(base.FOLDS) if args.fold == 'all' else [args.fold]
    source = sorted(p for package in ('full_v5_anchor_balance', 'full_v5_joint', 'full_v5')
                    for p in (Path('forecasting') / package).glob('*.py'))
    for path in source:
        snapshot = output / 'source' / path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    protocol = Path('docs/v5_anchor_architecture_20261022.md')
    shutil.copyfile(protocol, output / 'protocol.md')
    (output / 'source.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], text=True))
    manifest = dict(scope='retrospective_development_not_final_test', config=cfg, regions=regions,
        folds={f: base.FOLDS[f] for f in folds},
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): base.digest(p) for p in source}, diff_sha256=base.digest(output / 'source.diff'),
        protocol_sha256=base.digest(protocol),
        bundles={}, samples={}, oof_stage_mapping={}, availability_audit='archive_timestamp_proxies_only',
        evaluation_policy='all selected region/fold/variant fits complete before any evaluation scoring')
    base.write_json(output / 'manifest.json', manifest)
    for region in regions:
        store, paths = base.load_store(args.bundle_root, region, cfg['tree_seed'])
        manifest['bundles'].update({str(p): base.digest(p) for p in paths})
        for fold in folds:
            rows = base.split_rows(store.origins, fold, 'screen')
            key = f'{region}/{fold}'
            manifest['samples'][key] = {role: base.sample_manifest(store.origins, ids) for role, ids in rows.items()}
            manifest['oof_stage_mapping'][key] = {role: base.provenance_manifest(store, ids, paths) for role, ids in rows.items()}
        del store
    base.write_json(output / 'manifest.json', manifest)
    specs = [(region, folds, cfg, args.bundle_root, str(output)) for region in regions]
    base.write_json(output / 'progress.json', dict(stage='training_selection', regions=regions))
    completed = {}
    if args.workers == 1 or len(specs) == 1:
        for spec in specs:
            completed.update(train_region(spec))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(train_region, spec) for spec in specs]):
                completed.update(future.result())
                base.write_json(output / 'progress.json', dict(stage='training_selection', completed_fits=len(completed)))
    if any(base.digest(p) != expected for p, expected in manifest['source_sha256'].items()):
        raise ValueError('Source changed during training; evaluation aborted')
    base.write_json(output / 'selection_complete.json', completed)
    results = {}
    for region in regions:
        store, _ = base.load_store(args.bundle_root, region, cfg['tree_seed'])
        for fold in folds:
            rows = base.split_rows(store.origins, fold, 'screen')
            key = f'{region}/{fold}'
            results[key] = {}
            for variant in cfg['variants']:
                checkpoint_hashes = completed[f'{key}/{variant}']['checkpoints']
                if any(base.digest(p) != expected for p, expected in checkpoint_hashes.items()):
                    raise ValueError('Selected checkpoint changed before evaluation')
                base.write_json(output / 'progress.json', dict(stage='evaluation', region=region, fold=fold, variant=variant))
                results[key][variant] = base.evaluate(store, rows, variant, cfg, output / region / fold / variant,
                                                       model_builder=checkpoint_model)
        del store
    base.write_json(output / 'report.json', results)
    (output / 'report.md').write_text(base.report_markdown(results, cfg).replace('Anchor balance development comparison', 'Anchor architecture development comparison'))
    base.write_json(output / 'progress.json', dict(stage='complete', comparisons=len(completed)))


if __name__ == '__main__':
    main()
