"""Expanding-window front ends; every OOF component is fitted before its labels."""
import gc
from pathlib import Path
import numpy as np
import torch
from forecasting.aligned_experiment import atomic_json, read_json, sha
from forecasting.aligned_inputs import _array_hash
from .data import rebuild_bank, DeviceBank, prefix_rows, unix
from .training import train_frontend, load_model, infer

MODES = ('current', 'full', 'forced')
FOLDS = (('F1', '2017-07-01', '2018-01-01', '2020-01-01'),
         ('F2', '2019-07-01', '2020-01-01', '2021-01-01'),
         ('F3', '2020-07-01', '2021-01-01', '2022-01-01'))


def limit(rows, count):
    return rows if count is None or len(rows) <= count else rows[np.linspace(0, len(rows) - 1, count, dtype=int)]


def fit_or_load(bank, tr, sel, cfg, mode, folder, identity, fixed_epochs=None):
    folder = Path(folder)
    identity = dict(identity, mode=mode, fixed_epochs=fixed_epochs,
                    rows_hash=_array_hash(dict(train=tr, selection=sel)))
    if (folder / 'identity.json').exists():
        if read_json(folder / 'identity.json') != identity:
            raise ValueError('Frontend resume identity mismatch')
        fit = read_json(folder / 'fit.json')
        if sha(folder / 'model.pt') != fit['model_sha256']:
            raise ValueError('Frontend checkpoint changed')
        return load_model(folder / 'model.pt', cfg['neural'], mode, bank.device), fit
    model, fit = train_frontend(bank, tr, sel, cfg['neural'], mode, folder,
                               seed=2026, fixed_epochs=fixed_epochs)
    atomic_json(folder / 'identity.json', identity)
    return model, fit


def save_features(folder, rows, output, identity):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / 'rows.npy', rows, allow_pickle=False)
    for key, values in output.items():
        np.save(folder / f'{key}.npy', values, allow_pickle=False)
    atomic_json(folder / 'manifest.json', dict(identity=identity,
        files={p.name: sha(p) for p in folder.glob('*.npy')}))


def load_features(folder):
    folder = Path(folder)
    meta = read_json(folder / 'manifest.json')
    for name, digest in meta['files'].items():
        if sha(folder / name) != digest:
            raise ValueError('Frontend feature cache changed')
    return {Path(n).stem: np.load(folder / n, mmap_mode='r') for n in meta['files']}


def generate_frontends(root, region, memory, candidates, splits, config, base_config, manifest, device):
    directory = Path(root) / 'frontends' / region
    common_identity = dict(source_hash=manifest['source_hash'], config_hash=manifest['config_hash'], region=region)
    state_path = directory / 'state.json'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        done = read_json(directory / 'complete.json')
        if done['identity'] != common_identity:
            raise ValueError('Completed frontend identity changed')
        for path, digest in done['files'].items():
            if sha(directory / path) != digest:
                raise ValueError('Completed frontend output changed')
        return
    folds = FOLDS[:1] if config.get('smoke') else FOLDS
    max_train = config.get('max_train_origins')
    max_eval = config.get('max_eval_origins')
    fold_rows = []
    for fold, inner_end, fit_end, predict_end in folds:
        chosen = {}
        for stage, cutoff in (('selection', inner_end), ('refit', fit_end)):
            atomic_json(state_path, dict(status='running', fold=fold, stage=stage, prefix_end=cutoff))
            packed, audit = rebuild_bank(memory, base_config, region, cutoff, directory / 'banks' / cutoff)
            bank = DeviceBank(memory, packed, audit, candidates, device)
            del packed
            tr = limit(prefix_rows(memory, cutoff), max_train)
            sel = limit(prefix_rows(memory, fit_end, inner_end), max_eval) if stage == 'selection' else np.array([], dtype=np.int64)
            rows = limit(prefix_rows(memory, predict_end, fit_end), max_eval)
            for mode in MODES:
                atomic_json(state_path, dict(status='running', fold=fold, stage=stage, mode=mode, prefix_end=cutoff))
                folder = directory / mode / fold / stage
                identity = dict(common_identity, preprocessing_hash=audit['preprocessing_hash'], cutoff=cutoff)
                model, fit = fit_or_load(bank, tr, sel, config, mode, folder, identity,
                                        chosen[mode] if stage == 'refit' else None)
                if stage == 'selection':
                    chosen[mode] = fit['selected_epoch']
                else:
                    output = infer(model, bank, rows, config['neural']['batch_size'])
                    save_features(directory / mode / fold / 'oof', rows, output,
                                  dict(identity, model_sha256=fit['model_sha256']))
                del model
            del bank
            gc.collect(); torch.cuda.empty_cache()
        fold_rows.append(rows)
    oof_rows = np.concatenate(fold_rows)
    if len(np.unique(oof_rows)) != len(oof_rows):
        raise AssertionError('OOF folds overlap')
    for mode in MODES:
        pieces = [load_features(directory / mode / fold[0] / 'oof') for fold in folds]
        joined = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0] if k != 'rows'}
        save_features(directory / mode / 'oof', oof_rows, joined, common_identity)
    atomic_json(state_path, dict(status='running', stage='final', prefix_end='2022-01-01'))
    packed, audit = rebuild_bank(memory, base_config, region, '2022-01-01', directory / 'banks' / '2022-01-01')
    bank = DeviceBank(memory, packed, audit, candidates, device)
    del packed
    tr = limit(prefix_rows(memory, '2022-01-01'), max_train)
    sel = limit(np.searchsorted(memory.origins, splits['selection'].origin_unix), max_eval)
    for mode in MODES:
        atomic_json(state_path, dict(status='running', stage='final', mode=mode))
        identity = dict(common_identity, preprocessing_hash=audit['preprocessing_hash'], cutoff='2022-01-01')
        model, fit = fit_or_load(bank, tr, sel, config, mode, directory / mode / 'final', identity)
        for name in ('selection', 'calibration', 'test'):
            rows = limit(np.searchsorted(memory.origins, splits[name].origin_unix), max_eval)
            output = infer(model, bank, rows, config['neural']['batch_size'], audit_weights=True)
            save_features(directory / mode / name, rows, output,
                          dict(identity, model_sha256=fit['model_sha256']))
        del model
    del bank
    gc.collect(); torch.cuda.empty_cache()
    # Protect checkpoints, identities, prefix transforms, features and their audits.
    files = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
             if p.is_file() and p.name not in ('state.json', 'progress.json', 'complete.json')}
    atomic_json(directory / 'complete.json', dict(identity=common_identity, files=files,
        oof_origins=len(oof_rows), folds=[list(f) for f in folds]))
    atomic_json(state_path, dict(status='completed', oof_origins=len(oof_rows)))
