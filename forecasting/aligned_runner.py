"""One shared input builder and scorer for aligned baseline smoke experiments.

This CLI deliberately runs integration smoke checks only; ``fit_neural`` also
supports full training runs with checkpoints and progress logs. The CLI does not
claim tuned performance or overwrite historical experiments. Model internals,
including RevIN, input fusion, and per-horizon tree training, remain native.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from importlib.metadata import version
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .aligned_inputs import INPUT_KEYS, AlignedWindows, build_aligned_splits
from .aligned_scoring import (
    QUANTILE_LEVELS, assert_comparable, make_prediction_artifact,
    save_prediction_artifact, score_predictions,
)
from .datasets import load_forecast_config

NEURAL_MODELS = ('full', 'no_fields', 'mlp', 'linear', 'dlinear', 'patchtst', 'gatets_adapter',
                 'timexer', 'pricefm')
TREE_MODELS = ('xgboost', 'lightgbm', 'lasso_arx72_cp')


class PlainMLP(nn.Module):
    """Independent all-input control, with no dual-field implementation dependency."""
    def __init__(self, example, horizon, hidden=64):
        super().__init__()
        width = sum(int(np.prod(example[k].shape[1:])) for k in INPUT_KEYS)
        self.horizon = horizon
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Linear(hidden, horizon * len(QUANTILE_LEVELS)))

    def forward(self, batch):
        flat = torch.cat([batch[k].flatten(1) for k in INPUT_KEYS], dim=1)
        return self.network(flat).reshape(-1, self.horizon, len(QUANTILE_LEVELS))


def _native_batch(batch):
    # Layout mapping belongs to the model adapter. It preserves all values of
    # the canonical blocks and does not fit or recompute any feature transform.
    result = {k: batch[k] for k in INPUT_KEYS}
    result['ctf_exogenous'] = torch.cat(
        [result['ctf_exogenous'], result['quantile_exogenous']], dim=-1)
    return result


def _full_inputs(batch):
    keys = ('history_values', 'future_calendar', 'future_exogenous',
            'origin_context', 'ctf_exogenous', 'quantile_exogenous')
    return tuple(batch[k] if batch[k].shape[-1] else None for k in keys)


def _forward(model, kind, batch):
    if kind in ('full', 'no_fields'):
        output = model(*_full_inputs(batch))
        return output['quantile_forecast'], output
    inputs = {k: batch[k] for k in INPUT_KEYS}
    return model(inputs if kind in ('mlp', 'timexer', 'pricefm') else _native_batch(inputs)), None


def _loader(split, batch_size, *, shuffle=False, seed=2026):
    return DataLoader(AlignedWindows(split), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


class AlignedTensorBatches:
    """Index canonical arrays directly; no per-sample copies or new transforms.

    The shuffle RNG draws mirror PyTorch DataLoader/RandomSampler, preserving
    minibatch order when this performance option is toggled.
    """
    def __init__(self, split, batch_size, device, *, shuffle=False, seed=2026,
                 include_targets=False):
        self.batch_size, self.shuffle, self.size = batch_size, shuffle, len(split)
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(seed)
        self.tensors = {key: torch.from_numpy(value).to(self.device)
                        for key, value in split.inputs.items()}
        if include_targets:
            target = torch.from_numpy(split.target[..., None]).to(self.device)
            self.tensors.update(target_price=target, target_quantile=target,
                                target_price_raw=torch.from_numpy(split.actual[..., None]).to(self.device))

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            order = torch.randperm(self.size, generator=self.generator)
            torch.randperm(self.size, generator=self.generator)
            order = order.to(self.device)
        else:
            order = torch.arange(self.size, device=self.device)
        for start in range(0, self.size, self.batch_size):
            index = order[start:start + self.batch_size]
            yield {key: value[index] for key, value in self.tensors.items()}


def _tensor_bytes(split, *, include_targets=False):
    total = sum(value.nbytes for value in split.inputs.values())
    if include_targets:
        total += split.target.nbytes + split.actual.nbytes
    return total


def _preload_allowed(device, requested, needed_bytes):
    if requested not in (True, False, 'auto'):
        raise ValueError("preload_batches must be true, false or 'auto'")
    if requested is False:
        return False
    if device.type != 'cuda':
        return requested is True
    free, _ = torch.cuda.mem_get_info(device)
    allowed = needed_bytes <= min(int(free * .35), max(0, free - 2 * 1024 ** 3))
    if requested is True and not allowed:
        raise MemoryError('Preloaded shared inputs exceed the GPU memory safety allowance')
    return allowed


def _write_progress(path, event):
    if path is not None:
        with Path(path).open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()


def _save_best_checkpoint(path, payload):
    if path is not None:
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)


def _predict(model, kind, split, device, batch_size, *, loader=None):
    model.eval()
    quantiles, points = [], []
    with torch.no_grad():
        for values in (loader if loader is not None else _loader(split, batch_size)):
            batch = {k: v.to(device) for k, v in values.items()}
            q, output = _forward(model, kind, batch)
            quantiles.append(q.cpu().numpy())
            if output is not None:
                points.append(output['point_forecast'].squeeze(-1).cpu().numpy())
    mean, std = split.metadata['target_mean'], split.metadata['target_std']
    quantiles = np.concatenate(quantiles).astype(np.float64) * std + mean
    point = np.concatenate(points).astype(np.float64) * std + mean if points else None
    return quantiles, point


def fit_neural(kind, train, selection, evaluation, config, *, seed=2026,
               epochs=100, batch_size=32, learning_rate=1e-3, patience=15,
               selection_metric='pinball', device='cpu', checkpoint_path=None,
               progress_path=None, preload_batches='auto'):
    """Fit on train; choose on selection; evaluation labels never select weights.

    Optional best checkpoints preserve inference weights, input shapes and the
    fitted target scaler. They intentionally do not support optimizer resume.
    Progress is append-only JSONL with explicit budget/early-stopping outcomes.
    """
    from . import general_baselines
    from .gatets_baseline import build_gatets
    from .train import build_loss, build_model, set_seed

    if 'model' in config and not np.array_equal(
            np.asarray(config['model'].get('quantiles')), QUANTILE_LEVELS):
        raise ValueError('Configured model quantiles differ from the common scoring levels')
    if 'data' in config and config['data']['history_columns'][0] != config['data']['target_column']:
        raise ValueError('Native price-history adapters require the target in history column zero')
    if kind not in NEURAL_MODELS:
        raise ValueError(f'Unknown neural model: {kind}')
    if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
            for v in (epochs, batch_size, patience)) or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Training parameters must be positive')
    if selection_metric not in ('mae', 'pinball'):
        raise ValueError('selection_metric must be mae or pinball')
    for split in (train, selection, evaluation):
        for key in ('preprocessing_hash', 'input_hash', 'sample_hash', 'region', 'protocol_id'):
            if not isinstance(split.metadata.get(key), str) or not split.metadata[key].strip():
                raise ValueError(f'Missing required input metadata: {key}')
        n = len(split.origin_unix)
        if not n or split.target_time_unix.shape != (n, train.target.shape[1]):
            raise ValueError('Target timestamps must match input rows and horizon')
        if any(v.shape[0] != n or not np.isfinite(v).all() for v in split.inputs.values()):
            raise ValueError('Input blocks must contain aligned finite rows')
    for split in (train, selection):
        if split.target.shape != split.target_time_unix.shape or not np.isfinite(split.target).all():
            raise ValueError('Training and selection targets must be aligned and finite')
    for left, right in ((train, selection), (selection, evaluation)):
        if int(left.target_time_unix.max()) + 3600 > int(right.origin_unix.min()):
            raise ValueError('Training/selection/evaluation targets must be chronologically available')
        for key in ('preprocessing_hash', 'region', 'protocol_id'):
            if not left.metadata.get(key) or left.metadata[key] != right.metadata.get(key):
                raise ValueError(f'Splits must share fitted preprocessing and experiment identity: {key}')
        for key in ('target_mean', 'target_std'):
            if not np.isfinite(left.metadata[key]) or left.metadata[key] != right.metadata[key]:
                raise ValueError('Splits must share the same finite target scaler')
        if left.metadata['target_std'] <= 0:
            raise ValueError('Target scale must be positive')
        if set(left.inputs) != set(INPUT_KEYS) or set(right.inputs) != set(INPUT_KEYS):
            raise ValueError('Splits must have all shared input blocks')
        if any(left.inputs[k].shape[1:] != right.inputs[k].shape[1:] for k in INPUT_KEYS):
            raise ValueError('Splits must share input shapes')
    for path in (checkpoint_path, progress_path):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is not None and progress_path is not None and Path(checkpoint_path).resolve() == Path(progress_path).resolve():
        raise ValueError('Checkpoint and progress paths must differ')
    set_seed(seed)
    device = torch.device(device)
    example = next(iter(_loader(train, batch_size)))
    model_config = copy.deepcopy(config)
    model_config['training']['epochs'] = epochs
    if kind == 'no_fields':
        model_config['model']['field_forecast'] = False
    if kind in ('full', 'no_fields'):
        model, criterion = build_model(model_config), build_loss(model_config)
    elif kind == 'mlp':
        model, criterion = PlainMLP(example, train.actual.shape[1]), None
    elif kind == 'gatets_adapter':
        model, criterion = build_gatets(kind, _native_batch(example)), None
    elif kind == 'timexer':
        from .timexer_baseline import build_timexer
        model = build_timexer(example, **config.get('model_baselines', {}).get(kind, {}))
        criterion = None
    elif kind == 'pricefm':
        from .pricefm_baseline import build_pricefm
        model = build_pricefm(example, **config.get('model_baselines', {}).get(kind, {}))
        criterion = None
    else:
        model, criterion = general_baselines.build(kind, _native_batch(example)), None
    model.to(device)
    if criterion is not None:
        criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=config['training'].get('weight_decay', 1e-5))
    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
    best_value, best_epoch, best_state, stale = float('inf'), -1, None, 0
    curve = []
    preload_bytes = _tensor_bytes(train, include_targets=True) + _tensor_bytes(selection)
    use_preload = _preload_allowed(device, preload_batches, preload_bytes)
    if use_preload:
        train_loader = AlignedTensorBatches(train, batch_size, device, shuffle=True,
                                           seed=seed, include_targets=True)
        selection_loader = AlignedTensorBatches(selection, batch_size, device)
    else:
        train_loader = _loader(train, batch_size, shuffle=True, seed=seed)
        selection_loader = _loader(selection, batch_size)
    started = time.monotonic()
    _write_progress(progress_path, {'event': 'start', 'model': kind, 'seed': seed,
        'epoch_budget': epochs, 'patience': patience, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'device': str(device),
        'preloaded_batches': use_preload, 'preload_bytes_estimate': preload_bytes,
        'train_samples': len(train), 'selection_samples': len(selection)})
    stop_reason = 'epoch_budget_exhausted'
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        loss_sum, seen = 0., 0
        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        for values in train_loader:
            batch = {k: v.to(device) for k, v in values.items()}
            optimizer.zero_grad(set_to_none=True)
            q, output = _forward(model, kind, batch)
            if criterion is not None:
                loss = criterion(output, batch['history_values'], batch['target_price'],
                                 batch['target_quantile'], batch['target_price_raw'])['total']
            else:
                error = batch['target_quantile'] - q
                loss = torch.maximum(levels * error, (levels - 1) * error).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {kind} loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            count = batch['target_price'].shape[0]
            loss_sum += float(loss.detach()) * count
            seen += count
        q, _ = _predict(model, kind, selection, device, batch_size, loader=selection_loader)
        q = np.sort(q, axis=-1)
        if selection_metric == 'mae':
            value = float(np.abs(q[..., 2] - selection.actual).mean())
        else:
            error = selection.actual[..., None] - q
            value = float(np.maximum(QUANTILE_LEVELS * error,
                                     (QUANTILE_LEVELS - 1) * error).mean())
        if not np.isfinite(value):
            raise FloatingPointError(f'Non-finite {kind} selection metric')
        curve.append(value)
        if value < best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_best_checkpoint(checkpoint_path, {
                'model_state': best_state, 'model_kind': kind, 'model_config': model_config,
                'seed': seed, 'best_epoch': best_epoch, 'best_selection_value': best_value,
                'selection_metric': selection_metric, 'epoch_budget': epochs, 'patience': patience,
                'batch_size': batch_size, 'learning_rate': learning_rate,
                'input_schema': {key: {'shape': list(value.shape[1:]), 'dtype': str(value.dtype)}
                                 for key, value in train.inputs.items()},
                'forecast_horizon': train.target.shape[1], 'quantile_levels': QUANTILE_LEVELS.tolist(),
                'train_metadata': dict(train.metadata), 'selection_metadata': dict(selection.metadata),
                'optimizer_resume_supported': False,
            })
        else:
            stale += 1
        _write_progress(progress_path, {'event': 'epoch', 'epoch': epoch,
            'train_loss': loss_sum / seen, 'selection_value': value,
            'selection_metric': selection_metric, 'best_epoch': best_epoch,
            'best_selection_value': best_value, 'stale_epochs': stale,
            'seconds': time.monotonic() - epoch_started})
        if stale >= patience:
            stop_reason = 'early_stopping'
            break
    model.load_state_dict(best_state)
    if hasattr(model, 'set_epoch'):
        model.set_epoch(best_epoch)
    # Drop train/selection GPU inputs before allocating evaluation inputs.
    del train_loader, selection_loader
    evaluation_loader = (AlignedTensorBatches(evaluation, batch_size, device)
                         if use_preload and _preload_allowed(device, 'auto', _tensor_bytes(evaluation))
                         else None)
    quantiles, point = _predict(model, kind, evaluation, device, batch_size, loader=evaluation_loader)
    metadata = {'best_epoch': best_epoch, 'best_selection_value': best_value,
                'epoch_budget': epochs, 'patience': patience, 'stop_reason': stop_reason,
                'budget_cap_reached': len(curve) == epochs, 'early_stopped': stop_reason == 'early_stopping',
                'batch_size': batch_size, 'device': str(device),
                'preloaded_batches': use_preload, 'preload_bytes_estimate': preload_bytes,
                'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else None,
                'selection_metric': selection_metric,
                'selection_curve': curve, 'epochs_executed': len(curve),
                'learning_rate': learning_rate, 'parameters': sum(p.numel() for p in model.parameters()),
                'internal_loss': 'native_full_loss' if criterion is not None else 'pinball',
                'input_mapping': 'shared_blocks_model_native_fusion',
                'native_unused_blocks': [] if kind in ('mlp', 'timexer', 'pricefm') else ['history_calendar']}
    if hasattr(model, 'provenance'):
        metadata['model_provenance'] = model.provenance
    _write_progress(progress_path, {'event': 'complete', 'best_epoch': best_epoch,
        'best_selection_value': best_value, 'epochs_executed': len(curve),
        'stop_reason': stop_reason, 'budget_cap_reached': len(curve) == epochs,
        'seconds': time.monotonic() - started})
    return quantiles, point, metadata


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_smoke(config_path, output, *, regions=('NSW1',), seed=2026,
              models=(*NEURAL_MODELS, *TREE_MODELS), max_origins=64, epochs=2):
    from .aligned_baselines import fit_predict

    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; previous artifacts are never overwritten')
    if len(set(models)) != len(models) or not set(models) <= set(NEURAL_MODELS + TREE_MODELS):
        raise ValueError('Models must be unique supported names')
    config = load_forecast_config(config_path)
    if config.get('aligned_protocol', {}).get('stage') != 'integration_smoke':
        raise ValueError('This runner requires an integration_smoke configuration')
    if 'test' in config['data']['split_names']:
        raise ValueError('Smoke configuration must use development-only splits')
    cutoff = config['data'].get('load_end_exclusive')
    if cutoff is None:
        raise ValueError('Development cutoff is required')
    budget = {'epochs': epochs, 'batch_size': int(config['training']['batch_size']),
              'learning_rate': float(config['training']['learning_rate']),
              'patience': int(config['training'].get('patience', 2)),
              'device': config['training']['device'], 'selection_metric': 'pinball'}
    source = source_fingerprint()
    torch.set_num_threads(2)
    summary = {'stage': 'integration_smoke_not_quality_ranking', 'seed': seed,
               'source': source, 'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
               'max_origins_per_split': max_origins, 'resolved_neural_budget': budget,
               'environment': {'python': platform.python_version(), **{name: version(name) for name in
                               ('numpy', 'torch', 'xgboost', 'lightgbm', 'scikit-learn')}},
               'regions': {}}
    output.mkdir(parents=True)
    for region in regions:
        splits = build_aligned_splits(config_path, region,
                                     split_names=('train', 'selection', 'calibration', 'screening'),
                                     max_origins=max_origins, end_exclusive=cutoff)
        train, selection, calibration, evaluation = (splits[s] for s in
                                                     ('train', 'selection', 'calibration', 'screening'))
        reference, rows = None, {}
        for kind in models:
            start = time.monotonic()
            if kind in NEURAL_MODELS:
                quantiles, point, model_metadata = fit_neural(
                    kind, train, selection, evaluation, config, seed=seed, **budget)
            else:
                result = fit_predict(kind, train, selection, evaluation,
                                     calibration=calibration if kind == 'lasso_arx72_cp' else None,
                                     seed=seed, quantile_levels=tuple(QUANTILE_LEVELS),
                                     params=({'alphas': [0.01, 0.1], 'max_iter': 2000,
                                              'selection_metric': 'mae'} if kind == 'lasso_arx72_cp' else
                                             {'n_estimators': 8, 'early_stopping_rounds': 3,
                                              'n_jobs': 2, 'selection_metric': 'pinball'}))
                quantiles, point, model_metadata = result.quantiles, result.point, result.metadata
            metadata = dict(evaluation.metadata, seed=seed, source=f"{kind}:{source}", model=kind,
                            output_type='quantile' if quantiles is not None else 'point_only',
                            prediction_space='raw_rrp', smoke=True, model_metadata=model_metadata,
                            config_sha256=summary['config_sha256'])
            artifact = make_prediction_artifact(
                actual=evaluation.actual, origin_unix=evaluation.origin_unix,
                target_time_unix=evaluation.target_time_unix, metadata=metadata,
                quantiles=quantiles, point=point,
                quantile_levels=QUANTILE_LEVELS if quantiles is not None else None)
            if reference is not None:
                assert_comparable(reference, artifact)
            else:
                reference = artifact
            save_prediction_artifact(output / region / kind / 'screening.npz', artifact)
            rows[kind] = {'scores': score_predictions(artifact), 'seconds': time.monotonic() - start,
                          'model_metadata': model_metadata}
            print(f'{region} {kind}: aligned smoke passed', flush=True)
        summary['regions'][region] = {'input_metadata': evaluation.metadata, 'models': rows}
    if source_fingerprint() != source:
        raise RuntimeError('Source changed during smoke; rerun with stable code before accepting results')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/aemo_forecast_aligned_core_v1.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', nargs='+', default=['NSW1'])
    parser.add_argument('--models', nargs='+', default=list(NEURAL_MODELS + TREE_MODELS))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-origins', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    run_smoke(args.config, args.output, regions=args.regions, models=args.models,
              seed=args.seed, max_origins=args.max_origins, epochs=args.epochs)


if __name__ == '__main__':
    main()
