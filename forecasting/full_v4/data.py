"""Prefix-only preprocessing, rebuilt from archives including nonlinear thresholds."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from forecasting.aligned_inputs import INPUT_KEYS, _array_hash, _preprocessing_hash
from forecasting.datasets import AEMOForecastDataset, _load_region_frame
from forecasting.aligned_experiment import atomic_json, read_json, sha
from .schema import validate_inputs


def unix(date):
    return int(pd.Timestamp(date, tz='Australia/Brisbane').timestamp())


def prefix_rows(memory, end, start='2015-01-01'):
    """Every supervised delivery and its observed publication precedes the cutoff."""
    end = unix(end) if isinstance(end, str) else end
    start = unix(start) if isinstance(start, str) else start
    return np.flatnonzero((memory.origins >= start) & (memory.origins + 24 * 3600 <= end)
                          & memory.complete & (memory.max_availability < end))


def pack_inputs(inputs):
    validate_inputs(inputs)
    n = len(inputs['history_values'])
    history = np.concatenate((inputs['history_values'], inputs['history_calendar']), -1)
    future = np.concatenate((inputs['future_calendar'],
        np.broadcast_to(inputs['origin_context'][:, None], (n, 24, 1)),
        *(inputs[k] for k in INPUT_KEYS[-3:])), -1)
    flat = np.concatenate([inputs[k].reshape(n, -1) for k in INPUT_KEYS], -1)
    return dict(history=history.astype(np.float32), future=future.astype(np.float32),
                flat=flat.astype(np.float32))


def rebuild_bank(memory, base_config, region, prefix_end, directory):
    """Same origins/labels; only INTERNAL transforms change with the training prefix."""
    directory = Path(directory)
    identity = dict(prefix_end=prefix_end, region=region,
                    origins_hash=_array_hash({'origins': memory.origins}),
                    base_config_hash=_array_hash({'config': np.frombuffer(
                        __import__('json').dumps(base_config, sort_keys=True).encode(), dtype=np.uint8)}))
    if (directory / 'manifest.json').exists():
        audit = read_json(directory / 'manifest.json')
        if audit['identity'] != identity:
            raise ValueError('Prefix preprocessing identity changed')
        for name, digest in audit['files'].items():
            if sha(directory / name) != digest:
                raise ValueError('Prefix bank was modified')
        return {k: np.load(directory / f'{k}.npy', mmap_mode='r')
                for k in ('history', 'future', 'flat')}, audit
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(base_config)
    config['data'].update(split_names=['train', 'evaluation'], relabel_splits=True,
        split_boundaries=dict(train=dict(start='2015-01-01', end_exclusive=prefix_end),
                              evaluation=dict(start=prefix_end, end_exclusive='2025-01-01')),
        load_end_exclusive='2025-01-01')
    frame = _load_region_frame(config, region)
    ds = AEMOForecastDataset(frame, config, region, 'train')
    positions = np.searchsorted(ds.delivery_unix_seconds, memory.origins)
    if not np.array_equal(ds.delivery_unix_seconds[positions], memory.origins):
        raise ValueError('Raw data no longer aligns with canonical origins')
    history_rows = positions[:, None] + np.arange(-72, 0)
    future_rows = positions[:, None] + np.arange(24)
    if not np.array_equal(ds.target_values_raw[future_rows], memory.curves):
        raise ValueError('Raw labels differ from the frozen canonical bank')
    inputs = dict(history_values=ds.history_values[history_rows],
                  history_calendar=ds.calendar_values[history_rows],
                  future_calendar=ds.calendar_values[future_rows],
                  origin_context=ds.origin_context_values[positions, None])
    for key in INPUT_KEYS[-3:]:
        values = getattr(ds, f'{key}_by_origin')
        inputs[key] = np.stack([values[int(t)] for t in memory.origins])
    validate_inputs(inputs)
    names = list(ds.future_exogenous_feature_names)
    expected = ['max_spare_capacity_mw', 'predispatch_rrp', 'predispatch_total_demand_mw',
                'predispatch_available_generation_mw', 'predispatch_net_interchange_mw',
                'predispatch_covered', 'ic_net_import_mw', 'ic_import_room_mw',
                'ic_export_room_mw', 'ic_main_import_room_mw', 'ic_covered']
    if names != expected or list(ds.ctf_exogenous_feature_names) != ['net_load_mw']:
        raise ValueError('Evidence group schema differs from the audited feature names')
    packed = pack_inputs(inputs)
    if prefix_end == '2022-01-01':
        # The deployed front end must see exactly the baseline's canonical blocks.
        for key in INPUT_KEYS:
            if not np.array_equal(inputs[key], memory.inputs[key]):
                raise ValueError(f'Final preprocessing differs from canonical {key}')
    for k, array in packed.items():
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite prefix input {k}')
        np.save(directory / f'{k}.npy', array, allow_pickle=False)
    audit = dict(identity=identity, preprocessing_hash=_preprocessing_hash(ds, frame, config),
                 target_mean=float(ds.history_standardizer.mean[0]),
                 target_std=float(ds.history_standardizer.std[0]),
                 future_feature_names=names, ctf_feature_names=list(ds.ctf_exogenous_feature_names),
                 quantile_feature_names=list(ds.quantile_exogenous_feature_names),
                 fit_end_exclusive=prefix_end,
                 files={f'{k}.npy': sha(directory / f'{k}.npy') for k in packed})
    atomic_json(directory / 'manifest.json', audit)
    return packed, audit


class DeviceBank:
    def __init__(self, memory, packed, audit, indices, device):
        self.device, self.sigma = device, audit['target_std']
        self.history = torch.as_tensor(np.array(packed['history']), device=device)
        self.future = torch.as_tensor(np.array(packed['future']), device=device)
        self.flat = torch.as_tensor(np.array(packed['flat']), device=device)
        self.curves = torch.as_tensor(((memory.curves - memory.levels[:, None]) / self.sigma).astype(np.float32), device=device)
        self.indices = torch.as_tensor(indices, device=device)
        self.origins = torch.as_tensor(memory.origins, device=device)
        self.levels = memory.levels

    def batch(self, rows, *, with_memory=True):
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        result = dict(history=self.history[rows], future=self.future[rows])
        if with_memory:
            ix = self.indices[rows]
            valid = ix >= 0
            ix = ix.clamp(min=0)
            def gather(array):
                values = array[ix]
                mask = valid.reshape(*valid.shape, *([1] * (values.ndim - 2)))
                return torch.where(mask, values, 0.)
            result.update(candidate_flat=gather(self.flat), candidate_future=gather(self.future),
                candidate_curve=gather(self.curves), valid=valid,
                ages=torch.where(valid, (self.origins[rows, None] - self.origins[ix]) / (365.25 * 86400), 0.).float())
        return result
