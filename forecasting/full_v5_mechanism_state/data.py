"""Read the SAME vintage-aligned source fields; audit their shared-input identity.

OOF folds use different z-score scalers, so standardized values must never be
treated as comparable MW. Source fields below are already in the seven blocks.
"""
import json
from pathlib import Path
import numpy as np
import torch

from ..full_v5_anchor_balance import experiment as base
from ..datasets import _load_region_frame, _valid_origins, _fit_standardizer, _to_unix_seconds
from .policy import validate_inputs


def load_inputs(store, paths):
    configurations = [json.loads(p.with_name('config.json').read_text()) for p in paths]
    region = json.loads(paths[0].with_suffix('.json').read_text())['region']
    source_paths = {}
    for block in ('future_exogenous', 'predispatch_exogenous', 'interconnector_exogenous'):
        found = {str((Path(c['project_root']) / c[block]['region_files'][region]).resolve()) for c in configurations}
        if len(found) != 1:
            raise ValueError('OOF folds must share the same source archive')
        source_paths[block] = Path(found.pop())
    fields = {}
    net_load_archive = None
    for block, path in source_paths.items():
        with np.load(path, allow_pickle=False) as z:
            origins = z['forecast_origin_unix']
            if np.any(np.diff(origins) <= 0):
                raise ValueError('Archive origins must be sorted and unique')
            ids = np.searchsorted(origins, store.origins)
            if np.any(ids >= len(origins)) or not np.array_equal(origins[ids], store.origins):
                raise ValueError('Missing source origins')
            for key in ('source_run_unix', 'source_last_changed_unix', 'source_stpasa_run_unix'):
                if key in z and np.any(z[key][ids] > store.origins):
                    raise ValueError('Future source vintage')
            names = {'future_exogenous': ('max_spare_capacity_mw', 'demand50_mw', 'uigf_mw'),
                     'predispatch_exogenous': ('predispatch_covered',),
                     'interconnector_exogenous': ('ic_import_room_mw', 'ic_export_room_mw', 'ic_covered')}[block]
            fields.update({name: z[name][ids].astype(np.float64) for name in names})
            if block == 'future_exogenous':
                # Preserve the original float32 subtraction before z-scoring.
                net_load_archive = (origins.copy(), z['demand50_mw'] - z['uigf_mw'])
                fields['net_load_mw'] = net_load_archive[1][ids].astype(np.float64)
    x = np.stack((fields['max_spare_capacity_mw'], fields['net_load_mw'],
        fields['ic_import_room_mw'], fields['ic_export_room_mw'], fields['predispatch_covered'],
        fields['ic_covered']), axis=-1)
    validate_inputs(x)
    # Audit affine channels and replay net load's invertible soft saturation in
    # EACH OOF fold. Audit coefficients never enter prediction/selection.
    offset, errors = 0, []
    csv_sources = {}
    for path, cfg in zip(paths, configurations):
        with np.load(path, allow_pickle=False) as bundle:
            future = bundle['future']
        raw = x[offset:offset + len(future)]
        for j, column in enumerate((10, 21, 17, 18, 15, 20)):
            a, b = raw[..., j].ravel(), future[..., column].astype(np.float64).ravel()
            if column == 21:
                # Net load has an invertible log soft saturation in the shared
                # pipeline; it is NOT affine in the tails. Replay that exact
                # training-only transform, instead of bypassing its audit.
                frame = _load_region_frame(cfg, region)
                ctf = cfg['ctf_exogenous']
                if ctf.get('features') != ['net_load_mw'] or ctf.get('clip_to_train_quantiles') != [.005, .995] or ctf.get('out_of_range') != 'log':
                    raise ValueError('Unexpected shared net-load transform')
                csv_path = (Path(cfg['project_root']) / cfg['data']['region_files'][region]).resolve()
                csv_sources[str(csv_path)] = base.digest(csv_path)
                train_times = _to_unix_seconds(frame[cfg['data']['delivery_column']])[_valid_origins(frame, cfg, 'train')]
                all_times, values = net_load_archive
                train_ids = np.flatnonzero(np.isin(all_times, train_times))
                if not len(train_ids):
                    raise ValueError('No original training origins for net-load transform audit')
                scaler = _fit_standardizer(values[train_ids].reshape(-1, 1))
                transformed = scaler.transform(values[..., None])
                lower, upper = np.quantile(transformed[train_ids].reshape(-1, 1), [.005, .995], axis=0)
                transformed = (np.clip(transformed, lower, upper)
                    - np.log1p(np.maximum(lower-transformed, 0))
                    + np.log1p(np.maximum(transformed-upper, 0))).astype(np.float32)
                current_ids = np.searchsorted(all_times, store.origins[offset:offset+len(future)])
                expected = transformed[current_ids, :, 0].ravel()
                error = float(np.max(np.abs(b-expected) / (1+np.abs(b))))
            elif np.ptp(a) == 0:
                error = float(np.ptp(b))
            else:
                a = (a - a.mean()) / a.std()
                slope = np.mean(a * (b - b.mean()))
                error = float(np.max(np.abs(b - (slope * a + b.mean())) / (1 + np.abs(b))))
                if slope <= 0:
                    raise ValueError('Public input channel has wrong direction')
            if error > 2e-5:
                raise ValueError(f'Raw source differs from public input: {path}, column {column}, {error}')
            errors.append(error)
        offset += len(future)
    if offset != len(x):
        raise ValueError('Source length mismatch')
    return x, dict(sources={**{str(p): base.digest(p) for p in source_paths.values()}, **csv_sources},
        same_information_as_shared_inputs=True, max_transform_relative_error=max(errors),
        availability='archive timestamp proxies; not independently certified first publication')


class MechanismStore:
    def __init__(self, store, inputs):
        self.store = store
        self.inputs = torch.as_tensor(validate_inputs(inputs), dtype=torch.float64, device=store.device)

    def __getattr__(self, name):
        return getattr(self.store, name)

    def batch(self, rows):
        batch = self.store.batch(rows)
        batch['mechanism_inputs'] = self.inputs[torch.as_tensor(rows, device=self.store.device)]
        return batch
