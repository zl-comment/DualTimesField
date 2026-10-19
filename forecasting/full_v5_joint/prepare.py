"""Materialize honest historical tree forecasts; each fold refits preprocessing."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from ..aligned_inputs import build_aligned_splits, _array_hash
from ..aligned_baselines import fit_predict
from ..datasets import load_forecast_config
from .data import bundle_from_aligned, digest
from .protocol import Fold, timestamp, DAY


def purge(split, before):
    rows = np.flatnonzero(split.origin_unix + DAY < timestamp(before))
    if not len(rows):
        raise ValueError('Empty purged fold')
    inputs = {k: v[rows] for k, v in split.inputs.items()}
    origins, times, actual = split.origin_unix[rows], split.target_time_unix[rows], split.actual[rows]
    meta = dict(split.metadata, input_hash=_array_hash(inputs),
                sample_hash=_array_hash(dict(origin=origins, target_time=times, actual=actual)))
    return replace(split, inputs=inputs, actual=actual, target=split.target[rows],
                   origin_unix=origins, target_time_unix=times, metadata=meta, dataset=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/full_v5_joint/protocol.yaml')
    p.add_argument('--region', required=True, choices=['NSW1', 'QLD1', 'TAS1'])
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--final-half', action='store_true', help='2022H2 deployment fold; same final train/selection cutoff as neural modules')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    fold = Fold(args.year, final_half=args.final_half)
    base = load_forecast_config(cfg['base_config'])
    if 'data_project_root' in cfg:
        base['project_root'] = cfg['data_project_root']
    configuration = fold.configure(base)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.json').write_text(json.dumps(configuration, indent=2))
    splits = build_aligned_splits(configuration, args.region)
    train = purge(splits['train'], fold.train_end)
    selection = purge(splits['selection'], fold.prediction_start)
    evaluation = purge(splits['calibration'], fold.prediction_end)
    result = fit_predict('lightgbm', train, selection, evaluation, seed=cfg['seed'],
                         params=base['experiment']['baseline_parameters']['lightgbm'],
                         progress_path=root / 'progress.json', model_directory=root / 'models')
    arrays = bundle_from_aligned(evaluation, result.quantiles,
                                 timestamp(fold.train_end), int(selection.origin_unix.max() + DAY))
    path = root / 'bundle.npz'
    np.savez_compressed(path, **arrays)
    metadata = dict(scope='chronological_oof', region=args.region, year=args.year,
                    final_half=args.final_half, seed=cfg['seed'], array_sha256=digest(path),
                    configuration_sha256=digest(root / 'config.json'),
                    model_files={str(f.relative_to(root)): digest(f) for f in sorted((root/'models').rglob('*')) if f.is_file()},
                    train=train.metadata, selection=selection.metadata, prediction=evaluation.metadata,
                    trained_through=int(arrays['trained_before'][0]), selected_through=int(arrays['selected_before'][0]),
                    availability_audit='archive_timestamp_proxies_only', baseline=result.metadata)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(bundle=str(path), rows=len(evaluation)), indent=2))


if __name__ == '__main__':
    main()
