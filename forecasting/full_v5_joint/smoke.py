"""Real-data integration check; NOT the expanded OOF experiment or a leaderboard."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..aligned_inputs import AlignedSplit, INPUT_KEYS
from ..full_v5.data import load_cache_split
from .data import bundle_from_aligned, HistoryStore, digest
from .protocol import timestamp, segment_rows
from .train import fit, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(Path('configs/full_v5_joint/protocol.yaml').read_text())
    cfg.update(epochs=3, candidates=8)
    summary = {}
    for region in cfg['regions']:
        arrays, meta = load_cache_split('outputs/forecasting/aligned_baselines_20261012/data_cache', region, 'calibration')
        source = Path('outputs/forecasting/v5_calibrated_history_20261015_r3/baseline_predictions')/f'{region}_calibration.npz'
        with np.load(source, allow_pickle=False) as archive:
            if not np.array_equal(archive['origin_unix'], arrays['origin_unix']) or not np.array_equal(archive['target_time_unix'], arrays['target_time_unix']):
                raise ValueError('Archived tree forecasts do not align')
            q = archive['quantiles'].copy()
        # The archived tree was selected through2022H1; never use it for older OOF rows.
        rows = np.flatnonzero((arrays['origin_unix'] >= timestamp('2022-07-02')) &
                              (arrays['origin_unix'] < timestamp('2022-09-14')))
        split = AlignedSplit({k:arrays[k][rows] for k in INPUT_KEYS}, arrays['actual'][rows],
                             arrays['origin_unix'][rows], arrays['target_time_unix'][rows], meta, arrays['target'][rows])
        store = HistoryStore(bundle_from_aligned(split, q[rows], timestamp('2022-01-01'), timestamp('2022-07-01')),
                             candidates=cfg['candidates'], history_days=cfg['history_days'])
        train = segment_rows(store.origins, '2022-08-01', '2022-08-15')[::6]
        selection = segment_rows(store.origins, '2022-09-01', '2022-09-04')[::3]
        evaluation = segment_rows(store.origins, '2022-09-10', '2022-09-13')[::3]
        folder = root/region
        folder.mkdir()
        manifest = dict(scope='smoke_archived2022_not_formal_oof', region=region, seed=cfg['seed'],
                        baseline_sha256=digest(source), cache_preprocessing_hash=meta['preprocessing_hash'],
                        rows=dict(train=len(train), selection=len(selection), evaluation=len(evaluation)),
                        warning='Integration only: sparse origins,3epochs, short history. No SOTA or expanded-history claim.')
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
        model, epoch = fit(store, train, selection, cfg, folder)
        summary[region] = dict(best_epoch=epoch, metrics=evaluate(model, store, evaluation, cfg, folder))
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
