"""Replay saved checkpoints through the original CPU history-batch adapter.

Checks first, middle, and final original inference batches for every saved
neural seed and output; full prediction rescoring is handled by the report.
"""
import json
from pathlib import Path

import numpy as np
import torch

from forecasting.full_v5_anchor_balance import aligned_final as e
from forecasting.full_v5_anchor_balance.confirmation_model import build_model
from forecasting.full_v5_anchor_balance.selection import blend

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/forecasting/v5_latest_aligned_20261023_r2'


def region_audit(spec):
    region, device = spec
    e.c.configure_device(device)
    manifest = json.loads((RUN / 'manifest.json').read_text())
    cfg = manifest['config']
    completed = json.loads((RUN / 'selection_complete.json').read_text())
    e.verify_barrier(completed, cfg)
    e.set_stage('final-test')
    a, meta = e.load_cache_split(e.PARENT / 'data_cache', region, 'test')
    split = e.AlignedSplit({k: a[k] for k in e.INPUT_KEYS}, a['actual'], a['origin_unix'],
                           a['target_time_unix'], meta, a['target'])
    bundles = Path(manifest['bundle_root'])
    tree = e.FrozenLightGBMPredictor(bundles / 'oof' / region / '2022H2/models', seed=2026, region=region)
    baseline = tree.predict(split)
    history, _ = e.c.base.load_store(bundles, region, 2026)
    test = e.bundle_from_aligned(split, baseline, e.timestamp('2022-01-01'), e.timestamp('2022-07-01'))
    offset = len(history.origins)
    store = e.HistoryStore(e.combine_memory(history.a, test), candidates=cfg['candidates'], history_days=cfg['history_days'])
    batch_size = cfg['prediction_batch_size']
    starts = sorted({0, (len(split)//2//batch_size)*batch_size, ((len(split)-1)//batch_size)*batch_size})
    checked, max_difference = 0, 0.
    selected_rows = np.concatenate([np.arange(start, min(start + batch_size, len(split))) for start in starts])
    for seed in cfg['seeds']:
        base_path = RUN / 'runs/B_retrained' / f'seed{seed}' / region / 'test.npz'
        stored_base = e.load_prediction_artifact(base_path)
        np.testing.assert_array_equal(stored_base.quantiles, baseline)
        for mode, controls in (('guarded', e.OUTPUTS), ('raw', {'Full_joint_raw': {}})):
            checkpoint = e.fit_path(RUN, region, seed) / f'{mode}.pt'
            model, selected = e.c.base.restore_checkpoint(checkpoint, model_builder=build_model)
            model.to(device).eval()
            saved = {name: e.load_prediction_artifact(RUN / 'runs' / name / f'seed{seed}' / region / 'test.npz')
                     for name in controls}
            for start in starts:
                rows = np.arange(start, min(start + batch_size, len(split)))
                # Deliberately use the original HistoryStore.batch, not the
                # device-resident cache used by formal training/inference.
                batch = store.batch(offset + rows, device=device)
                assert 'actual' not in batch
                with torch.inference_mode():
                    for name, flags in controls.items():
                        values = model(batch, **flags)['prediction'].cpu().numpy()
                        if mode == 'guarded':
                            values = blend(baseline[rows], values, selected['alpha'])
                        expected = saved[name].quantiles[rows]
                        difference = float(np.max(np.abs(values - expected)))
                        max_difference = max(max_difference, difference)
                        np.testing.assert_array_equal(values, expected, err_msg=f'{region}/{seed}/{name}/{start}')
                        checked += len(rows)
    return {region: dict(exact=True, seeds=cfg['seeds'], batch_starts=starts,
        sampled_origins_per_output=len(selected_rows), neural_outputs_per_seed=5,
        compared_origin_outputs=checked, maximum_absolute_difference=max_difference,
        fixed_tree_predictions_reproduced_all_origins=True,
        adapter='original HistoryStore.batch; formal run used DeviceHistoryStore')}


def main():
    progress = json.loads((RUN / 'progress.json').read_text())
    if progress['stage'] != 'complete':
        raise ValueError('Complete formal evaluation required before checkpoint audit')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    e.verify_manifest(manifest)
    result = e.c.run_pool(region_audit, list(zip(manifest['config']['regions'], manifest['config']['devices'])))
    e.verify_manifest(manifest)
    output = ROOT / 'logs/forecasting/v5_latest_aligned_20261023_r2/checkpoint_replay.json'
    e.atomic_json(output, dict(scope='sampled original inference batches, all nine fits and five neural outputs',
        run_id=manifest['run_id'], generator_sha256=e.c.base.digest(Path(__file__)), regions=result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
