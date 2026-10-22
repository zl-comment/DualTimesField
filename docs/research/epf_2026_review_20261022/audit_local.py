"""Read-only rescoring of existing predictions; no training or model selection."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
sys.path.insert(0, str(REPO))
from forecasting.aligned_scoring import load_prediction_artifact, score_predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    archive = REPO / 'logs/forecasting/v5_joint_history_20261020_final_seed2026_r1'
    provenance = json.loads((archive / 'archive_provenance.json').read_text())
    for name, expected in provenance['sha256'].items():
        assert sha(archive / name) == expected, name
    run = Path(provenance['source'])
    summary = json.loads((archive / 'summary.json').read_text())
    output_rows, evidence = [], []
    for region in ('NSW1', 'QLD1', 'TAS1'):
        manifest = json.loads((archive / region / 'manifest.json').read_text())
        canonical = None
        for model, expected in summary['regions'][region].items():
            if model.startswith('baseline_'):
                info = manifest['baseline_artifacts'][model]
                path = Path(info['path'])
                assert sha(path) == info['sha256'], path
            else:
                path = run / region / f'{model}.npz'
            artifact = load_prediction_artifact(path)
            aligned = (artifact.origin_unix, artifact.target_time_unix, artifact.actual)
            if canonical is None:
                canonical = aligned
            else:
                assert all(np.array_equal(a, b) for a, b in zip(canonical, aligned)), (region, model)
            scores = score_predictions(artifact)
            for metric in ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_80', 'coverage_90'):
                assert abs(scores[metric] - expected[metric]) < 1e-9, (region, model, metric)
            output_rows.append(dict(region=region, model=model,
                **{m: scores[m] for m in ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_80', 'coverage_90')},
                n_origins=len(artifact.actual), n_cells=int(artifact.actual.size)))
            evidence.append(dict(region=region, model=model, path=str(path), sha256=sha(path),
                                 aligned_origins_targets_labels=True, scores_recomputed=True))
        print('Rescored', region, flush=True)
    models = sorted({r['model'] for r in output_rows})
    all_rows = []
    for model in models:
        cells = [r for r in output_rows if r['model'] == model]
        row = dict(model=model, seed=2026, period='2023–2024',
            **{m: float(np.mean([r[m] for r in cells])) for m in
               ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_80', 'coverage_90')})
        for r in cells:
            row[r['region'] + '_mae'] = r['mae']
        all_rows.append(row)
    excluded = {'B_retrained', 'Anchor_joint', 'event_off_same_head', 'time_shift_off_same_head'}
    for metric in ('mae', 'quantile_score_proxy', 'rmse_pooled'):
        order = sorted(all_rows, key=lambda r: r[metric])
        for i, r in enumerate(order, 1):
            r[metric + '_rank_all17'] = i
        subset = [r for r in order if r['model'] not in excluded]
        for i, r in enumerate(subset, 1):
            r[metric + '_rank_13'] = i
    all_rows.sort(key=lambda r: r['mae'])
    (OUT / 'local_rankings.json').write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    (OUT / 'local_regional.json').write_text(json.dumps(output_rows, indent=2, ensure_ascii=False))
    (OUT / 'local_audit.json').write_text(json.dumps(dict(passed=True, archives_hash_verified=len(provenance['sha256']),
        rescored_artifacts=len(evidence), n_models=len(models), n_regions=3,
        n_origins_per_region=17521, n_cells_per_region=420504,
        ranking_scope='Descriptive same samples and scoring, not same information or tuning budget; retrospective single seed',
        evidence=evidence), indent=2))
    print(json.dumps([r for r in all_rows if r['model'] in ('Full_joint', 'baseline_lightgbm', 'baseline_xgboost')], indent=2))


if __name__ == '__main__':
    main()
