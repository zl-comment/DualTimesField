"""Read-only retrospective diagnosis of frozen V3 predictions and priors.

No parameters are fitted, predictions replaced, or experiments selected here.
All outcome-conditioned slices and component replacements are diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from forecasting.aligned_experiment import read_json, sha, verify_snapshot, atomic_json
from forecasting.aligned_scoring import load_prediction_artifact
from forecasting.full_v3.experiment import completed
from forecasting.full_v3.priors import VARIANTS


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(mean=float(v.mean()), std=float(v.std()),
                p01=float(np.quantile(v, .01)), p50=float(np.median(v)),
                p99=float(np.quantile(v, .99)), abs_p99=float(np.quantile(np.abs(v), .99)))


def metrics(y, p):
    err = p - y
    return dict(mae=float(np.abs(err).mean()), bias=float(err.mean()),
                rmse=float(np.sqrt(np.square(err).mean())),
                total_variation=float(np.abs(np.diff(p, axis=1)).mean()))


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / 'jobs.json')
    verify_snapshot(root, manifest)
    base = Path(manifest['baseline_root'])
    assert sha(base / 'jobs.json') == manifest['baseline_manifest_hash']
    prior_rows, result_rows, slice_rows, round_rows, retrieval_rows, examples = [], [], [], [], [], []
    comparison_rows = []
    for region in ('NSW1', 'QLD1', 'TAS1'):
        print(f'Diagnosing {region}', flush=True)
        cache = base / 'data_cache' / region
        cache_hashes = read_json(cache / 'hashes.json')
        assert sha(cache / 'hashes.json') == manifest['cache_hashes'][region]
        region_dir = root / 'features' / region
        library_hashes = read_json(region_dir / 'library_hashes.json')
        assert sha(region_dir / 'library_hashes.json') == manifest['feature_hashes'][f'{region}/library_hashes.json']
        library = {}
        for key in ('origins','curves','event'):
            assert sha(region_dir / f'{key}.npy') == library_hashes[f'{key}.npy']
            library[key] = np.load(region_dir / f'{key}.npy',mmap_mode='r')
        for variant in VARIANTS:
            job = dict(variant=variant, region=region, seed=2026)
            directory = root / 'runs' / variant / 'seed2026' / region
            assert completed(directory, job, manifest)
            model = read_json(directory / 'models/manifest.json')['model_metadata']
            rounds = np.array(model['selected_rounds'])
            round_rows.append(dict(region=region, variant=variant, median=float(np.median(rounds)),
                minimum=int(rounds.min()), maximum=int(rounds.max()), at_2000=int((rounds == 2000).sum()),
                under_50=int((rounds < 50).sum()), feature_count=model['feature_count']))
        for split in ('train', 'selection', 'calibration', 'test'):
            folder = region_dir / split
            assert sha(folder / 'hashes.json') == manifest['feature_hashes'][f'{region}/{split}/hashes.json']
            hashes = read_json(folder / 'hashes.json')
            def load(name):
                assert sha(folder / f'{name}.npy') == hashes[f'{name}.npy']
                return np.load(folder / f'{name}.npy', mmap_mode='r')
            for key in ('actual.npy', 'metadata.json'):
                assert sha(cache / split / key) == cache_hashes[f'{split}/{key}']
            y = np.load(cache / split / 'actual.npy').astype(np.float64)
            meta = read_json(cache / split / 'metadata.json')
            priors = {v: np.asarray(load(v)) for v in VARIANTS}
            field = np.asarray(load('field_features'), dtype=np.float64) * meta['target_std']
            level = priors[VARIANTS[0]]
            bs, es, bc, ed = np.split(field, 4, axis=1)
            # Float32 persisted features approximate the float64 prior components.
            assert np.max(np.abs(priors[VARIANTS[3]] - level - bs - es)) < .02
            assert np.max(np.abs(priors[VARIANTS[4]] - level - bc - ed)) < .02
            for variant, p in priors.items():
                prior_rows.append(dict(region=region, split=split, variant=variant,
                    **metrics(y, p), residual_target=describe(y-p)))
            for name, p in [('background_shared', level+bs), ('background_conditional', level+bc),
                            ('background_conditional_event_shared', level+bc+es),
                            ('background_shared_event_conditional', level+bs+ed)]:
                prior_rows.append(dict(region=region, split=split, variant=name,
                    **metrics(y, p), residual_target=describe(y-p)))
            if split == 'train':
                train_lower = priors[VARIANTS[4]].min(axis=0)
                train_upper = priors[VARIANTS[4]].max(axis=0)
                continue
            predictions = {}
            for variant in VARIANTS:
                artifact = load_prediction_artifact(root / 'runs' / variant / 'seed2026' / region / f'{split}_predictions.npz')
                assert np.array_equal(artifact.actual, y)
                assert artifact.metadata['base_input_hash'] == meta['input_hash']
                predictions[variant] = np.sort(artifact.quantiles, axis=-1)[..., 2]
                p, q = priors[variant], predictions[variant]
                result_rows.append(dict(region=region, split=split, variant=variant, **metrics(y, q),
                    prior_mae=float(np.abs(p-y).mean()), residual_correction_abs=float(np.abs(q-p).mean()),
                    prior_error_reduced_fraction=float((np.abs(q-y) < np.abs(p-y)).mean())))
            reference = predictions[VARIANTS[1]]
            conditional = predictions[VARIANTS[4]]
            masks = {'negative': y < 0, 'ordinary_0_300': (y >= 0) & (y <= 300), 'high_gt300': y > 300,
                     'conditional_prior_increment_abs_le50': np.abs(priors[VARIANTS[4]]-level) <= 50,
                     'conditional_prior_increment_abs_50_300': (np.abs(priors[VARIANTS[4]]-level) > 50) & (np.abs(priors[VARIANTS[4]]-level) <= 300),
                     'conditional_prior_increment_abs_gt300': np.abs(priors[VARIANTS[4]]-level) > 300,
                     'conditional_prior_outside_train_horizon_range':
                         (priors[VARIANTS[4]]<train_lower)|(priors[VARIANTS[4]]>train_upper)}
            for name, mask in masks.items():
                for variant, q in predictions.items():
                    e, ref = (q-y)[mask], (reference-y)[mask]
                    slice_rows.append(dict(region=region, split=split, group=name, variant=variant,
                        count=int(mask.sum()), frequency=float(mask.mean()),
                        mae=float(np.abs(e).mean()) if e.size else None,
                        bias=float(e.mean()) if e.size else None,
                        mae_gap_vs_features=float((np.abs(e)-np.abs(ref)).mean()) if e.size else None,
                        contribution_to_overall_gap=float((np.abs(e)-np.abs(ref)).sum()/y.size)))
            shared = predictions[VARIANTS[3]]
            dp = priors[VARIANTS[4]]-priors[VARIANTS[3]]
            dr = (conditional-priors[VARIANTS[4]])-(shared-priors[VARIANTS[3]])
            comparison_rows.append(dict(region=region, split=split,
                conditional_minus_shared_mae=float((np.abs(conditional-y)-np.abs(shared-y)).mean()),
                prior_delta_abs=float(np.abs(dp).mean()), residual_delta_abs=float(np.abs(dr).mean()),
                prediction_delta_abs=float(np.abs(conditional-shared).mean()),
                residual_cancels_prior_delta_fraction=float((np.abs(dp+dr)<np.abs(dp)).mean()),
                fraction_conditional_beats_features=float((np.abs(conditional-y)<np.abs(reference-y)).mean())))
            if split != 'test':
                continue
            # Diagnose prior false alarms without pretending these are calibrated probabilities.
            for name, p in [('event_shared',es),('event_conditional',ed),
                            ('prior_shared',priors[VARIANTS[3]]),('prior_conditional',priors[VARIANTS[4]]),
                            ('prediction_conditional',conditional),('prediction_features',reference)]:
                alarm = p>300
                positives = y>300
                retrieval_rows.append(dict(region=region, measure=name,
                    mean_abs=float(np.abs(p).mean()), total_variation=float(np.abs(np.diff(p,axis=1)).mean()),
                    alarm_frequency=float(alarm.mean()),
                    alarm_precision=float(positives[alarm].mean()) if alarm.any() else None,
                    alarm_recall=float(alarm[positives].mean()) if positives.any() else None))
            ws, wc, wd = load('audit_w_shared'),load('audit_w_background'),load('audit_w_event')
            ids = load('audit_indices')
            release = load('audit_max_availability')
            # Slot indices are stable within each query's 24-hour horizon.
            dominant = np.argmax(wd,axis=1)
            retrieval_rows.append(dict(region=region, measure='weights',
                event_effective_candidates=float(np.mean(1/np.maximum((wd**2).sum(1),1e-30))),
                shared_effective_candidates=float(np.mean(1/np.maximum((ws**2).sum(1),1e-30))),
                event_top1_switch_rate=float((np.diff(dominant,axis=1)!=0).mean()),
                event_adjacent_weight_total_variation=float((.5*np.abs(np.diff(wd,axis=2)).sum(axis=1)).mean()),
                event_adjacent_support_change_rate=float(np.any(np.diff(wd>0,axis=2),axis=1).mean()),
                background_shared_top1_disagreement=float((np.argmax(ws,axis=1)!=np.argmax(wc,axis=1)).mean()),
                event_shared_top1_disagreement=float((dominant!=np.argmax(ws,axis=1)[:,None]).mean())))
            # One record per delivery hour, from the origin/horizon with largest error gap.
            regret = np.abs(conditional-y)-np.abs(reference-y)
            selected, seen = [], set()
            for flat in np.argsort(regret.ravel())[::-1]:
                i,h = np.unravel_index(flat, y.shape)
                stamp = int(artifact.target_time_unix[i,h])
                if stamp in seen:continue
                seen.add(stamp);selected.append((i,h))
                if len(selected)==10:break
            for i,h in selected:
                candidates = []
                for slot in np.argsort(wd[i,:,h])[-4:][::-1]:
                    j = int(ids[i,slot])
                    if j < 0:continue
                    candidates.append(dict(origin_unix=int(library['origins'][j]),
                        max_availability=int(release[i,slot]),weight=float(wd[i,slot,h]),
                        historical_price=float(library['curves'][j,h]),historical_event=float(library['event'][j,h])))
                examples.append(dict(region=region, origin_unix=int(artifact.origin_unix[i]),
                    target_unix=int(artifact.target_time_unix[i,h]), horizon=int(h+1),actual=float(y[i,h]),
                    features_prediction=float(reference[i,h]),conditional_prediction=float(conditional[i,h]),
                    conditional_prior=float(priors[VARIANTS[4]][i,h]),recent_level=float(level[i,h]),
                    background_increment=float(bc[i,h]),event=float(ed[i,h]),
                    residual_correction=float(conditional[i,h]-priors[VARIANTS[4]][i,h]),
                    absolute_error_gap=float(regret[i,h]),
                    training_prior_min_at_horizon=float(train_lower[h]),
                    training_prior_max_at_horizon=float(train_upper[h]),historical_event_candidates=candidates))
    result = dict(scope='retrospective diagnostic only, no new fit or parameter selection',
                  run_root=str(root), source_hash=manifest['source_hash'], config_hash=manifest['config_hash'],
                  prior_metrics=prior_rows, prediction_metrics=result_rows, slices=slice_rows,
                  rounds=round_rows, retrieval=retrieval_rows, shared_conditional=comparison_rows, examples=examples)
    atomic_json(output / 'diagnosis.json', result)
    for name, rows in [('prediction_metrics', result_rows), ('slices', slice_rows), ('rounds',round_rows), ('examples',examples)]:
        with (output / f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',default='outputs/forecasting/full_v3_20261013')
    parser.add_argument('--output',default='logs/forecasting/full_v3_diagnosis_20261014')
    args=parser.parse_args()
    result=run(args.run_root,args.output)
    print(f"Saved diagnosis to {args.output}",flush=True)
