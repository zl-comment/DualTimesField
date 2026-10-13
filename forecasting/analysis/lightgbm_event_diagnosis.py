"""Development-only diagnosis of frozen LightGBM; never open test splits.

Reports every region/seed separately. H1 was used for early stopping; H2 is
the existing calibration period, now also inspected for development diagnosis.
No fitting, threshold tuning, or model updates occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..aligned_experiment import atomic_json, read_json, sha, REGIONS, SEEDS
from ..aligned_scoring import QUANTILE_LEVELS
from ..common.cache import load_split
from ..lightgbm.predictor import FrozenLightGBMPredictor


def average(values):
    return float(np.mean(values)) if np.size(values) else None


def correlation(left, right):
    left, right = np.asarray(left).ravel(), np.asarray(right).ravel()
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    return list(zip(np.flatnonzero(mask & np.r_[True, ~mask[:-1]]),
                    np.flatnonzero(mask & np.r_[~mask[1:], True])))


def unique_labels(split):
    times, first, inverse = np.unique(split.target_time_unix.ravel(), return_index=True, return_inverse=True)
    labels = np.asarray(split.actual).ravel()
    values = labels[first]
    if not np.array_equal(values[inverse], labels):
        raise ValueError('Overlapping windows disagree on actual label')
    return times, values


def events(times, values, threshold):
    """Split episodes at missing hours; mark boundary/gap censored episodes."""
    mask = values > threshold
    gaps = np.diff(times) != 3600
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1] | gaps])
    ends = np.flatnonzero(mask & np.r_[~mask[1:] | gaps, True])
    result = []
    for start, end in zip(starts, ends):
        complete = (start > 0 and end < len(times) - 1 and times[start] - times[start - 1] == 3600
                    and times[end + 1] - times[end] == 3600)
        result.append(dict(start=int(times[start]), end=int(times[end]), duration=int(end-start+1),
            complete=bool(complete), peak=float(values[start:end+1].max()),
            peak_time=int(times[start+np.argmax(values[start:end+1])])) )
    return result


def detection(actual, predicted, threshold):
    truth, flag = actual > threshold, predicted > threshold
    tp, fp, fn = int((truth & flag).sum()), int((~truth & flag).sum()), int((truth & ~flag).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/int((~truth).sum()) if (~truth).any() else None)


def oracle_diagnosis(actual, point, threshold=300):
    """Hindsight upper bounds, on identical central 18 steps; not forecasts."""
    target, baseline = actual[:, 3:-3], point[:, 3:-3]
    base_loss = np.abs(target-baseline).mean(1)
    losses = np.stack([np.abs(target-point[:, 3+lag:21+lag]).mean(1) for lag in range(-3, 4)], 1)
    shifted_loss = losses.min(1)
    offset = np.median(target-baseline, axis=1)
    level_loss = np.abs(target-baseline-offset[:, None]).mean(1)
    result = {}
    for name, mask in [('all', np.ones(len(target), bool)), ('spike', (target > threshold).any(1)),
                       ('ordinary', ((target >= 0) & (target <= threshold)).all(1))]:
        result.update({f'{name}_windows':int(mask.sum()), f'{name}_base_mae':average(base_loss[mask]),
            f'{name}_oracle_lag_gain':average((base_loss-shifted_loss)[mask]),
            f'{name}_oracle_level_gain':average((base_loss-level_loss)[mask])})
    return result


def event_rows(split, quantiles, threshold, lead=6):
    times, values = unique_labels(split)
    rows = []
    lookup = {int(t):i for i,t in enumerate(split.origin_unix)}
    for episode in events(times, values, threshold):
        row = dict(episode, threshold=threshold, lead_hours=lead)
        origin = episode['start']-lead*3600
        i = lookup.get(origin)
        eligible = episode['complete'] and i is not None and episode['end'] <= int(split.target_time_unix[i, -1])
        row.update(eligible=eligible, origin_unix=origin)
        if not eligible:
            rows.append(row)
            continue
        target_times = split.target_time_unix[i]
        actual_mask = (target_times >= episode['start']) & (target_times <= episode['end'])
        local_mask = (target_times >= episode['start']-3*3600) & (target_times <= episode['end']+3*3600)
        local_indices = np.flatnonzero(local_mask)
        for name, q in [('q50', 2), ('q95', 4)]:
            prediction = quantiles[i, :, q]
            flag = prediction > threshold
            local_peak = local_indices[np.argmax(prediction[local_indices])]
            candidates = runs(flag)
            overlap = [int(actual_mask[a:b+1].sum()) for a,b in candidates]
            matched = bool(overlap and max(overlap)>0)
            row.update({f'{name}_on_time_detected':bool(flag[actual_mask].any()),
                f'{name}_nearby_detected':bool(flag[local_mask].any()),
                f'{name}_any_horizon_alert':bool(flag.any()),
                f'{name}_local_peak':float(prediction[local_peak]),
                f'{name}_peak_shortfall':episode['peak']-float(prediction[local_peak]),
                f'{name}_local_peak_lag_hours':float((target_times[local_peak]-episode['peak_time'])/3600),
                f'{name}_matched':matched})
            if matched:
                a,b = candidates[int(np.argmax(overlap))]
                row.update({f'{name}_onset_error_hours':float((target_times[a]-episode['start'])/3600),
                    f'{name}_duration_error_hours':int(b-a+1-episode['duration']),
                    f'{name}_matched_forecast_censored':bool(a==0 or b==23)})
        rows.append(row)
    return rows


def diagnose(split, quantiles, input_cuts):
    y, point = np.asarray(split.actual), quantiles[..., 2]
    residual = y-point
    centered = residual-residual.mean(1, keepdims=True)
    errors = y[..., None]-quantiles
    metrics = dict(origins=len(split), mae=float(np.abs(residual).mean()), bias_actual_minus_q50=float(residual.mean()),
        median_actual_minus_q50=float(np.median(residual)), qs=float(2*np.maximum(QUANTILE_LEVELS*errors,(QUANTILE_LEVELS-1)*errors).mean()),
        coverage90=float(((y>=quantiles[...,0]) & (y<=quantiles[...,4])).mean()),
        above_q95_rate=float((y>quantiles[...,4]).mean()), below_q05_rate=float((y<quantiles[...,0]).mean()),
        adjacent_residual_correlation=correlation(residual[:,:-1],residual[:,1:]),
        centered_adjacent_residual_correlation=correlation(centered[:,:-1],centered[:,1:]),
        predicted_total_variation=float(np.abs(np.diff(point,axis=1)).mean()),
        actual_total_variation=float(np.abs(np.diff(y,axis=1)).mean()))
    for name, mask in [('negative',y<0),('ordinary',(y>=0)&(y<=300)),('spike',y>300),('extreme',y>1000)]:
        metrics.update({f'{name}_fraction':float(mask.mean()), f'{name}_mae':average(np.abs(residual)[mask]),
            f'{name}_mae_contribution':float(np.where(mask,np.abs(residual),0).mean()),
            f'{name}_bias':average(residual[mask])})
    for threshold in (300,1000):
        for name, index in [('q50',2),('q95',4)]:
            metrics.update({f't{threshold}_{name}_{k}':v for k,v in detection(y,quantiles[...,index],threshold).items()})
    truth_runs = (y>300) & np.concatenate([np.ones((len(y),1),bool),y[:,:-1]<=300],axis=1)
    pred_runs = (point>300) & np.concatenate([np.ones((len(y),1),bool),point[:,:-1]<=300],axis=1)
    has_event=(y>300).any(1)
    metrics.update(actual_runs_per_window=float(truth_runs.sum(1).mean()), predicted_runs_per_window=float(pred_runs.sum(1).mean()),
        more_predicted_runs_on_event_windows=average((pred_runs.sum(1)>truth_runs.sum(1))[has_event]))
    metrics.update(oracle_diagnosis(y,point))
    conditions = []
    x=split.inputs['future_exogenous']
    masks={'all':np.ones(y.shape,bool),'low_spare_train_p10':x[...,0]<=input_cuts['spare_p10'],
           'high_predispatch_price_train_p90':x[...,1]>=input_cuts['predispatch_p90'],
           'high_predispatch_price_train_p99':x[...,1]>=input_cuts['predispatch_p99']}
    for name,mask in masks.items():
        low_point = mask & (point <= 300)
        conditions.append(dict(condition=name, cells=int(mask.sum()), spike_rate=average((y>300)[mask]),
            bias=average(residual[mask]), median_residual=float(np.median(residual[mask])) if mask.any() else None,
            above_q95_rate=average((y>quantiles[...,4])[mask]), mae=average(np.abs(residual)[mask]),
            above_q50_rate=average((y>point)[mask]), q50_above300_rate=average((point>300)[mask]),
            median_actual=float(np.median(y[mask])) if mask.any() else None,
            median_q50=float(np.median(point[mask])) if mask.any() else None,
            low_q50_cells=int(low_point.sum()), low_q50_actual_spike_rate=average((y>300)[low_point]),
            extreme_rate=average((y>1000)[mask])))
    calibration=[]
    for low,high in [(-np.inf,100),(100,300),(300,1000),(1000,np.inf)]:
        mask=(quantiles[...,4]>low)&(quantiles[...,4]<=high)
        calibration.append(dict(q95_bin=f'({low},{high}]',cells=int(mask.sum()),exceedance_rate=average((y>quantiles[...,4])[mask])))
    horizons=[]
    for h in range(24):
        row=dict(horizon=h+1,mae=float(np.abs(residual[:,h]).mean()),bias=float(residual[:,h].mean()),
                 above_q95_rate=float((y[:,h]>quantiles[:,h,4]).mean()))
        for name,index in [('q50',2),('q95',4)]:
            row.update({f'{name}_{k}':v for k,v in detection(y[:,h],quantiles[:,h,index],300).items()})
        horizons.append(row)
    # Select a single systematic shift in H1 later; never interpret oracle lag as deployable.
    target=y[:,3:-3]
    metrics.update({f'fixed_lag_{lag}_mae':float(np.abs(target-point[:,3+lag:21+lag]).mean()) for lag in range(-3,4)})
    return metrics,conditions,calibration,horizons


def region_job(args):
    root,output,region=args
    root,output=Path(root),Path(output)
    manifest=read_json(root/'jobs.json')
    train=load_split(root,manifest,region,'train')
    train_times,train_y=unique_labels(train)
    x=train.inputs['future_exogenous']
    cuts=dict(spare_p10=float(np.quantile(x[...,0],.1)),predispatch_p90=float(np.quantile(x[...,1],.9)),
              predispatch_p99=float(np.quantile(x[...,1],.99)))
    inventory=[]
    for name,split in [('train',train), *[(s,load_split(root,manifest,region,s)) for s in ('selection','calibration')]]:
        times,values=unique_labels(split)
        for threshold in (300,1000):
            episodes=events(times,values,threshold)
            durations=[e['duration'] for e in episodes if e['complete']]
            inventory.append(dict(region=region,split=name,threshold=threshold,hours=len(times),event_rate=float((values>threshold).mean()),
                episodes=len(episodes),complete_episodes=len(durations),median_duration=float(np.median(durations)) if durations else None,
                long_over18_count=sum(d>18 for d in durations),one_hour_fraction=average(np.asarray(durations)==1)))
    del train,x
    collections={k:[] for k in ('metrics','conditions','calibration','horizons','events')}
    provenance=[]
    for seed in SEEDS:
        started=time.monotonic()
        anchor=FrozenLightGBMPredictor(root/f'runs/lightgbm/seed{seed}/{region}/models',seed=seed,region=region,threads=4)
        for name in ('selection','calibration'):
            split=load_split(root,manifest,region,name)
            folder=output/'predictions'/f'seed{seed}'/region
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{name}.npz'; identity=dict(anchor_manifest=anchor.fingerprint,sample_hash=split.metadata['sample_hash'],input_hash=split.metadata['input_hash'])
            meta_path=folder/f'{name}.json'
            if path.exists() and meta_path.exists():
                saved=read_json(meta_path)
                if saved['identity']!=identity or saved['sha256']!=sha(path): raise ValueError('Diagnostic cache mismatch')
                with np.load(path,allow_pickle=False) as archive: quantiles=archive['quantiles']
            else:
                quantiles=anchor.predict(split)
                np.savez_compressed(path,quantiles=quantiles,origin_unix=split.origin_unix,target_time_unix=split.target_time_unix)
                atomic_json(meta_path,dict(identity=identity,sha256=sha(path)))
            metrics,conditions,calibration,horizons=diagnose(split,quantiles,cuts)
            identity_row=dict(region=region,seed=seed,split=name)
            for key,rows in [('metrics',[metrics]),('conditions',conditions),('calibration',calibration),('horizons',horizons),
                             ('events',event_rows(split,quantiles,300)+event_rows(split,quantiles,1000))]:
                collections[key].extend([dict(identity_row,**row) for row in rows])
            provenance.append(dict(identity_row,**identity,prediction_sha256=sha(path),preprocessing_hash=split.metadata['preprocessing_hash']))
        print(f'{region} seed={seed} diagnostic inference/analysis complete ({time.monotonic()-started:.1f}s)',flush=True)
    return dict(collections=collections,inventory=inventory,provenance=provenance,cuts=dict(region=region,**cuts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(region_job,[(str(args.baseline_root.resolve()),str(args.output.resolve()),r) for r in REGIONS]))
    for key in ('metrics','conditions','calibration','horizons','events'):
        pd.DataFrame([row for result in results for row in result['collections'][key]]).to_csv(args.output/f'{key}_per_seed.csv',index=False)
    pd.DataFrame([row for result in results for row in result['inventory']]).to_csv(args.output/'event_inventory.csv',index=False)
    atomic_json(args.output/'provenance.json',dict(reads_test=False,trains_models=False,
        baseline_jobs_sha256=sha(args.baseline_root/'jobs.json'),script_sha256=sha(__file__),
        helper_source_sha256={name:sha(Path(__file__).parents[1]/name) for name in
            ('lightgbm/predictor.py','common/cache.py','aligned_inputs.py','aligned_scoring.py','aligned_experiment.py')},
        periods=['train:inventory_and_fixed_feature_thresholds_only','selection:2022H1_used_for_early_stopping','calibration:2022H2_development_diagnosis'],
        thresholds=[300,1000],event_lead_hours=6,local_event_matching_tolerance_hours=3,
        event_eligibility='Complete observed episode, origin exists 6h before onset, episode ends within 24h forecast; excluded counts retained',
        q95_flag='Risk flag, not a point prediction or estimated event probability',
        oracle='Uses actual future labels; upper bounds only, not operational gains or learnability proof',
        independence='Origins overlap and seeds share labels; no independence or significance claim',
        input_conditions='Feature thresholds fit train only; associations are not causal importance or proof of incremental predictive information',
        cuts=[r['cuts'] for r in results],cells=[p for r in results for p in r['provenance']]))
    print(f'Diagnostic artifacts saved: {args.output}',flush=True)


if __name__=='__main__':
    main()
