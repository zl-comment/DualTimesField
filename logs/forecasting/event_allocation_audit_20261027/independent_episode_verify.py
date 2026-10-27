"""Independent episode audit: no imports from production diagnostics or scorers."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))

assert read(root/'progress.json')['stage']=='complete'
manifest = read(root/'manifest.json')
verification = read(root/'verification.json')
assert manifest['run_id']==verification['run_id']
for relative, expected in verification['artifact_sha256'].items():
    if relative.startswith('episodes/') or relative=='episodes_complete.json':
        assert sha(root/relative)==expected, relative
marker = read(root/'episodes_complete.json')
identity = {k:v for k,v in marker.items() if k!='completion_id'}
assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==marker['completion_id']
files = {str(path.relative_to(root)):sha(path) for path in (root/'episodes').rglob('*') if path.is_file()}
assert files==marker['artifact_sha256']
summary = read(root/'episodes/summary.json')
assert summary['threshold']==1000. and summary['quantiles']==['q50','q90']
index = read(root/'audit/prediction_manifest.json')
assert summary['cases']==len(index)==9
ids = ('case_id','region','seed','lead_hours','target_year_aest','model','quantile')
def key(row):
    return tuple(str(row[k]) for k in ids)
saved_rows = csv_rows(root/'episodes/hourly_and_episode_metrics.csv')
row_lookup = {key(row):row for row in saved_rows}
assert len(row_lookup)==len(saved_rows)==summary['summary_rows']
saved_peaks = csv_rows(root/'episodes/actual_episode_peaks.csv')
peak_lookup = {(key(row),int(row['episode'])):row for row in saved_peaks}
assert len(peak_lookup)==len(saved_peaks)==summary['actual_episode_peak_rows']
checked_rows, checked_peaks, checked_scalars, empty_groups, maximum_error = 0,0,0,0,0.
seen, seen_peaks = set(),set()

# Vectorized segment endpoints, independently of the production loop.
def segments(times, positive):
    years=(times+36000).astype('datetime64[s]').astype('datetime64[Y]')
    connected=(np.diff(times)==3600)&(years[1:]==years[:-1])
    starts=np.flatnonzero(positive & ~np.r_[False,positive[:-1]&connected])
    stops=np.flatnonzero(positive & ~np.r_[positive[1:]&connected,False])+1
    assert len(starts)==len(stops)
    return list(zip(starts,stops))

def check_value(saved, expected, context):
    global checked_scalars, maximum_error
    checked_scalars+=1
    if expected is None:
        assert saved=='', context
    elif isinstance(expected,(bool,np.bool_)):
        assert saved==str(bool(expected)), context
    elif isinstance(expected,(int,np.integer)):
        assert int(saved)==int(expected), context
    else:
        error=abs(float(saved)-float(expected))
        maximum_error=max(maximum_error,error)
        assert error<=1e-9,(context,error)

source_lookup = {row['case_id']:row for row in summary['source_predictions']}
assert len(source_lookup)==len(index)
for item in index:
    assert sha(item['path'])==item['sha256']
    assert source_lookup[item['case_id']]['path']==item['path']
    assert source_lookup[item['case_id']]['sha256']==item['sha256']
    with np.load(item['path'],allow_pickle=False) as data:
        y=np.asarray(data['actual'],dtype=np.float64)
        times=np.asarray(data['target_time_unix'],dtype=np.int64)
        origins=np.asarray(data['origin_unix'],dtype=np.int64)
        forecasts={name:np.asarray(data[array],dtype=np.float64) for name,array in
                   [('B','baseline'),('Full','original'),('event_off','event_off'),('upper_only','upper_only')]}
    assert times.shape==y.shape and y.shape[1]==24
    np.testing.assert_array_equal(times,origins[:,None]+3600*np.arange(24))
    np.testing.assert_array_equal(forecasts['Full'][...,3],forecasts['upper_only'][...,3])
    np.testing.assert_array_equal(forecasts['event_off'][...,2],forecasts['upper_only'][...,2])
    for lead in range(24):
        timeline=times[:,lead]
        assert np.all(np.diff(timeline)>0)
        years=(timeline+36000).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        for year in np.unique(years):
            selection=years==year
            target,actual=timeline[selection],y[selection,lead]
            truth=actual>=1000.
            actual_segments=segments(target,truth)
            for model,q in forecasts.items():
                for quantile,column in [('q50',2),('q90',3)]:
                    row_identity=dict(case_id=item['case_id'],region=item['region'],seed=item['seed'],
                                      lead_hours=lead,target_year_aest=int(year),model=model,quantile=quantile)
                    record_key=key(row_identity)
                    saved=row_lookup[record_key]
                    seen.add(record_key)
                    predicted=q[selection,lead,column]
                    forecast=predicted>=1000.
                    pred_segments=segments(target,forecast)
                    hit=sum(int(np.count_nonzero(forecast[s:e])>0) for s,e in actual_segments)
                    false_alarm=sum(int(np.count_nonzero(truth[s:e])==0) for s,e in pred_segments)
                    values=dict(cells=len(target),actual_positive_hours=np.count_nonzero(truth),
                        predicted_positive_hours=np.count_nonzero(forecast),
                        hourly_tp=np.count_nonzero(truth&forecast),hourly_fp=np.count_nonzero(~truth&forecast),
                        hourly_fn=np.count_nonzero(truth&~forecast),hourly_tn=np.count_nonzero(~truth&~forecast),
                        actual_episodes=len(actual_segments),detected_actual_episodes=hit,
                        missed_actual_episodes=len(actual_segments)-hit,predicted_episodes=len(pred_segments),
                        false_alarm_predicted_episodes=false_alarm,episode_peak_count=len(actual_segments))
                    errors,timing=[],[]
                    for number,(start,stop) in enumerate(actual_segments,1):
                        # Select the earliest matching peak index explicitly;
                        # do not import or call the production peak helper.
                        true_max=float(np.max(actual[start:stop]))
                        pred_max=float(np.max(predicted[start:stop]))
                        a=start+int(np.flatnonzero(actual[start:stop]==true_max)[0])
                        p=start+int(np.flatnonzero(predicted[start:stop]==pred_max)[0])
                        amplitude=pred_max-true_max
                        offset=float(target[p]-target[a])/3600
                        errors.append(amplitude)
                        timing.append(offset)
                        detail=dict(episode=number,start_unix=int(target[start]),end_unix=int(target[stop-1]),
                            hours=stop-start,actual_peak_time_unix=int(target[a]),actual_peak=true_max,
                            prediction_peak_time_unix=int(target[p]),prediction_peak=pred_max,
                            peak_amplitude_error=amplitude,peak_time_error_hours=offset,
                            prediction_at_actual_peak=float(predicted[a]),actual_peak_hour_error=float(predicted[a]-actual[a]),
                            detected=bool(np.count_nonzero(forecast[start:stop])>0))
                        peak_key=(record_key,number)
                        saved_peak=peak_lookup[peak_key]
                        seen_peaks.add(peak_key)
                        for field,value in detail.items():
                            check_value(saved_peak[field],value,(peak_key,field))
                        checked_peaks+=1
                    values.update(mean_peak_amplitude_error=float(np.mean(errors)) if errors else None,
                        mean_abs_peak_amplitude_error=float(np.mean(np.abs(errors))) if errors else None,
                        mean_peak_time_error_hours=float(np.mean(timing)) if timing else None,
                        mean_abs_peak_time_error_hours=float(np.mean(np.abs(timing))) if timing else None)
                    for field,value in values.items():
                        check_value(saved[field],value,(record_key,field))
                    empty_groups+=int(not actual_segments)
                    checked_rows+=1
    print('independently checked episodes:',item['case_id'],flush=True)
assert seen==set(row_lookup) and seen_peaks==set(peak_lookup)
assert empty_groups==summary['empty_episode_groups']
# The two exact prediction identities must propagate to every episode metric.
for row in saved_rows:
    if (row['model'],row['quantile']) not in [('Full','q90'),('event_off','q50')]:
        continue
    other=dict(row,model='upper_only')
    matched=row_lookup[key(other)]
    assert all(matched[field]==value for field,value in row.items() if field!='model')
for row in saved_peaks:
    if (row['model'],row['quantile']) not in [('Full','q90'),('event_off','q50')]:
        continue
    other=dict(row,model='upper_only')
    matched=peak_lookup[(key(other),int(row['episode']))]
    assert all(matched[field]==value for field,value in row.items() if field!='model')
result=dict(status='passed',run_id=manifest['run_id'],cases=len(index),
    independent_implementation='Standalone NumPy vectorized segment endpoints and explicit earliest-peak selection; no production helper imports',
    summary_rows_checked=checked_rows,episode_peak_rows_checked=checked_peaks,
    scalar_values_checked=checked_scalars,maximum_float_abs_error=maximum_error,
    empty_episode_groups_retained=empty_groups,threshold=1000.,
    full_upper_only_q90_exact=True,event_off_upper_only_q50_exact=True,
    unique_target_hours_within_each_lead=True,aest_target_year_boundaries_verified=True,
    missing_hour_boundaries_verified=True,original_predictions_and_metadata_unmodified=True,
    limitations=['Counts are retained separately by case, lead and AEST target year; no total of independent market episodes is inferred.',
        'Quantile threshold exceedances are deterministic diagnostics, not event probabilities.',
        'Peak timing is restricted to the observed actual episode window; it cannot detect forecast peaks outside that window.',
        'Neural seeds and forecast leads repeat the same target hours; zero-episode groups retain blank peak errors.'])
(root/'independent_episode_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
