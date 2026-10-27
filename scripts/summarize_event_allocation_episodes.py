"""Descriptive hourly and episode diagnostics for a completed frozen allocation.

An episode is an actual-price run >= 1000 on consecutive AEST target hours,
within one target year and one forecast lead. Forecast quantile threshold
exceedance is a deterministic diagnostic, not an estimated event probability.
No episode counts are pooled across leads, neural seeds, or regions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = {'B': 'baseline', 'Full': 'original', 'event_off': 'event_off',
          'upper_only': 'upper_only'}
QUANTILES = {'q50': 2, 'q90': 3}
THRESHOLD = 1000.
HOUR = 3600


def episode_ranges(times, positive):
    """Index ranges [start, stop), breaking on gaps and AEST year boundaries."""
    times = np.asarray(times)
    positive = np.asarray(positive)
    if times.ndim != 1 or positive.shape != times.shape or positive.dtype != bool:
        raise ValueError('Expected one-dimensional times and a same-shape boolean event mask')
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Target hours must be finite, unique and strictly increasing within each lead')
    years = (times+10*HOUR).astype('datetime64[s]').astype('datetime64[Y]')
    ranges, start = [], None
    for i, flag in enumerate(positive):
        contiguous = i > 0 and times[i]-times[i-1] == HOUR and years[i] == years[i-1]
        if start is not None and (not flag or not contiguous):
            ranges.append((start, i))
            start = None
        if flag and start is None:
            start = i
    if start is not None:
        ranges.append((start, len(times)))
    return ranges


def diagnose_series(times, actual, prediction):
    """Return one lead/year/quantile's metrics and actual-episode peak details.

    Hourly TP/FP/FN have the same hourly unit. Episode detected/missed counts
    use actual episodes; false-alarm counts use predicted episodes. These
    different episode denominators are deliberately not called one-to-one TP.
    Peak prediction and its time are restricted to the actual episode window.
    Ties select the earliest target hour, with no search outside the window.
    """
    times = np.asarray(times, dtype=np.int64)
    actual, prediction = (np.asarray(x, dtype=np.float64) for x in (actual, prediction))
    if actual.shape != times.shape or prediction.shape != times.shape:
        raise ValueError('Actual, prediction and target hours must have identical one-dimensional shapes')
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError('Actual and predicted prices must be finite')
    truth, forecast = actual >= THRESHOLD, prediction >= THRESHOLD
    observed_runs, predicted_runs = episode_ranges(times, truth), episode_ranges(times, forecast)
    detected = sum(bool(forecast[start:stop].any()) for start, stop in observed_runs)
    false_alarms = sum(not bool(truth[start:stop].any()) for start, stop in predicted_runs)
    result = dict(cells=int(len(times)), actual_positive_hours=int(truth.sum()),
        predicted_positive_hours=int(forecast.sum()),
        hourly_tp=int(np.sum(truth & forecast)), hourly_fp=int(np.sum(~truth & forecast)),
        hourly_fn=int(np.sum(truth & ~forecast)), hourly_tn=int(np.sum(~truth & ~forecast)),
        actual_episodes=len(observed_runs), detected_actual_episodes=detected,
        missed_actual_episodes=len(observed_runs)-detected,
        predicted_episodes=len(predicted_runs), false_alarm_predicted_episodes=false_alarms)
    details = []
    for episode, (start, stop) in enumerate(observed_runs, 1):
        actual_peak = start+int(np.argmax(actual[start:stop]))
        predicted_peak = start+int(np.argmax(prediction[start:stop]))
        details.append(dict(episode=episode, start_unix=int(times[start]), end_unix=int(times[stop-1]),
            hours=stop-start, actual_peak_time_unix=int(times[actual_peak]),
            actual_peak=float(actual[actual_peak]), prediction_peak_time_unix=int(times[predicted_peak]),
            prediction_peak=float(prediction[predicted_peak]),
            peak_amplitude_error=float(prediction[predicted_peak]-actual[actual_peak]),
            peak_time_error_hours=float((times[predicted_peak]-times[actual_peak])/HOUR),
            prediction_at_actual_peak=float(prediction[actual_peak]),
            actual_peak_hour_error=float(prediction[actual_peak]-actual[actual_peak]),
            detected=bool(forecast[start:stop].any())))
    # None means no actual episode, never a zero peak error or a successful forecast.
    result.update(episode_peak_count=len(details),
        mean_peak_amplitude_error=float(np.mean([d['peak_amplitude_error'] for d in details])) if details else None,
        mean_abs_peak_amplitude_error=float(np.mean([abs(d['peak_amplitude_error']) for d in details])) if details else None,
        mean_peak_time_error_hours=float(np.mean([d['peak_time_error_hours'] for d in details])) if details else None,
        mean_abs_peak_time_error_hours=float(np.mean([abs(d['peak_time_error_hours']) for d in details])) if details else None)
    return result, details


def _csv(path, rows, empty_fields):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_fields
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(output: Path, cfg: dict) -> dict:
    """Read frozen audit predictions; write only a new output/episodes directory."""
    if float(cfg.get('episode_threshold', THRESHOLD)) != THRESHOLD:
        raise ValueError('The episode threshold is fixed at 1000')
    if cfg.get('episode_quantiles', list(QUANTILES)) != list(QUANTILES):
        raise ValueError('Episode diagnostics are fixed to q50 and q90')
    output = Path(output).resolve()
    destination = output/'episodes'
    index = json.loads((output/'audit'/'prediction_manifest.json').read_text())
    if not isinstance(index, list) or not index:
        raise ValueError('A nonempty completed prediction manifest is required')
    destination.mkdir(parents=False, exist_ok=False)
    summaries, details, sources = [], [], []
    for item in index:
        path = Path(item['path'])
        with path.open('rb') as stream:
            actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_hash != item['sha256']:
            raise ValueError(f'Prediction hash changed: {path}')
        sources.append(dict(case_id=item['case_id'], path=str(path.resolve()), sha256=actual_hash))
        with np.load(path, allow_pickle=False) as data:
            target_times = np.asarray(data['target_time_unix'], dtype=np.int64)
            actual = np.asarray(data['actual'], dtype=np.float64)
            forecasts = {name: np.asarray(data[key], dtype=np.float64) for name, key in MODELS.items()}
        if actual.ndim != 2 or actual.shape[1] != 24 or target_times.shape != actual.shape:
            raise ValueError('Expected the complete [origin, 24] target grid')
        if any(q.shape != (*actual.shape, 5) for q in forecasts.values()):
            raise ValueError('Prediction grids differ from the archived actual grid')
        identity = {key: item[key] for key in ('case_id', 'region', 'seed')}
        for lead in range(24):
            times = target_times[:, lead]
            if np.any(np.diff(times) <= 0):
                raise ValueError('A target hour is repeated or out of order within a lead')
            years = (times+10*HOUR).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
            for year in np.unique(years):
                mask = years == year
                for model, forecast in forecasts.items():
                    for quantile, column in QUANTILES.items():
                        row_identity = dict(identity, lead_hours=lead, target_year_aest=int(year),
                                            model=model, quantile=quantile)
                        scores, peaks = diagnose_series(times[mask], actual[mask, lead], forecast[mask, lead, column])
                        summaries.append(dict(row_identity, **scores))
                        details.extend(dict(row_identity, **peak) for peak in peaks)
    _csv(destination/'hourly_and_episode_metrics.csv', summaries,
         ['case_id', 'region', 'seed', 'lead_hours', 'target_year_aest', 'model', 'quantile', 'cells'])
    _csv(destination/'actual_episode_peaks.csv', details,
         ['case_id', 'region', 'seed', 'lead_hours', 'target_year_aest', 'model', 'quantile',
          'episode', 'start_unix', 'end_unix', 'hours', 'actual_peak', 'prediction_peak'])
    result = dict(cases=len(index), summary_rows=len(summaries), actual_episode_peak_rows=len(details),
        threshold=THRESHOLD, quantiles=list(QUANTILES), models=list(MODELS),
        units='one case, one lead, one AEST target year; unique target hours within each lead',
        lead_hours='0 through 23 as encoded by target_time_unix-origin_unix',
        episode_definition='actual >= 1000, consecutive 3600-second target hours; split at missing hours and AEST target-year boundaries',
        hourly_confusion='TP/FP/FN/TN at the unique target-hour unit within each lead',
        episode_detection='actual episode detected if any forecast >= 1000 within its window; misses use actual episodes; false alarms use predicted episodes with no actual positive hour',
        peak_definition='peak amplitude and timing restricted to each actual episode window; earliest peak in a tie',
        empty_episode_groups=sum(row['actual_episodes']==0 for row in summaries),
        empty_peak_errors='null in JSON-compatible values / blank in CSV; no actual episode is not zero error',
        source_predictions=sources,
        limitations=[
            'Quantile threshold exceedance is not an event occurrence probability.',
            'Conditioning peak evaluation on an observed episode omits timing errors outside that window; hourly false alarms and misses are reported separately.',
            'Leads and neural seeds repeat target hours and episodes; these row counts are not independent event totals.',
            'Episodes at year boundaries or missing-hour gaps are split and can be censored.',
            'These diagnostics are descriptive and do not enter checkpoint selection or the screening decision.'])
    (destination/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.audit_root, {}), ensure_ascii=False, indent=2))
