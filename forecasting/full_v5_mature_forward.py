"""Forward evaluation and independently reconstructed candidate evidence.

Called only by the committed stage runner. Actual-price groups are reporting
masks applied after all predictions have been constructed, never model inputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .full_v5_mature_controls import build_residual_controls, apply_shared_beta
from .full_v5_mature_evaluation import (
    paired_scores, paired_cell_losses, daily_paired_losses,
    paired_day_block_inference, apply_decision_rules, REQUIRED_INTEGRITY_CHECKS,
)
from .full_v5_mature_sources import DAY, HOUR, purged_period_rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def independent_ids(data, row, method):
    """Boolean permission masks independently check searchsorted pool bounds."""
    origins, t = data['origin'], int(data['origin'][row])
    permission = origins+DAY <= t
    if method == 'R28':
        return np.flatnonzero(permission & (origins > t-28*DAY))
    same_hour = ((origins+10*HOUR)//HOUR % 24) == ((t+10*HOUR)//HOUR % 24)
    if method == 'Recent16':
        return np.flatnonzero(permission & same_hour)[-16:]
    pool = np.flatnonzero(permission & same_hour & (origins >= t-730*DAY))
    dims = [1, 2, 3, 6, 7, 8]
    # Retain the original FP32 reduction order, but reconstruct stable ordering
    # with Python's stable sort instead of the production argsort.
    distance = ((data['z'][pool][:, :, dims]-data['z'][row][:, dims])**2).mean(axis=(1, 2))
    chosen = list(pool[-8:])
    for _, candidate in sorted(zip(distance.tolist(), pool.tolist()), key=lambda pair:pair[0]):
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= 16:
            break
    return np.asarray(chosen, dtype=np.int64)


def audit_controls(data, rows, controls, beta):
    """Actual-library probes complement committed synthetic boundary tests."""
    rng = np.random.default_rng(20261027)
    positions = np.unique(np.r_[0, len(rows)-1, rng.choice(len(rows), min(64, len(rows)), replace=False)])
    evidence = dict(probe_positions=positions.tolist(), candidate_pairs=0, future_label_probes=[],
                    candidate_ids_equal=True, candidate_medians_equal=True,
                    all_candidate_permissions=True, beta_zero_exact=True,
                    quantiles_ordered=True, deterministic_simple_arms=True)
    baseline = data['baseline'][rows]
    for arm, result in controls.items():
        for p in positions:
            ids = independent_ids(data, int(rows[p]), arm)
            count = int(result['counts'][p])
            if not np.array_equal(ids, result['candidate_ids'][p, :count]):
                raise AssertionError(f'Independent {arm} candidate mismatch at {rows[p]}')
            residual = data['actual'][ids].astype(np.float64)-data['baseline'][ids, :, 2].astype(np.float64)
            expected = np.median(residual, axis=0) if len(ids) else np.zeros(24)
            np.testing.assert_array_equal(expected, result['median'][p])
            np.testing.assert_array_equal(result['candidate_fold_id'][p, :count], data['fold_id'][ids])
            np.testing.assert_array_equal(result['candidate_age_hours'][p, :count],
                                          (data['origin'][rows[p]]-data['origin'][ids])//HOUR)
            evidence['candidate_pairs'] += 1
        ids = result['candidate_ids']
        valid = ids >= 0
        age = data['origin'][rows, None]-data['origin'][np.maximum(ids, 0)]
        if np.any(age[valid] < DAY) or (arm=='R28' and np.any(age[valid] >= 28*DAY)):
            raise AssertionError('A saved candidate violates maturity or the strict rolling boundary')
        if arm=='Matched16' and np.any(age[valid] > 730*DAY):
            raise AssertionError('A matched candidate exceeds the original lookback')
        if arm!='R28' and np.any(age[valid] % DAY != 0):
            raise AssertionError('A same-hour candidate has a different origin hour')
        zero = apply_shared_beta(baseline, result['median'], 0.)
        if zero.dtype != baseline.dtype or zero.tobytes() != baseline.tobytes():
            raise AssertionError('Beta zero does not reproduce the baseline bitwise')
        q = apply_shared_beta(baseline, result['median'], beta)
        if np.any(np.diff(q, axis=-1) < 0):
            raise AssertionError('Quantiles crossed')
    for p in sorted(set([0, len(rows)//2, len(rows)-1])):
        row = int(rows[p])
        altered = dict(data, actual=data['actual'].copy(), neural_seed=9999)
        altered['actual'][data['origin']+DAY > data['origin'][row]] = np.nan
        rebuilt = build_residual_controls(altered, np.array([row]))
        for arm, result in controls.items():
            for field in ('median', 'counts', 'fallback'):
                np.testing.assert_array_equal(rebuilt[arm][field][0], result[field][p])
            count = int(result['counts'][p])
            np.testing.assert_array_equal(rebuilt[arm]['candidate_ids'][0, :count], result['candidate_ids'][p, :count])
            np.testing.assert_array_equal(apply_shared_beta(baseline[p:p+1], rebuilt[arm]['median'], beta),
                apply_shared_beta(baseline[p:p+1], result['median'][p:p+1], beta))
        evidence['future_label_probes'].append(dict(query_row=row, unavailable_labels_replaced='NaN',
            seed_metadata_changed_to=9999, all_three_predictions_unchanged=True))
    return evidence


def group_masks(origins, actual):
    local = (origins+10*HOUR).astype('datetime64[s]')
    years = local.astype('datetime64[Y]').astype(int)+1970
    quarters = local.astype('datetime64[M]').astype(int)%12//3+1
    groups = [('all', np.ones(actual.shape, dtype=bool))]
    for year in sorted(set(years.tolist())):
        mask = years == year
        groups.append((f'year/{year}', np.broadcast_to(mask[:, None], actual.shape)))
        for quarter in (1, 2, 3, 4):
            groups.append((f'quarter/{year}Q{quarter}', np.broadcast_to((mask & (quarters==quarter))[:, None], actual.shape)))
    for lead in range(24):
        groups.append((f'lead/{lead:02d}', np.broadcast_to(np.arange(24)==lead, actual.shape)))
    groups.extend([('actual/negative', actual<0), ('actual/0_to_300', (actual>=0)&(actual<300)),
                   ('actual/300_to_1000', (actual>=300)&(actual<1000)), ('actual/1000_plus', actual>=1000)])
    return groups


def _save_npz(path, **arrays):
    if Path(path).exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def evaluate_forward_stage(output, cfg, loaded, beta, stage, verify_barrier, run_id):
    """One deterministic prediction per simple arm; Full retains seed labels."""
    if stage not in ('L0','L1'):
        raise ValueError('Only the frozen L0 and conditionally permitted L1 stages exist')
    directory = Path(output)/stage
    directory.mkdir(exist_ok=False)
    evidence, groups = {}, []
    periods = ['2022H2'] if stage=='L0' else ['2023', '2024', '2023_2024']
    comparisons = ['R28_minus_B', 'Recent16_minus_B', 'Matched16_minus_B',
                   'Recent16_minus_R28', 'Matched16_minus_R28']
    inference_inputs = {period:{name:{} for name in comparisons} for period in periods}
    query_grid = None
    for region, data in loaded['regions'].items():
        print(f'[{stage}] {region}: build deterministic controls', flush=True)
        rows = data['query_rows']
        origins, y, baseline = (data[key][rows] for key in ('origin', 'actual', 'baseline'))
        if query_grid is None:
            query_grid = origins
        elif not np.array_equal(origins, query_grid):
            raise AssertionError('Regional query grids differ')
        if stage=='L0':
            expected = purged_period_rows(data['origin'], '2022-07-01', '2023-01-01')
        else:
            expected = np.concatenate([purged_period_rows(data['origin'], f'{yr}-01-01', f'{yr+1}-01-01') for yr in (2023,2024)])
        np.testing.assert_array_equal(rows, expected)
        controls = build_residual_controls(data, rows)
        evidence[region] = audit_controls(data, rows, controls, beta)
        evidence[region]['purge_verified'] = True
        predictions = {'B':baseline}
        predictions.update({arm:apply_shared_beta(baseline, result['median'], beta) for arm,result in controls.items()})
        for seed, reference in data['full_reference'].items():
            np.testing.assert_array_equal(reference['origin'], origins)
            predictions[f'Full_joint_seed{seed}'] = reference['quantiles']
        if set(data['full_reference']) != set(cfg['seeds']):
            raise AssertionError('Full historical reference seed matrix incomplete')
        target = directory/region
        target.mkdir()
        _save_npz(target/'predictions.npz', origin=origins, target_time_unix=origins[:,None]+np.arange(24)*HOUR,
            actual=y, query_rows=rows, library_origin=data['origin'], library_fold_id=data['fold_id'],
            beta=np.array(beta), **predictions)
        write_json(target/'full_reference_metadata.json', {str(seed):{key:value for key,value in ref.items()
            if key not in ('quantiles','origin')} for seed,ref in data['full_reference'].items()})
        for arm, result in controls.items():
            _save_npz(target/f'{arm}_candidates.npz', **{key:value for key,value in result.items() if key!='method'},
                correction=predictions[arm][...,2]-baseline[...,2])
        masks = group_masks(origins, y)
        for arm, q in predictions.items():
            if np.any(np.diff(q,axis=-1)<0):
                raise AssertionError('Unordered saved prediction')
            if arm != 'B':
                _save_npz(target/f'{arm}_cell_losses.npz', origin=origins, **paired_cell_losses(y, baseline, q))
                _save_npz(target/f'{arm}_daily_losses.npz', **daily_paired_losses(origins, y, baseline, q))
            fallback = controls[arm]['fallback'] if arm in controls else np.zeros(len(rows),dtype=bool)
            for name, mask in masks:
                scores = paired_scores(y[mask], baseline[mask], q[mask],
                    np.broadcast_to(fallback[:,None], y.shape)[mask]) if mask.any() else dict(cells=0,empty=True)
                groups.append(dict(stage=stage,region=region,arm=arm,group=name,**scores))
        years = (origins+10*HOUR).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        for period in periods:
            mask = np.ones(len(rows),dtype=bool) if period in ('2022H2','2023_2024') else years==int(period)
            for comparison in comparisons:
                arm, before = comparison.split('_minus_')
                d = daily_paired_losses(origins[mask],y[mask],predictions[before][mask],predictions[arm][mask])
                inference_inputs[period][comparison][region] = d
                _save_npz(target/f'{period}_{comparison}_paired_days.npz', **d)
        print(f'[{stage}] {region}: predictions, all slices and loss arrays saved',flush=True)
    write_csv(directory/'paired_groups.csv', groups)
    overview = []
    for period in periods:
        selected_group = 'all' if period in ('2022H2','2023_2024') else 'year/'+period
        for arm in ['B','R28','Recent16','Matched16']+[f'Full_joint_seed{seed}' for seed in cfg['seeds']]:
            records = [row for row in groups if row['group']==selected_group and row['arm']==arm]
            overview.append(dict(period=period,arm=arm,regions=len(records),
                **{metric:float(np.mean([row[f'after_{metric}'] for row in records]))
                   for metric in ('mae','qs','coverage80','coverage90','width80','width90')}))
    write_csv(directory/'macro_overview.csv',overview)
    reference_summary = []
    for period in periods:
        reference = [row for row in overview if row['period']==period and row['arm'].startswith('Full_')]
        r28 = next(row for row in overview if row['period']==period and row['arm']=='R28')
        reference_summary.append(dict(period=period,seeds=cfg['seeds'],
            interpretation='optimization sensitivity only; not independent market samples',
            metrics={metric:dict(mean=float(np.mean([row[metric] for row in reference])),
                seed_sd_ddof0=float(np.std([row[metric] for row in reference])),
                mean_minus_R28=float(np.mean([row[metric] for row in reference])-r28[metric]))
                for metric in ('mae','qs')}))
    write_json(directory/'full_descriptive_summary.json',reference_summary)
    write_json(directory/'integrity_evidence.json', evidence)
    write_json(directory/'source_manifest.json', dict(sources=loaded['sources'],metadata=loaded['metadata']))
    inference = {period:{name:paired_day_block_inference(regional,cfg) for name,regional in entries.items()}
                 for period,entries in inference_inputs.items()}
    write_json(directory/'paired_inference.json',inference)
    checked = verify_barrier(loaded['sources'])
    integrity = {key:True for key in REQUIRED_INTEGRITY_CHECKS}
    # These booleans are issued only after every throwing audit above and the
    # frozen source/protected-output barrier have succeeded. Persist evidence.
    reproduced = loaded['metadata']['score_reproduction']
    required_baselines = [f'{region}/B/2022H2' for region in cfg['regions']] if stage=='L0' else [
        f'{region}/{seed}/B_retrained/formal_unpurged' for region in cfg['regions'] for seed in cfg['seeds']]
    integrity['baseline_archive_reproduced'] = all(
        sum(row['description']==name and row['passed'] for row in reproduced)==1 for name in required_baselines)
    if not integrity['baseline_archive_reproduced']:
        raise AssertionError('Required regional archived B scores were not reproduced')
    write_json(directory/'integrity.json',dict(checks=integrity,barrier=checked,evidence='integrity_evidence.json',
        score_reproduction='source_manifest.json', committed_boundary_tests='tests/test_mature_residual_controls.py'))
    decisions = {}
    for period in periods:
        group = 'all' if period in ('2022H2','2023_2024') else 'year/'+period
        records = [row for row in groups if row['group']==group and row['arm']=='R28']
        decisions[period] = dict(apply_decision_rules(records,beta,integrity,cfg,stage=stage),period=period,run_id=run_id)
    if stage=='L0':
        decision = decisions['2022H2']
    else:
        passed = all(value['passed'] for value in decisions.values())
        decision = dict(stage='L1',primary_arm='R28',beta=beta,passed=passed,run_id=run_id,periods=decisions,
            decision='retain_R28_as_required_research_baseline' if passed else 'stop_current_v5_residual_route',
            evaluation_scope='retrospective forward evaluation on previously inspected periods; not blind test')
    write_json(directory/'decision.json',decision)
    return decision
