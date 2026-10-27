"""Independent stdlib/NumPy verification of completed selection and L0 outputs.

Never imports production prediction, retrieval, selection, or scoring code.
Never opens formal 2023/2024 arrays. If L1 ran, its numerical verification is
explicitly out of scope; this script does not claim complete L1 verification.
The original runtime artifact manifest is verified and remains unchanged.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DAY, HOUR = 86400, 3600
LEVELS = np.array([.05, .1, .5, .9, .95])
QNAMES = ('q05', 'q10', 'q50', 'q90', 'q95')
GRID = (0., .25, .5, .75, 1.)
ATOL = 1e-8  # Independent numerical reproduction, never a decision margin.


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone(timedelta(hours=10))).timestamp())


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Checks:
    def __init__(self):
        self.numeric_values = 0
        self.max_absolute_error = 0.
        self.hashes = {}

    def equal(self, actual, expected, label, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            raise AssertionError(f'{label}: shape mismatch {a.shape} != {b.shape}')
        if exact:
            if not np.array_equal(a, b):
                raise AssertionError(f'{label}: exact values differ')
            return
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f'{label}: nonfinite values')
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.max_absolute_error = max(self.max_absolute_error, error)
        self.numeric_values += int(a.size)
        if not np.allclose(a, b, rtol=0, atol=ATOL):
            raise AssertionError(f'{label}: maximum absolute error {error}')

    def hash(self, path, expected):
        path = Path(path).resolve()
        if not path.is_file():
            raise AssertionError(f'Missing source/artifact: {path}')
        observed = self.hashes.setdefault(str(path), digest(path)) if str(path) not in self.hashes else self.hashes[str(path)]
        if observed != expected:
            raise AssertionError(f'Source/artifact hash mismatch: {path}')


def primitives(y, q):
    y, q = np.asarray(y, np.float64), np.asarray(q, np.float64)
    if q.shape != (*y.shape, 5) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise AssertionError('Invalid saved outcome/prediction grid')
    if np.any(np.diff(q, axis=-1) < 0):
        raise AssertionError('Saved quantile crossing')
    e = y[..., None]-q
    pb = np.maximum(LEVELS*e, (LEVELS-1)*e)
    v = dict(mae=np.abs(e[..., 2]), qs=2*pb.mean(axis=-1),
             rmse=e[..., 2]**2, bias=-e[..., 2],
             coverage80=((y >= q[..., 1]) & (y <= q[..., 3])).astype(float),
             coverage90=((y >= q[..., 0]) & (y <= q[..., 4])).astype(float),
             width80=q[..., 3]-q[..., 1], width90=q[..., 4]-q[..., 0],
             under_q05=(y < q[..., 0]).astype(float), over_q95=(y > q[..., 4]).astype(float))
    v.update({f'pinball_{name}': pb[..., i] for i, name in enumerate(QNAMES)})
    return v


def score(values, mask=None):
    result = {key: float(np.mean(value if mask is None else value[mask])) for key, value in values.items()}
    result['rmse'] = float(np.sqrt(result['rmse']))
    return result


def paired(y, b, q, before, after, mask):
    result = {f'before_{k}':v for k,v in score(before, mask).items()}
    result.update({f'after_{k}':v for k,v in score(after, mask).items()})
    for key in ('mae', 'qs', *(f'pinball_{name}' for name in QNAMES)):
        delta = (after[key]-before[key])[mask]
        result.update({key+'_delta':float(delta.mean()), key+'_benefit':float(np.maximum(-delta, 0).mean()),
                       key+'_harm':float(np.maximum(delta, 0).mean())})
        if key in ('mae','qs'):
            result[key+'_harm_rate'] = float((delta > 1e-7).mean())
    for side, names in (('lower', QNAMES[:2]), ('upper', QNAMES[3:])):
        result[f'{side}_tail_qs_delta'] = float(2*np.mean([
            (after['pinball_'+name]-before['pinball_'+name])[mask].mean() for name in names]))
    residual, movement = (y-b[..., 2])[mask], (q[..., 2]-b[..., 2])[mask]
    result.update(wrong_direction_rate=float((movement*residual < 0).mean()),
        target_crossing_rate=float(((movement*residual > 0)&(np.abs(movement)>np.abs(residual))).mean()),
        harmful_overshoot_rate=float(((movement*residual > 0)&(np.abs(movement)>2*np.abs(residual))).mean()),
        correction_from_exact_baseline_rate=float(((residual==0)&(movement!=0)).mean()),
        zero_correction_rate=float((movement==0).mean()), mean_abs_median_delta=float(np.abs(movement).mean()),
        mean_abs_quantile_delta=float(np.abs(q-b)[mask].mean()), mean_signed_median_delta=float(movement.mean()))
    return result


def cell_pairs(y, b, q, before, after):
    result = {f'{side}_{key}':values[key] for side, values in (('before',before),('after',after))
              for key in ('mae','qs',*(f'pinball_{n}' for n in QNAMES),
                          'coverage80','coverage90','width80','width90')}
    for key in ('mae','qs'):
        delta = after[key]-before[key]
        result.update({key+'_delta':delta, key+'_benefit':np.maximum(-delta,0), key+'_harm':np.maximum(delta,0)})
    result['median_delta'] = q[..., 2]-b[..., 2]
    return result


def masks(origins, actual):
    local = (origins+10*HOUR).astype('datetime64[s]')
    year = local.astype('datetime64[Y]').astype(int)+1970
    quarter = local.astype('datetime64[M]').astype(int)%12//3+1
    out = {'all':np.ones(actual.shape,bool)}
    for yr in np.unique(year):
        out[f'year/{yr}'] = np.broadcast_to((year==yr)[:,None],actual.shape)
        for qr in range(1,5):
            out[f'quarter/{yr}Q{qr}'] = np.broadcast_to(((year==yr)&(quarter==qr))[:,None],actual.shape)
    out.update({f'lead/{h:02d}':np.broadcast_to(np.arange(24)==h,actual.shape) for h in range(24)})
    out.update({'actual/negative':actual<0, 'actual/0_to_300':(actual>=0)&(actual<300),
                'actual/300_to_1000':(actual>=300)&(actual<1000), 'actual/1000_plus':actual>=1000})
    return out


def daily(origins, cells):
    dates, index, counts = np.unique((origins+10*HOUR)//DAY, return_inverse=True, return_counts=True)
    result = dict(date_unix=dates*DAY-10*HOUR, cells=counts*24)
    for key, value in cells.items():
        result[key+'_sum'] = np.bincount(index, weights=value.sum(axis=1), minlength=len(dates))
    return result


def library(output, stage, region, checks):
    manifest = read_json(output/stage/'source_manifest.json')
    allowed = ('2020','2021','2022') if stage=='selection' else ('2020','2021','2022','2022H2')
    catalog = [item for item in manifest['metadata']['folds'] if item['region']==region]
    if tuple(item['fold'] for item in catalog) != allowed:
        raise AssertionError('Unexpected historical period or missing source fold')
    arrays = {k:[] for k in ('origin','actual','baseline','z','fold_id')}
    for item in catalog:
        checks.hash(item['bundle_path'],item['bundle_sha256'])
        with np.load(item['bundle_path'],allow_pickle=False) as archive:
            origin = archive['origin']
            if np.any(origin >= timestamp('2023-01-01')):
                raise AssertionError('Independent L0 verifier refuses formal-period source arrays')
            block = {key:archive[key] for key in ('actual','baseline','z')}
            for name in ('trained_before','selected_before'):
                if np.any(archive[name] >= origin):
                    raise AssertionError('Noncausal archived tree provenance')
        expected = np.arange(timestamp(item['start']),timestamp(item['end_exclusive'])-DAY,HOUR)
        checks.equal(origin,expected,'OOF purged hourly grid',exact=True)
        block.update(origin=origin,fold_id=np.full(len(origin),item['fold_id'],np.int16))
        for key in arrays:arrays[key].append(block[key])
    return {key:np.concatenate(values) for key,values in arrays.items()}


def expected_ids(data, row, arm):
    times, t = data['origin'], int(data['origin'][row])
    valid = times+DAY <= t
    if arm=='R28':return np.flatnonzero(valid & (times > t-28*DAY))
    same = ((times+10*HOUR)//HOUR%24)==((t+10*HOUR)//HOUR%24)
    if arm=='Recent16':return np.flatnonzero(valid & same)[-16:]
    pool = np.flatnonzero(valid & same & (times >= t-730*DAY))
    dims=[1,2,3,6,7,8]
    dist=((data['z'][pool][:,:,dims]-data['z'][row][:,dims])**2).mean(axis=(1,2))
    chosen=list(pool[-8:])
    for idx in sorted(range(len(pool)),key=lambda i:float(dist[i])):
        if int(pool[idx]) not in chosen:chosen.append(int(pool[idx]))
        if len(chosen)==16:break
    return np.asarray(chosen,np.int64)


def candidates(data, saved, query_rows, arm, checks):
    ids, counts = saved['candidate_ids'], saved['counts']
    if ids.shape[0] != len(query_rows) or counts.shape != (len(query_rows),):
        raise AssertionError('Candidate grid differs from queries')
    valid = np.arange(ids.shape[1])[None,:] < counts[:,None]
    if np.any((counts<0)|(counts>ids.shape[1])) or np.any(ids[~valid]!=-1):
        raise AssertionError('Invalid candidate count/padding')
    if np.any((ids[valid]<0)|(ids[valid]>=len(data['origin']))):
        raise AssertionError('Candidate index outside history library')
    origin = data['origin'][query_rows]
    safe=np.maximum(ids,0)
    age=origin[:,None]-data['origin'][safe]
    if np.any(age[valid]<DAY):raise AssertionError('Immature candidate')
    if arm=='R28' and np.any(age[valid]>=28*DAY):raise AssertionError('R28 lower boundary violated')
    if arm!='R28' and (np.any(age[valid]%DAY) or ids.shape[1]!=16):raise AssertionError('Same-hour/count rule violated')
    if arm=='Matched16' and np.any(age[valid]>730*DAY):raise AssertionError('Matched lookback violated')
    for i,count in enumerate(counts):
        if len(set(ids[i,:count].tolist())) != count:raise AssertionError('Duplicated candidate')
    checks.equal(saved['candidate_age_hours'][valid],age[valid]/HOUR,'candidate ages',exact=True)
    checks.equal(saved['candidate_fold_id'][valid],data['fold_id'][safe][valid],'candidate folds',exact=True)
    checks.equal(saved['fallback'],counts==0,'candidate fallbacks',exact=True)
    checks.equal(saved['query_rows'],query_rows,'candidate query rows',exact=True)
    checks.equal(saved['query_origin'],origin,'candidate query times',exact=True)
    # Fixed deterministic 64 samples, with endpoint probes in addition.
    positions=np.unique(np.r_[0,len(query_rows)-1,np.random.default_rng(20261027).choice(len(query_rows),min(64,len(query_rows)),replace=False)])
    for p in positions:
        wanted=expected_ids(data,int(query_rows[p]),arm)
        checks.equal(ids[p,:counts[p]],wanted,'independent candidate prefilter',exact=True)
        median=np.median(data['actual'][wanted].astype(float)-data['baseline'][wanted,:,2].astype(float),axis=0) if len(wanted) else np.zeros(24)
        checks.equal(saved['median'][p],median,'independent raw residual median',exact=True)
    if np.any(counts==0):checks.equal(saved['median'][counts==0],np.zeros((sum(counts==0),24)),'empty-pool median',exact=True)
    return len(positions)


def verify_beta(output,cfg,checks):
    choice=read_json(output/'selection/beta_selection.json')
    frozen=read_json(output/'selection/selection_frozen.json')
    checks.hash(output/'selection/beta_selection.json',frozen['sha256'])
    if not choice['selection_only'] or choice['H2_labels_opened'] or not frozen['frozen_before_H2_array_access']:
        raise AssertionError('Selection stage barrier metadata failed')
    loaded={}; probes=0
    for region in cfg['regions']:
        saved=npz(output/'selection'/f'{region}_R28.npz')
        data=library(output,'selection',region,checks)
        rows=np.flatnonzero((data['origin']>=timestamp('2022-01-01'))&(data['origin']+DAY<timestamp('2022-07-01')))
        if len(rows)!=4320:raise AssertionError('Wrong selection sample size')
        for key in ('origin','actual','baseline'):checks.equal(saved[key],data[key][rows],'selection source alignment',exact=True)
        checks.equal(saved['library_origin'],data['origin'],'selection library times',exact=True)
        checks.equal(saved['library_fold_id'],data['fold_id'],'selection library folds',exact=True)
        probes+=candidates(data,saved,rows,'R28',checks)
        loaded[region]=saved
    curves=[]
    for beta in GRID:
        per_region={}
        for region,s in loaded.items():
            q=s['baseline'].astype(float)+beta*np.clip(s['median'],-30,30)[...,None]
            values=score(primitives(s['actual'],q))
            per_region[region]={key:values[key] for key in ('mae','qs')}
        macro={key:float(np.mean([v[key] for v in per_region.values()])) for key in ('mae','qs')}
        if beta==0:base=macro
        feasible=all(macro[key]<=base[key]+1e-10 for key in ('mae','qs'))
        terms=[macro[key]/base[key] if base[key]>0 else 1. if macro[key]==0 else float('inf') for key in ('mae','qs')]
        objective=sum(terms)
        curves.append(dict(beta=beta,regional=per_region,macro=macro,eligible=feasible,objective=objective))
    if len(choice['curve'])!=5:raise AssertionError('Selection curve not the fixed five beta values')
    for actual, expected in zip(choice['curve'],curves):
        if actual['beta']!=expected['beta'] or actual['eligible']!=expected['eligible']:raise AssertionError('Selection curve feasibility mismatch')
        for key in ('mae','qs'):
            checks.equal(actual['macro'][key],expected['macro'][key],'beta macro')
            for region in cfg['regions']:checks.equal(actual['regional'][region][key],expected['regional'][region][key],'beta regional')
        if np.isfinite(expected['objective']):checks.equal(actual['standardized_relative_loss_sum'],expected['objective'],'beta objective')
        elif actual['standardized_relative_loss_sum'] is not None:raise AssertionError('Nonfinite beta objective not represented as null')
    eligible=[v for v in curves if v['eligible'] and np.isfinite(v['objective'])]
    minimum=min(v['objective'] for v in eligible)
    beta=min(v['beta'] for v in eligible if v['objective']<=minimum+1e-12)
    if choice['beta']!=beta or frozen['beta']!=beta:raise AssertionError('Shared beta selection differs')
    csv_rows=list(csv.DictReader((output/'selection/beta_curve.csv').open()))
    if len(csv_rows)!=5:raise AssertionError('CSV beta curve incomplete')
    for r,c in zip(csv_rows,curves):
        checks.equal(float(r['beta']),c['beta'],'CSV beta')
        for key in ('mae','qs'):checks.equal(float(r['macro_'+key]),c['macro'][key],'CSV macro')
    return beta,probes


def verify_l0(output,cfg,beta,checks):
    directory=output/'L0'; rows=list(csv.DictReader((directory/'paired_groups.csv').open()))
    group_index={(r['region'],r['arm'],r['group']):r for r in rows}
    if len(group_index)!=len(rows):raise AssertionError('Duplicate paired score rows')
    expected_group_keys=set(); overall={}; day_tables={}; probes=0
    for region in cfg['regions']:
        data=library(output,'L0',region,checks)
        p=npz(directory/region/'predictions.npz')
        query=np.flatnonzero((data['origin']>=timestamp('2022-07-01'))&(data['origin']+DAY<timestamp('2023-01-01')))
        if len(query)!=4392:raise AssertionError('Wrong L0 sample size')
        for key in ('origin','actual'):checks.equal(p[key],data[key][query],'L0 source alignment',exact=True)
        checks.equal(p['B'],data['baseline'][query],'L0 tree alignment',exact=True)
        checks.equal(p['query_rows'],query,'L0 purged queries',exact=True)
        checks.equal(p['library_origin'],data['origin'],'L0 library times',exact=True)
        checks.equal(p['library_fold_id'],data['fold_id'],'L0 library folds',exact=True)
        checks.equal(p['target_time_unix'],p['origin'][:,None]+HOUR*np.arange(24),'target times',exact=True)
        if float(p['beta'])!=beta:raise AssertionError('Stage changed beta')
        controls={}
        for arm in ('R28','Recent16','Matched16'):
            c=npz(directory/region/f'{arm}_candidates.npz');controls[arm]=c
            probes+=candidates(data,c,query,arm,checks)
            expected=p['B'].copy() if beta==0 else p['B'].astype(float)+beta*np.clip(c['median'],-30,30)[...,None]
            checks.equal(p[arm],expected,'saved common-shift control',exact=True)
            if beta==0 and (p[arm].dtype!=p['B'].dtype or p[arm].tobytes()!=p['B'].tobytes()):raise AssertionError('Beta-zero exact fallback failed')
            checks.equal(c['correction'],p[arm][...,2]-p['B'][...,2],'saved correction',exact=True)
        references=read_json(directory/region/'full_reference_metadata.json')
        if set(references)!=set(map(str,cfg['seeds'])):raise AssertionError('Full reference seed matrix incomplete')
        for seed,meta in references.items():
            source=Path(meta['prediction_path'])
            if source.name!='reference_guarded.npz' or '2022' not in source.parts:raise AssertionError('Unexpected L0 Full source')
            checks.hash(source,meta['prediction_sha256'])
            archived=npz(source); alignment=np.searchsorted(archived['origin_unix'],p['origin'])
            checks.equal(archived['origin_unix'][alignment],p['origin'],'Full query alignment',exact=True)
            checks.equal(archived['quantiles'][alignment],p[f'Full_joint_seed{seed}'],'Full frozen output',exact=True)
            if float(archived['alpha'])!=meta['alpha'] or int(archived['epoch'])!=meta['epoch']:raise AssertionError('Full selection identity changed')
        names=['B','R28','Recent16','Matched16']+[f'Full_joint_seed{seed}' for seed in cfg['seeds']]
        primitive={arm:primitives(p['actual'],p[arm]) for arm in names}
        all_masks=masks(p['origin'],p['actual']); before=primitive['B']
        for arm in names:
            for name,mask in all_masks.items():
                key=(region,arm,name);expected_group_keys.add(key);saved=group_index[key]
                if int(saved['cells'])!=int(mask.sum()) or (saved['empty']=='True')!=bool(not mask.any()):raise AssertionError('Empty/count group mismatch')
                if not mask.any():continue
                values=paired(p['actual'],p['B'].astype(float),p[arm].astype(float),before,primitive[arm],mask)
                for metric,value in values.items():checks.equal(float(saved[metric]),value,'paired group '+metric)
                fallback=controls[arm]['fallback'] if arm in controls else np.zeros(len(query),bool)
                checks.equal(float(saved['fallback_rate']),np.broadcast_to(fallback[:,None],mask.shape)[mask].mean(),'fallback rate')
                if name=='all':overall[(region,arm)]=values
            if arm=='B':continue
            cells=cell_pairs(p['actual'],p['B'].astype(float),p[arm].astype(float),before,primitive[arm])
            saved_cells=npz(directory/region/f'{arm}_cell_losses.npz')
            checks.equal(saved_cells.pop('origin'),p['origin'],'cell origins',exact=True)
            if set(saved_cells)!=set(cells):raise AssertionError('Cell-loss field matrix differs')
            for field,v in cells.items():checks.equal(saved_cells[field],v,'cell '+field)
            expected_daily=daily(p['origin'],cells);saved_daily=npz(directory/region/f'{arm}_daily_losses.npz')
            if set(saved_daily)!=set(expected_daily):raise AssertionError('Daily-loss field matrix differs')
            for field,v in expected_daily.items():checks.equal(saved_daily[field],v,'daily '+field)
        for comparison in ('R28_minus_B','Recent16_minus_B','Matched16_minus_B','Recent16_minus_R28','Matched16_minus_R28'):
            arm,base=comparison.split('_minus_')
            d=daily(p['origin'],cell_pairs(p['actual'],p[base].astype(float),p[arm].astype(float),primitive[base],primitive[arm]))
            archived=npz(directory/region/f'2022H2_{comparison}_paired_days.npz')
            for field,v in d.items():checks.equal(archived[field],v,'paired days '+field)
            day_tables[(region,comparison)]=d
    if expected_group_keys!=set(group_index):raise AssertionError('Unexpected/missing group row')
    overview=list(csv.DictReader((directory/'macro_overview.csv').open()))
    for row in overview:
        if row['period']!='2022H2':raise AssertionError('L0 overview contains another period')
        for key in ('mae','qs','coverage80','coverage90','width80','width90'):
            checks.equal(float(row[key]),np.mean([overall[(r,row['arm'])]['after_'+key] for r in cfg['regions']]),'macro overview')
    inference=read_json(directory/'paired_inference.json')['2022H2']
    for comparison,entry in inference.items():
        for metric in ('mae','qs'):
            regional=[]
            for region in cfg['regions']:
                d=day_tables[(region,comparison)]
                estimate=float((d['after_'+metric+'_sum']-d['before_'+metric+'_sum']).sum()/d['cells'].sum())
                regional.append(estimate);checks.equal(entry['point_region'][region][metric],estimate,'inference regional point')
            checks.equal(entry['point_macro'][metric],np.mean(regional),'inference macro point')
        if [b['block_days'] for b in entry['blocks']]!=[7,14,28]:raise AssertionError('Bootstrap block plan changed')
    integrity=read_json(directory/'integrity.json')['checks']
    if set(integrity)!=set(cfg['required_integrity_checks']) or not all(v is True for v in integrity.values()):raise AssertionError('Integrity checks missing or failed')
    flags=[beta>0]
    base={m:np.mean([overall[(r,'R28')]['before_'+m] for r in cfg['regions']]) for m in ('mae','qs')}
    after={m:np.mean([overall[(r,'R28')]['after_'+m] for r in cfg['regions']]) for m in ('mae','qs')}
    flags.extend(after[m]<=base[m]+1e-10 for m in base)
    flags.append(any(after[m]<base[m]-1e-10 for m in base))
    for region in cfg['regions']:
        row=overall[(region,'R28')]
        flags.extend(row['after_'+m]<=row['before_'+m]*1.01+1e-10 for m in ('mae','qs'))
        flags.extend(abs(row[f'after_coverage{c}']-c/100)-abs(row[f'before_coverage{c}']-c/100)<=.01+1e-10 for c in (80,90))
    passed=all(flags);decision=read_json(directory/'decision.json')
    if decision['passed'] is not passed or decision['beta']!=beta:raise AssertionError('L0 independent decision mismatch')
    expected='proceed_to_frozen_L1' if passed else 'stop_current_v5_residual_route'
    if decision['decision']!=expected:raise AssertionError('L0 decision disposition mismatch')
    if bool(decision['failed_criteria'])==passed:raise AssertionError('Decision failure list inconsistent')
    for criterion in decision['criteria']:
        name=criterion['name']
        if name in ('complete_single_R28_per_region','complete_integrity_manifest'):
            expected_flag=True
        elif name=='positive_shared_beta':expected_flag=beta>0
        elif name.startswith('integrity/'):
            expected_flag=integrity[name.split('/',1)[1]] is True
        elif name=='regional_loss_limit':
            row=overall[(criterion['region'],'R28')];metric=criterion['metric']
            expected_flag=row['after_'+metric]<=row['before_'+metric]*1.01+1e-10
        elif name=='regional_coverage_deviation':
            row=overall[(criterion['region'],'R28')];level=criterion['level']
            expected_flag=abs(row[f'after_coverage{level}']-level/100)-abs(row[f'before_coverage{level}']-level/100)<=.01+1e-10
        elif name=='macro_loss_nonincrease':
            metric=criterion['metric'];expected_flag=after[metric]<=base[metric]+1e-10
        elif name=='at_least_one_macro_loss_strictly_better':
            expected_flag=any(after[m]<base[m]-1e-10 for m in base)
        else:raise AssertionError('Unknown decision criterion: '+name)
        if criterion['passed']!=bool(expected_flag):raise AssertionError('Individual criterion differs: '+name)
    if decision['failed_criteria']!=[c for c in decision['criteria'] if not c['passed']]:raise AssertionError('Failure criterion records differ')
    for m in base:
        checks.equal(decision['macro_before'][m],base[m],'decision macro B')
        checks.equal(decision['macro_after'][m],after[m],'decision macro R28')
    return dict(passed=passed,beta=beta,macro_before=base,macro_after=after,
                groups=len(rows),candidate_median_probes=probes,decision=expected)


def verify(output):
    started=time.monotonic();usage=resource.getrusage(resource.RUSAGE_SELF)
    cpu_start=usage.ru_utime+usage.ru_stime
    output=Path(output).resolve();checks=Checks()
    completion=read_json(output/'completion.json');manifest=read_json(output/'manifest.json')
    artifacts=read_json(output/'artifact_manifest.json');cfg=manifest['configuration']
    original_identity={key:value for key,value in manifest.items() if key!='run_id'}
    expected_run_id=hashlib.sha256(json.dumps(original_identity,sort_keys=True).encode()).hexdigest()
    if manifest['run_id']!=expected_run_id:raise AssertionError('Frozen manifest run_id digest does not reconstruct')
    if completion['run_id']!=manifest['run_id'] or artifacts['run_id']!=manifest['run_id']:raise AssertionError('Run identities differ')
    stages=completion['completed']
    if stages not in (['selection','L0'],['selection','L0','L1']):raise AssertionError('Only a completed selection/L0 run is verifiable')
    for relative,sha in artifacts['files'].items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output):raise AssertionError('Artifact escapes run directory')
        checks.hash(path,sha)
    for path,sha in manifest['sources'].items():
        checks.hash(path,sha);checks.hash(output/'source'/Path(path).relative_to(cfg['root']),sha)
    for path,sha in manifest['protected_inputs'].items():checks.hash(path,sha)
    for stage in stages:
        sm=read_json(output/stage/'source_manifest.json')
        for record in sm['sources']:
            checks.hash(record['path'],record['sha256'])
            if record.get('expected_sha256') is not None and record['expected_sha256']!=record['sha256']:raise AssertionError('Source expected/hash mismatch')
    beta,selection_probes=verify_beta(output,cfg,checks)
    result=verify_l0(output,cfg,beta,checks)
    if not result['passed']:
        if stages!=['selection','L0'] or (output/'L1').exists():raise AssertionError('L1 accessed despite L0 failure')
        stopped=read_json(output/'L1_not_run.json')
        checks.hash(output/'L0/decision.json',stopped['L0_decision_sha256'])
        if stopped['formal_arrays_opened'] or stopped['new_formal_predictions']!=0:raise AssertionError('L1 failure barrier violated')
    elif 'L1' not in stages:raise AssertionError('Completed passed-L0 run lacks its conditional L1 completion')
    resources=read_json(output/'resources.json')
    if resources['cpu_core_hours']>8 or resources['wall_seconds']>3*3600 or resources['gpu_seconds']!=0:raise AssertionError('Resource budget exceeded')
    usage=resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds=usage.ru_utime+usage.ru_stime-cpu_start
    return dict(passed=True,verified=True,scope='selection_and_L0_independent_numeric_verification',
        run_id=manifest['run_id'],beta=beta,L0=result,selection_candidate_median_probes=selection_probes,
        hashes_verified=len(checks.hashes),numeric_values_verified=checks.numeric_values,
        maximum_absolute_metric_error=checks.max_absolute_error,reproduction_absolute_tolerance=ATOL,
        decision_numerical_tolerance=1e-10,verified_stages=['selection','L0'],
        unverified_numeric_stages=['L1'] if 'L1' in stages else [],
        resources=dict(wall_seconds=time.monotonic()-started,cpu_seconds=cpu_seconds,
                       cpu_core_hours=cpu_seconds/3600,peak_rss_kib=usage.ru_maxrss,gpu_seconds=0),
        formal_arrays_opened_by_verifier=False,
        limits=['The block-bootstrap confidence interval draws were not independently repeated; daily losses and point estimates were recomputed.',
                'All candidate permissions were checked; candidate prefilter identities and raw medians were independently recomputed at fixed 64 random positions plus endpoints per region and arm.',
                'Protected formal prediction files may be byte-hashed without opening or decompressing their arrays.',
                'If L1 completed, its numerical results need separate verification; this report never claims L1 numerical coverage.'])


def main():
    if len(sys.argv)>2:raise ValueError('Usage: verify_mature_residual_controls.py [completed_output_directory]')
    output=Path(sys.argv[1]) if len(sys.argv)==2 else ROOT/'outputs/forecasting/mature_residual_controls_20261027'
    report_path=output/'independent_verification.json'
    if report_path.exists():raise FileExistsError('Independent verification already exists')
    original_manifest=digest(output/'artifact_manifest.json')
    result=verify(output)
    result['verification_provenance']=dict(original_artifact_manifest_sha256=original_manifest,
        verifier_path=str(Path(__file__).resolve()),verifier_sha256=digest(__file__))
    with report_path.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    if digest(output/'artifact_manifest.json')!=original_manifest:raise AssertionError('Runtime artifact manifest changed')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
