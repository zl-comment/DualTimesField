"""Bounded maintenance diagnostics; model selection reads only 2022H1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = ROOT.parent/'DualTimesField-foundation-baselines'
RUN = FOUNDATION/'outputs/forecasting/foundation_baselines_20261023_r1'
REGIONS = ('NSW1','QLD1','TAS1')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def data_audit(output):
    import pandas as pd
    result={'publication_audit_complete':False,'sources':{},'limitations':[
        'Archive LASTCHANGED denotes record modification, not certified public release time.',
        'Hourly actual prices/demand and downloaded gas schedule history may contain later revisions; publication vintages not retained.',
        'Gas input is the arithmetic mean of five scheduled prices, not the settlement-weighted daily imbalance price.']}
    for r in REGIONS:
        for kind in ('pdpasa','predispatch','predispatch_interconnector'):
            path=ROOT/f'data/aemo_exogenous/{r.lower()}_{kind}.npz'
            with np.load(path,allow_pickle=False) as a:
                origins=a['forecast_origin_unix'];checks={}
                for key in a.files:
                    if key.startswith('source_') and key.endswith('_unix'):
                        source=a[key];ref=origins.reshape((len(origins),)+(1,)*(source.ndim-1))
                        active=source>0
                        checks[key]={'future_count':int(np.sum((source>ref)&active)),
                            'minimum_lead_seconds':int(np.min(np.broadcast_to(ref,source.shape)[active]-source[active])) if active.any() else None}
                result['sources'][str(path.relative_to(ROOT))]={'sha256':digest(path),'origins':len(origins),'checks':checks}
        path=ROOT.parent/f'electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem/{r.lower()}_hourly.csv'
        df=pd.read_csv(path,usecols=['delivery_start_aest','available_at_aest'])
        delta=(pd.to_datetime(df.available_at_aest,utc=True)-pd.to_datetime(df.delivery_start_aest,utc=True)).dt.total_seconds()
        result['sources'][str(path)]={'sha256':digest(path),'rows':len(df),'availability_offset_seconds':sorted(delta.unique().tolist()),'status':'declared_hour_end_proxy_not_original_release_log'}
    gas=ROOT/'data/aemo_exogenous/raw_gas/dwgm_prices.csv'
    df=pd.read_csv(gas);counts=df.groupby('Gas_Date').size()
    result['gas']={'sha256':digest(gas),'columns':list(df),'schedule_hours':sorted(df.Hour.unique().tolist()),'schedule_counts_per_day':sorted(counts.unique().tolist()),'window':'mean last 7 completed gas days; gas day begins 06:00 AEST','public_release_timestamps_present':False}
    result['timestamp_violations']=sum(v['future_count'] for a in result['sources'].values() for v in a.get('checks',{}).values())
    save(output/'data_availability.json',result)
    table=json.loads((ROOT/'logs/forecasting/main_comparison_20261023/verification.json').read_text())
    core=[a for a in table['artifacts'] if a['model'] in ('baseline_lightgbm','baseline_xgboost','baseline_patchtst')]
    for a in core:assert digest(a['path'])==a['sha256']
    save(output/'core_freeze.json',{'models':['lightgbm','xgboost','patchtst'],'selection':'original 2022H1 rules, no new HPO','test':'2023-2024 retrospective','artifacts':core,'inputs':result['sources']})
    print('Data timestamp audit:',result['timestamp_violations'],'violations; public release remains unverified',flush=True)

def timesfm(output,region,count):
    import torch
    from forecasting.common.cache import load_split
    from forecasting.timesfm25_xreg import model as native
    torch.set_num_threads(4)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((RUN/'jobs.json').read_text())
    split=load_split(RUN,manifest,region,'selection')
    # Fixed, outcome-independent sample; count=0 means complete selection.
    indices=np.arange(len(split)) if count==0 else np.unique(np.linspace(0,len(split)-1,count).astype(int))
    times=split.origin_unix[indices]
    assert times.max()<np.datetime64('2022-07-01').astype('datetime64[s]').astype(int)
    model=native.load_model('cuda:0');native.compile_model(model)
    mean,std=split.metadata['target_mean'],split.metadata['target_std']
    price=split.inputs['history_values'][indices,:,0].astype(np.float64)*std+mean
    calendar=np.concatenate([split.inputs['history_calendar'][indices],split.inputs['future_calendar'][indices]],axis=1)
    gas=split.inputs['origin_context'][indices,0]
    predictions={k:[] for k in (('original','ridge1','bare') if count else ('original','ridge1'))};started=time.monotonic()
    for i in range(len(indices)):
        inputs=[price[i]];dynamic={f'calendar_{c}':[calendar[i,:,c]] for c in range(calendar.shape[-1])}
        for name,ridge in [('original',0.),('ridge1',1.)]:
            point,_=model.forecast_with_covariates(inputs=inputs,dynamic_numerical_covariates=dynamic,
                static_numerical_covariates={'gas_7d_log_std':[float(gas[i])]},xreg_mode='xreg + timesfm',
                normalize_xreg_target_per_input=True,ridge=ridge,max_rows_per_col=0,force_on_cpu=False)
            predictions[name].append(np.asarray(point[0])[-24:])
        if count:
            point,_=model.forecast(horizon=24,inputs=inputs)
            predictions['bare'].append(np.asarray(point[0])[-24:])
        if (i+1)%32==0 or i+1==len(indices):
            save(output/f'{region}_progress.json',{'completed':i+1,'total':len(indices),'seconds':time.monotonic()-started})
            print(region,i+1,'/',len(indices),flush=True)
    actual=np.asarray(split.actual[indices],dtype=np.float64)
    predictions={k:np.stack(v).astype(np.float64) for k,v in predictions.items()}
    scores={k:{'mae':float(np.mean(np.abs(v-actual))),'rmse':float(np.sqrt(np.mean((v-actual)**2))),'max_abs_prediction':float(np.max(np.abs(v)))} for k,v in predictions.items()}
    np.savez_compressed(output/f'{region}.npz',indices=indices,origin_unix=times,actual=actual,**predictions)
    save(output/f'{region}.json',{'region':region,'split':'selection','count':len(indices),'scores':scores,'seconds':time.monotonic()-started,'weights':native.weight_identity(os.environ['HF_HOME']),'script_sha256':digest(__file__)})
    print(scores,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['data','timesfm']);p.add_argument('--output',type=Path,required=True);p.add_argument('--region',choices=REGIONS);p.add_argument('--count',type=int,default=128);a=p.parse_args()
    if a.mode=='data':data_audit(a.output)
    else:timesfm(a.output,a.region,a.count)
