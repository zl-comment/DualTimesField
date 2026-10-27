"""Summarize completed frozen L0 outputs; no new inference, fitting or selection."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
summary = json.loads((root/'replay/summary.json').read_text())
verification = json.loads((root/'verification.json').read_text())
if verification['completed'] != ['saved','replay']:
    raise ValueError('Require a completed frozen audit')
for path, expected in verification['artifact_sha256'].items():
    if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
        raise ValueError('Modified audit artifact: '+path)

def dump(name, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def losses(y,q):
    error=np.asarray(y,dtype=np.float64)[...,None]-np.asarray(q,dtype=np.float64)
    levels=np.array([.05,.1,.5,.9,.95])
    pb=np.maximum(levels*error,(levels-1)*error)
    return np.abs(error[...,2]),2*pb.mean(axis=-1)

contrasts=[]
strata=[]
for case in summary['cases']:
    identity={k:case[k] for k in ['region','fold','mode','seed','n_origins','alpha','epoch']}
    arms=case['arms']
    for representation in ['unshrunk','chosen']:
        row=dict(identity,representation=representation)
        for arm, result in arms.items():
            for metric in ['mae_delta','qs_delta','mae_benefit','mae_harm','qs_benefit','qs_harm','mean_abs_quantile_delta','mean_abs_median_delta','after_coverage80','after_coverage90','after_width80','after_width90']:
                row[f'{arm}_{metric}']=result[representation][metric]
        for name, left, right in [('state_minus_original','state','original'),('state_minus_constant','state','constant'),('external_state_minus_uniform','external_final_state','external_matched_uniform')]:
            for metric in ['mae_delta','qs_delta']:
                row[f'{name}_{metric}']=arms[left][representation][metric]-arms[right][representation][metric]
        for arm in ['original','state','constant','coverage_only']:
            for metric, value in arms[arm]['movement_'+('chosen' if representation=='chosen' else 'unshrunk')].items():
                row[f'{arm}_{metric}']=value
        contrasts.append(row)
    for arm, result in arms.items():
        with np.load(result['path']) as a:
            y=a['actual'].astype(np.float64)
            b=a['baseline'].astype(np.float64)
            s=a['states']; g=a['evidence_gates'][...,0]
            masks={'all':np.ones(y.shape,bool),'negative':y<0,'0_to_100':(y>=0)&(y<100),'100_to_300':(y>=100)&(y<300),'300_to_1000':(y>=300)&(y<1000),'1000_plus':y>=1000,'covered_weak':(s!=5)&(g<=.25),'covered_stronger':(s!=5)&(g>.25),'uncovered':s==5}
            mb,qb=losses(y,b)
            for representation,key in [('unshrunk','raw_quantiles'),('chosen','quantiles')]:
                m,q=losses(y,a[key])
                for group, mask in masks.items():
                    row=dict(identity,arm=arm,representation=representation,group=group,cells=int(mask.sum()),empty=not mask.any())
                    if mask.any():
                        for name,d in [('mae',m-mb),('qs',q-qb)]:
                            row[name+'_delta']=float(d[mask].mean())
                            row[name+'_benefit']=float(np.maximum(-d[mask],0).mean())
                            row[name+'_harm']=float(np.maximum(d[mask],0).mean())
                    strata.append(row)
dump('replay_contrasts.csv',contrasts)
dump('replay_strata.csv',strata)
aggregates=[]
for mode in ['raw','guarded']:
    for representation in ['unshrunk','chosen']:
        rows=[r for r in contrasts if r['mode']==mode and r['representation']==representation]
        summary_row=dict(mode=mode,representation=representation,cases=len(rows),zero_alpha_cases=sum(r['alpha']==0 for r in rows))
        for prefix in ['state_minus_original','state_minus_constant','external_state_minus_uniform']:
            for metric in ['mae_delta','qs_delta']:
                key=prefix+'_'+metric
                values=np.array([r[key] for r in rows])
                summary_row[key+'_macro']=float(values.mean())
                summary_row[key+'_improved_cases']=int((values<-1e-8).sum())
                summary_row[key+'_worse_cases']=int((values>1e-8).sum())
                summary_row[key+'_ties']=int((np.abs(values)<=1e-8).sum())
        for arm in ['original','state','constant','coverage_only','external_final_state','external_matched_uniform']:
            for metric in ['mae_delta','qs_delta']:
                summary_row[arm+'_'+metric]=float(np.mean([r[arm+'_'+metric] for r in rows]))
        aggregates.append(summary_row)
dump('replay_macro.csv',aggregates)
metadata=dict(run_id=verification['run_id'],input_root=str(root),scope='descriptive summaries of frozen L0 outputs; no new experiment or selection',group_boundaries='negative; [0,100); [100,300); [300,1000); [1000,infinity)',price_groups='outcome-only retrospective diagnostics, never routing inputs',units='origin-horizon cells overlap; 6 cases per mode, not independent market repetitions',source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['manifest.json','verification.json','replay/summary.json']},artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
(out/'summary_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(aggregates,indent=2))
