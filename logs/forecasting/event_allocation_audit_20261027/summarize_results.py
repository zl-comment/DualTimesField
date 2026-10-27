"""Human-readable summaries derived only from the completed frozen CSVs."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
root=Path(sys.argv[1])
def readcsv(name):
    with (root/name).open() as stream: return list(csv.DictReader(stream))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def savecsv(name,rows):
    with (root/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
rows,regional,macro=map(readcsv,('scores.csv','regional.csv','macro.csv'))
comparisons=['upper_only_vs_original','event_off_vs_original','upper_only_vs_event_off']
metrics=['mae','qs','pinball_q05','pinball_q10','pinball_q50','pinball_q90','pinball_q95','coverage80','coverage90','width80','width90','rmse','bias']
macroall={r['comparison']:r for r in macro if r['group']=='all'}
models={}
for comparison,row in macroall.items():
    for prefix,model in [('baseline','B'),('before',row['before_arm']),('after',row['after_arm'])]:
        val={m:float(row[f'{prefix}_{m}_mean']) for m in metrics}
        if model in models:
            assert all(abs(val[m]-models[model][m])<1e-10 for m in metrics)
        models[model]=val
summary=[]
for row in regional:
    if row['group'] not in ('all','actual/negative','actual/0_to_300','actual/0_to_100','actual/1000_plus'): continue
    item={k:row[k] for k in ('region','comparison','group','cells')}
    item.update({k:float(row[k+'_mean']) for k in ['mae_delta','qs_delta','mae_benefit','mae_harm','qs_benefit','qs_harm']})
    summary.append(item)
savecsv('primary_summary.csv',summary)
disjoint=['actual/negative','actual/0_to_100','actual/100_to_300','actual/300_to_1000','actual/1000_plus']
lookup={(r['case_id'],r['comparison'],r['group']):r for r in rows}
contributions=[]
for comparison in comparisons:
    for group in disjoint:
        selected=[r for r in rows if r['comparison']==comparison and r['group']==group]
        regions=sorted({r['region'] for r in selected})
        means={}
        for field in ('mae_delta','qs_delta'):
            means[field]=float(np.mean([np.mean([float(r[field])*int(r['cells'])/int(lookup[(r['case_id'],comparison,'all')]['cells']) for r in selected if r['region']==region]) for region in regions]))
        share=float(np.mean([np.mean([int(r['cells'])/int(lookup[(r['case_id'],comparison,'all')]['cells']) for r in selected if r['region']==region]) for region in regions]))
        contributions.append(dict(comparison=comparison,group=group,mean_cell_share=share,**{k+'_contribution':v for k,v in means.items()}))
closure={}
for comparison in comparisons:
    for metric in ('mae','qs'):
        error=abs(sum(r[metric+'_delta_contribution'] for r in contributions if r['comparison']==comparison)-float(macroall[comparison][metric+'_delta_mean']))
        assert error<1e-10
        closure[comparison+'/'+metric]=error
savecsv('price_contributions.csv',contributions)
result={'models':models,'macro_comparisons':{k:{m:float(r[m+'_mean']) for m in ['mae_delta','qs_delta','mae_benefit','mae_harm','qs_benefit','qs_harm','pinball_q05_delta','pinball_q10_delta','pinball_q50_delta','pinball_q90_delta','pinball_q95_delta']} for k,r in macroall.items()},'decision':json.loads((root/'decision.json').read_text()),'price_contribution_closure_error':closure}
(root/'readable_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
prov={'script_sha256':sha(Path(__file__)),'input_sha256':{n:sha(root/n) for n in ['scores.csv','regional.csv','macro.csv','decision.json']},'output_sha256':{n:sha(root/n) for n in ['primary_summary.csv','price_contributions.csv','readable_summary.json']}}
(root/'summary_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'models':models,'comparisons':result['macro_comparisons'],'decision':result['decision']['decision'],'failures':result['decision']['failed_criteria']},ensure_ascii=False,indent=2))
