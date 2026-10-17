import argparse
import json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='/root/workspace/Code/DualTimesField/outputs/forecasting/v5_adaptive_dual_20261017_r2')
parser.add_argument('--output',default='/tmp/adaptive_history_transfer_analysis_r2.json')
args=parser.parse_args()
P=Path(args.run)
S=Path('/root/workspace/Code/DualTimesField/outputs/forecasting/v5_calibrated_history_20261015_r3')
H=3600; cutoff=1669816800
output={}
for region in ('NSW1','QLD1','TAS1'):
 if not (P/f'{region}.json').exists():continue
 result=json.loads((P/f'{region}.json').read_text())
 correction=np.load(P/'features'/f'{region}_adaptive_corrections.npz')
 old=np.load(P/'features'/f'{region}_old_corrections.npz')
 weights=np.repeat(result['fits']['H_adaptive_full']['alpha'],6)
 rr={}
 for seg in ('v1','v2'):
  a=np.load(S/'development/predictions/C_roll'/f'{region}_{seg}.npz')
  origins=a['origin_unix'];use=np.ones(len(origins),bool) if seg=='v2' else origins+24*H<cutoff
  origins=origins[use];y=a['actual'][use];q=a['quantiles'][use];ix=np.searchsorted(correction['origin_unix'],origins)
  assert np.array_equal(correction['origin_unix'][ix],origins)
  raw=correction['rawmatched'][ix];ad=correction['dualfull'][ix];tr=correction['dualtrend'][ix];od=old['dualfull'][ix];delta=ad*weights[None];p=q[:,:,2]+delta;err=y-q[:,:,2]
  active=np.abs(delta)>1e-9
  stats={'origins':len(y),'correlation_raw_adaptive':float(np.corrcoef(raw.ravel(),ad.ravel())[0,1]),'mean_abs_adaptive_minus_raw_correction':float(np.abs(ad-raw).mean()),'mean_abs_old_minus_raw_correction':float(np.abs(od-raw).mean()),'raw_correction_mean_abs':float(np.abs(raw).mean()),'adaptive_correction_mean_abs':float(np.abs(ad).mean()),'mean_applied_shift':float(delta.mean()),'mean_abs_applied_shift':float(np.abs(delta).mean()),'positive_shift_fraction':float((delta>0).mean()),'required_mean_shift':float(err.mean()),'required_median_shift':float(np.median(err)),'correct_direction_among_active':float((delta[active]*err[active]>0).mean()),'mean_abs_event_effect_before_weight':float(np.abs(ad-tr).mean()),'mean_abs_event_effect_after_weight':float(np.abs((ad-tr)*weights).mean()),'bands':{}}
  for band,mask in [('all',np.ones(y.shape,bool)),('negative',y<0),('normal',(y>=0)&(y<=300)),('spike',y>300)]:
   base=np.abs(q[:,:,2]-y);cand=np.abs(p-y)
   stats['bands'][band]={'count':int(mask.sum()),'baseline_mae':float(base[mask].mean()),'adaptive_mae':float(cand[mask].mean()),'mae_delta':float((cand-base)[mask].mean()),'mae_delta_contribution':float(np.where(mask,cand-base,0).mean()),'mean_applied_shift':float(delta[mask].mean()),'mean_required_shift':float(err[mask].mean())}
  rr[seg]=stats
 # Saved December already independently scored; no fitting performed here.
 rr['fitted_objective_relative_improvement']=1-sum(g['objective_fit'] for g in result['fits']['H_adaptive_full']['diagnostics']['groups'])/sum(g['objective_zero'] for g in result['fits']['H_adaptive_full']['diagnostics']['groups'])
 rr['dec_scores']={m:{'mae':v['metrics']['mae'],'qs':v['metrics']['quantile_score_proxy']} for m,v in result['scores'].items()}
 rr['reconstruction']=result['reconstruction']
 rr['unique_dec_spike_hours']=result['unique_test_spike_hours']
 rr['same_head_event_contribution']={}
 for band in ('negative','range_0_300','above_300'):
  full=result['scores']['H_adaptive_full']['bands'][band]; off=result['scores']['H_adaptive_event_off']['bands'][band]
  rr['same_head_event_contribution'][band]={'n_cells':full['n_cells'],'full_mae':full['conditional_mae'],'event_off_mae':off['conditional_mae'],'full_minus_off_conditional_mae':full['conditional_mae']-off['conditional_mae'],'full_minus_off_total_mae_contribution':full['mae_contribution']-off['mae_contribution']}
 rr['q50_alarm_diagnostics']={m:result['scores'][m]['spike'] for m in ('C_roll','H_adaptive_full','H_adaptive_event_off')}
 output[region]=rr
Path(args.output).write_text(json.dumps(output,indent=2))
for r,v in output.items():
 print(r)
 for seg in ('v1','v2'):
  q=v[seg];print(seg,json.dumps({k:a for k,a in q.items() if k!='bands'}));print('bands',json.dumps(q['bands']))
 print('fit objective improvement',v['fitted_objective_relative_improvement'])
print('saved',args.output)
