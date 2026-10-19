"""Read-only saved-artifact analysis; does not fit or select any predictor."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',default=str(Path(__file__).resolve().parent));args=p.parse_args();root=Path(args.run)
references=('C_roll','H_previous_global','TE_fixed_pulses','T_only','TE_event_off','TE_time_shift_off','TE_unshifted')
output={'period':'November2022 retrospective development; gates fitted October','regions':{}}
for region in ('NSW1','QLD1','TAS1'):
 a=json.loads((root/f'{region}.json').read_text());pred=np.load(root/'predictions'/f'{region}_TE_full.npz');y=pred['actual'];q=pred['quantiles'];g=np.load(root/'gates'/f'{region}_TE_full.npz');m=np.load(root/'features'/f'{region}_full.npz');ix=np.searchsorted(m['origin_unix'],g['origin_unix']);assert np.array_equal(m['origin_unix'][ix],g['origin_unix']);event=m['event'][ix];gate=g['gate_event'];proposal=event>1e-9;accepted=event*gate>1e-9
 r={'unique_actual_spike_hours':a['unique_spike_hours'],'scores':{},'comparisons':{},'event_adoption':{}}
 for name,v in a['scores'].items():r['scores'][name]={'mae':v['metrics']['mae'],'qs':v['metrics']['quantile_score_proxy'],'q95_pinball':v['metrics']['pinball_by_quantile']['0.95'],'q50_spike_alarm':v['spike']}
 for band,mask in [('all',np.ones(y.shape,bool)),('negative',y<0),('normal',(y>=0)&(y<=300)),('spike',y>300)]:
  eligible=proposal&mask; r['event_adoption'][band]={'n_cells':int(mask.sum()),'proposal_nonzero_fraction':float(proposal[mask].mean()),'gate_exact_zero_fraction':float((gate[mask]==0).mean()),'adopted_nonzero_fraction':float(accepted[mask].mean()),'gate_rejects_nonzero_proposal_fraction':float((gate[eligible]==0).mean()) if eligible.any() else None,'mean_event_proposal':float(event[mask].mean()),'mean_gated_event_term':float((event*gate)[mask].mean())}
 for ref in references:
  v=a['scores'][ref];full=a['scores']['TE_full'];r['comparisons'][ref]={'mae_full_minus_reference':full['metrics']['mae']-v['metrics']['mae'],'qs_full_minus_reference':full['metrics']['quantile_score_proxy']-v['metrics']['quantile_score_proxy'],'bands':{}}
  for band in ('negative','range_0_300','above_300'):
   f=full['bands'][band];b=v['bands'][band];r['comparisons'][ref]['bands'][band]={'full_conditional_mae':f['conditional_mae'],'reference_conditional_mae':b['conditional_mae'],'conditional_mae_delta':f['conditional_mae']-b['conditional_mae'],'total_mae_contribution_delta':f['mae_contribution']-b['mae_contribution'],'n_cells':f['n_cells']}
 for intervention in ('TE_event_off','TE_time_shift_off'):
  same=np.load(root/'gates'/f'{region}_{intervention}.npz');assert np.array_equal(g['gate_event'],same['gate_event']);assert np.array_equal(g['gate_trend'],same['gate_trend'])
 r['samehead_gates_exactly_equal']=True
 r['fit_objective_improvement_fraction']=1-a['fits']['TE_full']['final_selected_objective']/a['fits']['TE_full']['null_objective']
 output['regions'][region]=r
output['macro']={}
for ref in references:
 output['macro'][ref]={'mae_full_minus_reference':float(np.mean([r['comparisons'][ref]['mae_full_minus_reference'] for r in output['regions'].values()])),'qs_full_minus_reference':float(np.mean([r['comparisons'][ref]['qs_full_minus_reference'] for r in output['regions'].values()])),'bands':{band:float(np.mean([r['comparisons'][ref]['bands'][band]['total_mae_contribution_delta'] for r in output['regions'].values()])) for band in ('negative','range_0_300','above_300')}}
(root/'mechanism_contributions.json').write_text(json.dumps(output,indent=2)+'\n')
for region,r in output['regions'].items():
 print(region,'spikehours',r['unique_actual_spike_hours'],'adoption',r['event_adoption'])
 for ref in ('C_roll','TE_event_off','TE_time_shift_off'):print(ref,r['comparisons'][ref])
 print('ALARMS',{k:r['scores'][k]['q50_spike_alarm'] for k in ('C_roll','TE_full','TE_event_off','TE_time_shift_off')})
print('MACRO',output['macro'])
