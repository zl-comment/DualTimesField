"""Independent immutable-source and causal artifact validation, no training."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
PROJECT=Path('/root/workspace/Code/DualTimesField');sys.path.insert(0,str(PROJECT))
from forecasting.aligned_scoring import load_prediction_artifact,assert_comparable,score_predictions
from forecasting.full_v5.conditional_event_experiment import MODELS,summarize,check_shared_candidates
from forecasting.full_v5.conditional_transfer_gate import ConditionalTransferGate,FIT_CUTOFF,FIT_START
from forecasting.full_v5.history_correction import apply_history_correction
from forecasting.full_v5.history_quickcheck import load_frozen_region
from forecasting.full_v5 import schema
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else PROJECT/'outputs/forecasting/v5_conditional_event_transfer_20261017_r1'
PARENT=PROJECT/'outputs/forecasting/v5_calibrated_history_20261015_r3'
ENC=PROJECT/'outputs/forecasting/v5_adaptive_dual_20261017_r2'
END=schema.unix_timestamp('2022-12-01');DAY=86400

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b,name):
 if not np.array_equal(a,b):raise AssertionError(name)
def close(a,b,name,atol=1e-10):
 if not np.allclose(a,b,rtol=1e-10,atol=atol):raise AssertionError((name,float(np.max(np.abs(a-b)))))
needed=[OUT/'summary.json',OUT/'report.md',*[OUT/f'{r}.json' for r in ('NSW1','QLD1','TAS1')]]
if not all(p.exists() for p in needed):raise RuntimeError('Run incomplete')
identity=read(OUT/'identity.json');parent=read(PARENT/'identity/identity.json');encidentity=read(ENC/'identity.json')
assert identity['parent_identity_sha']==sha(PARENT/'identity/identity.json')
assert identity['encoder_identity_sha']==sha(ENC/'identity.json')
for path,digest in identity['source_hashes'].items():
 assert sha(PROJECT/path)==digest,('source',path)
 assert sha(OUT/'source'/path)==digest,('snapshot',path)
for path,digest in encidentity['source_hashes'].items():assert sha(PROJECT/path)==digest,('frozen ancestor source',path)
manifest=read(PROJECT/'logs/forecasting/v5_adaptive_dual_20261017_r2/artifact_manifest.json')['artifacts']
counts={'source_snapshots':len(identity['source_hashes']),'frozen_encoder_checkpoints':0,'gate_checkpoints':0,'predictions_rescored':0,'forecast_replays':0,'event_proposal_replays':0,'event_candidate_references':0,'trend_candidate_references':0,'frozen_parent_files':0}
regions={};results=[];torch.set_num_threads(2)
for region in ('NSW1','QLD1','TAS1'):
 r=read(OUT/f'{region}.json');results.append(r)
 arrays,meta,bq,z,anchor,rows,hashes,gaps=load_frozen_region(PROJECT,PARENT,region,parent)
 assert hashes==r['frozen_hashes']
 for path,digest in hashes.items():assert sha(PARENT/path)==digest;counts['frozen_parent_files']+=1
 keep=arrays['origin_unix']+DAY<END
 origins=arrays['origin_unix'][keep];actual=arrays['actual'][keep];targets=arrays['target_time_unix'][keep]
 bq,anchor=bq[keep],anchor[keep];assess=origins>=FIT_CUTOFF
 expected_fit=(origins>=FIT_START)&(origins+DAY<FIT_CUTOFF)
 assert origins[assess].min()>=FIT_CUTOFF and (origins[assess]+DAY).max()<END
 assert r['assessment_rows']==int(assess.sum())
 cp=ENC/'models'/f'{region}_adaptive.pt'
 assert sha(cp)==r['encoder_checkpoint_sha256']==manifest[f'models/{region}_adaptive.pt']['sha256']
 payload=torch.load(cp,map_location='cpu',weights_only=True)
 assert payload['audit']==r['encoder_audit']
 assert payload['audit']['latest_fit_label_available']<schema.unix_timestamp('2022-09-01')
 counts['frozen_encoder_checkpoints']+=1
 oldpath=ENC/'features'/f'{region}_adaptive_corrections.npz'
 assert sha(oldpath)==r['reused_corrections_sha256']==manifest[f'features/{region}_adaptive_corrections.npz']['sha256']
 memories={name:dict(np.load(OUT/'features'/f'{region}_{name}.npz')) for name in ('full','unshifted','fixed')}
 check_shared_candidates(memories)
 for name,m in memories.items():
  equal(m['origin_unix'],origins,'memory origins');assert np.isfinite(m['features']).all();assert np.max(np.abs(m['features']))<=1+1e-9
  for idname,counter in [('event_record_indices','event_candidate_references'),('trend_ids','trend_candidate_references')]:
   ids=m[idname];valid=ids>=0;current=origins.reshape((len(origins),)+(1,)*(ids.ndim-1));candidate=origins[ids.clip(0)]
   assert np.all((candidate+DAY<=current)[valid]),('maturity',name,idname)
   assert np.all((candidate>=FIT_START)[valid]),('library cutoff',name,idname)
   assert np.all((candidate>current-365*DAY)[valid]),('lookback',name,idname)
   assert np.all(((candidate//3600+10)%24==(current//3600+10)%24)[valid]),('same origin hour',name,idname)
   counts[counter]+=int(valid.sum())
  encoded=m['encoded_at']>=0
  assert np.all(m['encoded_at'][encoded]>=origins[encoded]+DAY)
  ids=m['event_record_indices'];valid=ids>=0;source_h=m['event_source_h'];target_h=m['event_target_h']
  assert np.all((source_h[valid]>=0)&(source_h[valid]<24));assert np.all((target_h[valid]>=0)&(target_h[valid]<24))
  event_target_timestamp=origins[ids.clip(0)]+(source_h+1)*3600
  for i in range(len(origins)):
   selected=event_target_timestamp[i,valid[i]]
   assert len(np.unique(selected))==len(selected),('duplicate event timestamps',name,i)
  past=actual[ids.clip(0),source_h.clip(0)];forecast=anchor[ids.clip(0),source_h.clip(0),2]
  assert np.all(past[valid]>300) and np.all((past-forecast)[valid]>0),('event eligibility',name)
  assert np.all(m['event_amplitudes'][valid]<= (past-forecast)[valid]+1e-8),('raw missed amplitude bound',name)
  weights=m['event_weights'];assert np.all(weights>=0);close(weights.sum(1)[valid.any(1)],np.ones(int(valid.any(1).sum())),'weight sums')
  widths=np.where(valid,m['event_widths'],1.)
  pulse=m['event_amplitudes'][...,None]*np.exp(-.5*((np.arange(24)[None,None]-m['event_centers'][...,None])/widths[...,None])**2)
  reconstructed=np.clip((weights[...,None]*pulse).sum(1),0,150)
  close(reconstructed,m['event'],('event replay',name));counts['event_proposal_replays']+=1
  close(m['features'][...,20],m['trend']/150,'trend feature');close(m['features'][...,21],m['event']/150,'event feature')
  assert r['memory_audits'][name]['maturity_violations']==0
 f,u=memories['full'],memories['unshifted']
 valid=f['event_record_indices']>=0
 close((f['event_centers']-u['event_centers'])[valid],(f['event_target_h']-u['event_target_h'])[valid],'pure center translation')
 arts={name:load_prediction_artifact(OUT/'predictions'/f'{region}_{name}.npz') for name in MODELS};reference=arts['B']
 equal(reference.origin_unix,origins[assess],'assessment origins');equal(reference.actual,actual[assess],'assessment labels');equal(reference.target_time_unix,targets[assess],'assessment targets')
 equal(reference.quantiles,bq[assess],'frozen baseline');equal(arts['C_roll'].quantiles,anchor[assess],'frozen Croll')
 for name,a in arts.items():
  assert_comparable(reference,a,allow_different_seed=False)
  assert score_predictions(a)==r['scores'][name]['metrics'],('rescoring',name)
  counts['predictions_rescored']+=1
 old=dict(np.load(oldpath));idx=np.searchsorted(old['origin_unix'],origins);equal(old['origin_unix'][idx],origins,'control origins')
 for name,key in [('H_recent','recentraw'),('H_previous_global','dualfull')]:
  fit=r['fits'][name];assert fit['diagnostics']['fit_cutoff']==FIT_CUTOFF;assert fit['diagnostics']['latest_fit_label_available']<FIT_CUTOFF;assert fit['diagnostics']['fit_origin_count']==int(expected_fit.sum())
  q=apply_history_correction(anchor,old[key][idx],fit['alpha']);equal(q[assess],arts[name].quantiles,('control replay',name));counts['forecast_replays']+=1
 parts={};heads={}
 mapping={'T_only':'full','TE_unshifted':'unshifted','TE_full':'full','TE_fixed_pulses':'fixed'}
 for name,memoryname in mapping.items():
  path=OUT/'models'/f'{region}_{name}.json';assert sha(path)==r['gate_checkpoint_hashes'][name]
  head,audit=ConditionalTransferGate.load_checkpoint(path);assert audit==r['fits'][name];assert audit['fit_cutoff']==FIT_CUTOFF and audit['latest_fit_label_available']<FIT_CUTOFF;assert audit['fit_rows']==int(expected_fit.sum())
  assert audit['final_selected_objective']<=audit['null_objective']+1e-12
  if audit['selected_null']:assert np.all(head.linear.weight.detach().numpy()==0) and np.all(head.linear.bias.detach().numpy()==0)
  m=memories[memoryname];parts[name]=head.predict(m['features'],m['trend'],m['event'],anchor);heads[name]=head;counts['gate_checkpoints']+=1
 full=memories['full'];head=heads['TE_full']
 parts['TE_event_off']=head.predict(full['features'],full['trend'],full['event'],anchor,event_off=True)
 parts['TE_time_shift_off']=head.predict(full['features'],full['trend'],memories['unshifted']['event'],anchor)
 for name,pred in parts.items():
  equal(pred['quantiles'][assess],arts[name].quantiles,('gate replay',name));counts['forecast_replays']+=1
  saved=dict(np.load(OUT/'gates'/f'{region}_{name}.npz'));equal(saved['origin_unix'],origins[assess],'gate origin')
  for key in ('gate_trend','gate_event','delta'):equal(pred[key][assess],saved[key],('saved part replay',name,key))
  assert np.max(np.abs(pred['delta']))<=150
 for name in ('TE_event_off','TE_time_shift_off'):
  for key in ('gate_trend','gate_event'):equal(parts[name][key],parts['TE_full'][key],'intervention gate invariance')
 direct=(parts['TE_full']['quantiles']-parts['TE_event_off']['quantiles'])[assess,:,2]
 raw=parts['TE_full']['gate_event'][assess]*full['event'][assess]
 regions[region]={'maturity_violations':0,'assessment_rows':int(assess.sum()),'unique_spike_target_hours':len(np.unique(targets[assess][actual[assess]>300])),
  'event_gate_positive_fraction':float(np.mean(parts['TE_full']['gate_event'][assess]>1e-9)),
  'event_proposal_positive_fraction':float(np.mean(full['event'][assess]>1e-9)),
  'actual_event_delta_nonzero_fraction':float(np.mean(np.abs(direct)>1e-9)),
  'actual_event_delta_mean_abs':float(np.abs(direct).mean()),
  'raw_gated_event_mean_abs':float(np.abs(raw).mean()),
  'all_forecast_replays_exact':True}
assert summarize(results)==read(OUT/'summary.json'),'macro and acceptance'
for path,digest in identity['source_hashes'].items():assert sha(PROJECT/path)==digest,('source changed during verification',path)
result={'passed':True,'counts':counts,'regions':regions,'script_sha256':sha(Path(__file__)),
 'scope':'Immutable sources/snapshots; frozen encoder/parent/cache; exact scoring/head/event-proposal replay; causal same-hour unique historical events; October fitting and November assessment. Does not independently establish first-publication timestamps.'}
(OUT/'independent_verification_script.py').write_text(Path(__file__).read_text())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
