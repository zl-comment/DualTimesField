"""Independent read-only artifact validator; writes only verification JSON."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
PROJECT=Path('/root/workspace/Code/DualTimesField')
sys.path.insert(0,str(PROJECT))
from forecasting.aligned_scoring import load_prediction_artifact,assert_comparable,score_predictions
from forecasting.full_v5.adaptive_dual_history import AdaptiveHistoricalDualField,FIT_CUTOFF
from forecasting.full_v5.dual_history_model import HistoricalDualField
from forecasting.full_v5.dual_history_experiment import price_lags
from forecasting.full_v5.history_correction import apply_history_correction
from forecasting.full_v5.history_quickcheck import load_frozen_region
from forecasting.full_v5.adaptive_dual_experiment import MODELS,summarize
from forecasting.full_v5 import data
import yaml
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else PROJECT/'outputs/forecasting/v5_adaptive_dual_20261017_r2'
PARENT=PROJECT/'outputs/forecasting/v5_calibrated_history_20261015_r3'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b,label):
 if not np.array_equal(a,b):raise AssertionError(label)
def close(a,b,label,atol=1e-9):
 if not np.allclose(a,b,rtol=1e-10,atol=atol):raise AssertionError(label)
for p in [OUT/'summary.json',OUT/'report.md',*[OUT/f'{r}.json' for r in ('NSW1','QLD1','TAS1')]]:
 if not p.exists():raise RuntimeError(f'Run not yet complete: {p}')
identity=read(OUT/'identity.json');parent=read(PARENT/'identity/identity.json')
assert sha(PARENT/'identity/identity.json')==identity['parent_identity_sha']
for path,digest in identity['source_hashes'].items():
 assert sha(PROJECT/path)==digest,('current source',path)
 assert sha(OUT/'source'/path)==digest,('snapshot source',path)
basepath=PROJECT/'outputs/forecasting/aligned_baselines_20261012/config.yaml'
assert sha(basepath)==identity['baseline_config_sha']==parent['baseline_run']['original_config_hash']
baseconfig=yaml.safe_load(basepath.read_text())
results=[];validations={};counts={'source_snapshots':len(identity['source_hashes']),'checkpoints':0,'predictions_rescored':0,'forecast_replays':0,'checkpoint_encoded_rows_replayed':0,'frozen_parent_files':0,'candidate_assignments':0}
torch.set_num_threads(2)
for region in ('NSW1','QLD1','TAS1'):
 r=read(OUT/f'{region}.json');results.append(r)
 arrays,meta,bq,z,anchor,rows,frozen,gaps=load_frozen_region(PROJECT,PARENT,region,parent)
 for path,digest in r['frozen_hashes'].items():
  assert sha(PARENT/path)==digest;counts['frozen_parent_files']+=1
 assert r['frozen_hashes']==frozen
 origins=arrays['origin_unix'];actual=arrays['actual'];idx=rows['v2'];lags=price_lags(arrays,meta,baseconfig)
 arts={name:load_prediction_artifact(OUT/'predictions'/f'{region}_{name}.npz') for name in MODELS}
 reference=arts['B']
 equal(reference.origin_unix,origins[idx],'origins');equal(reference.actual,actual[idx],'actual');equal(reference.target_time_unix,arrays['target_time_unix'][idx],'targets')
 equal(reference.quantiles,bq[idx],'frozen B');equal(arts['C_roll'].quantiles,anchor[idx],'frozen Croll')
 for name,a in arts.items():
  assert_comparable(reference,a,allow_different_seed=False)
  assert score_predictions(a)==r['scores'][name]['metrics'],('scores',region,name)
  counts['predictions_rescored']+=1
 unique=len(np.unique(reference.target_time_unix[reference.actual>300]));assert unique==r['unique_test_spike_hours']
 memories={name:dict(np.load(OUT/'features'/f'{region}_{name}_corrections.npz')) for name in ('adaptive','fixed','old')}
 encodings={name:dict(np.load(OUT/'features'/f'{region}_{name}_encoded.npz')) for name in memories}
 sample_replay={}
 for name,m in memories.items():
  equal(m['origin_unix'],origins,'correction origins');equal(m['matched_indices'],memories['adaptive']['matched_indices'],'shared matched ids');equal(m['recent_indices'],memories['adaptive']['recent_indices'],'shared recent ids')
  for key in ('rawmatched','recentraw','fixedsmooth'):equal(m[key],memories['adaptive'][key],'unchanged control')
  ids=m['matched_indices'];active=ids>=0;candidate_origin=origins[ids.clip(0)]
  assert np.all((candidate_origin+86400<=origins[:,None,None])[active]),('maturity',name)
  assert np.all(((candidate_origin//3600+10)%24==(origins[:,None,None]//3600+10)%24)[active]),('same-hour',name)
  assert np.all((candidate_origin>=1664546400)[active]),('history-start',name)
  assert np.all((candidate_origin>origins[:,None,None]-365*86400)[active]),('lookback',name)
  counts['candidate_assignments']+=int(active.sum())
  e=encodings[name];equal(origins[e['indices']],e['origins'],'encoded indices')
  equal(e['available_at'],e['origins']+86400,'availability')
  assert np.all(e['encoded_at']>=e['available_at'])
  close(e['trend']+e['event'],e['reconstruction'],'fields sum',atol=1e-3)
  if name!='old':close(e['reconstruction'].mean(1),actual[e['indices']].mean(1),'mean preservation',atol=.001)
  ckpt_name='old' if name=='old' else 'adaptive';cp=OUT/'models'/f'{region}_{ckpt_name}.pt'
  assert sha(cp)==r['checkpoint_hashes'][ckpt_name]
  cls=HistoricalDualField if name=='old' else AdaptiveHistoricalDualField
  model,audit=cls.load_checkpoint(cp)
  assert audit['latest_fit_label_available']<FIT_CUTOFF
  assert audit['fit_cutoff']==FIT_CUTOFF
  if name!='fixed':counts['checkpoints']+=1
  chosen=np.unique(np.linspace(0,len(e['indices'])-1,16).astype(int));maxerr=0.
  for j in chosen:
   gi=int(e['indices'][j]);kw={'fixed':True} if name=='fixed' else {}
   replay=model.encode_matured(origins[gi:gi+1],actual[gi:gi+1],z[gi:gi+1],lags[gi:gi+1],as_of=int(e['encoded_at'][j]),**kw)
   for key in ('trend','event','reconstruction'):
    close(replay[key][0],e[key][j],'checkpoint replay',atol=1e-4)
    maxerr=max(maxerr,float(np.max(np.abs(replay[key][0]-e[key][j]))))
   counts['checkpoint_encoded_rows_replayed']+=1
  sample_replay[name]=maxerr
 mapping={'H_recent':('adaptive','recentraw'),'H_raw':('adaptive','rawmatched'),'H_smooth':('adaptive','fixedsmooth'),'H_old_dual':('old','dualfull'),'H_fixed_pulses':('fixed','dualfull'),'H_adaptive_trend':('adaptive','dualtrend'),'H_adaptive_full':('adaptive','dualfull'),'H_adaptive_event_off':('adaptive','dualtrend')}
 for name,(memory,key) in mapping.items():
  head='H_adaptive_full' if name=='H_adaptive_event_off' else name
  fit=r['fits'][head];assert fit['diagnostics']['latest_fit_label_available']<fit['diagnostics']['fit_cutoff']
  q=apply_history_correction(anchor,memories[memory][key],fit['alpha'])
  equal(q[idx],arts[name].quantiles,('head replay',name));counts['forecast_replays']+=1
 rec=r['reconstruction'];assert rec['first_origin']>=FIT_CUTOFF and rec['latest_label_available']<1664546400
 assert not rec['selection_used']
 validations[region]={'maturity_violations':0,'checkpoint_replay_max_abs':sample_replay,'unique_spike_target_hours':unique,'forecast_replay_exact':True}
assert summarize(results)==read(OUT/'summary.json'),'macro and gates'
previous=PROJECT/'outputs/forecasting/v5_adaptive_dual_20261017_r1'
precision_only={}
for region in ('NSW1','QLD1','TAS1'):
 for kind in ('old','adaptive'):
  name=f'{region}_{kind}.pt'
  before=torch.load(previous/'models'/name,map_location='cpu',weights_only=True)
  after=torch.load(OUT/'models'/name,map_location='cpu',weights_only=True)
  assert before['audit']==after['audit'],('training audit changed',name)
  assert before['config']==after['config'] and before['fit_cutoff']==after['fit_cutoff']
  assert before['state_dict'].keys()==after['state_dict'].keys()
  assert all(torch.equal(v,after['state_dict'][key]) for key,v in before['state_dict'].items()),('trained tensors changed',name)
  precision_only[name]={'training_audit_identical':True,'all_state_tensors_identical':True,'tensor_count':len(before['state_dict'])}

verification={'passed':True,'counts':counts,'regions':validations,'r1_r2_precision_only_training_checks':precision_only,'script_sha256':sha(Path(__file__)),'scope':'Independent sources/current-snapshot, data/cache, parent, checkpoint, maturity, equal-neighbor, rescoring, exact-head-replay validation; reconstruction checkpoint replay sampled16rows per region per representation; no retraining or parameter selection.'}
(OUT/'independent_verification_script.py').write_text(Path(__file__).read_text())
(OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
