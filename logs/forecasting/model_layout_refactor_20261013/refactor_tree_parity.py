import sys,os,json
from pathlib import Path
sys.path.insert(0,os.getcwd())
import numpy as np
from forecasting.aligned_inputs import INPUT_KEYS,AlignedSplit
from forecasting.aligned_baselines import fit_predict
out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
def split(start):
 rng=np.random.default_rng(start);shapes=[(4,2),(4,2),(2,2),(1,),(2,2),(2,1),(2,3)]
 inputs={k:rng.normal(size=(64,*s)).astype('float32') for k,s in zip(INPUT_KEYS,shapes)}
 target=2*inputs['future_exogenous'][...,0]; origins=(start+np.arange(64))*3600;times=origins[:,None]+np.arange(2)*3600
 meta=dict(region='synthetic',protocol_id='parity',preprocessing_hash='same',input_hash=str(start),sample_hash=str(start),target_mean=10.,target_std=3.)
 return AlignedSplit(inputs,target*3+10,origins,times,meta,target)
train,select,evaluation=[split(s) for s in (0,100,200)]
report={}
for model in ('lightgbm','xgboost'):
 for scope in ('per_horizon','shared'):
  params=dict(n_estimators=8,early_stopping_rounds=3,n_jobs=1,selection_scope=scope)
  params.update(dict(num_leaves=5,min_child_samples=3) if model=='lightgbm' else dict(max_depth=3))
  r=fit_predict(model,train,select,evaluation,params=params,seed=2026)
  name=f'{model}_{scope}';np.save(out/f'{name}.npy',r.quantiles);report[name]=r.metadata
  assert np.std(r.quantiles[:,:,2])>0
(out/'metadata.json').write_text(json.dumps(report,indent=2))
print('Validated nonconstant forecasts for both tree models and both selection scopes')
