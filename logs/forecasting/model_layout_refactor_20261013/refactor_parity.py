import sys, os, copy, json
from pathlib import Path
sys.path.insert(0,os.getcwd())
import numpy as np
import torch
import yaml
from forecasting.aligned_inputs import INPUT_KEYS, AlignedSplit
from forecasting.aligned_runner import NEURAL_MODELS, fit_neural
from forecasting.aligned_baselines import fit_predict

torch.set_num_threads(2)
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
configpath=Path('configs/experiments/aemo_forecast_aligned_20261012.yaml')
if not configpath.exists(): configpath=Path('configs/aemo_forecast_aligned_20261012.yaml')
config=yaml.safe_load(configpath.read_text())
config['model_baselines']={'timexer':dict(patch_len=24,d_model=16,n_heads=4,e_layers=2,d_ff=32,dropout=0.)}
def split(start):
 rng=np.random.default_rng(start); n=12
 shapes=[(72,2),(72,10),(24,10),(1,),(24,11),(24,1),(24,3)]
 inputs={k:rng.normal(size=(n,*shape)).astype(np.float32) for k,shape in zip(INPUT_KEYS,shapes)}
 target=rng.normal(size=(n,24)).astype(np.float32)
 origin=(start+np.arange(n))*3600; times=origin[:,None]+np.arange(24)*3600
 return AlignedSplit(inputs,target*10+20,origin,times,dict(preprocessing_hash='same',input_hash=str(start),sample_hash=str(start),region='NSW1',protocol_id='parity',target_mean=20.,target_std=10.),target)
train,selection,calibration,evaluation=[split(s) for s in (0,100,200,300)]
summary={}
for kind in NEURAL_MODELS:
 folder={'gatets_adapter':'gatets'}.get(kind,kind)
 specific=Path('configs')/folder/'aligned.yaml'
 training_config=yaml.safe_load(specific.read_text()) if specific.exists() else copy.deepcopy(config)
 if kind=='timexer':training_config['model_baselines']={'timexer':config['model_baselines']['timexer']}
 q,point,metadata=fit_neural(kind,train,selection,evaluation,training_config,epochs=1,batch_size=6,seed=2026,checkpoint_path=out/f'{kind}.pt')
 values={'quantiles':q}
 if point is not None: values['point']=point
 np.savez(out/f'{kind}.npz',**values)
 summary[kind]=metadata
 print(kind,flush=True)
for kind in ['lightgbm','xgboost','lasso_arx72_cp']:
 params=dict(n_estimators=4,early_stopping_rounds=2,n_jobs=2,selection_interval=1)
 if kind=='lasso_arx72_cp': params=dict(alphas=[.1],max_iter=3000)
 r=fit_predict(kind,train,selection,evaluation,seed=2026,calibration=calibration if kind=='lasso_arx72_cp' else None,params=params)
 np.savez(out/f'{kind}.npz',quantiles=r.quantiles,point=r.point)
 summary[kind]=r.metadata
 print(kind,flush=True)
(out/'metadata.json').write_text(json.dumps(summary,indent=2))
