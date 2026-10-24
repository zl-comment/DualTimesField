"""Real-weight development-only check of canonical batch behavior (one horizon)."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from forecasting.common.cache import load_split
from forecasting.aligned_inputs import flatten_inputs
from forecasting.tabpfn_v3 import model

root=Path(__file__).resolve().parents[2]
run=root.parent/'DualTimesField-foundation-baselines/outputs/forecasting/foundation_baselines_20261023_r1'
out=root/'outputs/forecasting/baseline_maintenance_20261024/tabpfn'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
manifest=json.loads((run/'jobs.json').read_text())
train=load_split(run,manifest,'NSW1','train');dev=load_split(run,manifest,'NSW1','selection')
x=flatten_inputs(train).astype(np.float32)
y=train.target[:,0].astype(np.float64)*train.metadata['target_std']+train.metadata['target_mean']
reg=model.create_regressor(model.resolve_checkpoint_path(os.environ['HF_HOME']),device='cuda:0',random_state=2026)
start=time.monotonic();print('fit one horizon, complete training context',x.shape,flush=True);reg.fit(x,y)
q=flatten_inputs(dev).astype(np.float32)[:4096];ids=np.array([0,3,7,31,63,127,255,511])
print('production-shape development query',q.shape,flush=True)
full=model.predict_quantiles(reg,q)
print('independent production-shape replay',flush=True)
replay=model.predict_quantiles(reg,q)
print('noncanonical diagnostic query',flush=True)
small=model.predict_quantiles(reg,q[ids])
result={'region':'NSW1','horizon':1,'seed':2026,'split':'selection','train_rows':len(x),'production_query_rows':len(q),'sample_indices':ids.tolist(),'full_replay_max_abs_difference':float(np.max(np.abs(full-replay))),'canonical_subset_max_abs_difference':float(np.max(np.abs(full[ids]-replay[ids]))),'noncanonical_subset_max_abs_difference':float(np.max(np.abs(full[ids]-small))),'seconds':time.monotonic()-start,'scope':'one horizon development evidence; not all 24 horizons or all test cells'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'development.npz',full=full,replay=replay,small=small,ids=ids)
print(result,flush=True)
