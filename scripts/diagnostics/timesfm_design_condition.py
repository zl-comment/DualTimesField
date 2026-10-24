"""Inspect XReg design geometry on outcome-independent selection samples."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
import numpy as np
from timesfm.utils.xreg_lib import BatchedInContextXRegLinear
from forecasting.common.cache import load_split
from scripts.diagnostics.baseline_maintenance import ROOT, RUN, REGIONS, save, digest

manifest=json.loads((RUN/'jobs.json').read_text()); result={}
for region in REGIONS:
    split=load_split(RUN,manifest,region,'selection')
    ids=np.unique(np.linspace(0,len(split)-1,128).astype(int))
    ranks=[]; conditions=[]; leverage={0:[],1:[]}; constant_gas=[]
    for i in ids:
        hist=split.inputs['history_calendar'][i]; future=split.inputs['future_calendar'][i]
        xreg=BatchedInContextXRegLinear(targets=[np.zeros(72)],train_lens=[72],test_lens=[24],
            train_dynamic_numerical_covariates={f'cal_{c}':[hist[:,c]] for c in range(10)},
            test_dynamic_numerical_covariates={f'cal_{c}':[future[:,c]] for c in range(10)},
            static_numerical_covariates={'gas_7d_log_std':[float(split.inputs['origin_context'][i,0])]})
        _,x,z=xreg.create_covariate_matrix()
        ranks.append(int(np.linalg.matrix_rank(x)))
        singular=np.linalg.svd(x,compute_uv=False); active=singular[singular>singular[0]*1e-6]
        conditions.append(float(active[0]/active[-1]))
        constant_gas.append(float(np.max(np.abs(x[:,-1]))))
        for ridge in (0,1):
            # Float64 mathematical diagnostic, not a replica of JAX's float32 solver.
            operator=z@np.linalg.pinv(x.T@x+ridge*np.eye(x.shape[1]),rcond=1e-6)@x.T
            leverage[ridge].append(float(np.linalg.norm(operator,axis=1).max()))
    result[region]={'sample_count':len(ids),'columns':x.shape[1],'rank_min':min(ranks),'rank_max':max(ranks),
        'nontrivial_singular_condition_median':float(np.median(conditions)),
        'gas_centered_column_max_abs':max(constant_gas),
        'horizon_operator_row_norm':{str(k):{'median':float(np.median(v)),'max':max(v)} for k,v in leverage.items()}}
save(ROOT/'logs/forecasting/baseline_maintenance_20261024/timesfm_design.json',
    {'split':'selection','label_values_used':False,'script_sha256':digest(__file__),'regions':result,
     'scope':'design-only float64 diagnostic; not a causal attribution of all MAE differences'})
print(json.dumps(result,indent=2))
