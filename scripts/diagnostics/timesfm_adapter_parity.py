"""Check the production adapter against the development diagnostic on fixed rows."""
import json
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from forecasting.common.cache import load_split
from forecasting.common.foundation_api import PredictionView
from forecasting.timesfm25_xreg.adapter import Adapter
from scripts.diagnostics.baseline_maintenance import ROOT, RUN, save, digest

torch.set_num_threads(4)
manifest=json.loads((RUN/'jobs.json').read_text())
split=load_split(RUN,manifest,'NSW1','selection')
view=PredictionView(split.inputs,split.origin_unix,split.target_time_unix,split.metadata)
config=yaml.safe_load((ROOT/'configs/timesfm25_xreg/aligned.yaml').read_text());config['project_root']=str(ROOT)
context=SimpleNamespace(config=config,device='cuda:0',test_view=view)
path=ROOT/'outputs/forecasting/baseline_maintenance_20261024/timesfm_full_selection/NSW1.npz'
indices=np.arange(8); diffs={}
with np.load(path,allow_pickle=False) as stored:
    for name,ridge in [('original',0.),('ridge1',1.)]:
        config['foundation_models']['timesfm25_xreg']['ridge']=ridge
        adapter=Adapter(context)
        output=adapter.predict_chunk(indices)['point']; expected=stored[name][indices]
        diff=float(np.abs(output-expected).max());scale=max(1.,float(np.abs(expected).max()))
        diffs[name]={'max_abs_diff':diff,'relative_to_max_abs_prediction':diff/scale}
        assert diff/scale<=1e-3
        del adapter
save(ROOT/'logs/forecasting/baseline_maintenance_20261024/timesfm_adapter_parity.json',
    {'split':'selection','region':'NSW1','indices':indices.tolist(),'differences':diffs,
     'reference_sha256':digest(path),'script_sha256':digest(__file__),
     'scope':'fixed first eight development origins; two predefined configurations; no test labels'})
print(diffs,flush=True)
