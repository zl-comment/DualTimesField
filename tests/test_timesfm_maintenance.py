from types import SimpleNamespace
import numpy as np
import pytest
from scripts.diagnostics.freeze_timesfm_maintenance import gate
from forecasting.timesfm25_xreg import model
from forecasting.timesfm25_xreg.adapter import Adapter


def test_gate_requires_macro_gain_and_each_region_guard():
    good={r:{'original':100.,'ridge1':90.} for r in ('NSW1','QLD1','TAS1')}
    assert gate(good)['accepted']
    good['NSW1']['ridge1']=106.
    assert not gate(good)['accepted']  # macro still improves; one region fails
    with pytest.raises(ValueError):gate({'NSW1':good['NSW1']})


def test_configured_ridge_reaches_native_call_without_changing_mode():
    calls=[]
    fake=SimpleNamespace(forecast_with_covariates=lambda **kw:(calls.append(kw),None))
    model.predict_with_xreg(fake,inputs=[np.zeros(72)],dynamic_numerical_covariates={},static_numerical_covariates={},ridge=1.)
    assert calls[0]['ridge']==1.
    assert calls[0]['xreg_mode']=='xreg + timesfm'
    assert calls[0]['max_rows_per_col']==0


@pytest.mark.parametrize('ridge',[float('nan'),-1.,2.])
def test_adapter_rejects_unplanned_ridge_before_loading_weights(ridge):
    context=SimpleNamespace(config={'foundation_models':{'timesfm25_xreg':{'ridge':ridge,'xreg_mode':'xreg + timesfm'}}})
    with pytest.raises(ValueError):Adapter(context)


def test_test_run_requires_accepted_frozen_development_decision(tmp_path):
    import json
    from forecasting.foundation_experiment import verify_maintenance_selection
    from forecasting.aligned_experiment import sha
    path=tmp_path/'selection.json'
    config={'experiment':{'models':['timesfm25_xreg']},'foundation_models':{'timesfm25_xreg':{'ridge':1.}}}
    with pytest.raises(KeyError):verify_maintenance_selection(config,tmp_path)
    path.write_text(json.dumps({'accepted':False,'test_used_for_selection':False,'selected_ridge':1.}))
    config['maintenance_selection']={'path':'selection.json','sha256':sha(path)}
    with pytest.raises(ValueError,match='gate'):verify_maintenance_selection(config,tmp_path)
    path.write_text(json.dumps({'accepted':True,'test_used_for_selection':False,'selected_ridge':1.}))
    config['maintenance_selection']['sha256']=sha(path)
    assert verify_maintenance_selection(config,tmp_path)==path
    path.write_text('{}')
    with pytest.raises(ValueError,match='fingerprint'):verify_maintenance_selection(config,tmp_path)
