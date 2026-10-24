from types import SimpleNamespace
import numpy as np
import pytest
from forecasting.tabpfn_v3.adapter import Adapter
from forecasting.tabpfn_v3 import model
from forecasting.aligned_inputs import INPUT_KEYS
from forecasting.common.foundation_api import PredictionView


def adapter(n=11):
    a=Adapter.__new__(Adapter)
    inputs={key:np.arange(n,dtype=float)[:,None,None] for key in INPUT_KEYS}
    a.context=SimpleNamespace(test_view=PredictionView(inputs,np.arange(n),np.zeros((n,24)),{}))
    a.batch_queries=4;a._regressors=list(range(24));a._canonical_cache=None
    return a


def test_queries_preserve_production_shapes_order_duplicates_and_last_chunk(monkeypatch):
    # Deliberately batch-size-sensitive predictor; independent of companion values.
    monkeypatch.setattr(model,'predict_quantiles',lambda reg,x: np.repeat((x[:,0]+len(x)+reg)[:,None],5,axis=1))
    a=adapter()
    full=a.predict_chunk(np.arange(11))['quantiles']
    a._canonical_cache=None
    ids=np.array([10,0,5,5,3,8])
    np.testing.assert_array_equal(a.predict_chunk(ids)['quantiles'],full[ids])
    for start in range(0,11,4):
        ids=np.arange(start,min(start+4,11))
        expected=np.stack([model.predict_quantiles(r,np.repeat(ids[:,None],7,axis=1)) for r in range(24)],axis=1)
        np.testing.assert_array_equal(full[ids],expected)


@pytest.mark.parametrize('ids',[np.array([-1]),np.array([11]),np.array([0.5]),np.array([],dtype=int)])
def test_invalid_indices_are_rejected(ids):
    with pytest.raises(ValueError):adapter().predict_chunk(ids)
