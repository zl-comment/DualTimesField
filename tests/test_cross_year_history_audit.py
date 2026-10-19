"""Calendar and historical-availability invariants for the analogue audit."""
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

folder=Path(__file__).resolve().parents[1]/'scripts/diagnostics'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('cross_year_audit',folder/'cross_year_history_audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def source():
    start=int(datetime(2015,1,1,tzinfo=audit.AEST).timestamp())
    end=int(datetime(2023,1,1,tzinfo=audit.AEST).timestamp())
    return start,(end-start)//3600


def test_november11_uses_same_date_and_hour_in_each_prior_year():
    start,length=source()
    t=int(datetime(2022,11,11,12,tzinfo=audit.AEST).timestamp())
    exact,weekday,valid=audit.annual_indices(np.array([t]),start,length)
    assert valid.tolist()==[True]
    for k in range(3):
        d=datetime.fromtimestamp(start+int(exact[0,k])*3600,audit.AEST)
        assert (d.year,d.month,d.day,d.hour)==(2021-k,11,11,12)
        w=datetime.fromtimestamp(start+int(weekday[0,k])*3600,audit.AEST)
        assert w.weekday()==4
        assert abs((w-d).days)<=3
        assert start+int(weekday[0,k])*3600+86400<=t


def test_leap_day_and_insufficient_history_excluded():
    start,length=source()
    origins=np.array([int(datetime(2020,2,29,tzinfo=audit.AEST).timestamp()),
                      int(datetime(2018,1,1,tzinfo=audit.AEST).timestamp())])
    _,_,valid=audit.annual_indices(origins,start,length)
    assert not valid.any()


def test_level_context_uses_only_preceding_72_hours():
    price=np.arange(200,dtype=float)
    original=audit.mean_before(price,np.array([100]))
    price[100:]=1e9
    np.testing.assert_array_equal(audit.mean_before(price,np.array([100])),original)
    assert original[0]==np.arange(28,100).mean()


def test_saved_probe_correction_is_bounded_and_zero_model_returns_zero():
    model={'mean':[1.,2.],'std':[1.,1.],'weights':[0.,0.],'bias':0.}
    x=np.array([[1e10,-1e10],[0,0]])
    np.testing.assert_array_equal(audit.predict(model,x),[0.,0.])
    model['weights']=[1.,0.]
    np.testing.assert_array_equal(audit.predict(model,x),[150.,-150.])
