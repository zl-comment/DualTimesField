"""Pre-check for step 3 (calendar-anchored CTF extrapolation).

For the detected-event DGF trunk (seed 2026), fits a ridge regression of the
model's target-space residuals on two candidate feature groups and scores the
corrected forecasts on the test split:

* trend: a per-sample weighted ridge fit of the de-evented history on a
  clock-anchored harmonic basis (level, linear trend, 3 daily harmonics),
  extrapolated over the 24 forecast hours;
* net load x hour: the net-load forecast times daily harmonics of the clock
  hour, which horizon-indexed linear heads cannot form.

The fit uses the split named in collect(region, ...); the committed version
uses the 2022 validation split. Run from the repository root with
PYTHONPATH=. after training configs/aemo_forecast_detected_dgf.yaml.
"""
import math, torch, numpy as np
from sklearn.linear_model import Ridge
from forecasting.datasets import build_region_datasets, load_forecast_config
from forecasting.train import build_model, DeviceBatches
dev=torch.device('cuda'); cfg='configs/aemo_forecast_detected_dgf.yaml'; config=load_forecast_config(cfg)

def anchored_features(x_price, events, cal, netload, tau=24.0, lam=1e-2, K=3):
    B,T=x_price.shape; H=cal.shape[1]
    theta0=torch.atan2(cal[:,0,0],cal[:,0,1])                  # clock phase of first forecast hour
    past=torch.arange(T,device=dev,dtype=torch.float32); fut=torch.arange(H,device=dev,dtype=torch.float32)
    th_p=theta0[:,None]-2*math.pi*(T-past[None])/24; th_f=theta0[:,None]+2*math.pi*fut[None]/24
    def basis(th,step):
        cols=[torch.ones_like(th),(step-(T-1))/T*torch.ones_like(th)]
        for k in range(1,K+1): cols+= [torch.cos(k*th),torch.sin(k*th)]
        return torch.stack(cols,-1)
    Pp=basis(th_p,past[None]); Pf=basis(th_f,T+fut[None])
    w=torch.exp(-(T-1-past)/tau)[None,:,None]
    A=(Pp*w).transpose(1,2)@Pp+lam*torch.eye(Pp.shape[-1],device=dev); b=(Pp*w).transpose(1,2)@(x_price-events)[...,None]
    c=torch.linalg.solve(A,b); trend=(Pf@c)[...,0]
    inter=torch.stack([netload*torch.cos(th_f),netload*torch.sin(th_f),netload*torch.cos(2*th_f),netload*torch.sin(2*th_f)],-1)
    return trend, inter

def collect(region, split):
    ds=build_region_datasets(cfg,region)[split]; m=build_model(config).to(dev)
    ck=torch.load(f'outputs/forecasting/detected_dgf/{region}/best_model.pt',map_location=dev,weights_only=False); m.load_state_dict(ck['model_state']); m.set_epoch(ck.get('model_epoch',ck['best_epoch'])); m.eval()
    P=[];Y=[];TR=[];IN=[];NL=[]
    with torch.no_grad():
        for b in DeviceBatches(ds,4096,False,dev):
            out=m(b['history_values'],b['future_calendar'],b['future_exogenous'],b['origin_context'],b['ctf_exogenous'],b['quantile_exogenous'])
            tr,inter=anchored_features(b['history_values'][...,0],out['event_signal'][...,0],b['future_calendar'],b['ctf_exogenous'][...,0])
            P.append(out['point_forecast'][...,0].cpu()); Y.append(b['target_price'][...,0].cpu()); TR.append(tr.cpu()); IN.append(inter.cpu()); NL.append(b['ctf_exogenous'][...,0].cpu())
    cat=lambda L: torch.cat(L).numpy()
    return ds, cat(P), cat(Y), cat(TR), cat(IN), cat(NL)

for region in ['NSW1','QLD1','TAS1']:
    dtr,Ptr,Ytr,TRtr,INtr,NLtr=collect(region,'validation'); dte,Pte,Yte,TRte,INte,NLte=collect(region,'test')
    to_price=lambda z: dte.denormalize_target(torch.from_numpy(z)).numpy()
    actual=to_price(Yte); base=np.abs(to_price(Pte)-actual).mean()
    res=[]
    for name,(ftr,fte) in {'trend':(TRtr[...,None],TRte[...,None]),
                           'netload x hour':(INtr,INte),
                           'both':(np.concatenate([TRtr[...,None],INtr],-1),np.concatenate([TRte[...,None],INte],-1))}.items():
        # include the model's own point forecast as a feature so the fit only adds what is missing
        Xtr=np.concatenate([ftr,Ptr[...,None]],-1).reshape(-1,ftr.shape[-1]+1); Xte=np.concatenate([fte,Pte[...,None]],-1).reshape(-1,fte.shape[-1]+1)
        r=Ridge(alpha=1.0).fit(Xtr,(Ytr-Ptr).ravel())
        corr=Pte+r.predict(Xte).reshape(Pte.shape)
        res.append(f"{name}: MAE {np.abs(to_price(corr)-actual).mean():.2f}")
    print(f"{region}: model MAE {base:.2f} | "+' | '.join(res), flush=True)
