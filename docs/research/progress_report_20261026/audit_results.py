from pathlib import Path
import csv,json,hashlib
import numpy as np
R=Path('/root/workspace/Code/DualTimesField'); O=Path('/root/workspace/Code/research_progress_report_20261026'); C=R/'logs/forecasting/main_comparison_20261023'
def read(p):
 with open(p) as f: return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
main=read(C/'main.csv');reg=read(C/'regional.csv');per=read(C/'per_seed.csv');v=json.load(open(C/'verification.json')); models={x['model'] for x in main}; pe={(x['model'],x['region'],int(x['seed'])):x for x in per}; out=[];refs={}; inventory=[]; audit=[]
for a in v['artifacts']:
 m=a['model'];r=a['region'];s=a['seed']
 if m not in models:continue
 p=Path(a['path']); h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==a['sha256'],p
 with np.load(p,allow_pickle=False) as z:
  y=z['actual'];t=z['target_time_unix'];o=z['origin_unix'];q=np.sort(z['quantiles'],axis=-1) if 'quantiles' in z.files else None
  point=q[...,2] if q is not None else z['native_point'];err=point-y
  if r not in refs:
   refs[r]=(y.copy(),t.copy(),o.copy())
   for name,mask in {'negative':y<0,'normal_0_to_300':(y>=0)&(y<=300),'high_gt300':y>300,'extreme_gt1000':y>1000}.items():
    inventory.append(dict(region=r,band=name,origin_horizon_cells=int(mask.sum()),total_cells=int(y.size),unique_delivery_hours=int(np.unique(t[mask]).size),cell_fraction=float(mask.mean())))
  else:
   assert np.array_equal(y,refs[r][0]) and np.array_equal(t,refs[r][1]) and np.array_equal(o,refs[r][2])
  mae=float(np.abs(err).mean());assert abs(mae-float(pe[m,r,s]['mae']))<1e-8,(m,r,s,mae)
  masks={'negative':y<0,'normal_0_to_300':(y>=0)&(y<=300),'high_gt300':y>300,'extreme_gt1000':y>1000,'window_moderate_300_1000':np.broadcast_to(((y.max(1)>=300)&(y.max(1)<1000))[:,None],y.shape),'window_extreme_ge1000':np.broadcast_to((y.max(1)>=1000)[:,None],y.shape),'window_sustained_mean_ge300':np.broadcast_to((y.mean(1)>=300)[:,None],y.shape),'window_calm_max_lt300':np.broadcast_to((y.max(1)<300)[:,None],y.shape)}
  for name,mask in masks.items():
   e=err[mask]; yy=y[mask]
   row=dict(model=m,region=r,seed=s,band=name,n_cells=int(mask.sum()),n_origins=int(mask.any(1).sum()),mae=float(np.abs(e).mean()),bias=float(e.mean()),rmse=float(np.sqrt((e**2).mean())),quantile_score=None,coverage_80=None,coverage_90=None)
   if q is not None:
    qq=q[mask]; residual=yy[:,None]-qq;levels=np.array([.05,.1,.5,.9,.95]);row.update(quantile_score=float(2*np.maximum(levels*residual,(levels-1)*residual).mean()),coverage_80=float(((yy>=qq[:,1])&(yy<=qq[:,3])).mean()),coverage_90=float(((yy>=qq[:,0])&(yy<=qq[:,4])).mean()))
   out.append(row)
  audit.append(dict(model=m,region=r,seed=s,path=str(p),sha256=h,mae_reproduced=mae))
 print(m,r,s,flush=True)
write(O/'latest_spike_diagnostics_per_seed.csv',out);write(O/'latest_spike_inventory.csv',inventory)
metrics=['mae','bias','rmse','quantile_score','coverage_80','coverage_90'];groups={}
for row in out:groups.setdefault((row['model'],row['region'],row['band']),[]).append(row)
regional=[]
for (m,r,b),rows in groups.items():
 d=dict(model=m,region=r,band=b,n_cells=rows[0]['n_cells'],n_origins=rows[0]['n_origins'])
 for k in metrics:
  vals=[x[k] for x in rows if x[k] is not None];d[k+'_mean']=float(np.mean(vals)) if vals else None;d[k+'_std']=float(np.std(vals,ddof=1)) if vals else None
 regional.append(d)
write(O/'latest_spike_diagnostics_regional.csv',regional)
macro=[]
for m in sorted(models):
 for b in masks:
  d=dict(model=m,band=b)
  for k in metrics:
   vs=[]
   for s in [2026,2027,2028]:
    a=[x[k] for x in out if x['model']==m and x['band']==b and x['seed']==s and x[k] is not None]
    if a:vs.append(float(np.mean(a)))
   d[k+'_mean']=float(np.mean(vs)) if vs else None;d[k+'_std']=float(np.std(vs,ddof=1)) if vs else None
  macro.append(d)
write(O/'latest_spike_diagnostics_macro.csv',macro)
json.dump({'artifacts_checked':audit,'rows':len(out),'protocol':'2023-2024 latest current-main-table artifacts; SHA256 checked; actual/origin/target timestamps bitwise aligned across all models and seeds within region; sorted q50 for Q and native point for P; 3-region equal average within each seed followed by 3-seed mean/sample SD; retrospective report-only diagnosis, no selection or training; thresholds actual >300, >1000; windows separate; overlapping origin-hour cells are not independent events'},open(O/'latest_spike_diagnostics_provenance.json','w'),ensure_ascii=False,indent=2)
print('DONE',len(audit),len(out))
