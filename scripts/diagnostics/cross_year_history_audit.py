"""Calendar-analogue descriptive audit and fixed, chronological q50 probes."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from v5_loss_attribution import AEST, ROOT, PARENT, RUN, REGIONS, load, sha, table, verify_freeze

OUT = ROOT / 'logs/forecasting/cross_year_history_audit_20261019'
ARTIFACT = ROOT / 'outputs/forecasting/cross_year_history_audit_20261019'
DATA = ROOT.parent / 'electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem'
HOUR = 3600
END = int(datetime(2023, 1, 1, tzinfo=AEST).timestamp())


def read_prices(region):
    """Do not parse target prices from 2023 onward."""
    origins, price = [], []
    path = DATA / f'{region.lower()}_hourly.csv'
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row['delivery_start_aest'][:4] >= '2023':
                break
            d = datetime.fromisoformat(row['delivery_start_aest'])
            available = datetime.fromisoformat(row['available_at_aest'])
            assert (available-d).total_seconds() == HOUR
            assert row['region'] == region
            origins.append(int(d.timestamp()))
            price.append(float(row['rrp_aud_per_mwh']))
    origins, price = np.array(origins, dtype=np.int64), np.array(price)
    assert np.all(np.diff(origins) == HOUR) and np.isfinite(price).all()
    digest = __import__('hashlib').sha256(origins.tobytes()+price.tobytes()).hexdigest()
    return origins, price, {'path': str(path), 'parsed_start': str(datetime.fromtimestamp(origins[0], AEST)),
                             'parsed_end': str(datetime.fromtimestamp(origins[-1], AEST)), 'parsed_values_sha256': digest}


def annual_indices(origins, source_start, length):
    """Same-date and closest same-weekday origins for each of three prior years."""
    exact = np.full((len(origins), 3), -1, dtype=np.int64)
    weekday = exact.copy()
    valid = np.ones(len(origins), dtype=bool)
    for i, t in enumerate(origins):
        current = datetime.fromtimestamp(int(t), AEST)
        for k in range(1, 4):
            try:
                past = current.replace(year=current.year-k)
            except ValueError:  # February29 has no exact analogue in all three years.
                valid[i] = False
                continue
            shift = (current.weekday()-past.weekday()+3) % 7 - 3
            matched = past + timedelta(days=shift)
            assert matched.weekday() == current.weekday()
            for target, value in ((exact, past), (weekday, matched)):
                target[i, k-1] = (int(value.timestamp())-source_start)//HOUR
                j = target[i, k-1]
                if j < 72 or j+24 > length or int(value.timestamp())+24*HOUR > t:
                    valid[i] = False
    return exact, weekday, valid


def mean_before(price, indices):
    sums = np.r_[0., np.cumsum(price)]
    assert np.all(indices >= 72)
    return (sums[indices]-sums[indices-72])/72


def curve_stats(y, p):
    err = np.abs(y-p)
    yc, pc = y-y.mean(axis=1, keepdims=True), p-p.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(yc, axis=1)*np.linalg.norm(pc, axis=1)
    keep = denominator > 1e-9
    correlation = (yc[keep]*pc[keep]).sum(axis=1)/denominator[keep]
    return {'mae': float(err.mean()), 'median_centered_curve_correlation': float(np.median(correlation)),
            'valid_correlation_windows': int(keep.sum()),
            'ordinary_mae': float(err[(y >= 0) & (y <= 300)].mean()),
            'spike_mae': float(err[y > 300].mean()) if (y > 300).any() else None,
            'spike_prediction_recall': float((p[y > 300] > 300).mean()) if (y > 300).any() else None}


def descriptive(origins, price):
    current = np.flatnonzero((origins >= int(datetime(2018, 1, 1, tzinfo=AEST).timestamp())) & (origins+24*HOUR <= END))
    a, w, valid = annual_indices(origins[current], int(origins[0]), len(price))
    current, a, w = current[valid], a[valid], w[valid]
    curves = np.lib.stride_tricks.sliding_window_view(price, 24)
    y = curves[current]
    three_years = np.median(curves[a], axis=1)
    recent = np.median(curves[current[:, None]-np.array([24, 48, 72])], axis=1)
    annual_shift = np.median(curves[a] + (mean_before(price, current)[:, None]-mean_before(price, a))[..., None], axis=1)
    predictions = {'previous_day': curves[current-24], 'recent_three_days': recent,
                   'recent_three_weekdays': np.median(curves[current[:, None]-np.array([168, 336, 504])], axis=1),
                   'previous_year_same_date': curves[a[:, 0]], 'three_years_same_date': three_years,
                   'three_years_near_same_weekday': np.median(curves[w], axis=1),
                   'three_years_same_date_level_adjusted': annual_shift}
    years = np.array([datetime.fromtimestamp(int(t), AEST).year for t in origins[current]])
    result = {'origins': len(current), 'scores': {k: curve_stats(y,p) for k,p in predictions.items()}, 'by_year': {}}
    for year in range(2018, 2023):
        rows = years == year
        result['by_year'][str(year)] = {'origins': int(rows.sum()), 'scores': {k:curve_stats(y[rows],p[rows]) for k,p in predictions.items()}}
    return result


def fit_probe(x, target):
    """Same fixed q50 objective and budget in every feature ablation."""
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-8, 1., std)
    x = torch.tensor((x-mean)/std, dtype=torch.float64)
    target = torch.tensor(target/300, dtype=torch.float64)
    model = torch.nn.Linear(x.shape[1], 1, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    def objective():
        pred = model(x).squeeze(-1).clamp(-.5,.5)
        params = torch.cat([p.reshape(-1) for p in model.parameters()])
        return .5*(target-pred).abs().mean()+.001*params.square().mean()

    trace=[]
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss=objective()
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        if step in (0, 49, 99, 199):trace.append({'step':step+1,'objective_before_update':float(loss.detach())})
    with torch.no_grad():
        final=float(objective()); null=float(.5*target.abs().mean())
        if final >= null:
            for parameter in model.parameters(): parameter.zero_()
        weights=model.weight.detach().numpy()[0]; bias=float(model.bias.detach()[0])
    return {'mean':mean.tolist(),'std':std.tolist(),'weights':weights.tolist(),'bias':bias,
            'fit_rows':len(x),'features':x.shape[1],'steps':200,'trace':trace,'trained_objective':final,
            'null_objective':null,'selected_null':final>=null}


def predict(model, x):
    correction=((x-np.asarray(model['mean']))/np.asarray(model['std']))@np.asarray(model['weights'])+model['bias']
    return np.clip(300*correction, -150, 150)


def probes(region, origins, price):
    artifacts=[load(PARENT/'B'/f'{region}_{split}.npz') for split in ('dfit_fit','v1')]
    origin=np.concatenate([a['origin_unix'] for a in artifacts])
    assert np.all(np.diff(origin)>0)
    y=np.concatenate([a['actual'] for a in artifacts]).astype(float)
    b=np.concatenate([a['quantiles'] for a in artifacts])
    # The latest state archive uses strict label availability before Dec1;
    # exclude the older v1 artifact's final boundary origin before indexing.
    available=origin+24*HOUR<int(datetime(2022,12,1,tzinfo=AEST).timestamp())
    origin,y,b=origin[available],y[available],b[available]
    indices=np.searchsorted(origins,origin)
    np.testing.assert_array_equal(origins[indices],origin)
    y_source=np.lib.stride_tricks.sliding_window_view(price,24)[indices]
    np.testing.assert_allclose(y_source,y,rtol=1e-6,atol=1e-3)
    memory=load(RUN/'features'/f'{region}_full.npz')
    mi=np.searchsorted(memory['origin_unix'],origin)
    np.testing.assert_array_equal(memory['origin_unix'][mi],origin)
    a,w,valid=annual_indices(origin,int(origins[0]),len(price))
    assert valid.all()
    lag=price[indices[:,None]+np.arange(-72,0)]
    summary=np.column_stack((lag.mean(axis=1),lag.std(axis=1),lag[:,-1],lag[:,-24:].mean(axis=1)))
    daily=np.stack((lag[:,-24:],lag[:,-48:-24],lag[:,:24]),axis=-1)
    core=np.concatenate((memory['features'][mi,:,:16][:,:,1:],np.broadcast_to(summary[:,None],(len(origin),24,4))/300,daily/300),axis=-1)
    curves=np.lib.stride_tricks.sliding_window_view(price,24)
    feature_sets={'current_only':core}
    for name,ids in (('annual_same_date',a),('annual_same_weekday',w)):
        extra=np.concatenate((curves[ids].transpose(0,2,1),np.broadcast_to(mean_before(price,ids)[:,None],(len(origin),24,3))),axis=-1)/300
        feature_sets[name]=np.concatenate((core,extra),axis=-1)
    cut=int(datetime(2022,10,1,tzinfo=AEST).timestamp())
    fit=origin+24*HOUR<cut
    assert np.all(origin[fit]>=int(datetime(2022,7,1,tzinfo=AEST).timestamp()))
    result={'fit_origins':int(fit.sum()),'fit_label_cutoff':str(datetime.fromtimestamp(cut,AEST)),
            'models':{},'months':{},'source_archive_sha256':{f'{region}_{s}':sha(PARENT/'B'/f'{region}_{s}.npz') for s in ('dfit_fit','v1')}}
    all_preds={'LightGBM':b[:,:,2]}
    for name,x in feature_sets.items():
        model=fit_probe(x[fit].reshape(-1,x.shape[-1]),(y-b[:,:,2])[fit].ravel())
        result['models'][name]=model
        p=b[:,:,2]+predict(model,x.reshape(-1,x.shape[-1])).reshape(y.shape)
        all_preds[name]=p
    for month in (10,11):
        start=int(datetime(2022,month,1,tzinfo=AEST).timestamp());end=int(datetime(2022,month+1,1,tzinfo=AEST).timestamp())
        rows=(origin>=start)&(origin+24*HOUR<end)
        result['months'][str(month)]={'origins':int(rows.sum()),'mae':{name:float(np.abs(y[rows]-p[rows]).mean()) for name,p in all_preds.items()},
                                     'bands':{name:{band:float(np.abs(y[rows]-p[rows])[mask].sum()/y[rows].size) for band,mask in [('negative',y[rows]<0),('ordinary',(y[rows]>=0)&(y[rows]<=300)),('spike',y[rows]>300)]} for name,p in all_preds.items()}}
        np.savez_compressed(ARTIFACT/f'{region}_{month}_probes.npz',origin_unix=origin[rows],actual=y[rows],**{k:v[rows] for k,v in all_preds.items()})
    return result


def november_example(series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    colors=plt.cm.viridis(np.linspace(.05,.95,8))
    rows=[]
    for ri,region in enumerate(REGIONS):
        origins,price=series[region]
        for yi,year in enumerate(range(2015,2023)):
            t=int(datetime(year,11,11,tzinfo=AEST).timestamp()); j=int(np.searchsorted(origins,t))
            assert origins[j]==t
            curve=price[j:j+24]
            for h,value in enumerate(curve):rows.append([region,year,h,float(value)])
            for col,values in ((0,curve),(1,curve-curve.mean())):
                axes[ri,col].plot(np.arange(24),values,color=colors[yi],lw=2 if year==2022 else 1.1,alpha=.95,label=str(year))
                axes[ri,col].grid(alpha=.2)
                axes[ri,col].set_title(f'{region}: Nov 11 '+('raw prices' if col==0 else 'daily mean removed'))
                axes[ri,col].set_ylabel('AUD/MWh')
    for ax in axes[-1]:ax.set_xlabel('Delivery hour (AEST)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=8)
    fig.suptitle('Same calendar date across years: descriptive example, not a forecasting score')
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(OUT/'november11_across_years.png',dpi=160)
    fig.savefig(OUT/'november11_across_years.pdf')
    plt.close(fig)
    with (OUT/'november11_prices.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['region','year','hour_AEST','price']);writer.writerows(rows)


def report(result):
    lines=['# 跨年同日期历史检查', '', '冻结模型保持不变；只分析2015–2022，当前结果为回顾性开发。所有年度候选仅使用查询之前已成熟的价格。', '',
           '## 朴素历史曲线直接预测：2018–2022', '', '所有方法使用相同有效逐小时窗口，完整结果按年份保存在summary.json。均价和尖峰可能跨年改变，原曲线直接复制不等于特征无价值。', '']
    names=list(result['regions']['NSW1']['descriptive']['scores'])
    rows=[]
    for name in names:
        vals=[result['regions'][r]['descriptive']['scores'][name]['mae'] for r in REGIONS]
        rows.append([name]+[f'{v:.4f}' for v in vals]+[f'{np.mean(vals):.4f}'])
    lines+=table(['方法','NSW1 MAE','QLD1 MAE','TAS1 MAE','宏MAE'],rows)
    lines+=['','## 日内形状相关（每个24小时窗口去均值）','','相关性只是描述，不是增量预测证据，也没有据此选择模型。','']
    lines+=table(['方法','NSW1相关中位数','QLD1相关中位数','TAS1相关中位数'],[[name]+[f"{result['regions'][r]['descriptive']['scores'][name]['median_centered_curve_correlation']:.4f}" for r in REGIONS] for name in names])
    lines+=['','## 固定q50线性修正探针：7–9月拟合，10/11月评价','','current_only包含当前状态和72小时信息；annual_same_date与annual_same_weekday分别只增加六个年度价格特征。三个模型预算、目标、拟合期一致，均非完整双场模型。','']
    rows=[]
    for r in REGIONS:
        for month in ('10','11'):
            d=result['regions'][r]['probe']['months'][month]
            rows.append([r,month,d['origins']]+[f"{d['mae'][n]:.4f}" for n in ('LightGBM','current_only','annual_same_date','annual_same_weekday')]+[f"{d['mae']['annual_same_date']-d['mae']['current_only']:+.4f}"])
    lines+=table(['地区','2022月份','窗口数','LightGBM','当前信息修正','增加同日期','增加同星期','同日期增量ΔMAE'],rows)
    lines+=['','## 11月11日实例','','见november11_across_years.png/PDF和逐小时CSV。三地区2015–2022全年度曲线均展示，没有按效果筛选。这个单日不能代表全年。','',
            '## 解释边界','','- 历史价格来自已知过去，即使属于LightGBM训练年份也可合法作为输入；本探针未使用该模型在训练年份的残差。',
            '- 这是固定线性q50探针，不能证明或否定可训练双场、非线性检索、联合优化的效果。没有使用10/11月选择参数。',
            '- 年度日期存在星期错位；闰年2月29日没有完整同日期匹配时从所有朴素方法共同样本中排除。',
            '- 拟合标准化只用7–9月。训练目标统一q50 pinball加L2，200步，最后一步或零修正仅按训练目标选择。',
            '- 未给统计显著性结论；逐小时24h预测存在标签重叠，不能按独立单元计数。',
            '- 原始数据和外生可用性沿用已有档案代理；没有新市场情绪或边际机组数据。','']
    (OUT/'report.md').write_text('\n'.join(lines))


def main():
    freeze=verify_freeze()
    OUT.mkdir(exist_ok=False); ARTIFACT.mkdir(exist_ok=False)
    result={'frozen_commit':freeze['frozen_commit'],'protocol_sha256':sha(ROOT/'docs/cross_year_history_audit_20261019.md'),
            'script_sha256':sha(Path(__file__)),'regions':{}}
    series={}
    for region in REGIONS:
        print(region, 'loading pre-2023 prices and annual patterns',flush=True)
        origins,price,source=read_prices(region);series[region]=(origins,price)
        d=descriptive(origins,price)
        print(region, 'fitting three fixed q50 probes',flush=True)
        p=probes(region,origins,price)
        result['regions'][region]={'source':source,'descriptive':d,'probe':p}
        print(region,{m:v['mae'] for m,v in p['months'].items()},flush=True)
    november_example(series)
    verify_freeze();result['frozen_files_unchanged']=True
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    report(result)
    print(OUT/'report.md',flush=True)


if __name__=='__main__':
    main()
