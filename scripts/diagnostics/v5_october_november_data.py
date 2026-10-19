"""Describe saved October/November labels, errors and actual history dates.

No model fitting; October fitted-head results are not a forward backtest.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from v5_loss_attribution import AEST, OUT, PARENT, REGIONS, RUN, load, sha, table, verify_freeze


def stamp(month, day=1, hour=0):
    return int(datetime(2022, month, day, hour, tzinfo=AEST).timestamp())


def date(value):
    return datetime.fromtimestamp(int(value), AEST).isoformat()


def main():
    frozen = verify_freeze()
    result = {'frozen_commit': frozen['frozen_commit'], 'regions': {},
              'price_scope': 'Full calendar month, unique delivery hours: 744 October /720 November',
              'prediction_scope': 'Origins with full labels available strictly before next month: 720 October /696 November',
              'history_start': '2022-10-01T00:00:00+10:00',
              'fit_warning': 'Gates fitted using whole October. October predictions are IN-SAMPLE, not a chronological daily deployment backtest.'}
    for region in REGIONS:
        b = load(PARENT / 'B' / f'{region}_v1+v2.npz')
        c = load(PARENT / 'C_roll' / f'{region}_v1+v2.npz')
        m = load(RUN / 'features' / f'{region}_full.npz')
        np.testing.assert_array_equal(b['origin_unix'], c['origin_unix'])
        np.testing.assert_array_equal(b['actual'], c['actual'])
        origins, labels = b['origin_unix'], b['actual'].astype(float)
        times = origins[:, None] + np.arange(24) * 3600
        unique, first, inverse = np.unique(times.ravel(), return_index=True, return_inverse=True)
        prices = labels.ravel()[first]
        np.testing.assert_array_equal(prices[inverse], labels.ravel())
        r = {'months': {}, 'history_examples': {}}
        for month in (10, 11):
            start, end = stamp(month), stamp(month+1)
            values = prices[(unique >= start) & (unique < end)]
            assert len(values) == (744 if month == 10 else 720)
            rows = (origins >= start) & (origins + 86400 < end)
            ix = np.searchsorted(m['origin_unix'], origins[rows])
            np.testing.assert_array_equal(m['origin_unix'][ix], origins[rows])
            errors = {}
            for name, p in (('LightGBM', b), ('C_roll', c)):
                err = labels[rows] - p['quantiles'][rows, :, 2]
                errors[name] = {'mean_actual_minus_prediction': float(err.mean()),
                                'median_actual_minus_prediction': float(np.median(err)),
                                'underprediction_fraction': float((err > 0).mean()),
                                'mae': float(np.abs(err).mean())}
            states = {name: {'mean': float(m['features'][ix, :, i].mean()),
                             'std': float(m['features'][ix, :, i].std())}
                      for i, name in ((3, 'predispatch_price'), (4, 'demand'), (5, 'available_generation'),
                                      (6, 'spare_capacity'), (7, 'net_load'), (8, 'import_room'))}
            ids = m['trend_ids'][ix]
            valid = ids >= 0
            selected = m['origin_unix'][ids[valid]]
            query_times = np.broadcast_to(origins[rows, None, None], ids.shape)[valid]
            assert np.all(selected + 86400 <= query_times)
            ages = (query_times - selected) / 86400
            r['months'][str(month)] = {
                'price': {'hours': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
                          'p95': float(np.quantile(values, .95)), 'maximum': float(values.max()),
                          'spike_hours_gt300': int((values > 300).sum()), 'negative_hours': int((values < 0).sum())},
                'prediction_origins': int(rows.sum()), 'errors': errors,
                'state_values_clipped_training_zscore_divided_by_3': states,
                'trend_history': {'earliest': date(selected.min()), 'latest': date(selected.max()),
                                  'median_age_days': float(np.median(ages)),
                                  'reference_fraction_from_October': float((selected < stamp(11)).mean()),
                                  'has_any_selected_trend_fraction': float(valid.any(axis=(1, 2)).mean()),
                                  'note': 'reference counts across origin/horizon/neighbors, not unique events or weighted influence'}}
        for month, day in ((10, 5), (10, 15), (11, 15)):
            query = stamp(month, day, 12)
            ix = np.searchsorted(m['origin_unix'], query)
            assert m['origin_unix'][ix] == query
            ids = np.unique(m['trend_ids'][ix][m['trend_ids'][ix] >= 0])
            dates = m['origin_unix'][ids]
            r['history_examples'][date(query)] = {'selected_unique_records_across_24_leads': len(ids),
                                                  'earliest_selected': date(dates.min()) if len(ids) else None,
                                                  'latest_selected': date(dates.max()) if len(ids) else None}
        result['regions'][region] = r
    verify_freeze()
    result['frozen_files_unchanged'] = True
    result['script_sha256'] = sha(Path(__file__))
    (OUT / 'october_november_data.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# 10月与11月数据差异及历史来源核对', '',
             '当前版本未实现跨年同日期历史检索。历史修正库起点固定为2022-10-01；候选需与当前起报小时一致、24小时标签已成熟，至少8个候选。365天是最多回看范围，并不意味着库中实际有365天数据。', '',
             '## 月度价格标签（去重小时）', '',
             '价格表覆盖完整自然月：10月744小时，11月720小时；误差表按评分窗口计算，分别720/696个origin，不能将重叠预测单元当独立小时。', '']
    rows = []
    for region,r in result['regions'].items():
        for month, d in r['months'].items():
            p = d['price']
            rows.append([region,month]+[f"{p[k]:.2f}" for k in ('mean','median','p95','maximum')]+[p['spike_hours_gt300'],p['negative_hours']])
    lines += table(['地区','2022月份','均价','中位数','P95','最大值','>300小时数','负价小时数'],rows)
    lines += ['', '## 基础预测剩余误差', '', '误差定义为实际价格−预测：正值是低估、负值是高估。下表为中位数，避免少量极端价格主导均值。', '']
    rows=[]
    for region,r in result['regions'].items():
        a,b = r['months']['10']['errors'],r['months']['11']['errors']
        rows.append([region]+[f"{d[k]['median_actual_minus_prediction']:+.2f}" for k in ('LightGBM','C_roll') for d in (a,b)])
    lines += table(['地区','LightGBM 10月','LightGBM 11月','C_roll 10月','C_roll 11月'],rows)
    lines += ['', '## 历史来源实际例子', '', '下表为各预测步选中记录的并集，因此记录总数可能超过单个预测步K=16；不是同月同日跨年度记录。', '']
    rows=[]
    for region,r in result['regions'].items():
        for query,v in r['history_examples'].items():
            rows.append([region,query,v['selected_unique_records_across_24_leads'],v['earliest_selected'],v['latest_selected']])
    lines += table(['地区','查询时刻AEST','选中记录数（24步并集）','最早选中起报时刻','最晚选中起报时刻'],rows)
    lines += ['', '## 哪些数据用于训练', '',
              '- LightGBM：2015–2021拟合，2022H1原早停/选择。',
              '- C_roll：2022年7–9月拟合。',
              '- 历史双场：2022年7–8月训练，9月重建诊断；输入为已知历史24小时价格、该记录之前72小时价格及当时状态。',
              '- 最新采用模块：2022年10月720个成熟窗口训练80步；输入为当时状态、72小时摘要、历史趋势/事件提案及检索信息；标签为该窗口未来24小时实际价格，目标为加权分位损失和正则。',
              '- 检索库：从10月1日开始，随时间逐步加入成熟记录；11月可读取已成熟的10月和较早11月记录，门控参数不重新训练。', '',
              '**10月结果是整月门控训练后的训练内重放。即便每条输入的历史记录合法，它也不是10月每天在当时已训练好模型所得的前推预测。11月使用的是10月拟合后的固定门控。**', '',
              '## 对反转的解释边界', '',
              '10月普通价格与基础模型低估均较明显；11月普通价格下降，C_roll中位残差由正转负。继续迁移10月的向上修正就可能在11月造成高估。TAS1的11月中位价大幅下降，同时最大价格升高，不能用“尖峰更少”概括。', '',
              '外生状态也发生变化，具体标准化值保存于JSON（训练期z-score裁剪后除3），不等于物理MW或价格值。例如QLD可用发电容量均值明显下降、TAS备用容量均值下降、NSW进口空间均值上升。这里只描述数据变化，不能据此断言价格变化的市场因果原因。', '',
              '当前实现没有检验用户提出的往年同日期参照假说。原始历史价格本身可以用于历史输入；若要构造往年“样本外预测误差”，必须使用当时只由更早数据拟合/选择的预测模型，不能直接用覆盖那些年份训练的模型的训练内残差。', '']
    (OUT / 'october_november_data.md').write_text('\n'.join(lines))
    print(OUT / 'october_november_data.md')


if __name__ == '__main__':
    main()
