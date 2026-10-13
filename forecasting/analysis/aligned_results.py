"""Reproduce tables and exploratory diagnostics for a complete aligned run.

This reads frozen predictions only. It never fits, chooses, or ensembles models.
Run: python -m forecasting.analysis.aligned_results --run-root ... --output ...
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.aligned_report import build_aligned_report, REGIONS, SEEDS
from forecasting.aligned_scoring import load_prediction_artifact, QUANTILE_LEVELS
from forecasting.exploration_evaluate import Forecast, compare_sources, holm_adjust

BANDS = ('negative', '0_to_100', '100_to_300', 'spike')
METRICS = ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_90', 'interval_width_90')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_dump(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def macro_seed_stats(regions, metric):
    values = [np.mean([regions[r]['scores']['per_seed'][str(seed)][metric]
                       for r in REGIONS]) for seed in SEEDS]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def masks(actual):
    return dict(zip(BANDS, (actual < 0, (actual >= 0) & (actual <= 100),
                            (actual > 100) & (actual <= 300), actual > 300)))


def slice_scores(actual, quantiles, mask, target_times):
    n = int(mask.sum())
    if not n:
        raise ValueError('Empty diagnostic slice')
    error = quantiles[..., 2] - actual
    delta = actual[..., None] - quantiles
    pinball = np.maximum(QUANTILE_LEVELS * delta, (QUANTILE_LEVELS - 1) * delta)
    return {'n_cells': n, 'n_unique_delivery_hours': int(np.unique(target_times[mask]).size),
            'share': n / actual.size, 'mae': float(np.abs(error[mask]).mean()),
            'bias': float(error[mask].mean()),
            'quantile_score_proxy': float(2 * pinball[mask].mean()),
            'q05_pinball': float(pinball[..., 0][mask].mean()),
            'q95_pinball': float(pinball[..., 4][mask].mean())}


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]])


def analyse(root, out, replicates=9999):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('Verifying all 90 artifacts, checkpoints, scores and frozen sources', flush=True)
    report = build_aligned_report(root)
    if not report['comparison_available']:
        raise ValueError('The complete matrix must pass validation before analysis')
    saved = json.loads((root / 'report.json').read_text())
    if report != saved:
        raise ValueError('Recomputed report differs from the completed run report')
    with gzip.open(out / 'verified_report.json.gz', 'wt') as stream:
        json.dump(report, stream, allow_nan=False)
    tables = report['tables']['Q']
    order = sorted(tables, key=lambda model: tables[model]['macro_regions']['mae'])
    summary, regional, cells, training, bands, temporal = [], [], [], [], [], []
    sources = {}
    manifest = json.loads((root / 'jobs.json').read_text())
    jobs = {(j['model'], j['region'], j['seed']): j for j in manifest['jobs']}
    for model in order:
        regions = tables[model]['regions']
        row = {'model': model, 'mae_rank': len(summary) + 1}
        for metric in METRICS:
            row[metric], row[metric + '_seed_sd'] = macro_seed_stats(regions, metric)
        row['raw_output_crossing_rate'] = tables[model]['macro_regions']['crossing_rate']
        row['fit_hours_sum'] = sum(j['seconds'] for j in report['jobs'] if j['model'] == model) / 3600
        row['minimum_distinct_seed_predictions'] = min(regions[r]['seed_evidence']['distinct_scored_prediction_count'] for r in REGIONS)
        summary.append(row)
        sources[model] = {}
        for region in REGIONS:
            scores = regions[region]['scores']
            regional.append({'model': model, 'region': region,
                             **{k: scores['seed_summary'][k]['mean'] for k in METRICS},
                             'mae_seed_sd': scores['seed_summary']['mae']['std']})
            sources[model][region] = {}
            for seed in SEEDS:
                directory = root / jobs[(model, region, seed)]['output']
                a = load_prediction_artifact(directory / 'test.npz')
                q = np.sort(a.quantiles, axis=-1)
                sources[model][region][seed] = Forecast(q[..., 2], q, a.actual, a.origin_unix,
                                                       scores['per_seed'][str(seed)]['crossing_rate'])
                ident = dict(model=model, region=region, seed=seed)
                cells.append({**ident, **{k: scores['per_seed'][str(seed)][k] for k in METRICS}})
                block_rows = []
                for band, mask in masks(a.actual).items():
                    item = {**ident, 'band': band, **slice_scores(a.actual, q, mask, a.target_time_unix)}
                    bands.append(item)
                    block_rows.append(item)
                # Exact decomposition of overall error, including all overlapping horizons.
                for metric in ('mae', 'quantile_score_proxy'):
                    np.testing.assert_allclose(sum(b['share'] * b[metric] for b in block_rows),
                                               scores['per_seed'][str(seed)][metric], rtol=1e-12)
                if model in ('full', 'lightgbm', 'xgboost', 'no_fields', 'patchtst'):
                    months = np.asarray(pd.to_datetime(a.target_time_unix.reshape(-1), unit='s', utc=True)
                                        .tz_convert('Australia/Brisbane').strftime('%Y-%m')).reshape(a.actual.shape)
                    for month in np.unique(months):
                        temporal.append({**ident, 'family': 'delivery_month_aest', 'slice': month,
                                         **slice_scores(a.actual, q, months == month, a.target_time_unix)})
                    for start, end in ((0, 6), (6, 12), (12, 18), (18, 24)):
                        mask = np.zeros(a.actual.shape, dtype=bool); mask[:, start:end] = True
                        temporal.append({**ident, 'family': 'horizon_hours', 'slice': f'{start+1}-{end}',
                                         **slice_scores(a.actual, q, mask, a.target_time_unix)})
                path = directory / 'progress.jsonl'
                if path.exists():
                    events = [json.loads(line) for line in path.read_text().splitlines()]
                    finish = next(e for e in reversed(events) if e['event'] == 'complete')
                    training.append({**ident, 'best_epoch_one_based': finish['best_epoch'] + 1,
                                     'epochs_executed': finish['epochs_executed'],
                                     'stop_reason': finish['stop_reason'],
                                     'best_selection_pinball': finish['best_selection_value']})
        print('Scored', model, flush=True)
    for name, rows in [('main', summary), ('regional', regional), ('per_seed', cells),
                       ('price_bands_per_seed', bands), ('temporal_per_seed', temporal), ('training', training)]:
        csv_dump(out / f'{name}.csv', rows)
    band_macro = []
    for model in order:
        for band in BANDS:
            selected = [r for r in bands if r['model'] == model and r['band'] == band]
            band_macro.append({'model': model, 'band': band,
                               **{k: float(np.mean([r[k] for r in selected]))
                                  for k in ('share', 'mae', 'bias', 'quantile_score_proxy', 'q05_pinball', 'q95_pinball')}})
    csv_dump(out / 'price_bands_macro.csv', band_macro)
    gaps = []
    for reference in ('lightgbm', 'xgboost', 'no_fields'):
        for metric in ('mae', 'quantile_score_proxy'):
            for band in BANDS:
                contributions = []
                for region in REGIONS:
                    for seed in SEEDS:
                        f = next(r for r in bands if (r['model'], r['region'], r['seed'], r['band']) == ('full', region, seed, band))
                        b = next(r for r in bands if (r['model'], r['region'], r['seed'], r['band']) == (reference, region, seed, band))
                        contributions.append(f['share'] * (f[metric] - b[metric]))
                gaps.append({'reference': reference, 'metric': metric, 'band': band,
                             'full_minus_reference_contribution': float(np.mean(contributions))})
            np.testing.assert_allclose(sum(g['full_minus_reference_contribution'] for g in gaps
                                           if g['reference'] == reference and g['metric'] == metric),
                                       tables['full']['macro_regions'][metric] - tables[reference]['macro_regions'][metric], rtol=1e-10)
    csv_dump(out / 'gap_contributions.csv', gaps)
    # Declare the complete comparison family before calculating p values.
    specs = [(model, 'full', ('mae', 'crps_quantile_approx')) for model in order if model != 'full']
    specs += [(model, 'full', ('pinball_q05_negative_hours', 'pinball_q95_spike_hours'))
              for model in ('lightgbm', 'xgboost', 'no_fields')]
    specs += [('xgboost', 'lightgbm', ('mae', 'crps_quantile_approx'))]
    comparisons = {}
    for hours in (168, 24):
        family = {}
        for reference, candidate, metrics in specs:
            key = candidate + '_minus_' + reference + ('_tail' if metrics[0].startswith('pinball') else '')
            family[key] = compare_sources(sources[reference], sources[candidate], replicates=replicates,
                                          block_hours=hours, metrics=metrics)
            print('Bootstrap', hours, key, flush=True)
        holm_adjust(family)
        comparisons[str(hours)] = family
    dump(out / 'comparisons.json', comparisons)
    inference_rows = []
    for hours, family in comparisons.items():
        for contrast, result in family.items():
            for metric, regions in result['metrics'].items():
                for region, row in regions.items():
                    inference_rows.append({'block_hours': hours, 'contrast': contrast, 'metric': metric,
                                           'region': region, **{k: row[k] for k in ('difference', 'ci_low', 'ci_high', 'p_value', 'p_value_holm', 'reject_holm')}})
    csv_dump(out / 'comparisons.csv', inference_rows)
    make_figure(out, summary, gaps)
    main_rows = [[r['model'], f"{r['mae']:.3f} ± {r['mae_seed_sd']:.3f}",
                  f"{r['rmse_pooled']:.3f}", f"{r['quantile_score_proxy']:.3f} ± {r['quantile_score_proxy_seed_sd']:.3f}",
                  f"{r['coverage_90']:.2%}", f"{r['interval_width_90']:.2f}"] for r in summary]
    text = ['# 2026-10-12 静态共同输入基线结果', '',
            '本报告只使用 aligned_baselines_20261012 的 90 格结果；全部完成并重新通过源码、配置、checkpoint、预测哈希和重评分校验。旧实验不参与本表。', '',
            '## 口径', '',
            'NSW1/QLD1/TAS1；seed 2026/2027/2028。每小时以 72h 历史预测未来 24h。训练 2015–2021；选择 2022H1；CP 校准 2022H2；回顾性评分 2023–2024。每地区每 seed 17,521 origins、420,504 个 origin×horizon 单元；这些单元因重叠窗口并不独立。', '',
            '点预测统一为排序后的 q50，原始 AUD/MWh。QS = 五个固定分位数的平均 pinball × 2，不是精确 CRPS。先独立评分各 seed，再等权平均地区；± 为先按同一 seed 平均地区后计算的三个 seed 样本标准差，不是时间置信区间。RMSE 列是各地区/seed 的 pooled RMSE 均值，不是将三地区平方误差合并后的 RMSE。', '',
            '固定配置比较；没有完成公平 HPO。Lasso-ARX72+CP 是 72h 适配版，不是原版 LEAR。其三个 seed 的预测完全相同，不提供三份独立随机证据。外生发布时间仍使用归档元数据代理。', '',
            '## 主表（按宏平均 MAE 排序）', '',
            markdown_table(['模型', 'MAE ± seed SD', '地区平均 pooled RMSE', 'QS ± seed SD', '90%覆盖率', '90%区间宽度'], main_rows), '',
            '## 地区 MAE（每地区三个 seed）', '',
            markdown_table(['模型', *REGIONS], [[model, *[f"{tables[model]['regions'][r]['scores']['seed_summary']['mae']['mean']:.3f} ± {tables[model]['regions'][r]['scores']['seed_summary']['mae']['std']:.3f}" for r in REGIONS]] for model in order]), '',
            '## Full 的误差来源', '',
            '以下是按实际价格事后切片的误差分解，不是输入或模块的因果贡献；正值表示 Full 更差。各价格区间贡献之和精确等于总体 MAE 差。', '',
            markdown_table(['参照', *BANDS, '合计'], [[ref, *[f"{next(g['full_minus_reference_contribution'] for g in gaps if g['reference']==ref and g['metric']=='mae' and g['band']==b):+.3f}" for b in BANDS], f"{tables['full']['macro_regions']['mae']-tables[ref]['macro_regions']['mae']:+.3f}"] for ref in ('lightgbm', 'xgboost', 'no_fields')]), '',
            '## 尖峰切片（实际价格 >300）', '',
            '各地区先平均 seed 的切片损失，再等权平均地区。q95 在真实尖峰上的损失仅描述该切片；不是全分布校准或尖峰发生概率检验，不能忽略非尖峰误报。独立交付小时数另存 per-seed 切片 CSV。', '',
            markdown_table(['模型', '尖峰 q50 MAE', '尖峰 q95 pinball'], [[r['model'], f"{r['mae']:.3f}", f"{r['q95_pinball']:.3f}"] for r in band_macro if r['band']=='spike']), '',
            '## 配对统计（探索性）', '',
            f'配对非循环 moving-block bootstrap，{replicates:,} 次，主块长 168h、24h 敏感性。沿完整 origin 时间轴抽样，三个地区使用同样块起点，先平均 seed 损失而非预测；尾部按重采样损失总和/事件数计算，保留无事件 origin。', '',
            '比较族：Full 与其他九模型的 MAE/QS；Full 与 LightGBM/XGBoost/no_fields 的负价 q05、尖峰 q95；LightGBM 与 XGBoost 的 MAE/QS。每个块长内对全部 104 个地区/宏平均检验统一 Holm 修正。95% CI 是未作多重比较校正的百分位区间，显著性仅看 Holm p；24h 是敏感性分析，不从两个块长择优。', '',
            '这些比较及切片在已查看部分结果后制定，属于探索性统计。区间仅反映给定已训练模型的时间样本不确定性；三个 seed 不充分刻画训练随机性。不包含 DM/GW 检验，不宣称完整 SOTA 统计协议已执行。', '',
            markdown_table(['比较', '指标', '差值', '168h 95% CI', 'Holm p', '24h Holm p'],
                           [[r['contrast'], r['metric'], f"{r['difference']:+.3f}", f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", f"{r['p_value_holm']:.4f}", f"{comparisons['24'][r['contrast']]['metrics'][r['metric']]['macro_regions']['p_value_holm']:.4f}"]
                            for r in inference_rows if r['block_hours']=='168' and r['region']=='macro_regions']), '',
            '## 训练与诊断附件', '',
            'NSW1 Full 的三个 seed 都在第一轮达到所选最佳 checkpoint；这提示优化/目标/验证期适应问题，但不证明过拟合的唯一原因。no_fields 关闭双场对预测的贡献，仍执行双场拟合及重构损失；它不是删掉全部双场计算的等成本对照。', '',
            'crossing_rate 在共同评分排序前计算，但 Full/no_fields 的 forward 已内部排序，零值不证明未排序专家从不交叉。各模型完整数据在 CSV 中保留，不把该列当独立架构排名。', '',
            markdown_table(['模型', '累计 worker 小时', '最少不同 seed 预测数'], [[r['model'], f"{r['fit_hours_sum']:.2f}", r['minimum_distinct_seed_predictions']] for r in summary]), '',
            'worker 时间含载入、训练、预测和保存；GPU/CPU 实现不同，不能用该列直接证明算法效率优劣。', '',
            '![主指标和 Full 误差贡献](overview.png)', '',
            '文件：main.csv / regional.csv / per_seed.csv / price_bands_per_seed.csv / price_bands_macro.csv / temporal_per_seed.csv / gap_contributions.csv / training.csv / comparisons.csv / comparisons.json / verified_report.json.gz / provenance.json。', '',
            '复现命令：', '', '```bash',
            '.venv/bin/python -m forecasting.analysis.aligned_results \\',
            '  --run-root outputs/forecasting/aligned_baselines_20261012 \\',
            '  --output logs/forecasting/aligned_baselines_20261012/results', '```', '']
    (out / 'report.md').write_text('\n'.join(text))
    provenance = {'run_id': report['run_id'], 'training_source_hash': manifest['source_hash'],
                  'config_hash': manifest['config_hash'], 'bootstrap_replicates': replicates,
                  'analysis_source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'exploration_evaluate.py',
                     Path(__file__).resolve().parents[1] / 'aligned_report.py', Path(__file__).resolve().parents[1] / 'aligned_scoring.py')},
                  'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'provenance.json'}}
    dump(out / 'provenance.json', provenance)
    print('Completed report:', out / 'report.md', flush=True)


def make_figure(out, summary, gaps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    names = [r['model'] for r in summary]
    colors = ['#d26735' if n == 'full' else '#367c9b' for n in names]
    for ax, metric, title in zip(axes[:2], ('mae', 'quantile_score_proxy'), ('Macro MAE (q50)', 'Macro quantile score proxy')):
        ax.barh(names, [r[metric] for r in summary], xerr=[r[metric+'_seed_sd'] for r in summary], color=colors, capsize=2)
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel('AUD/MWh; lower is better')
    ax = axes[2]
    xx = np.arange(4)
    for offset, ref in enumerate(('lightgbm', 'xgboost', 'no_fields')):
        values = [next(r['full_minus_reference_contribution'] for r in gaps if r['reference']==ref and r['metric']=='mae' and r['band']==b) for b in BANDS]
        ax.bar(xx+(offset-1)*.25, values, width=.25, label='vs '+ref)
    ax.set_xticks(xx, ('<0', '0–100', '100–300', '>300')); ax.axhline(0, color='black', linewidth=.8)
    ax.set_title('Full MAE gap contributions'); ax.set_xlabel('Actual price band (AUD/MWh)')
    ax.set_ylabel('Positive: Full has larger error'); ax.legend(fontsize=8)
    fig.savefig(out / 'overview.png', dpi=180); fig.savefig(out / 'overview.pdf'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=9999)
    args = parser.parse_args()
    analyse(args.run_root, args.output, args.replicates)
