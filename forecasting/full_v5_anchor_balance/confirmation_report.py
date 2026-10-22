"""Fixed confirmation gates and seed dispersion, with no evaluation-time tuning."""
import csv
import json
from pathlib import Path

import numpy as np

from . import experiment as base
from .selection import RTOL


def noninferior(value, reference):
    return value <= reference + RTOL * abs(reference)


def _metric(results, region, fold, seed, variant, mode, group, component, metric):
    result = results[f'{region}/{fold}/{seed}/{variant}']
    if component == 'baseline':
        return result['B'][group][metric]
    return result[mode]['metrics'][group][component][metric]


def acceptance(results, cfg, variant, mode, seeds):
    """The original four gates on means of the specified neural seeds."""
    folds = {}
    for fold in cfg['folds']:
        def get(region, arm, group, component, metric):
            return float(np.mean([_metric(results, region, fold, seed, arm, mode,
                                           group, component, metric) for seed in seeds]))
        checks = dict(macro_noninferior_to_B_and_balanced=True, macro_strictly_improves_B=False,
                      each_region_noninferior_to_B=True, each_spring_noninferior_to_B=True)
        for metric in ('mae', 'qs'):
            values = [get(r, variant, 'all', 'full', metric) for r in cfg['regions']]
            baseline = [get(r, variant, 'all', 'baseline', metric) for r in cfg['regions']]
            balanced = [get(r, 'balanced', 'all', 'full', metric) for r in cfg['regions']]
            mean, b, ref = map(lambda x: float(np.mean(x)), (values, baseline, balanced))
            checks['macro_noninferior_to_B_and_balanced'] &= noninferior(mean, b) and noninferior(mean, ref)
            checks['macro_strictly_improves_B'] |= mean < b - RTOL * abs(b)
            checks['each_region_noninferior_to_B'] &= all(noninferior(v, b) for v, b in zip(values, baseline))
            checks['each_spring_noninferior_to_B'] &= all(noninferior(
                get(r, variant, 'season/spring', 'full', metric),
                get(r, variant, 'season/spring', 'baseline', metric)) for r in cfg['regions'])
        folds[fold] = dict(checks=checks, passed=all(checks.values()))
    return dict(folds=folds, passed=all(f['passed'] for f in folds.values()))


def summarize(results, cfg):
    summary, gates = {}, {}
    for variant in cfg['variants']:
        summary[variant], gates[variant] = {}, {}
        for mode in ('raw', 'guarded'):
            averaged = acceptance(results, cfg, variant, mode, cfg['seeds'])
            per_seed = {str(seed): acceptance(results, cfg, variant, mode, [seed]) for seed in cfg['seeds']}
            gates[variant][mode] = dict(seed_mean=averaged, per_seed=per_seed,
                robust_pass=averaged['passed'] and all(row['passed'] for row in per_seed.values()))
            summary[variant][mode] = {}
            for fold in cfg['folds']:
                row = {}
                for component in ('baseline', 'anchor', 'full'):
                    row[component] = {}
                    for metric in ('mae', 'qs', 'bias', 'coverage_80', 'coverage_90'):
                        values = [float(np.mean([_metric(results, r, fold, seed, variant, mode, 'all', component, metric)
                                                 for r in cfg['regions']])) for seed in cfg['seeds']]
                        row[component][metric] = dict(mean=float(np.mean(values)),
                            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0., per_seed=values)
                selected = [results[f'{r}/{fold}/{s}/{variant}'][mode]['selection']
                            for r in cfg['regions'] for s in cfg['seeds']]
                row['fallback_count'] = sum(int(s['fallback']) for s in selected)
                row['boundary_checkpoint_count'] = sum(s['epoch'] >= cfg['epochs'] - 1 for s in selected)
                row['alpha_counts'] = {str(a): sum(s['alpha'] == a for s in selected) for a in cfg['alphas']}
                summary[variant][mode][fold] = row
    return summary, gates


def verify_training_orders(output, cfg):
    """Confirm sample order and full budget matched across arms for each seed."""
    checks = []
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                reference = None
                for variant in cfg['variants']:
                    path = output / region / fold / f'seed_{seed}' / variant / 'training.json'
                    log = json.loads(path.read_text())
                    orders = [row['order_sha256'] for row in log]
                    if len(orders) != cfg['epochs'] or (reference is not None and orders != reference):
                        raise ValueError('Unmatched confirmation training budget/order')
                    reference = orders
                checks.append(f'{region}/{fold}/{seed}')
    return dict(matched=True, comparisons=checks, epochs=cfg['epochs'])


def write_report(output, results, cfg):
    output = Path(output)
    audit = verify_training_orders(output, cfg)
    summary, gates = summarize(results, cfg)
    base.write_json(output / 'summary.json', summary)
    base.write_json(output / 'acceptance.json', gates)
    base.write_json(output / 'training_order_audit.json', audit)
    columns = ['region', 'fold', 'seed', 'variant', 'mode', 'component', 'group',
               'epoch', 'alpha', 'fallback', 'n_origins', 'mae', 'qs', 'bias', 'coverage_80', 'coverage_90']
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, result in sorted(results.items()):
            region, fold, seed, variant = key.split('/')
            for mode in ('raw', 'guarded'):
                selected = result[mode]['selection']
                for group, metrics in result[mode]['metrics'].items():
                    for component in ('baseline', 'anchor', 'full'):
                        writer.writerow(dict(region=region, fold=fold, seed=seed, variant=variant, mode=mode,
                            component=component, group=group, n_origins=metrics['n_origins'],
                            **{k: selected[k] for k in ('epoch', 'alpha', 'fallback')},
                            **{k: metrics[component][k] for k in ('mae', 'qs', 'bias', 'coverage_80', 'coverage_90')}))
    lines = ['# Anchor 三种方案确认实验与非对称尺度头', '',
        '范围：回顾性开发实验；2021/2022 已用于前期诊断，不能称为未见测试集。',
        '同一七折树 OOF、全小时起点、24 小时目标、三地区、三个神经随机种子、每组固定 20 轮。',
        '以下为地区等权宏平均，再对三个种子求均值和样本标准差；不是三模型预测集成。',
        'QS = 2 × 五分位点平均 pinball，MAE 使用 q50，均在原始价格单位计算。', '',
        '| 方案 | 输出 | 年份 | Full MAE（均值 ± SD） | Full QS（均值 ± SD） | 回退数/9 | 最后两轮选中数/9 |',
        '|---|---|---|---:|---:|---:|---:|']
    for variant, modes in summary.items():
        for mode, folds in modes.items():
            for fold, row in folds.items():
                mae, qs = row['full']['mae'], row['full']['qs']
                lines.append(f'| {variant} | {mode} | {fold} | {mae["mean"]:.4f} ± {mae["std"]:.4f} | '
                    f'{qs["mean"]:.4f} ± {qs["std"]:.4f} | {row["fallback_count"]} | {row["boundary_checkpoint_count"]} |')
    lines += ['', '| 方案 | 输出 | 种子均值通过全部门槛 | 三个种子分别通过全部门槛 | 稳定通过 |',
              '|---|---|---|---|---|']
    for variant, modes in gates.items():
        for mode, gate in modes.items():
            lines.append(f'| {variant} | {mode} | {gate["seed_mean"]["passed"]} | '
                         f'{all(s["passed"] for s in gate["per_seed"].values())} | {gate["robust_pass"]} |')
    lines += ['', '验收保留原四条件：宏平均 MAE/QS 不劣于 B 和同种子 balanced；至少一个宏指标严格优于 B；'
        '每地区不劣于 B；每地区春季不劣于 B。2021/2022 均需通过。数值容差沿用 RTOL=1e-7。',
        '稳定通过还要求三个种子各自通过；均值通过而部分种子失败仅记为不稳定候选。',
        'raw 与 guarded 分别验收；guarded 的成功不能自动解释为原始网络结构成功。',
        '最后两轮仍被选中的次数用于提示预算边界，不能仅因完成 20 轮便断言已经收敛。',
        '三个种子只改变神经初始化与训练顺序，树种子固定；未计算市场总体置信区间或显著性。',
        '所有分组、尾部 pinball、覆盖率、区间宽度、Anchor 位移与历史校正抵消量均为事后诊断，不能用于重选 checkpoint/alpha。',
        'NPZ 保留原始/守护预测、起点、目标时间和实际值；results.csv 包含地区/年份/种子/季节/价格分组的完整结果。',
        '数据可用性审计沿用归档时间戳代理，不能据此宣称已核实真实发布延迟。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
