"""Report mechanism evidence separately from predictive acceptance."""
import csv
import json
from pathlib import Path
import numpy as np
from ..full_v5_anchor_balance import confirmation_report as old
from ..full_v5_anchor_balance import experiment as base
from .runner import arm_path, OLD


def write_report(output, results, cfg):
    output = Path(output)
    order_audit = old.verify_training_orders(output, cfg)
    base.write_json(output/'training_order_audit.json', order_audit)
    rows = []
    for key, result in results.items():
        region, fold, seed, arm = key.split('/')
        for mode in ('raw', 'guarded'):
            for group, metrics in result[mode]['mechanism'].items():
                rows.append(dict(region=region, fold=fold, seed=seed, arm=arm, mode=mode, group=group, **metrics))
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'mechanism.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if cfg['scope'] == 'integration_smoke_not_evidence':
        base.write_json(output/'acceptance.json', dict(eligible=False, reason='short integration smoke; no scientific acceptance'))
        (output/'report.md').write_text('# 机理状态约束：真实数据短流程验收\n\n'
            'NSW1 / 2021 开发折，三个神经 seed、三个臂，每臂2轮。\n'
            '完成梯度、选择冻结、恢复与预测链路检查；不能作为有效性证据，不替换正式结果。\n')
        return
    combined = dict(results)
    root = Path('outputs/forecasting/v5_anchor_confirmation_20261022_r1').resolve()
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                combined[f'{region}/{fold}/{seed}/balanced'] = json.loads((arm_path(root, region, fold, seed, 'balanced')/'metrics.json').read_text())
                combined[f'{region}/{fold}/{seed}/asymmetric_reference'] = json.loads((arm_path(OLD, region, fold, seed, 'reference')/'metrics.json').read_text())
    summary, gates = old.summarize(combined, cfg)
    for arm in cfg['variants']:
        for mode in ('raw', 'guarded'):
            gate = gates[arm][mode]
            for selected, seeds in [(gate['seed_mean'], cfg['seeds']), *[(gate['per_seed'][str(s)], [s]) for s in cfg['seeds']]]:
                for fold, row in selected['folds'].items():
                    for reference in ('bounded_reference', 'asymmetric_reference'):
                        checks = []
                        for metric in ('mae', 'qs'):
                            def macro(v):
                                return np.mean([combined[f'{r}/{fold}/{s}/{v}'][mode]['metrics']['all']['full'][metric]
                                                for r in cfg['regions'] for s in seeds])
                            checks.append(old.noninferior(macro(arm), macro(reference)))
                        row['checks'][f'macro_noninferior_to_{reference}'] = bool(all(checks))
                    row['passed'] = all(row['checks'].values())
                selected['passed'] = all(r['passed'] for r in selected['folds'].values())
            gate['robust_pass'] = gate['seed_mean']['passed'] and all(r['passed'] for r in gate['per_seed'].values())
    base.write_json(output/'summary.json', summary)
    base.write_json(output/'acceptance.json', gates)
    lines = ['# 机理状态约束开发确认', '',
        '供需状态内的跨时间残差证据只用于约束三个校正幅度，不强制残差方向。',
        '机制证据见 mechanism.csv：伤害、获益、方向、区间与状态样本量必须一起解释。',
        '预测验收见 acceptance.json；通过预测门槛不自动证明机制假设或论文创新。', '',
        '| 臂 | 年份 | guarded MAE | guarded QS |', '|---|---|---:|---:|']
    for arm in cfg['variants']:
        for fold, row in summary[arm]['guarded'].items():
            lines.append(f'| {arm} | {fold} | {row["full"]["mae"]["mean"]:.4f} | {row["full"]["qs"]["mean"]:.4f} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
