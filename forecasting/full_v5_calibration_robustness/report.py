"""Retain all old gates, add the current-reference gate, and report failures."""
import csv
import json
from pathlib import Path

import numpy as np

from ..full_v5_anchor_balance import confirmation_report as old
from ..full_v5_anchor_balance import experiment as base
from .experiment import ARMS, TRAJECTORIES, arm_path


def acceptance(results, cfg, arm, mode, seeds):
    result = old.acceptance(results, cfg, arm, mode, seeds)
    for fold, row in result['folds'].items():
        good = True
        for metric in ('mae','qs'):
            def macro(variant):
                return float(np.mean([results[f'{r}/{fold}/{s}/{variant}'][mode]['metrics']['all']['full'][metric]
                                      for r in cfg['regions'] for s in seeds]))
            good &= old.noninferior(macro(arm), macro('reference'))
        row['checks']['macro_noninferior_to_current_reference'] = good
        row['passed'] = all(row['checks'].values())
    result['passed'] = all(row['passed'] for row in result['folds'].values())
    return result


def summarize(results, cfg):
    combined = dict(results)
    for r in cfg['regions']:
        for f in cfg['folds']:
            for s in cfg['seeds']:
                path = arm_path(cfg['old_reference'],r,f,s,'balanced') / 'metrics.json'
                combined[f'{r}/{f}/{s}/balanced'] = json.loads(path.read_text())
    summary, _ = old.summarize(combined, cfg)
    gates = {}
    for arm in ARMS:
        gates[arm] = {}
        for mode in ('raw','guarded'):
            mean = acceptance(combined,cfg,arm,mode,cfg['seeds'])
            individual = {str(s):acceptance(combined,cfg,arm,mode,[s]) for s in cfg['seeds']}
            gates[arm][mode] = dict(seed_mean=mean,per_seed=individual,
                robust_pass=mean['passed'] and all(v['passed'] for v in individual.values()))
    return summary, gates


def audit_orders(output,cfg):
    checks = []
    for r in cfg['regions']:
        for f in cfg['folds']:
            for s in cfg['seeds']:
                logs = [json.loads((arm_path(output,r,f,s,v)/'training.json').read_text()) for v in TRAJECTORIES]
                orders = [[row['order_sha256'] for row in log] for log in logs]
                if any(len(log)!=cfg['epochs'] for log in logs) or orders[0]!=orders[1]:
                    raise ValueError('Training budget/orders differ between physical trajectories')
                replay = json.loads((arm_path(output,r,f,s,'reference')/'reference_replay.json').read_text())
                if not replay['exact']:
                    raise ValueError('Historical reference replay failed')
                checks.append(f'{r}/{f}/{s}')
    return dict(matched=True, reference_replays_exact=True, combinations=checks,
                epochs_per_trajectory=cfg['epochs'], selection_only_uses_same_trajectory=True)


def write_report(output,results,cfg):
    output = Path(output)
    audit = audit_orders(output,cfg)
    summary,gates = summarize(results,cfg)
    base.write_json(output/'summary.json',summary)
    base.write_json(output/'acceptance.json',gates)
    base.write_json(output/'training_order_audit.json',audit)
    columns = ['region','fold','seed','variant','mode','component','group','epoch','alpha','fallback',
               'n_origins','mae','qs','bias','coverage_80','coverage_90']
    with (output/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for key,result in sorted(results.items()):
            region,fold,seed,arm=key.split('/')
            for mode in ('raw','guarded'):
                selected=result[mode]['selection']
                for group,metrics in result[mode]['metrics'].items():
                    for component in ('baseline','anchor','full'):
                        writer.writerow(dict(region=region,fold=fold,seed=seed,variant=arm,mode=mode,
                            component=component,group=group,n_origins=metrics['n_origins'],
                            **{k:selected[k] for k in ('epoch','alpha','fallback')},
                            **{k:metrics[component][k] for k in ('mae','qs','bias','coverage_80','coverage_90')}))
    lines=['# 两项校正稳健性开发实验', '',
        '三个逻辑方案：原 asymmetric_scale 对照、仅改四季度选择、仅加校正幅度边界。',
        '36 次物理拟合、54 个逻辑方案；每条轨迹 20 epochs，三个地区、两个开发年、三个神经 seed。',
        'multiwindow 与 reference 使用完全相同的训练轨迹，raw 诊断重复同一输出，不是独立训练证据。',
        '旧 72 次确认失败原样保留；本轮 2021/2022 为回顾性开发，未更新正式主表。', '',
        '| 方案 | 输出 | 评估年 | 宏 MAE ± SD | 宏 QS ± SD | 回退/9 | 末两轮选中/9 |',
        '|---|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for mode in ('guarded','raw'):
            for fold,row in summary[arm][mode].items():
                mae,qs=row['full']['mae'],row['full']['qs']
                lines.append(f'| {arm} | {mode} | {fold} | {mae["mean"]:.4f} ± {mae["std"]:.4f} | '
                    f'{qs["mean"]:.4f} ± {qs["std"]:.4f} | {row["fallback_count"]} | {row["boundary_checkpoint_count"]} |')
    lines += ['', '| 方案 | 输出 | 种子均值全部通过 | 每 seed 全部通过 | 稳定通过 |',
              '|---|---|---|---|---|']
    for arm in ARMS:
        for mode,gate in gates[arm].items():
            lines.append(f'| {arm} | {mode} | {gate["seed_mean"]["passed"]} | '
                f'{all(v["passed"] for v in gate["per_seed"].values())} | {gate["robust_pass"]} |')
    lines += ['', '旧四门槛全部保留：宏 MAE/QS 不劣于 B/旧 balanced、至少一项严格优于 B、'
        '每地区不劣于 B、每地区春季不劣于 B；新增宏 MAE/QS 不劣于当前 reference。',
        '两年与三个 seed 均需通过。回退、单年通过、个别 seed 通过不视为稳定成功。',
        '参考轨迹的每轮损失/训练顺序/原选模和 raw/guarded/最终参数均已与旧档精确核对。',
        '均值为每 seed 三地区等权后再汇总，未做预测集成或显著性检验。树 seed 固定为 2026。',
        '所有参数、窗口和边界均在本轮训练前冻结；不根据本轮评估修改这些设置。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
