"""Publish the audited, development-selected repair regardless of test ranking.

Run only after the repair's nine cells and independent replay audit complete.
Historical predictions and original foundation archives are never overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from forecasting.aligned_report import build_aligned_report
from forecasting.foundation_audit import collect
from scripts.diagnostics.baseline_maintenance import ROOT, RUN as ORIGINAL, REGIONS, digest, save
from scripts import update_latest_main_comparison as main_table


def publish(run):
    run=run.resolve()
    manifest=json.loads((run/'jobs.json').read_text())
    selected=manifest['foundation']['external_development_selection']
    entry={'run_root':str(run.relative_to(ROOT)), 'manifest_sha256':digest(run/'jobs.json'),
           'selection':{'path':selected['path'],'sha256':selected['sha256']}}
    main_table.validate_maintenance_source('timesfm25_xreg',entry)
    audit=collect(run)
    report=build_aligned_report(run)
    if not report['comparison_available'] or report['progress']['completed_cells']!=9:
        raise ValueError('All nine repair cells and independent audit must pass')
    comparisons={}
    for region in REGIONS:
        scores={}
        for name,source in [('original_ridge0',ORIGINAL),('selected_ridge1',run)]:
            cells=[json.loads((source/f'runs/timesfm25_xreg/seed{seed}/{region}/metrics.json').read_text())
                   for seed in (2026,2027,2028)]
            scores[name]={metric:float(np.mean([c[metric] for c in cells]))
                          for metric in ('mae','rmse_pooled')}
        comparisons[region]=scores
    macro={name:{metric:float(np.mean([comparisons[r][name][metric] for r in REGIONS]))
                 for metric in ('mae','rmse_pooled')}
           for name in ('original_ridge0','selected_ridge1')}
    archive=ROOT/'logs/forecasting/timesfm25_xreg_ridge1_20261024_r1'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('jobs.json','config.yaml','maintenance_selection.json','progress.json'):
        shutil.copyfile(run/name,archive/name)
    save(archive/'audit.json',audit)
    save(archive/'report.json',report)
    save(archive/'comparison.json',{'selection_source':entry,'test':'2023-2024 retrospective',
        'regions':comparisons,'macro':macro,'promoted_by':'frozen development rule, not test score',
        'deterministic_seeds_are_not_independent_training':True})
    index=ROOT/'configs/experiments/current_foundation_sources.json'
    previous=index.read_text(); registry=json.loads(previous)
    if registry.get('timesfm25_xreg') not in (None,entry):
        raise ValueError('A different repair is already registered; explicit review required')
    registry['timesfm25_xreg']=entry
    save(index,registry)
    try:
        main_table.main([])
    except BaseException:
        index.write_text(previous)
        raise
    lines=['# TimesFM XReg 固定正则修复：正式测试归档','',
        '两个预定设置在固定抽样与完整2022H1开发集确认后，先提交冻结配置，再运行原2023—2024测试。'
        '三个地区、三个seed共9格完成并通过重评分、分块重组、独立抽样重放与原测试样本对齐。'
        '正式结果不参与配置选择；原ridge0预测保留。','',
        '| 地区 | 原 ridge0 MAE | 冻结 ridge1 MAE | MAE变化 |',
        '|---|---:|---:|---:|']
    for region in (*REGIONS,'Macro'):
        row=macro if region=='Macro' else comparisons[region]
        old,new=(row[k]['mae'] for k in ('original_ridge0','selected_ridge1'))
        lines.append(f'| {region} | {old:.6f} | {new:.6f} | {(new/old-1)*100:+.2f}% |')
    lines+=['','MAE先逐seed计算，再作地区/seed等权汇总。该模型无随机训练，seed重复不代表独立证据；'
        'P表仅原生中位数评分，没有虚构五分位QS。仅使用价格/日历等受限映射，静态气价被中心化移除。'
        '输入真实公开发布时间、修订版本及预训练重叠仍未完整审计，不作严格实时或盲测SOTA结论。','',
        '- [当前统一主表](../main_comparison_20261023/README.md)',
        '- [逐地区比较及来源](comparison.json) · [完整审计](audit.json)',
        '- [冻结开发选择](maintenance_selection.json) · [冻结配置](config.yaml) · [运行清单](jobs.json)',
        '- [维护范围、TabPFN修复与数据审计](../baseline_maintenance_20261024/README.md)','']
    (archive/'README.md').write_text('\n'.join(lines))
    maintenance=ROOT/'logs/forecasting/baseline_maintenance_20261024/README.md'
    text=maintenance.read_text().replace('完整2022H1确认进行中。','完整2022H1确认已完成并通过预定门槛。')
    if '## 正式修复结果' not in text:
        text+='\n\n## 正式修复结果\n\nTimesFM ridge1 的9格正式测试与独立审计已完成，并已替换统一主表的 TimesFM 来源。'
        text+=f"宏MAE由 {macro['original_ridge0']['mae']:.6f} 变为 {macro['selected_ridge1']['mae']:.6f}。"
        text+='详见[正式修复归档](../timesfm25_xreg_ridge1_20261024_r1/README.md)。其他模型原始预测不改写；主表全部来源重新对齐、评分。\n'
        maintenance.write_text(text)
    foundation=ROOT/'logs/forecasting/foundation_baselines_20261023/README.md'
    text=foundation.read_text()
    note='**TimesFM 原 ridge0 行已被开发期冻结的 ridge1 正式结果替换；本目录保留原运行档案。最新来源见[修复结果](../timesfm25_xreg_ridge1_20261024_r1/README.md)。**'
    if note not in text:
        foundation.write_text(text.replace('## 正式结果',note+'\n\n## 正式结果',1))
    print(json.dumps({'published':True,'macro':macro,'archive':str(archive)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'outputs/forecasting/timesfm25_xreg_ridge1_20261024_r1')
    publish(parser.parse_args().run_root)
