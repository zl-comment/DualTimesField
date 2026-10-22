"""Summarize fixed screening criteria; never selects or changes a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def no_worse(value, reference):
    return value <= reference + 1e-7 * abs(reference)


def summarize(run):
    data = json.loads((run / 'report.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    expected = {f'{r}/{f}' for r in ('NSW1', 'QLD1', 'TAS1') for f in ('2021', '2022')}
    if set(data) != expected:
        raise ValueError('Complete three-region, two-fold results required')
    rows, criteria = [], {}
    for fold in ('2021', '2022'):
        cells = [data[f'{region}/{fold}'] for region in ('NSW1', 'QLD1', 'TAS1')]
        for cell in cells:
            if set(cell) != {'legacy', 'balanced', 'identity_anchor'}:
                raise ValueError('All three preregistered variants required')
        b = {metric: sum(c['legacy']['B']['all'][metric] for c in cells) / 3
             for metric in ('mae', 'qs')}
        rows.append(dict(fold=fold, variant='B', mode='baseline', **b, fallback_count=0))
        macro = {}
        for variant in ('legacy', 'balanced', 'identity_anchor'):
            for mode in ('raw', 'guarded'):
                metrics = {m: sum(c[variant][mode]['metrics']['all']['full'][m] for c in cells) / 3
                           for m in ('mae', 'qs')}
                macro[(variant, mode)] = metrics
                fallback = sum(c[variant][mode]['selection']['fallback'] for c in cells)
                rows.append(dict(fold=fold, variant=variant, mode=mode, **metrics, fallback_count=fallback))
        balanced, legacy = macro[('balanced', 'guarded')], macro[('legacy', 'guarded')]
        checks = dict(
            macro_noninferior_to_B_and_legacy=all(no_worse(balanced[m], ref[m])
                for m in ('mae', 'qs') for ref in (b, legacy)),
            macro_strictly_improves_B=any(balanced[m] < b[m] - 1e-7 * abs(b[m]) for m in ('mae', 'qs')),
            each_region_noninferior_to_B=all(no_worse(c['balanced']['guarded']['metrics']['all']['full'][m],
                c['balanced']['B']['all'][m]) for c in cells for m in ('mae', 'qs')),
            each_spring_noninferior_to_B=all(no_worse(c['balanced']['guarded']['metrics']['season/spring']['full'][m],
                c['balanced']['B']['season/spring'][m]) for c in cells for m in ('mae', 'qs')))
        criteria[fold] = dict(checks=checks, passed=all(checks.values()))
    return data, manifest, rows, dict(folds=criteria, passed=all(x['passed'] for x in criteria.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    data, manifest, rows, acceptance = summarize(args.run)
    hashes = {}
    for name in ('manifest.json', 'selection_complete.json', 'report.json', 'report.md'):
        source = args.run / name
        shutil.copyfile(source, destination / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for path in args.run.glob('*/*/*/training.json'):
        target = destination / 'training' / path.relative_to(args.run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    (destination / 'run_pointer.json').write_text(json.dumps(dict(run=str(args.run.resolve()),
        files_sha256=hashes, profile=manifest['config']['profile'],
        prediction_and_checkpoint_policy='Retained locally under run; not committed'), indent=2) + '\n')
    with (destination / 'macro.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    cfg = manifest['config']
    lines = ['# Anchor 平衡修复：首轮开发筛选', '',
             f"三地区等权平均；{cfg['profile']} 采样、{cfg['epochs']} epoch、神经 seed={cfg['seed']}、树 seed={cfg['tree_seed']}。不是全量收敛或显著性结论。", '',
             '| 评估年 | 方案 | 选择方式 | 宏MAE | 宏QS | 回退地区数 |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['fold']} | {row['variant']} | {row['mode']} | {row['mae']:.4f} | {row['qs']:.4f} | {row['fallback_count']} |")
    lines += ['', f"预注册筛选标准：{'通过' if acceptance['passed'] else '未通过'}。", '',
              '各地区、季度、春季与预测价格带诊断见 report.json；逐地区主表见 report.md。',
              '每个 raw/guarded 的选中 epoch 与 alpha 都保留。回退只保证选择期，不保证后续年份。',
              '本轮未读取2023–2024最终测试。没有根据本轮评估重新调整权重或收缩候选。', '',
              '具体未通过项：']
    for fold, result in acceptance['folds'].items():
        lines.append(f"- {fold}: " + (', '.join(k for k, v in result['checks'].items() if not v) or '全部满足'))
    (destination / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(acceptance, indent=2))


if __name__ == '__main__':
    main()
