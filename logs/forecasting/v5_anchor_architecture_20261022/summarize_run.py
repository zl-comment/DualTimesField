"""Archive the fixed architecture comparison; never tune or select checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

VARIANTS = ('balanced', 'identity_anchor', 'location_only', 'separate_heads', 'staged', 'wide_head')
REGIONS = ('NSW1', 'QLD1', 'TAS1')
FOLDS = ('2021', '2022')


def no_worse(value, reference):
    return value <= reference + 1e-7 * abs(reference)


def improves(value, reference):
    return value < reference - 1e-7 * abs(reference)


def summarize(run):
    data = json.loads((run / 'report.json').read_text())
    if set(data) != {f'{r}/{f}' for r in REGIONS for f in FOLDS}:
        raise ValueError('All six region/fold cells are required')
    rows, criteria, signals = [], {v: {} for v in VARIANTS[1:]}, {v: {} for v in VARIANTS[1:]}
    for fold in FOLDS:
        cells = [data[f'{r}/{fold}'] for r in REGIONS]
        if any(set(c) != set(VARIANTS) for c in cells):
            raise ValueError('All six architecture variants are required')
        base = {m: sum(c['balanced']['B']['all'][m] for c in cells) / 3 for m in ('mae', 'qs')}
        rows.append(dict(fold=fold, variant='B', mode='baseline', anchor_mae=base['mae'],
                         anchor_qs=base['qs'], **base, fallback_count=0))
        macro = {}
        for variant in VARIANTS:
            for mode in ('raw', 'guarded'):
                metrics = {m: sum(c[variant][mode]['metrics']['all']['full'][m] for c in cells) / 3
                           for m in ('mae', 'qs')}
                macro[variant, mode] = metrics
                anchor = {f'anchor_{m}': sum(c[variant][mode]['metrics']['all']['anchor'][m]
                          for c in cells) / 3 for m in ('mae', 'qs')}
                rows.append(dict(fold=fold, variant=variant, mode=mode, **anchor, **metrics,
                                 fallback_count=sum(c[variant][mode]['selection']['fallback'] for c in cells)))
        for variant in VARIANTS[1:]:
            chosen = macro[variant, 'guarded']
            checks = dict(
                macro_noninferior_to_B_and_balanced=all(no_worse(chosen[m], ref[m])
                    for m in ('mae', 'qs') for ref in (base, macro['balanced', 'guarded'])),
                macro_strictly_improves_B=any(improves(chosen[m], base[m]) for m in ('mae', 'qs')),
                each_region_noninferior_to_B=all(no_worse(c[variant]['guarded']['metrics']['all']['full'][m],
                    c[variant]['B']['all'][m]) for c in cells for m in ('mae', 'qs')),
                each_spring_noninferior_to_B=all(no_worse(c[variant]['guarded']['metrics']['season/spring']['full'][m],
                    c[variant]['B']['season/spring'][m]) for c in cells for m in ('mae', 'qs')))
            criteria[variant][fold] = dict(checks=checks, passed=all(checks.values()))
            raw, ref = macro[variant, 'raw'], macro['balanced', 'raw']
            signals[variant][fold] = (all(no_worse(raw[m], ref[m]) for m in ('mae', 'qs'))
                                     and any(improves(raw[m], ref[m]) for m in ('mae', 'qs')))
    acceptance = {v: dict(folds=folds, passed=all(f['passed'] for f in folds.values()))
                  for v, folds in criteria.items()}
    mechanism = {v: dict(folds=folds, consistent=all(folds.values())) for v, folds in signals.items()}
    return rows, acceptance, mechanism


def calibration_diagnostics(run):
    rows = []
    for region in REGIONS:
        for fold in FOLDS:
            for variant in VARIANTS:
                with np.load(run / region / fold / variant / f'{variant}_raw.npz') as saved:
                    a, b = saved['raw_anchor'].astype('float64'), saved['baseline'].astype('float64')
                    delta = a[..., 2] - b[..., 2]
                    widths = b[..., 4] - b[..., 0]
                    usable = widths != 0
                    ratio = (a[..., 4] - a[..., 0])[usable] / widths[usable]
                    rows.append(dict(region=region, fold=fold, variant=variant,
                        delta_mean=float(delta.mean()), delta_abs_mean=float(np.abs(delta).mean()),
                        width_ratio_mean=float(ratio.mean()) if ratio.size else None,
                        width_ratio_p05=float(np.quantile(ratio, .05)) if ratio.size else None,
                        width_ratio_p95=float(np.quantile(ratio, .95)) if ratio.size else None,
                        width_ratio_excluded=int((~usable).sum()), n_cells=int(delta.size)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run, destination = args.run, Path(__file__).resolve().parent
    rows, acceptance, mechanism = summarize(run)
    hashes = {}
    for name in ('manifest.json', 'selection_complete.json', 'report.json', 'report.md'):
        source = run / name
        shutil.copyfile(source, destination / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source in run.glob('*/*/*/training.json'):
        target = destination / 'training' / source.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name, content in (('acceptance', acceptance), ('mechanism_signals', mechanism),
                          ('calibration', calibration_diagnostics(run))):
        (destination / f'{name}.json').write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    (destination / 'run_pointer.json').write_text(json.dumps(dict(run=str(run.resolve()),
        files_sha256=hashes, prediction_and_checkpoint_policy='Retained locally under run; not committed'), indent=2) + '\n')
    with (destination / 'macro.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Anchor 结构与训练耦合：开发筛选结果', '',
             '三地区等权；screen 采样、6 轮（staged 为 3+3）、神经和树 seed=2026。',
             '本轮所有历史/联合训练统一平衡损失，不能直接与首轮三轮实验作结构归因。', '',
             '| 年份 | 方案 | 选择 | Anchor MAE | Full MAE | Full QS | 回退地区数 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['fold']} | {row['variant']} | {row['mode']} | {row['anchor_mae']:.4f} | "
                     f"{row['mae']:.4f} | {row['qs']:.4f} | {row['fallback_count']} |")
    lines += ['', '## 预先固定的筛选标准', '']
    for variant, result in acceptance.items():
        failures = [f"{f}: " + ', '.join(k for k, ok in x['checks'].items() if not ok)
                    for f, x in result['folds'].items() if not x['passed']]
        lines.append(f"- {variant}: " + ('通过' if result['passed'] else '未通过；' + '；'.join(failures)))
    lines += ['', '## raw 跨年机制线索', '',
              '要求两个年度宏 MAE/QS 均不差于 balanced，且每年至少一项严格改善；不是因果确证。', '']
    for variant, result in mechanism.items():
        lines.append(f"- {variant}: {'符合' if result['consistent'] else '不符合'}（{result['folds']}）")
    lines += ['', '全部检查点和 alpha 在评估前确定。未读取 2023–2024；未按本轮评估重新调参。',
              '六轮单种子筛选不能证明收敛或显著性；回退 B 不构成学习收益。',
              '详细地区/季节/价格带在 report.json；校正幅度在 calibration.json；逐轮记录在 training/。']
    (destination / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(acceptance=acceptance, mechanism=mechanism), indent=2))


if __name__ == '__main__':
    main()
