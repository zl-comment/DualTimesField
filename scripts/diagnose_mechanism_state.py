"""Fit training-only state evidence, then diagnose existing bounded predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from forecasting.full_v5_mechanism_state import runner as r
from forecasting.full_v5_mechanism_state.diagnostics import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    cfg = r.configuration(['cpu']*3)
    manifest = r.prepare(output, cfg, Path(r.base.BUNDLE_ROOT).resolve())
    # All six policies frozen before opening ANY diagnostic prediction labels.
    r.verify_manifest(manifest)
    rows, sources = [], {}
    for region in cfg['regions']:
        store, paths = r.base.load_store(r.base.BUNDLE_ROOT, region, cfg['tree_seed'])
        inputs, _ = r.load_inputs(store, paths)
        for fold in cfg['folds']:
            ids = r.split_rows(store, fold, cfg)['evaluation']
            policy = json.loads((output/region/fold/'policy.json').read_text())
            for seed in cfg['seeds']:
                for mode in ('raw', 'guarded'):
                    path = r.arm_path(r.OLD, region, fold, seed, 'bounded_calibration')/f'bounded_calibration_{mode}.npz'
                    sources[str(path)] = r.base.digest(path)
                    with np.load(path, allow_pickle=False) as z:
                        if not np.array_equal(z['origin_unix'], store.origins[ids]) or not np.array_equal(z['actual'], store.a['actual'][ids]):
                            raise ValueError('Diagnostic origin/label mismatch')
                        if not np.array_equal(z['baseline'], store.a['baseline'][ids]):
                            raise ValueError('Diagnostic baseline mismatch')
                        report = diagnose(z['actual'], z['baseline'], z['anchor'], z['quantiles'], inputs[ids], policy)
                    for group, values in report.items():
                        rows.append(dict(region=region, fold=fold, seed=seed, mode=mode, group=group, **values))
        del store, inputs
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'diagnosis.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    comparisons = []
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for mode in ('raw', 'guarded'):
                groups = {g: [row for row in rows if row['region']==region and row['fold']==fold and row['mode']==mode and row['group']==g]
                          for g in ('weak_location', 'stronger_location')}
                usable = all(len(v)==3 and all(not row['empty'] for row in v) for v in groups.values())
                means = {g: float(np.mean([row['anchor_harm'] for row in v])) for g,v in groups.items()} if usable else {}
                comparisons.append(dict(region=region, fold=fold, mode=mode, comparable=usable, harm=means,
                    weak_more_harm=bool(means['weak_location'] > means['stronger_location']) if usable else None))
    r.verify_manifest(manifest)
    if any(r.base.digest(p)!=h for p,h in sources.items()):
        raise ValueError('Diagnostic source changed')
    r.base.write_json(output/'verification.json', dict(prediction_sources=sources, policies_frozen_before_diagnosis=True,
        original_predictions_unchanged=True, comparisons=comparisons, causality_claim=False,
        conclusion='descriptive association only; retain non-supporting and empty strata'))
    lines = ['# 供需状态与校正证据：无需重训诊断', '',
        '训练期固定状态和三门，随后分析旧 bounded 的开发输出。没有新模型性能结果。',
        '弱位置证据定义 g<=.25。伤害为逐单元 max(Anchor绝对误差−树绝对误差,0) 的均值。',
        '伤害同时受原树误差与状态构成影响；关联不能确定原因。不可比较/反向结果必须保留。', '',
        '| 地区 | 年 | 输出 | 弱组伤害 | 其余组伤害 | 弱组更高 |', '|---|---|---|---:|---:|---|']
    for c in comparisons:
        h = c['harm']
        lines.append(f'| {c["region"]} | {c["fold"]} | {c["mode"]} | {h.get("weak_location", "NA")} | {h.get("stronger_location", "NA")} | {c["weak_more_harm"]} |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
