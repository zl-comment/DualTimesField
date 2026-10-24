"""Audit complete latest Full_joint predictions and publish three-seed tables.

Run from the repository root: PYTHONPATH=. .venv/bin/python scripts/update_latest_main_comparison.py
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from forecasting.aligned_scoring import assert_comparable, load_prediction_artifact, score_predictions
from forecasting.full_v5_anchor_balance import aligned_final as experiment
from forecasting.full_v5_anchor_balance.selection import selection_key

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/forecasting/v5_latest_aligned_20261023_r2'
ARCHIVE = ROOT / 'logs/forecasting/v5_latest_aligned_20261023_r2'
BASELINES = ROOT / 'logs/forecasting/ddnn_jsu_unclipped_20261022_r1'
OUTPUT = ROOT / 'logs/forecasting/main_comparison_20261023'
REGIONS = ('NSW1', 'QLD1', 'TAS1')
SEEDS = (2026, 2027, 2028)
METRICS = ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_80', 'coverage_90',
           'interval_width_80', 'interval_width_90')
ROLES = dict(B_retrained='fixed_tree_reference', Full_joint='latest_full_method',
             Full_joint_raw='raw_output_diagnostic', Anchor_joint='same_checkpoint_intervention',
             event_off_same_head='same_checkpoint_intervention', time_shift_off_same_head='same_checkpoint_intervention')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(per_seed):
    """Region-macro per seed, then sample SD across seeds (never ensemble)."""
    macro, regional, main = [], [], []
    models = sorted({r['model'] for r in per_seed})
    grid = {(r['model'], r['region'], r['seed']) for r in per_seed}
    require(len(grid) == len(per_seed), 'Duplicate model/region/seed result')
    require(grid == {(m, r, s) for m in models for r in REGIONS for s in SEEDS}, 'Incomplete three-seed grid')
    for model in models:
        rows = [r for r in per_seed if r['model'] == model]
        identity = {k: rows[0][k] for k in ('model', 'role', 'model_version', 'input_scope', 'seed_scope', 'table', 'pretraining_overlap')}
        available = [m for m in METRICS if all(r.get(m) is not None for r in rows)]
        for seed in SEEDS:
            selected = [r for r in rows if r['seed'] == seed]
            macro.append(dict(identity, seed=seed,
                **{m: float(np.mean([r[m] for r in selected])) if m in available else None for m in METRICS}))
        for region in REGIONS:
            selected = [r for r in rows if r['region'] == region]
            row = dict(identity, region=region, seeds=3)
            for m in METRICS:
                if m not in available:
                    row[m + '_mean'] = row[m + '_std'] = None
                    continue
                row[m + '_mean'] = float(np.mean([r[m] for r in selected]))
                row[m + '_std'] = float(np.std([r[m] for r in selected], ddof=1))
            regional.append(row)
        selected = [r for r in macro if r['model'] == model]
        row = dict(identity, seeds=3, cells=9, fallback_cells=sum(r['fallback'] for r in rows))
        for m in METRICS:
            if m not in available:
                row[m + '_mean'] = row[m + '_std'] = None
                continue
            row[m + '_mean'] = float(np.mean([r[m] for r in selected]))
            row[m + '_std'] = float(np.std([r[m] for r in selected], ddof=1))
        main.append(row)
    main.sort(key=lambda r: (r['mae_mean'], r['model']))
    order = {r['model']: i for i, r in enumerate(main)}
    macro.sort(key=lambda r: (order[r['model']], r['seed']))
    regional.sort(key=lambda r: (order[r['model']], REGIONS.index(r['region'])))
    return main, regional, macro


def verify_frozen_joint_manifest(manifest, run=RUN):
    """Reporting verifies the executed snapshot, not later working-tree edits."""
    digest = experiment.c.base.digest
    for original, expected in manifest['source_sha256'].items():
        relative = Path(original).relative_to(ROOT)
        require(digest(run / 'source' / relative) == expected, f'Frozen source snapshot changed: {relative}')
    for path, expected in manifest['inputs_sha256'].items():
        require(digest(path) == expected, f'Frozen input changed: {path}')
    require(digest(run / 'protocol.md') == manifest['protocol_sha256'], 'Frozen protocol snapshot changed')


def main(argv=None):
    progress = read(RUN / 'progress.json')
    require(progress['stage'] == 'complete', 'Latest run incomplete; do not publish replacement tables')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-run', type=Path, default=ROOT.parent / 'DualTimesField-foundation-baselines/outputs/forecasting/foundation_baselines_20261023_r1')
    args = parser.parse_args(argv)
    from forecasting.foundation_audit import collect
    from forecasting.aligned_report import build_aligned_report
    from forecasting.common.foundation_registry import FOUNDATION_REGISTRY
    foundation_run = args.foundation_run.resolve()
    foundation_audit = collect(foundation_run)
    foundation_report = build_aligned_report(foundation_run)
    require(foundation_report['comparison_available'] and foundation_report['progress']['completed_cells'] == 36,
            'Foundation grid incomplete')
    foundation_manifest = read(foundation_run / 'jobs.json')
    progress = read(RUN / 'progress.json')
    require(progress['stage'] == 'complete', 'Latest run incomplete; do not publish replacement tables')
    manifest = read(RUN / 'manifest.json')
    verify_frozen_joint_manifest(manifest, RUN)
    selections = read(RUN / 'selection_complete.json')
    experiment.verify_barrier(selections, manifest['config'])
    digest = experiment.c.base.digest
    require(progress['selection_sha256'] == digest(RUN / 'selection_complete.json'), 'Selection freeze changed')
    require(progress['report_sha256'] == digest(RUN / 'report.json'), 'Latest report changed')
    latest = read(RUN / 'report.json')
    require(set(latest) == {f'{m}/{s}/{r}' for m in experiment.ALL_OUTPUTS for s in SEEDS for r in REGIONS},
            'Latest output grid incomplete')
    training = {}
    for key, record in selections.items():
        region, seed = key.split('/')
        path = experiment.fit_path(RUN, region, int(seed)) / 'training.json'
        log = read(path)
        require(len(log) == 20, 'Training budget incomplete')
        require(min(log, key=lambda row: row['raw']['full']['qs'])['raw'] == record['selection']['raw'],
                'Raw selection does not match validation QS')
        require(min((row['guarded'] for row in log), key=selection_key) == record['selection']['guarded'],
                'Guarded selection does not match validation-only rule')
        training[str(path.relative_to(RUN))] = log
    boundary = {mode: sum(record['selection'][mode]['epoch'] >= manifest['config']['epochs'] - 1
                           for record in selections.values()) for mode in ('raw', 'guarded')}

    report = read(BASELINES / 'combined_report.json')
    require(report['comparison_available'] and report['progress']['completed_cells'] == 153,
            'Baseline report incomplete')
    require(report['cross_run_alignment_validation'] == 'passed_all_162_artifacts', 'Missing baseline source audit')
    baseline_rows = csv_rows(BASELINES / 'combined_report_tables/per_seed.csv')
    require(len(baseline_rows) == 153, 'Expected 153 baseline rows')
    require(Path(report['model_source_selection']['ddnn_jsu']).name == 'ddnn_jsu_unclipped_20261022_r1',
            'Superseded clipped DDNN source selected')
    references = {r: load_prediction_artifact(Path(report['model_source_selection']['lightgbm']) /
        'runs/lightgbm/seed2026' / r / 'test.npz') for r in REGIONS}
    per_seed, provenance = [], []
    score_differences = []
    def append(model, region, seed, path, recorded, role, version, source, seed_scope,
               *, alpha=None, fallback=False, checkpoint_sha256=None, expected_sha=None, input_scope=None, pretraining_overlap='not_applicable'):
        artifact = load_prediction_artifact(path)
        if artifact.table == 'Q':
            assert_comparable(references[region], artifact, allow_different_seed=True)
        else:
            for key in ('protocol_id', 'split', 'region', 'input_hash', 'sample_hash', 'preprocessing_hash', 'prediction_space'):
                require(artifact.metadata[key] == references[region].metadata[key], f'Point-table metadata mismatch: {key}')
            for key in ('origin_unix', 'target_time_unix', 'actual'):
                require(np.array_equal(getattr(artifact, key), getattr(references[region], key)), f'Point-table sample mismatch: {key}')
        require(artifact.metadata['seed'] == seed, 'Artifact seed mismatch')
        scores = score_predictions(artifact)
        require(scores['n_origins'] == 17521 and scores['horizon'] == 24, 'Wrong test shape')
        for metric in METRICS:
            if metric not in scores:
                require(artifact.table == 'P' and metric not in ('mae', 'rmse_pooled'), 'Missing required metric')
                continue
            difference = abs(scores[metric] - float(recorded[metric]))
            score_differences.append(difference)
            matches = difference <= 1e-10 if role == 'foundation_baseline' else difference == 0.0
            require(matches, f'Re-score mismatch: {model}/{region}/{seed}/{metric}')
        sha = digest(path)
        require(expected_sha is None or sha == expected_sha, 'Latest artifact hash changed')
        if checkpoint_sha256 is not None:
            require(artifact.metadata['checkpoint_sha256'] == checkpoint_sha256, 'Checkpoint source mismatch')
            require(artifact.metadata['model_version'] == 'asymmetric_balance_20261023', 'Old joint version selected')
            require(artifact.metadata['source'] == manifest['run_id'], 'Run source mismatch')
            require(artifact.metadata['alpha'] == alpha, 'Frozen alpha mismatch')
        scope = ('public_inputs_plus_mature_history_and_errors'
                 if model in ROLES and model != 'B_retrained' else 'public_inputs')
        per_seed.append(dict(model=model, role=role, model_version=version, input_scope=input_scope or scope,
            seed_scope=seed_scope, table=artifact.table, pretraining_overlap=pretraining_overlap, region=region, seed=seed, test_period='2023-2024',
            n_origins=17521, horizon=24, alpha=alpha, fallback=bool(fallback),
            **{m: scores.get(m) for m in METRICS}))
        provenance.append(dict(model=model, region=region, seed=seed, source=source, path=str(path), sha256=sha))
    for row in baseline_rows:
        model, region, seed = row['model'], row['region'], int(row['seed'])
        root = Path(report['model_source_selection'][model])
        require(root.name == row['run_id'], 'Baseline source CSV mismatch')
        append('baseline_' + model, region, seed, root / 'runs' / model / f'seed{seed}' / region / 'test.npz',
            row, 'historical_structure_control' if model in ('full', 'no_fields') else 'static_baseline',
            row['run_id'], dict(run_id=row['run_id'], source_hash=row['source_hash'], config_hash=row['config_hash']),
            'original baseline seed definition; deterministic methods may repeat')
    print('Re-scored and verified all 153 unchanged baseline cells', flush=True)
    for row in latest.values():
        model, region, seed = (row[k] for k in ('model', 'region', 'seed'))
        mode = 'raw' if model == 'Full_joint_raw' else 'guarded'
        selection = selections[f'{region}/{seed}']['selection'][mode]
        expected_alpha = 0.0 if model == 'B_retrained' else selection['alpha']
        expected_fallback = False if model == 'B_retrained' else selection['fallback']
        checkpoint = experiment.fit_path(RUN, region, seed) / f'{mode}.pt'
        require(row['checkpoint_sha256'] == digest(checkpoint), 'Recorded checkpoint changed')
        require(row['alpha'] == expected_alpha and row['fallback'] == expected_fallback, 'Frozen output selection changed')
        append(model, region, seed, Path(row['path']), row['scores'], ROLES[model],
            'asymmetric_balance_20261023', dict(run_id=manifest['run_id'], checkpoint_sha256=digest(checkpoint)),
            'fixed tree 2026 repeated reference' if model == 'B_retrained' else manifest['config']['seed_scope'],
            alpha=row['alpha'], fallback=row['fallback'], checkpoint_sha256=digest(checkpoint), expected_sha=row['sha256'])
    print('Re-scored and verified all 54 latest joint outputs', flush=True)
    for job in foundation_manifest['jobs']:
        model, region, seed = job['model'], job['region'], job['seed']
        spec = FOUNDATION_REGISTRY[model]
        directory = foundation_run / job['output']
        status = read(directory / 'status.json')
        append(model, region, seed, directory / 'test.npz', read(directory / 'metrics.json'),
               'foundation_baseline', foundation_manifest['run_id'],
               dict(run_id=foundation_manifest['run_id'], source_hash=foundation_manifest['source_hash'],
                    config_hash=foundation_manifest['config_hash']), spec.seed_scope,
               input_scope=spec.input_scope, pretraining_overlap='unaudited',
               expected_sha=status['artifact_hashes']['test.npz'])
    print('Re-scored and aligned all 36 foundation cells with the original LightGBM test samples', flush=True)
    main_rows, regional, macro = summarize(per_seed)
    require(len(main_rows) == 27 and len(per_seed) == 243, 'Unexpected current table size')
    old_macro = csv_rows(BASELINES / 'combined_report_tables/main.csv')
    for old in old_macro:
        new = next(r for r in main_rows if r['model'] == 'baseline_' + old['model'])
        for metric in METRICS:
            require(np.isclose(new[metric + '_mean'], float(old[metric]), rtol=0, atol=1e-12),
                    'Baseline aggregate changed beyond summation rounding')
    historical = [r for r in main_rows if r['role'] == 'historical_structure_control']
    historical_cells = [r for r in per_seed if r['role'] == 'historical_structure_control']
    main_rows = [r for r in main_rows if r['role'] != 'historical_structure_control']
    regional = [r for r in regional if r['role'] != 'historical_structure_control']
    macro = [r for r in macro if r['role'] != 'historical_structure_control']
    per_seed = [r for r in per_seed if r['role'] != 'historical_structure_control']
    require(len(main_rows) == 25 and len(per_seed) == 225 and len(historical_cells) == 18,
            'Unexpected current/historical table partition')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (('main.csv', main_rows), ('regional.csv', regional),
                           ('macro_per_seed.csv', macro), ('per_seed.csv', per_seed),
                           ('historical_controls.csv', historical), ('historical_controls_per_seed.csv', historical_cells)):
        write_csv(OUTPUT / filename, rows)
    audit = dict(state='complete', outputs=25, cells=225, source_outputs=27, source_cells=243,
        historical_control_outputs=2, historical_control_cells=18, neural_fits=9, latest_artifacts=54,
        baseline_cells=153, foundation_cells=36, foundation_audit=foundation_audit,
        foundation_run=str(foundation_run), foundation_manifest_sha256=digest(foundation_run / "jobs.json"),
        baseline_cell_scores_exactly_unchanged=True, metric_max_abs_difference=max(score_differences), foundation_rescore_absolute_tolerance=1e-10,
        all_three_seeds_complete=True, raw_and_guarded_selection_independently_verified=True,
        selected_in_last_two_epochs=boundary,
        old_joint_outputs_replaced=True, tree_seed=2026, neural_seeds=list(SEEDS),
        identical_information_scope=False, generator_sha256=digest(Path(__file__)),
        run_manifest_sha256=digest(RUN / 'manifest.json'), selection_sha256=digest(RUN / 'selection_complete.json'),
        baseline_report_sha256=digest(BASELINES / 'combined_report.json'), artifacts=provenance,
        output_sha256={p.name: digest(p) for p in OUTPUT.glob('*.csv')})
    (OUTPUT / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    for name in ('manifest.json', 'progress.json', 'selection_complete.json', 'report.json', 'protocol.md'):
        shutil.copyfile(RUN / name, ARCHIVE / name)
    (ARCHIVE / 'training_logs.json.gz').write_bytes(gzip.compress(json.dumps(training).encode(), mtime=0))
    (ARCHIVE / 'README.md').write_text(
        '# 新版 Full_joint 三神经种子正式协议评估\n\n'
        '九次拟合全部完成，每次 20 轮；九次选择冻结后才加载测试集。最新完整结果见'
        '[当前主表](../main_comparison_20261023/README.md)。\n\n'
        '本目录保留配置、源码/输入身份、选择记录、训练日志和逐格指标。大文件与冻结源码位于 '
        '`outputs/forecasting/v5_latest_aligned_20261023_r2`。树 seed 固定 2026，'
        '神经种子为 2026/2027/2028，不能将种子标准差解释为整个树＋神经流程的随机波动。\n')
    lines = ['# 当前主表：最新 Full_joint 与三个 seed 的基线（2026-10-24 更新）', '',
        '所有当前输出均在 2023—2024 年、同三地区、同一测试起点与共同评分下汇总。'
        '每个 seed 分别评分，先做三地区等权宏平均，再报告三个 seed 的均值 ± 样本标准差；没有先平均预测。', '',
        '新版 Full_joint 为平衡损失＋非对称 Anchor 尺度头，主输出采用选择期冻结的 guarded checkpoint/alpha。'
        'Full_joint_raw 使用独立按选择期 QS 选定的 checkpoint；Anchor_joint/event_off/time_shift_off 是新版 guarded 同检查点关闭干预。'
        '全部替换此前只完成 seed 2026 的旧联合模型，版本号为 asymmetric_balance_20261023。', '',
        '**种子范围：**基线沿用原来的 2026/2027/2028；联合方法改变神经初始化和训练顺序，树固定 seed 2026。'
        'B_retrained 重复同一冻结树，不能视为三个独立随机训练。'
        '联合方法还使用因果成熟历史与误差，信息范围大于静态基线；这是回顾性比较，不是盲测或显著性认证。', '',
        '当前表共 25 个输出、225 格，包括 15 个静态基线、4 个基础模型和联合方法的 6 个输出。'
        '旧静态 baseline_full、baseline_no_fields 的 18 格移入历史结构对照附表，不占当前主表行。'
        '连同该附表，共核验 27 个来源输出、243 格。'
        '最近四方案的 2021/2022 开发分数保留原档，不混入本表；选择最新结构不代表它已通过开发验收。', '',
        '新增基础模型全部标记预训练重叠未审计：TabPFN 使用完整公共输入，其他三者使用受限输入。TabPFN 固定生产分块可重现，但改变批次大小的抽样重放未通过原容差。三种时序基础模型的 seed 重复不能解释为独立随机训练。', '',
        '| 输出 | 表/信息组 | 宏 MAE（均值 ± SD） | 宏 QS（均值 ± SD） | 地区平均 RMSE | 回退格/9 |',
        '|---|---|---:|---:|---:|---:|']
    for row in main_rows:
        qs = '—' if row['quantile_score_proxy_mean'] is None else f"{row['quantile_score_proxy_mean']:.4f} ± {row['quantile_score_proxy_std']:.4f}"
        scope = ('历史扩展' if row['input_scope'] == 'public_inputs_plus_mature_history_and_errors'
                 else '受限输入' if row['input_scope'].startswith('restricted_') else '公共输入')
        lines.append(f"| {row['model']} | {row['table']}/{scope} | {row['mae_mean']:.4f} ± {row['mae_std']:.4f} | "
                     f"{qs} | {row['rmse_pooled_mean']:.4f} | {row['fallback_cells']} |")
    lines += ['', 'Q 表 MAE 使用排序后的 q50，P 表使用原生中位数；P 表无五分位 QS，不与 Q 表混排概率指标；QS=2×公共五分位平均 pinball。RMSE 为地区各自 pooled RMSE 的算术平均。'
        '“回退”只适用于 guarded 联合输出，表示选择期 alpha=0，不能作为结构有效的证据。', '',
        f"预算边界：raw 有 {boundary['raw']}/9、guarded 有 {boundary['guarded']}/9 次选择落在最后两轮，"
        '本轮仍保留预定 20 轮预算，不能宣称已经收敛。', '',
        '- [25 行三 seed 主表](main.csv) · [75 行地区汇总](regional.csv)',
        '- [75 行各 seed 宏平均](macro_per_seed.csv) · [225 格原始评分](per_seed.csv)',
        '- [旧静态 Full/no_fields 历史对照](historical_controls.csv) · [历史对照的 18 格评分](historical_controls_per_seed.csv)',
        '- [来源、选择与重评分核验](verification.json) · [训练和冻结档案](../v5_latest_aligned_20261023_r2/README.md)',
        '- [新增基础模型的实现、审计及限制](../foundation_baselines_20261023/README.md)',
        '- [上一版单 seed 比较（历史）](../main_comparison_20261022/README.md) · [四方案开发确认](../v5_anchor_confirmation_20261022/README.md)', '',
        '复算：`PYTHONPATH=. .venv/bin/python scripts/update_latest_main_comparison.py`。'
        '未完成九格训练/选择或任意预测来源/分数不一致时拒绝更新。', '']
    (OUTPUT / 'README.md').write_text('\n'.join(lines))
    (OUTPUT / 'VALIDATION.md').write_text('# 当前主表验证（2026-10-24）\n\n25 个输出、225 格；另保留 18 格历史结构对照。所有 243 个来源预测均重新评分并对齐原测试样本。每个模型覆盖三个地区和三个 seed；确定性 seed 重复不代表独立重复实验。\n\n更新命令：`PYTHONPATH=. .venv/bin/python scripts/update_latest_main_comparison.py`；可用 `--foundation-run` 指定基础模型原始运行目录。缺失或失败审计、缺格、来源哈希或评分不一致均拒绝发布。\n\n详见 [机器可读核验](verification.json)、[主表](README.md)和[基础模型审查边界](../foundation_baselines_20261023/README.md)。\n')
    print(f'Published {len(main_rows)} three-seed rows and {len(per_seed)} scored cells')


if __name__ == '__main__':
    main()
