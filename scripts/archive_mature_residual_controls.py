"""Archive a completed, independently verified mature-residual run locally.

This script never opens an NPZ, changes the run directory, overwrites an archive,
or invokes Git/network operations. All summaries come from verified JSON/CSV.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import statistics
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked_path(root, relative):
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError(f'Unsafe manifest-relative path: {relative}')
    path = root/name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing, nonregular or escaping artifact: {path}')
    return path


def verify_output(output):
    """Validate the entire completed artifact set before creating a directory."""
    output = Path(output).resolve()
    manifest = read_json(checked_path(output, 'manifest.json'))
    identity = {key: value for key, value in manifest.items() if key != 'run_id'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != manifest['run_id']:
        raise ValueError('Run manifest fingerprint changed')
    cfg, run_id = manifest['configuration'], manifest['run_id']
    root = Path(cfg['root']).resolve()
    if output != (root/cfg['output_root']).resolve():
        raise ValueError('Output differs from the frozen configured run directory')
    completion = read_json(checked_path(output, 'completion.json'))
    index = read_json(checked_path(output, 'artifact_manifest.json'))
    independent = read_json(checked_path(output, 'independent_verification.json'))
    if any(item.get('run_id') != run_id for item in (completion, index, independent)):
        raise ValueError('Completion, artifact manifest or independent verification belongs to another run')
    if independent.get('passed') is not True:
        raise ValueError('Independent verification has not passed')
    provenance = independent.get('verification_provenance', {})
    if provenance.get('original_artifact_manifest_sha256') != digest(output/'artifact_manifest.json'):
        raise ValueError('Independent verification is not bound to this artifact manifest')
    verifier_key = str(root/'scripts/verify_mature_residual_controls.py')
    if provenance.get('verifier_sha256') != manifest['sources'].get(verifier_key):
        raise ValueError('Independent verifier differs from the frozen source snapshot')
    if any(stage in independent.get('unverified_numeric_stages', []) for stage in ('selection', 'L0')):
        raise ValueError('Selection and L0 require independent numerical verification')
    if completion.get('main_table_modified') is not False or completion.get('pushed') is not False:
        raise ValueError('Completion does not certify local-only unchanged-main-table scope')
    if (output/'failure.json').exists():
        raise ValueError('A failed run cannot be archived as a completed result')
    files = index.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing complete artifact hash manifest')
    for relative, expected in files.items():
        path = checked_path(output, relative)
        if not isinstance(expected, str) or digest(path) != expected:
            raise ValueError(f'Artifact changed: {relative}')
    observed = set()
    for path in output.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Run contains a symlink: {path}')
        if path.is_file():
            observed.add(str(path.relative_to(output)))
    additions = {'artifact_manifest.json', 'independent_verification.json'}
    if set(files) & additions or observed != set(files) | additions:
        raise ValueError(f'Incomplete artifact manifest or unexpected files: {sorted(observed ^ (set(files) | additions))}')
    selection = read_json(checked_path(output, 'selection/beta_selection.json'))
    frozen = read_json(checked_path(output, 'selection/selection_frozen.json'))
    if (selection.get('run_id') != run_id or frozen.get('run_id') != run_id
            or frozen['sha256'] != digest(output/'selection/beta_selection.json')
            or frozen['beta'] != selection['beta'] or frozen.get('frozen_before_H2_array_access') is not True
            or selection.get('H2_labels_opened') is not False):
        raise ValueError('Frozen H1 beta-selection barrier is inconsistent')
    if [entry['beta'] for entry in selection['curve']] != cfg['beta_grid']:
        raise ValueError('The complete five-beta curve is absent')
    l0 = read_json(checked_path(output, 'L0/decision.json'))
    if (l0.get('run_id') != run_id or l0.get('stage') != 'L0' or l0.get('primary_arm') != 'R28'
            or l0.get('beta') != selection['beta']):
        raise ValueError('L0 identity or frozen beta differs')
    stages = ['selection', 'L0']
    l1 = None
    if l0['passed'] is False:
        absent = read_json(checked_path(output, 'L1_not_run.json'))
        if ((output/'L1').exists() or absent.get('run_id') != run_id or absent.get('reason') != 'L0_failed'
                or absent.get('formal_arrays_opened') is not False or absent.get('new_formal_predictions') != 0
                or absent.get('L0_decision_sha256') != digest(output/'L0/decision.json')):
            raise ValueError('A failed L0 did not preserve the formal-access barrier')
    elif l0['passed'] is True:
        l1 = read_json(checked_path(output, 'L1/decision.json'))
        if (l1.get('run_id') != run_id or l1.get('stage') != 'L1' or l1.get('primary_arm') != 'R28'
                or l1.get('beta') != selection['beta'] or (output/'L1_not_run.json').exists()):
            raise ValueError('L1 identity, beta or access record is inconsistent')
        stages.append('L1')
    else:
        raise ValueError('L0 passed must be an explicit boolean')
    last = l1 if l1 is not None else l0
    if (completion['completed'] != stages or completion['disposition'] != last['decision']
            or completion['final_stage_passed'] != last['passed']):
        raise ValueError('Final completion differs from the stage decisions')
    for absolute, expected in manifest['sources'].items():
        relative = Path(absolute).relative_to(root)
        copy = checked_path(output, str(Path('source')/relative))
        if digest(copy) != expected:
            raise ValueError(f'Frozen source snapshot changed: {relative}')
    return dict(output=output, root=root, cfg=cfg, manifest=manifest, completion=completion,
        artifacts=index, independent=independent, selection=selection, l0=l0, l1=l1,
        verified_hashes={**files, **{name: digest(output/name) for name in additions}})


def number(value, precision=6):
    if value is None or value == '':
        return 'NA'
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('Nonfinite reported metric')
    return f'{parsed:.{precision}f}'


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def macro_rows(rows, cfg, group):
    arms = ['B', 'R28', 'Recent16', 'Matched16', *[f'Full_joint_seed{seed}' for seed in cfg['seeds']]]
    result = []
    for arm in arms:
        selected = [row for row in rows if row['arm'] == arm and row['group'] == group]
        if (len(selected) != len(cfg['regions']) or {row['region'] for row in selected} != set(cfg['regions'])
                or any(row['empty'] != 'False' or int(row['cells']) <= 0 for row in selected)):
            raise ValueError(f'Incomplete nonempty macro table: {group}/{arm}')
        keys = ('after_mae', 'after_qs', 'mae_delta', 'qs_delta', 'after_coverage80',
                'after_coverage90', 'after_width80', 'after_width90')
        result.append(dict(arm=arm, **{key: statistics.fmean(float(row[key]) for row in selected) for key in keys}))
    full = [row for row in result if row['arm'].startswith('Full_joint_seed')]
    result.append(dict(arm='Full_joint_三seed描述均值',
        **{key: statistics.fmean(row[key] for row in full) for key in full[0] if key != 'arm'}))
    return result


def decision_lines(decision, title):
    lines = [title, f"判定：{'通过' if decision['passed'] else '失败'}；{decision['decision']}"]
    if 'periods' in decision:
        for name, item in decision['periods'].items():
            lines.extend(decision_lines(item, f'期间 {name}'))
        return lines
    before, after = decision.get('macro_before', {}), decision.get('macro_after', {})
    for metric in ('mae', 'qs'):
        if metric in before and metric in after:
            delta = None if before[metric] is None or after[metric] is None else after[metric]-before[metric]
            lines.append(f"宏 {metric.upper()}：B={number(before[metric])}，R28={number(after[metric])}，Δ={number(delta)}")
    failed = decision.get('failed_criteria', [])
    lines.append(f"失败条件：{len(failed)}/{len(decision.get('criteria', []))}")
    for item in failed:
        details = {key: value for key, value in item.items() if key != 'passed'}
        lines.append('  '+json.dumps(details, ensure_ascii=False, sort_keys=True))
    return lines


def result_text(verified, evidence):
    cfg, output = verified['cfg'], verified['output']
    manifest, completion = verified['manifest'], verified['completion']
    lines = ['成熟残差信息价值阶梯：本地冻结结果归档',
        f"基准提交：{manifest['base_commit']}", f"研究分支：{manifest['branch']}",
        f"运行源码提交：{manifest['head']}", f"run_id：{manifest['run_id']}",
        f"原始输出：{output}", '', '2022H1 唯一共享 β 选择曲线（地区等权）',
        'β | 宏MAE | 宏QS | 标准化相对损失和 | 可行 | 选中']
    for row in verified['selection']['curve']:
        lines.append(' | '.join((number(row['beta'], 2), number(row['macro']['mae']), number(row['macro']['qs']),
            number(row['standardized_relative_loss_sum'], 12), str(row['eligible']), str(row['beta'] == verified['selection']['beta']))))
    lines.extend([f"最终共享 β={verified['selection']['beta']}；clip=30；Recent16、Matched16原样复用。",
        'H1选择曲线先落盘、冻结哈希，然后才加载H2标签。', ''])
    for stage, decision in (('L0', verified['l0']), ('L1', verified['l1'])):
        if decision is None:
            continue
        lines.extend(decision_lines(decision, f'{stage} 冻结判定'))
        rows = csv_rows(output/stage/'paired_groups.csv')
        groups = [('2022H2', 'all')] if stage == 'L0' else [('2023', 'year/2023'), ('2024', 'year/2024'), ('2023_2024', 'all')]
        for period, group in groups:
            lines.extend(['', f'{stage}/{period} 宏表（先各地区评分，再地区等权）',
                '方法 | MAE | QS | ΔMAE对B | ΔQS对B | C80 | C90 | W80 | W90'])
            for row in macro_rows(rows, cfg, group):
                lines.append(' | '.join([row['arm'], *[number(row[key]) for key in
                    ('after_mae', 'after_qs', 'mae_delta', 'qs_delta', 'after_coverage80', 'after_coverage90', 'after_width80', 'after_width90')]]))
            lines.extend(['', f'{stage}/{period} R28 地区损失及覆盖保护表',
                '地区 | B_MAE | R28_MAE | 相对变化% | B_QS | R28_QS | 相对变化% | B_C80 | R28_C80 | C80绝对偏差恶化 | B_C90 | R28_C90 | C90绝对偏差恶化'])
            selected = {row['region']: row for row in rows if row['group'] == group and row['arm'] == 'R28'}
            for region in cfg['regions']:
                row = selected[region]
                values = [region]
                for loss in ('mae', 'qs'):
                    b, q = float(row[f'before_{loss}']), float(row[f'after_{loss}'])
                    relative = 100*(q/b-1) if b else 0. if q == 0 else None
                    values.extend((number(b), number(q), number(relative)))
                for level in (80, 90):
                    b, q = float(row[f'before_coverage{level}']), float(row[f'after_coverage{level}'])
                    values.extend((number(b), number(q), number(abs(q-level/100)-abs(b-level/100))))
                lines.append(' | '.join(values))
        inference = read_json(output/stage/'paired_inference.json')
        lines.extend(['', f'{stage} R28−B：日期同步配对块 95% 区间（负值为改善）',
            '期间 | 块长天 | 地区/宏 | ΔMAE估计 | 下界 | 上界 | ΔQS估计 | 下界 | 上界'])
        for period, _ in groups:
            report = inference[period]['R28_minus_B']
            if [block['block_days'] for block in report['blocks']] != [7, 14, 28]:
                raise ValueError('Required block sensitivities missing')
            for block in report['blocks']:
                if not block['available']:
                    lines.append(f"{period} | {block['block_days']} | 不可用：{block['reason']}")
                    continue
                for name, metrics in [('宏', block['macro']), *[(region, block['regional'][region]) for region in cfg['regions']]]:
                    values = [period, str(block['block_days']), name]
                    values.extend(number(metrics[loss][key]) for loss in ('mae', 'qs') for key in ('estimate', 'lower', 'upper'))
                    lines.append(' | '.join(values))
        lines.append('7天块为主，14/28天仅敏感性；区间不参与门槛判定，也不把重叠单元或神经seed视为独立市场样本。')
    if not verified['l0']['passed']:
        lines.extend(['', 'L0 未通过：L1 未运行；2023/2024 正式数组未打开，未生成新的正式简单臂预测。',
            'B 保持默认模型；停止当前 V5 残差校正路线新增状态门、收益接受器、尺度及事件实验。'])
    elif verified['l1'] is not None and not verified['l1']['passed']:
        lines.extend(['', 'L0通过、L1失败：成熟残差收益未稳定迁移；不改β、地区规则或价格路由挽救，B保持默认。'])
    else:
        lines.extend(['', 'L0与L1均通过：R28成为后续研究的必要竞争基线；不构成模型创新、双场必要性或神经重训许可。'])
    resources = read_json(output/'resources.json')
    test_resources = evidence.get('resources', evidence.get('resource'))
    if test_resources is None:
        test_resources = {key: value for key, value in evidence.items() if any(word in key for word in ('seconds', 'hours', 'rss'))}
    lines.extend(['', '资源与完整性证据',
        '主运行资源：'+json.dumps(resources, ensure_ascii=False, sort_keys=True),
        '测试资源：'+json.dumps(test_resources if test_resources else {'note': '测试证据未单列资源；不推算或虚构'}, ensure_ascii=False, sort_keys=True),
        '独立核验资源：'+json.dumps(verified['independent'].get('resources', {'note': '见独立核验完整JSON'}), ensure_ascii=False, sort_keys=True),
        f"旧文件保持不变：{completion['protected_files_unchanged']}；运行源码快照保持不变：{completion['source_snapshots_unchanged']}。",
        f"独立核验：passed={verified['independent']['passed']}；完整报告见 independent_verification.json。",
        '独立数值核验范围：'+str(verified['independent'].get('scope', '以完整核验JSON为准')),
        '尚未独立数值复算的阶段：'+json.dumps(verified['independent'].get('unverified_numeric_stages', []), ensure_ascii=False),
        '测试输出摘要：'+(evidence.get('stdout', '').strip().splitlines()[-1] if evidence.get('stdout', '').strip() else '见冻结测试证据'),
        '测试证明：'+json.dumps({key: value for key, value in evidence.items() if key not in ('tested_sources', 'stdout', 'stderr')}, ensure_ascii=False, sort_keys=True),
        '神经训练=0；LightGBM重训=0；GPU=0；新模型推理=0；未改正式主表；未推送。',
        '', '解释边界',
        '2022H1/H2遵循已归档的树更新日程，不能称为全年同一棵冻结树。',
        'L0 Full为2022开发asymmetric/reference的原guarded三seed，区别于最终正式fit；只作历史参照。',
        'B/R28/Recent16/Matched16各只有一个确定性模型；Full三seed均值仅描述优化敏感性。',
        'Matched16保留原跨折FP32坐标，没有重拟合共同标准化；无法据此断言双场表示必要。',
        '价格组仅为事后解释；未按当前真实价格、负价或尖峰进行路由。',
        '若执行L1，它仍是已查看时期上的回顾性前推，不是盲测。',
        '失败约束当前协议和研究路线，不等于证明所有条件残差信息均无价值。',
        '', '复现信息见 reproduction.json；NPZ留在原outputs，位置/大小/SHA256见 npz_references.json。',
        '本归档仅复制已验证结果和冻结文本；未读取任何预测NPZ或正式原数组。'])
    if 'L1' in verified['independent'].get('unverified_numeric_stages', []):
        lines.extend(['', '核验限制：L1数值尚未由独立实现复算。该阶段在此仅完成完整产物哈希验证和原JSON/CSV归档；',
                      'independent_verification.passed 不表示 L1 已独立数值核验通过。'])
    return '\n'.join(lines)+'\n'


def archive_run(output):
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_started = before.ru_utime+before.ru_stime
    verified = verify_output(output)
    output, cfg, root = verified['output'], verified['cfg'], verified['root']
    relative_archive = Path(cfg['archive_root'])
    if relative_archive.is_absolute() or '..' in relative_archive.parts:
        raise ValueError('archive_root must be the frozen repository-relative path')
    destination = root/relative_archive
    if not destination.resolve().is_relative_to(root/'logs/forecasting') or destination.resolve().is_relative_to(output):
        raise ValueError('Unexpected archive destination')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to overwrite an existing archive: {destination}')
    reproduction_paths = [Path('configs/mature_residual_controls.json'), Path(cfg['protocol_path']),
        Path(cfg['requested_plan_path']), Path('docs/mature_residual_controls_test_evidence_20261027.json')]
    snapshots = {}
    for relative in reproduction_paths:
        source = checked_path(output, str(Path('source')/relative))
        expected = verified['manifest']['sources'][str(root/relative)]
        if digest(source) != expected:
            raise ValueError(f'Reproduction text snapshot changed: {relative}')
        snapshots[relative] = source
    frozen_cfg = read_json(snapshots[Path('configs/mature_residual_controls.json')])
    if frozen_cfg != cfg:
        raise ValueError('Frozen config snapshot differs from the executed configuration')
    evidence = read_json(snapshots[Path('docs/mature_residual_controls_test_evidence_20261027.json')])
    if evidence.get('passed') is not True or evidence.get('exit_code') != 0:
        raise ValueError('Frozen test evidence did not pass')
    results = result_text(verified, evidence)
    copy_items = {}
    npz = []
    for relative, expected in verified['verified_hashes'].items():
        path = checked_path(output, relative)
        if Path(relative).parts[0] == 'source':
            continue
        if path.suffix in ('.json', '.csv'):
            copy_items[Path(relative)] = (path, expected)
        elif path.suffix == '.npz':
            npz.append(dict(path=str(path), relative_to_output=relative, sha256=expected, bytes=path.stat().st_size,
                            copied_to_archive=False))
    for relative, source in snapshots.items():
        copy_items[Path('reproduction')/relative] = (source, digest(source))
    # All validation/rendering above is read-only. An existing or partially
    # created archive is never reused; no except/finally writes into output.
    destination.mkdir(parents=True, exist_ok=False)
    for relative, (source, expected) in copy_items.items():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError(f'Copy verification failed: {relative}')
    (destination/'results.txt').write_text(results)
    (destination/'source_commit.txt').write_text(verified['manifest']['head']+'\n')
    write_json(destination/'npz_references.json', dict(run_id=verified['manifest']['run_id'],
        output_root=str(output), count=len(npz), total_bytes=sum(item['bytes'] for item in npz), files=sorted(npz, key=lambda item:item['path'])))
    write_json(destination/'reproduction.json', dict(run_id=verified['manifest']['run_id'],
        base_commit=verified['manifest']['base_commit'], source_commit=verified['manifest']['head'],
        branch=verified['manifest']['branch'], output_root=str(output), archive_root=str(destination),
        source_manifest='manifest.json', source_snapshots_retained_at=str(output/'source'),
        beta=verified['selection']['beta'], artifact_manifest_sha256=digest(output/'artifact_manifest.json'),
        independent_verification_sha256=digest(output/'independent_verification.json'),
        unverified_numeric_stages=verified['independent'].get('unverified_numeric_stages', []),
        archive_script_sha256=digest(Path(__file__)), output_files_changed=False, numpy_arrays_read=False,
        reproduction_command='.venv/bin/python scripts/compare_mature_residual_controls.py',
        reproduction_warning='The frozen runner rejects existing outputs. A separately named reproduction needs an explicitly versioned protocol; never overwrite this run.',
        archive_command=f'.venv/bin/python scripts/archive_mature_residual_controls.py --output {output}',
        main_table_modified=False, pushed=False))
    # Rehash the input set before certification, to detect concurrent mutation
    # without altering any original file.
    for relative, expected in verified['verified_hashes'].items():
        if digest(checked_path(output, relative)) != expected:
            raise ValueError(f'Output changed while being archived: {relative}')
    files = {str(path.relative_to(destination)): dict(sha256=digest(path), bytes=path.stat().st_size)
             for path in sorted(destination.rglob('*')) if path.is_file()}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = usage.ru_utime+usage.ru_stime-cpu_started
    archive_resources = dict(wall_seconds=time.monotonic()-started, cpu_seconds=cpu_seconds,
        cpu_core_hours=cpu_seconds/3600, peak_rss_kib=usage.ru_maxrss,
        gpu_seconds=0, gpu_hours=0, npz_decompressed=0, original_files_modified=0,
        timing_scope='archive_run entry through verification, copying and archive-file hashing; final manifest write excluded')
    write_json(destination/'archive_manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        run_id=verified['manifest']['run_id'], source_commit=verified['manifest']['head'],
        original_output=str(output), original_artifact_manifest_sha256=digest(output/'artifact_manifest.json'),
        independent_verification_passed=True,
        unverified_numeric_stages=verified['independent'].get('unverified_numeric_stages', []),
        resources=archive_resources, files=files,
        self_excluded='archive_manifest.json', npz_copied=0, output_files_modified=0, pushed=False))
    return dict(archive=str(destination), files=len(files), npz_references=len(npz), output_files_modified=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive_run(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
