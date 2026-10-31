#!/usr/bin/env python3
"""S0 source coverage audit. Reads demand metadata, never predicts or scores prices."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.agent_forecasting.totaldemand_information import coverage, read_month

AEST = timezone(timedelta(hours=10))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def historical_inventory(root: Path, regions: list[str], year: int) -> dict:
    columns = ['delivery_start_aest', 'available_at_aest', 'region', 'total_demand_mw',
               'source_interval_minutes', 'source_interval_count', 'split']
    result = {}
    for region in regions:
        path = root / f'{region.lower()}_hourly.csv'
        frame = pd.read_csv(path, usecols=columns)
        # These ISO timestamps already carry the market UTC+10 offset.
        year_rows = frame.loc[frame['delivery_start_aest'].str.startswith(str(year))]
        intervals = year_rows.groupby(['source_interval_minutes', 'source_interval_count']).size()
        starts = pd.to_datetime(year_rows['delivery_start_aest'], utc=True)
        available = pd.to_datetime(year_rows['available_at_aest'], utc=True)
        result[region] = {
            'path': str(path), 'sha256': sha256(path), 'rows': len(frame),
            'year_rows': len(year_rows),
            'source_intervals': [
                {'minutes': int(key[0]), 'count': int(key[1]), 'hours': int(count)}
                for key, count in intervals.items()
            ],
            'availability_equals_hour_end': bool((available == starts + pd.Timedelta(hours=1)).all()),
            'columns_projected': columns,
            'price_columns_materialized': False,
        }
    return result


def run(config_path: Path) -> None:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(config_path.read_text())
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    source_hashes = {name: sha256(ROOT / name) for name in cfg['source_files']}
    manifest = []
    frames = []
    raw_root = Path(cfg['raw_root'])
    for month in cfg['months']:
        path = raw_root / f'PREDISPATCHREGIONSUM_{month}.zip'
        if not path.is_file():
            manifest.append({'month': month, 'path': str(path), 'status': 'missing'})
            print(f'{month}: missing source package', flush=True)
            continue
        rows = read_month(path, cfg['regions'])
        manifest.append({
            'month': month, 'path': str(path), 'status': 'read',
            'bytes': path.stat().st_size, 'sha256': sha256(path), 'retained_rows': len(rows),
            'runnos': [int(x) for x in sorted(rows['runno'].unique())],
        })
        frames.append(rows)
        print(f'{month}: {len(rows)} demand metadata rows', flush=True)
    rows = pd.concat(frames, ignore_index=True)
    dates = pd.date_range(f"{cfg['year']}-01-01", f"{cfg['year']}-12-31", freq='D', tz=AEST)
    origins = [int(day.timestamp()) for day in dates]
    records = coverage(rows, origins, cfg['regions'])
    for record in records:
        record['origin_aest'] = datetime.fromtimestamp(record['origin_unix'], AEST).isoformat()
    pd.DataFrame(records).to_csv(output / 'coverage.csv', index=False)
    audit_dates = set(cfg['audit_dates'])
    selected = [r for r in records if r['origin_aest'][:10] in audit_dates]
    write_json(output / 'timing_audit.json', selected)
    write_json(output / 'source_manifest.json', manifest)

    # One bounded directory request, without the old builder's retry/fill logic.
    url = cfg['october_listing_url']
    with urllib.request.urlopen(url, timeout=cfg['resources']['listing_timeout_seconds']) as response:
        listing = response.read()
        listing_status = response.status
        resolved_url = response.url
    (output / 'october_directory.html').write_bytes(listing)
    listing_info = {
        'url': url, 'resolved_url': resolved_url, 'status': listing_status,
        'bytes': len(listing), 'sha256': hashlib.sha256(listing).hexdigest(),
        'has_regionsum_package': b'PREDISPATCHREGIONSUM' in listing.upper(),
        'received_at_utc': datetime.now(timezone.utc).isoformat(),
        'is_historical_publication_receipt': False,
    }
    history = historical_inventory(Path(cfg['historical_root']), cfg['regions'], cfg['year'])
    quantity = {
        **cfg['quantity_review'],
        'review_status': 'UNVERIFIED' if cfg['quantity_review']['unresolved'] else 'VERIFIED_BY_REVIEW',
        'historical_inventory': history,
        'collector_source': {'path': cfg['collector_source'], 'sha256': sha256(Path(cfg['collector_source']))},
        'review_basis': 'source code lineage plus official definitions; equivalence remains unproven',
    }
    write_json(output / 'quantity_contract.json', quantity)

    regional = {}
    for region in cfg['regions']:
        part = [r for r in records if r['region'] == region]
        missing = [r['origin_aest'][:10] for r in part if not r['covered']]
        regional[region] = {
            'planned': len(part), 'covered25': sum(r['covered'] for r in part),
            'coverage25': sum(r['covered'] for r in part) / len(part),
            'target24_covered': sum(r['target24_covered'] for r in part),
            'bridge_only_missing': sum(r['target24_only'] for r in part),
            'missing_dates': missing,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    structural_pass = all(r['coverage25'] >= cfg['minimum_coverage'] for r in regional.values())
    if quantity['unresolved']:
        verdict, route = 'DATA_SEMANTICS_BLOCKED', 'BLOCKED_BEFORE_H_OR_P'
    elif structural_pass:
        verdict, route = 'H_SOURCE_READY', 'H'
    else:
        verdict, route = 'H_COVERAGE_INSUFFICIENT', 'P_REQUIRES_RECEIPTS'
    summary = {
        'stage': cfg['stage'], 'execution_commit': head, 'branch': branch,
        'source_sha256': source_hashes, 'started_at_utc': started_utc,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'verdict': verdict, 'advance_s1': route == 'H',
        'route': route, 'regions': regional,
        'structural_coverage_threshold': cfg['minimum_coverage'],
        'structural_coverage_pass': structural_pass,
        'publication_basis': 'archive_proxy', 'strict_asof_verified': False,
        'quantity_unresolved': quantity['unresolved'], 'october_directory': listing_info,
        'metadata_rows': len(rows), 'audit_units': len(selected),
        'price_scores_computed': 0, 'model_calls': 0, 'training_calls': 0,
        'requested_final_seeds': cfg['downstream_requested_seeds'], 'seeds_executed': [],
        'resources': {
            'wall_seconds': time.monotonic() - started, 'cpu_user_seconds': usage.ru_utime,
            'cpu_system_seconds': usage.ru_stime, 'peak_rss_bytes': usage.ru_maxrss * 1024,
            'local_archive_bytes': sum(m.get('bytes', 0) for m in manifest),
            'archive_gets': 0, 'listing_gets': 1, 'listing_bytes': len(listing),
            'scope': 'this source-audit process only; excludes research-agent and browser tools',
        },
    }
    write_json(output / 'summary.json', summary)
    lines = [
        'R10 S0 来源审计结果', f'执行提交：{head}', f'分支：{branch}',
        f'结论：{verdict}；路线：{route}；S1来源准备成立：{route == "H"}。',
        '未计算价格误差，未调用预测模型或实验LLM；不是新的价格预测负结果。',
        '历史是 PRICE_AND_DEMAND/TRADE 的12个五分钟值均值；候选是 REGION_SOLUTION/INTERVENTION=0 的两个半小时值均值。',
        '语义未解决项：' + ', '.join(quantity['unresolved']),
        '未来收取不能自动修复已有的定义差异。结构覆盖和语义审查分别报告。',
        '', '三地区全年结构覆盖（archive_proxy，不是first_seen）：',
    ]
    for region, r in regional.items():
        lines.append(f"{region}: {r['covered25']}/{r['planned']} = {r['coverage25']:.2%}; "
                     f"目标24h完整={r['target24_covered']}; 仅桥接缺失={r['bridge_only_missing']}")
    lines += [
        f"官方2022-10目录列有REGIONSUM包：{listing_info['has_regionsum_package']}",
        '要求的三地区、三个seed实验尚未执行，不能把已有r9结果或本次来源扫描算作完成。',
        'S0本身不启动预测；后续受上述来源判定约束，不自动填充、改编码、改日期或换来源。',
        f"资源：wall={summary['resources']['wall_seconds']:.3f}s，CPU(user/system)="
        f"{usage.ru_utime:.3f}/{usage.ru_stime:.3f}s，峰值RSS={usage.ru_maxrss * 1024}bytes；"
        f"新增ZIP下载=0，目录GET=1/{len(listing)}bytes。",
    ]
    (output / 'results_cn.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'verdict': summary['verdict'], 'regions': regional,
                      'resources': summary['resources']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
