#!/usr/bin/env python3
"""Audit predeclared historical years from local demand archives; no price scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.agent_forecasting.totaldemand_information import AEST, coverage, read_month


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(config_path):
    start = time.monotonic()
    cfg = json.loads(Path(config_path).read_text())
    output = ROOT / cfg['output_root']
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cache, manifest, results = {}, [], {}
    for year in cfg['years']:
        months = [f'{year - 1}12'] + [f'{year}{month:02d}' for month in range(1, 13)]
        frames = []
        for month in months:
            if month in cache:
                frame = cache[month]
            else:
                path = Path(cfg['raw_root']) / f'PREDISPATCHREGIONSUM_{month}.zip'
                if not path.exists():
                    manifest.append({'month': month, 'path': str(path), 'status': 'missing'})
                    continue
                frame = read_month(path, cfg['regions'])
                manifest.append({'month': month, 'path': str(path), 'status': 'read',
                                 'bytes': path.stat().st_size, 'sha256': sha256(path),
                                 'retained_rows': len(frame)})
                if month.endswith('12'):
                    cache[month] = frame
                print(f'{month}: {len(frame)} demand rows', flush=True)
            frames.append(frame)
        rows = pd.concat(frames, ignore_index=True)
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D', tz=AEST)
        records = coverage(rows, [int(day.timestamp()) for day in dates], cfg['regions'])
        for record in records:
            record['origin_aest'] = pd.Timestamp(record['origin_unix'], unit='s', tz='UTC').tz_convert(AEST).isoformat()
        pd.DataFrame(records).to_csv(output / f'coverage_{year}.csv', index=False)
        regional = {}
        for region in cfg['regions']:
            part = [r for r in records if r['region'] == region]
            regional[region] = {'planned': len(part), 'covered25': sum(r['covered'] for r in part),
                'coverage25': sum(r['covered'] for r in part) / len(part),
                'target24_covered': sum(r['target24_covered'] for r in part),
                'missing_dates': [r['origin_aest'][:10] for r in part if not r['covered']]}
        results[str(year)] = {'regions': regional,
            'structural_pass': all(r['coverage25'] >= cfg['minimum_coverage'] for r in regional.values())}
        print(json.dumps({'year': year, **results[str(year)]}), flush=True)
        del rows, frames
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {'execution_commit': head, 'years': results,
        'source_sha256': {name: sha256(ROOT / name) for name in cfg['source_files']},
        'minimum_coverage': cfg['minimum_coverage'],
        'publication_basis': 'archive_proxy', 'strict_asof_verified': False,
        'quantity_equivalence_verified': False, 'candidate_eligibility_assessed': False,
        'model_calls': 0, 'price_scores': 0, 'network_requests': 0,
        'resources': {'wall_seconds': time.monotonic() - start, 'cpu_user_seconds': usage.ru_utime,
            'cpu_system_seconds': usage.ru_stime, 'peak_rss_bytes': usage.ru_maxrss * 1024,
            'source_bytes': sum(m.get('bytes', 0) for m in manifest),
            'scope': 'this source-audit process only'}}
    write_json(output / 'source_manifest.json', manifest)
    write_json(output / 'summary.json', summary)
    print(json.dumps({'resources': summary['resources']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
