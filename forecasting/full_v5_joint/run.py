"""Durable full-run supervisor. Launch from a frozen source snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads((root/'launch.json').read_text())
    regions = manifest['regions']
    lock = threading.Lock()
    status = dict(started_at=now(), supervisor_pid=os.getpid(), state='running', jobs={})

    def update(key=None, **values):
        with lock:
            (status['jobs'].setdefault(key, {}) if key else status).update(values)
            status['updated_at'] = now()
            temp = root/'status.json.tmp'
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(root/'status.json')

    def execute(key, module, options):
        log = root/'logs'/f'{key}.log'
        command = [sys.executable, '-u', '-m', module, '--config', args.config, *options]
        update(key, state='starting', started_at=now(), command=command, log=str(log))
        try:
            with log.open('w') as handle:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                update(key, state='running', pid=child.pid)
                code = child.wait()
            update(key, state='complete' if code == 0 else 'failed', returncode=code, ended_at=now())
            return code == 0
        except Exception:
            update(key, state='failed', error=traceback.format_exc(), ended_at=now())
            return False

    def train_region(region):
        bundles = [str(root/'oof'/region/label/'bundle.npz') for label in labels]
        success = execute(f'{region}_joint', 'forecasting.full_v5_joint.train',
                          ['--region', region, '--bundles', *bundles, '--output', str(root/'joint'/region)])
        return region, success

    labels = [str(year) for year in manifest['oof_years']] + ['2022H2']
    pending = {r: len(labels) for r in regions}
    failed = set()
    training = []
    update()
    with ThreadPoolExecutor(max_workers=args.workers) as trees, ThreadPoolExecutor(max_workers=3) as neural:
        futures = {}
        for label in labels:
            for region in regions:
                key = f'{region}_{label}'
                update(key, state='queued')
                options = ['--region', region, '--year', label[:4], '--output', str(root/'oof'/region/label)]
                if label == '2022H2':
                    options.append('--final-half')
                futures[trees.submit(execute, key, 'forecasting.full_v5_joint.prepare', options)] = region
        for future in as_completed(futures):
            region = futures[future]
            if not future.result():
                failed.add(region)
            pending[region] -= 1
            if pending[region] == 0:
                if region in failed:
                    update(f'{region}_joint', state='blocked_by_failed_oof')
                else:
                    training.append(neural.submit(train_region, region))
        for future in as_completed(training):
            region, success = future.result()
            if not success:
                failed.add(region)
    metrics = {}
    for region in regions:
        path = root/'joint'/region/'metrics.json'
        if path.exists():
            metrics[region] = json.loads(path.read_text())
    (root/'summary.json').write_text(json.dumps(dict(complete=len(metrics)==len(regions), regions=metrics), indent=2))
    lines = ['# Joint history full development run', '',
             'Seed2026. Evaluation2022H2; retrospective, not a blind final test.', '',
             '| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |',
             '|---|---:|---:|---:|---:|---:|']
    if len(metrics) == len(regions):
        for model in metrics[regions[0]]:
            mae = [metrics[r][model]['mae'] for r in regions]
            qs = [metrics[r][model]['qs'] for r in regions]
            lines.append('| '+model+' | '+' | '.join(f'{v:.6f}' for v in [*mae, sum(mae)/3, sum(qs)/3])+' |')
    else:
        lines.extend(['', f'Incomplete: failed regions {sorted(failed)}. See status.json and logs.'])
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    update(state='failed' if failed else 'complete', ended_at=now(), failed_regions=sorted(failed))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
