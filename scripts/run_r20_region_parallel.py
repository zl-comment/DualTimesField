"""Run three unchanged R20 regional replays concurrently, each in time order."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.adaptive_agent.replay import (
    BASELINE_ROOT, digest, select_indices, sha_file, source_identity, write_json,
)
import numpy as np

REGIONS = ('NSW1', 'QLD1', 'TAS1')
SOURCE_SHA = 'c7708f1d90c4df9c71775fba2a2ecc4cfb0f0fc8ce63b6a324ef65d1d35fe8ef'
LIVE_PYTHON = '/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python'
AUDIT_PYTHON = '/root/workspace/Code/DualTimesField/.venv/bin/python'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def plan(output, split, origins):
    if source_identity()['sha256'] != SOURCE_SHA:
        raise ValueError('Frozen R20 source or skills changed')
    counts = {}
    for region in REGIONS:
        axis = np.load(BASELINE_ROOT / 'data_cache' / region / split / 'origin_unix.npy', mmap_mode='r')
        counts[region] = len(select_indices(len(axis), origins))
    commands = {region: [LIVE_PYTHON, '-u', '-m', 'forecasting.adaptive_agent.replay',
                        '--execute', '--policy', 'r20_iterative_input', '--split', split,
                        '--origins', origins, '--regions', region, '--seed', '2026',
                        '--output', str(output / region)] for region in REGIONS}
    return {'schema': 'r20-independent-regional-replays-v1', 'split': split, 'seed': 2026,
            'origins_per_region': origins, 'region_counts': counts,
            'expected_origins': sum(counts.values()), 'commands': commands,
            'source_sha256': SOURCE_SHA, 'launcher_sha256': sha_file(__file__),
            'execution': 'Three processes; strictly chronological within each region; isolated ledgers.',
            'scoring': 'Each region scores after all its predictions; aggregate only after all three audits pass.',
            'lineage': 'd0446604abde9fd4ead850f009c98dcacdfae3c1; frozen R20; no predictor or skill changes.'}


def combine_reports(reports, counts):
    """Never average percentiles or accept partial/mismatched regional reports."""
    if set(reports) != set(REGIONS) or set(counts) != set(REGIONS):
        raise ValueError('Exactly three regional reports are required')
    regions = {}
    for region in REGIONS:
        report = reports[region]
        if (report.get('status') != 'COMPLETE'
                or report.get('origins') != counts[region]
                or report.get('expected_origins') != counts[region]
                or set(report.get('regions', {})) != {region}
                or report['regions'][region]['count'] != counts[region]
                or report.get('policy') != 'r20_iterative_input'):
            raise ValueError('Incomplete or mismatched regional report: ' + region)
        regions[region] = report['regions'][region]
    summary = {'status': 'COMPLETE', 'origins': sum(counts.values()),
               'expected_origins': sum(counts.values()), 'regions': regions,
               'region_equal_mean': {}, 'policy': 'r20_iterative_input',
               'selected_candidates': {}, 'region_manifest_ids': {
                   r: reports[r]['manifest_id'] for r in REGIONS}}
    for kind in ('BASE', 'AGENT'):
        keys = set(regions[REGIONS[0]][kind])
        if any(set(regions[r][kind]) != keys for r in REGIONS):
            raise ValueError('Regional metric schema differs')
        summary['region_equal_mean'][kind] = {
            key: float(np.mean([regions[r][kind][key] for r in REGIONS])) for key in sorted(keys)}
    for key in ('api_calls', 'api_failure_count', 'fallback_count', 'failed_origins',
                'prediction_differs_from_base', 'revision_count', 'origins_with_revision'):
        summary[key] = sum(reports[r][key] for r in REGIONS)
    for report in reports.values():
        for name, count in report['selected_candidates'].items():
            summary['selected_candidates'][name] = summary['selected_candidates'].get(name, 0) + count
    return summary


def stop_children(children):
    for child in children.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for child in children.values():
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('selection', 'test'), default='test')
    parser.add_argument('--origins', default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = plan(output, args.split, args.origins)
    if args.plan_only:
        print(json.dumps(frozen, indent=2))
        return 0
    if args.resume:
        if json.loads((output / 'parallel_manifest.json').read_text()) != frozen:
            raise ValueError('Parallel resume configuration/source/launcher changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'parallel_manifest.json', frozen)
    with (output / 'parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children, handles = {}, []
        started = time.monotonic()
        state = {'status': 'STARTING', 'started_utc': timestamp(), 'pid': os.getpid(),
                 'expected_origins': frozen['expected_origins'], 'resume': args.resume,
                 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                 'worktree': str(ROOT)}
        def publish(status):
            state.update(status=status, updated_utc=timestamp(),
                         elapsed_this_invocation_seconds=time.monotonic() - started)
            write_json(output / 'progress.json', state)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Supervisor received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        try:
            publish('STARTING')
            for region, command in frozen['commands'].items():
                if args.resume and (output / region).exists():
                    command = command + ['--resume']
                handle = (output / (region + '.log')).open('a')
                handles.append(handle)
                children[region] = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
            state['worker_pids'] = {r: p.pid for r, p in children.items()}
            while True:
                progress = {}
                for region in REGIONS:
                    path = output / region / 'progress.json'
                    progress[region] = json.loads(path.read_text()) if path.exists() else {'status': 'STARTING', 'completed': 0}
                state['regions'] = progress
                state['completed'] = sum(p.get('completed', 0) for p in progress.values())
                failures = {r: p.returncode for r, p in children.items() if p.poll() not in (None, 0)}
                if failures:
                    raise RuntimeError('Regional workers failed: ' + json.dumps(failures))
                publish('PREDICTING')
                if all(p.poll() == 0 for p in children.values()):
                    break
                time.sleep(5)
            publish('AUDITING')
            reports, audits, common_identity = {}, {}, None
            for region in REGIONS:
                directory = output / region
                manifest = json.loads((directory / 'run_manifest.json').read_text())
                if (manifest['source']['sha256'] != SOURCE_SHA
                        or manifest['configuration']['regions'] != [region]
                        or manifest['configuration']['split'] != args.split
                        or manifest['configuration']['seed'] != 2026):
                    raise ValueError('Regional frozen configuration mismatch: ' + region)
                identity = {key: manifest[key] for key in ('release', 'source', 'skills', 'budgets', 'evidence', 'packages')}
                identity['shared_models'] = {key: manifest['models'][key] for key in ('kev', 'chronos')}
                if common_identity is not None and identity != common_identity:
                    raise ValueError('Shared policy/model/runtime differs between regions')
                common_identity = identity
                # Keep the audit in its own log and never replace an earlier audit.
                audit = directory / ('execution_audit_' + str(time.time_ns()) + '.json')
                with (output / (region + '.audit.log')).open('a') as handle:
                    subprocess.run([AUDIT_PYTHON, str(ROOT / 'scripts/audit_r20_replay.py'),
                                    '--run', str(directory), '--output', str(audit)], cwd=ROOT,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                audits[region] = str(audit)
                reports[region] = json.loads((directory / 'report.json').read_text())
            summary = combine_reports(reports, frozen['region_counts'])
            if source_identity()['sha256'] != SOURCE_SHA or sha_file(__file__) != frozen['launcher_sha256']:
                raise ValueError('Frozen source/launcher changed during execution')
            summary.update(parallel_manifest_id=digest(frozen), execution_audits=audits,
                           finished_utc=timestamp(),
                           elapsed_this_invocation_seconds=time.monotonic() - started)
            write_json(output / 'report.json', summary)
            publish('COMPLETE')
            return 0
        except BaseException as exc:
            stop_children(children)
            state['error'] = type(exc).__name__ + ': ' + str(exc)
            publish('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED')
            raise
        finally:
            for handle in handles:
                handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
