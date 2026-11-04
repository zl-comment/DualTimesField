"""Launch the frozen R20 formal test arm, then audit completed predictions.

No policy/model/skill edits, no retries that change evidence, and no training.
Use --resume after an interrupted invocation; underlying replay verifies identity.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.adaptive_agent.replay import source_identity, write_json

SOURCE_SHA = 'c7708f1d90c4df9c71775fba2a2ecc4cfb0f0fc8ce63b6a324ef65d1d35fe8ef'
OUTPUT = Path('/root/workspace/Code/DualTimesField/outputs/forecasting/r20_formal_test_seed2026_20261104')
LIVE_PYTHON = '/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python'
AUDIT_PYTHON = '/root/workspace/Code/DualTimesField/.venv/bin/python'
STATUS = ROOT / 'logs/forecasting/r20_formal_test_seed2026_20261104.launch.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    command = [LIVE_PYTHON, '-u', '-m', 'forecasting.adaptive_agent.replay', '--execute',
               '--policy', 'r20_iterative_input', '--split', 'test', '--origins', 'all',
               '--regions', 'NSW1', 'QLD1', 'TAS1', '--seed', '2026', '--output', str(OUTPUT)]
    if args.resume:
        command.append('--resume')
    if source_identity()['sha256'] != SOURCE_SHA:
        raise ValueError('R20 executable source/skills differ from the frozen development version')
    if args.plan_only:
        print(json.dumps({'status': 'PLAN_ONLY', 'command': command, 'expected_origins': 52563,
                          'source_sha256': SOURCE_SHA, 'output': str(OUTPUT)}, indent=2))
        return 0
    if OUTPUT.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit --resume; never overwrite an experiment')
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    state = {'status': 'PREDICTING', 'command': command, 'pid': os.getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'source_sha256': SOURCE_SHA, 'output': str(OUTPUT), 'expected_origins': 52563,
             'resume': args.resume,
             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()}
    write_json(STATUS, state)
    try:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise RuntimeError('Prediction process exited with code ' + str(result.returncode))
        report = json.loads((OUTPUT / 'report.json').read_text())
        if report['status'] != 'COMPLETE' or report['origins'] != 52563 or report['expected_origins'] != 52563:
            raise ValueError('Formal run did not complete the entire fixed denominator')
        state['status'] = 'AUDITING'
        write_json(STATUS, state)
        audit_path = OUTPUT / ('execution_audit_' + str(time.time_ns()) + '.json')
        audit = subprocess.run([AUDIT_PYTHON, str(ROOT / 'scripts/audit_r20_replay.py'),
                                '--run', str(OUTPUT), '--output', str(audit_path)], cwd=ROOT)
        if audit.returncode:
            raise RuntimeError('Execution audit exited with code ' + str(audit.returncode))
        state.update(status='COMPLETE', completed_origins=52563, execution_audit=str(audit_path),
                     report=str(OUTPUT / 'report.json'), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(STATUS, state)
    except BaseException as exc:
        state.update(status='FAILED', error=type(exc).__name__ + ': ' + str(exc),
                     failed_utc=datetime.now(timezone.utc).isoformat())
        write_json(STATUS, state)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
