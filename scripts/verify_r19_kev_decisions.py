"""Four prespecified synthetic numerical-evidence checks against live Kev.

This does not estimate forecasting accuracy or use historical evaluation labels.
Only the relative diagnostics change within each pair of cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from forecasting.api_agent.transport import KevClient
    from forecasting.adaptive_agent.evidence_policy import load_skills
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    skills = load_skills()
    instruction = skills['compare_candidates']['instructions']
    client = KevClient(timeout=30)
    default, alternative = 'lightgbm:full:72', 'chronos:price_only:24'
    criteria = {'commit:' + default: 'Submit the existing frozen LightGBM prediction.',
                'commit:' + alternative: 'Submit the existing frozen Chronos prediction.'}
    cases = []
    for support in (8, 4):
        for better in (default, alternative):
            evidence = {}
            for candidate in (default, alternative):
                evidence[candidate] = {
                    'sample_count': support, 'missing_count': 0,
                    'data_roles': ['post_selection_holdout'],
                    'metrics': {'qs': 10 if candidate == better else 50,
                                'mae': 20 if candidate == better else 90,
                                'coverage90': .9, 'width90': 100}}
            state = {'fixture': 'SYNTHETIC ONLY; these are not measured experimental results.',
                     'phase': 'final_selection', 'default_candidate_id': default,
                     'candidate_ids': [default, alternative], 'future_actuals_available': False,
                     'paired_diagnostic': {'diagnostic_kind': 'retrospective_frozen_recompute',
                         'same_episodes_for_both_candidates': True, 'horizon_hours': 24,
                         'support_count': support, 'candidates': evidence},
                     'other_evidence': 'No conflicting numerical evidence or failures; both forecasts valid.'}
            action, record = client.choose(state, instruction, criteria)
            cases.append({'support': support, 'synthetic_better_candidate': better,
                          'selected': action, 'responds_to_supplied_evidence': action == 'commit:' + better,
                          'record': record})
    report = {'kind': 'synthetic_live_api_behavior_check', 'cases': cases,
              'responds_in_all_cases': all(c['responds_to_supplied_evidence'] for c in cases),
              'real_forecasting_gain_tested': False,
              'skill_sha256': skills['compare_candidates']['sha256'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
              'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'responds_in_all_cases': report['responds_in_all_cases'],
                      'cases': [{k: v for k, v in c.items() if k != 'record'} for c in cases]}, indent=2))


if __name__ == '__main__':
    main()
