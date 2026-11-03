"""User-requested interim analysis of the completed seed, without editing R17.

This explicitly supersedes waiting for nine cells for this interim review only.
All three seed2026 cells must be complete and verified before opening labels.
"""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from forecasting.api_agent.formal_r17 import (ARMS, CACHE, MAIN, METRICS, ROOT,
    load_region, paired_block_interval, per_origin_qs, require_complete, verify_hashes, write_json)
from forecasting.api_agent.pilot import sha_file
from forecasting.api_agent.workflow import digest
from forecasting.aligned_scoring import make_prediction_artifact, score_predictions, QUANTILE_LEVELS


def main():
    run = ROOT / 'outputs/forecasting/kev_r17_formal_20261103'
    out = ROOT / 'logs/forecasting/kev_r17_seed2026_review_20261103'
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / 'manifest.json').read_text())
    cfg = dict(manifest['config'], seeds=[2026])
    shards = require_complete(run, cfg, manifest['source_fingerprint'])
    verify_hashes(manifest['source_hashes'], ROOT)
    verify_hashes(manifest['input_hashes'])
    verify_hashes({MAIN / 'main.csv': cfg['main_table_sha256'],
                   MAIN / 'verification.json': cfg['verification_sha256']})
    rows, behavior, deltas, region_reports, stability = [], [], [], {}, {}
    for (_, region), directories in shards.items():
        meta, arrays = load_region(region, cfg)
        parts = {key: [] for key in (*ARMS, 'origin_unix', 'target_time_unix',
                                     *(a + '_profiles' for a in ARMS))}
        tools, stops, final_actions = Counter(), Counter(), Counter()
        calls, fallbacks, changed, tokens = 0, 0, 0, 0
        for directory in directories:
            with np.load(directory / 'predictions.npz', allow_pickle=False) as z:
                for key in parts:
                    parts[key].append(z[key])
            with gzip.open(directory / 'decisions.jsonl.gz', 'rt') as stream:
                for line in stream:
                    trace = json.loads(line)
                    tools.update(t['tool'] for t in trace['tool_calls'])
                    stops[trace['stop_reason']] += 1
                    calls += len(trace['calls'])
                    fallbacks += trace['fallback'] is not None
                    changed += trace['selected_profile'] != trace['fixed_choice']
                    for call in trace['calls']:
                        response = call.get('response', {})
                        tokens += response.get('usage', {}).get('input_tokens', 0)
                    if trace['calls']:
                        answer = trace['calls'][-1].get('response', {}).get('answers', {}).get('action', {})
                        final_actions[answer.get('choice', 'request_failed')] += 1
        data = {k: np.concatenate(v) for k, v in parts.items()}
        for key in ('origin_unix', 'target_time_unix'):
            if not np.array_equal(data[key], arrays[key]):
                raise ValueError('Saved axes do not match formal cache')
        original = next(r for r in manifest['original_baselines'] if r['seed'] == 2026 and r['region'] == region)
        verify_hashes({original['path']: original['sha256']})
        with np.load(original['path'], allow_pickle=False) as z:
            if not np.array_equal(data['BASE'], z['quantiles']):
                raise ValueError('BASE does not exactly match original artifact')
        label = CACHE / region / 'test/actual.npy'
        actual = np.load(label, allow_pickle=False)
        scores = {}
        for arm in ARMS:
            metadata = dict(meta, input_hash=digest({'common_input_hash': meta['input_hash'],
                'profiles': data[arm + '_profiles'].tolist()}), base_input_hash=meta['input_hash'],
                seed=2026, source='r17_user_requested_interim:' + arm,
                output_type='quantile', prediction_space='raw_rrp')
            artifact = make_prediction_artifact(actual=actual, quantiles=data[arm],
                origin_unix=data['origin_unix'], target_time_unix=data['target_time_unix'],
                quantile_levels=QUANTILE_LEVELS, metadata=metadata)
            scores[arm] = {k: v for k, v in score_predictions(artifact).items() if k in METRICS}
            rows.append({'region': region, 'seed': 2026, 'arm': arm, **scores[arm]})
        delta = per_origin_qs(actual, data['AGENT']) - per_origin_qs(actual, data['FIXED'])
        deltas.append(delta)
        region_reports[region] = {
            'scores': scores, 'paired_uncertainty': paired_block_interval(delta),
            'agent_vs_fixed': {'wins': int((delta < 0).sum()), 'ties': int((delta == 0).sum()),
                               'losses': int((delta > 0).sum())},
            'profile_counts': {a: dict(Counter(data[a + '_profiles'].tolist())) for a in ARMS},
            'label_sha256': sha_file(label)}
        behavior.append({'region': region, 'api_calls': calls, 'fallbacks': fallbacks,
                         'changed_from_fixed': changed, 'input_tokens': tokens,
                         'tools': dict(tools), 'stops': dict(stops), 'last_actions': dict(final_actions)})
        # No scoring of the incomplete second seed; compare saved predictions only.
        partial = {'origins': 0, 'equal_prediction_origins': {a: 0 for a in ARMS}}
        for path in sorted((run / 'seed2027' / region).glob('*/complete.json')):
            record = json.loads(path.read_text())
            verify_hashes(record['files'], path.parent)
            start, stop = record['start'], record['stop']
            with np.load(path.parent / 'predictions.npz', allow_pickle=False) as z:
                partial['origins'] += stop - start
                for arm in ARMS:
                    partial['equal_prediction_origins'][arm] += int(
                        np.all(z[arm] == data[arm][start:stop], axis=(1, 2)).sum())
        stability[region] = partial
        print(json.dumps({'region': region, 'scores': scores, 'behavior': behavior[-1]}), flush=True)
    summary = {arm: {m: float(np.mean([r[m] for r in rows if r['arm'] == arm])) for m in METRICS} for arm in ARMS}
    qs = {a: summary[a]['quantile_score_proxy'] for a in ARMS}
    report = {'status': 'SEED2026_INTERIM_REVIEW_COMPLETE',
              'scope': 'Full 2023-2024 grid, 3 regions, seed2026 only, 52563 origins; NOT a completed three-seed experiment.',
              'authorization': 'User requested early seed2026 scoring to decide whether to stop later seeds.',
              'source_fingerprint': manifest['source_fingerprint'],
              'analysis_script_sha256': sha_file(Path(__file__)),
              'summary': summary, 'regions': region_reports, 'behavior': behavior,
              'qs_gain_vs_base_percent': 100 * (qs['BASE'] - qs['AGENT']) / qs['BASE'],
              'qs_gain_vs_fixed_percent': 100 * (qs['FIXED'] - qs['AGENT']) / qs['FIXED'],
              'paired_uncertainty': paired_block_interval(np.mean(deltas, axis=0)),
              'partial_seed2027_prediction_agreement': stability,
              'limitations': ['User-requested early look; no three-seed claim.',
                              'Any subsequent tuning uses an already exposed historical test set.',
                              'Kev argmax has no sampling seed; repeated runs are not independent trained models.']}
    write_json(out / 'report.json', report)
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ('summary', 'qs_gain_vs_base_percent',
          'qs_gain_vs_fixed_percent', 'paired_uncertainty', 'partial_seed2027_prediction_agreement')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
