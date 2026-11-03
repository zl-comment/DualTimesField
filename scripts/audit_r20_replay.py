"""Audit a completed replay from persisted predictions/traces only; never read labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_r20_replays import read_arm
from forecasting.adaptive_agent.replay import digest, sha_file


def audit_origin(manifest, row):
    record, trace, directory = row['record'], row['trace'], row['directory']
    policy = manifest['configuration'].get('policy', 'r18')
    findings, notes = [], []
    def check(ok, rule, detail=''):
        if not ok:
            findings.append({'rule': rule, 'detail': detail})
    budgets = manifest['budgets']
    costs = trace.get('costs', {})
    for key in ('api_calls', 'tool_calls', 'candidates', 'episodes', 'historical_evaluations'):
        value = costs.get(key)
        check(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= budgets[key],
              'budget:' + key, str(value))
    tools, calls = trace.get('tools', []), trace.get('calls', [])
    check(costs.get('api_calls') == len(calls), 'api_call_accounting')
    check(costs.get('tool_calls') == len(tools), 'tool_call_accounting')
    check(trace.get('context_id') == record['context_id'], 'trace_context')
    final_state = trace.get('final_state', {})
    check(final_state.get('costs') == costs, 'final_state_costs')
    candidate_ids = final_state.get('candidate_ids', [])
    check(len(candidate_ids) == len(set(candidate_ids)) <= costs.get('candidates', -1), 'candidate_accounting')
    if 'materialized_candidate_count' in trace:
        check(trace['materialized_candidate_count'] == len(candidate_ids), 'materialized_candidate_count')
    if len(candidate_ids) != costs.get('candidates'):
        check(bool(trace.get('fallback')), 'unmaterialized_candidate_reservation_requires_fallback')
        notes.append({'rule': 'candidate_budget_includes_reserved_or_failed_materialization',
                      'charged': costs.get('candidates'), 'materialized': len(candidate_ids)})
    check(trace.get('selected_candidate_id') == record['agent']['candidate_id'], 'selected_candidate_id')
    check(record['agent']['candidate_id'] in candidate_ids, 'selected_candidate_materialized')
    api_errors, max_bytes = 0, 0
    for i, call in enumerate(calls):
        response = call.get('record', call)
        api_errors += 'error' in response
        request = response.get('request')
        check(isinstance(request, dict), 'saved_api_request', str(i))
        if not isinstance(request, dict):
            continue
        size = len(json.dumps(request, allow_nan=False).encode())
        max_bytes = max(max_bytes, size)
        check(size <= 24000, 'api_request_bytes', f'call={i};bytes={size}')
        if 'request_bytes' in call:
            check(size == call['request_bytes'], 'recorded_request_bytes', str(i))
        state = request.get('state', {})
        check(state.get('context_id') == record['context_id'], 'api_context', str(i))
        check(state.get('region') == record['region'], 'api_region', str(i))
        check(state.get('origin_unix') == record['origin_unix'] == state.get('cutoff'), 'api_cutoff', str(i))
        check(state.get('memory_watermark', record['origin_unix'] + 1) <= record['origin_unix'],
              'api_memory_watermark', str(i))
        if 'action' in call:
            check(call['action'] in request.get('questions', {}).get('action', {}).get('criteria', {}),
                  'api_action_offered', str(i))
    historical_charged, episode_charged, tool_errors, estimated_events = 0, 0, 0, 0
    estimated_costs = Counter()
    diagnostic_tools = []
    for i, tool in enumerate(tools):
        result = tool.get('result', {})
        actual = tool.get('charged_costs', result.get('costs', {}))
        reserved = tool.get('reserved_costs', {})
        before, after = tool.get('costs_before', {}), tool.get('costs_after', {})
        tool_errors += tool.get('status') in ('error', 'deadline_exceeded_after_result') or 'error' in tool
        if tool.get('charged_costs_basis') == 'estimated_upper_bound':
            estimated_events += 1
            estimated_costs.update(actual)
            check(tool.get('actual_costs_status') == 'unknown_no_tool_result', 'unknown_actual_cost_disclosed')
            check(actual == reserved, 'exception_reservation_upper_bound')
        for key in ('candidates', 'episodes', 'historical_evaluations'):
            check(0 <= actual.get(key, 0) <= reserved.get(key, 0), 'tool_reservation:' + key, str(i))
        if after:
            for key in ('api_calls', 'tool_calls', 'candidates', 'episodes', 'historical_evaluations'):
                check(before.get(key, 0) <= after.get(key, 0) <= costs.get(key, -1),
                      'tool_monotonic_cost:' + key, str(i))
            historical_charged += actual.get('historical_evaluations', 0)
            episode_charged += actual.get('episodes', 0)
        elif result:
            notes.append({'rule': 'tool_result_without_accounting_completion', 'tool': tool.get('action')})
        if tool.get('action', '').startswith('diagnose:'):
            diagnostic_tools.append((i, tool))
    check(historical_charged == costs.get('historical_evaluations'), 'historical_charge_accounting')
    check(episode_charged == costs.get('episodes'), 'episode_charge_accounting')
    if 'resource_accounting' in trace:
        check(estimated_events == len(trace['resource_accounting'].get('unknown_actual_cost_events', [])),
              'unknown_cost_event_count')

    # Model arrays for every materialized R20 candidate are committed alongside
    # the selected/default arrays. Earlier R19 bundles intentionally lack them.
    materialized = {}
    archive = directory / 'materialized_candidates.npz'
    if archive.exists():
        with np.load(archive, allow_pickle=False) as data:
            check(set(data.files) == {'candidate_ids', 'forecast_ids', 'quantiles', 'target_time_unix'},
                  'materialized_archive_schema')
            ids, identities, q = data['candidate_ids'].tolist(), data['forecast_ids'].tolist(), data['quantiles']
            check(len(ids) == len(set(ids)) == len(identities), 'materialized_ids_unique')
            check(q.shape == (len(ids), 24, 5) and np.isfinite(q).all(), 'materialized_shape_finite')
            check(not np.any(np.diff(q, axis=-1) < 0), 'materialized_quantile_order')
            check(np.array_equal(data['target_time_unix'], row['arrays']['target_time_unix']), 'materialized_target_axis')
            materialized = {key: (q[i].copy(), identities[i]) for i, key in enumerate(ids)}
        check(set(candidate_ids).issubset(materialized), 'all_state_candidates_archived')
        if set(materialized) - set(candidate_ids):
            notes.append({'rule': 'generated_but_unregistered_candidates',
                          'candidate_ids': sorted(set(materialized) - set(candidate_ids))})
        for kind in ('base', 'agent'):
            key = record[kind]['candidate_id']
            check(key in materialized, 'committed_candidate_in_archive', kind)
            if key in materialized:
                q, identity = materialized[key]
                check(np.array_equal(q, row['arrays'][kind]), 'selected_array_equals_materialized', kind)
                check(identity == record[kind]['identity'], 'selected_identity_equals_materialized', kind)
        for tool in tools:
            action = tool.get('action', '')
            if not action.startswith('pool:') or not tool.get('result'):
                continue
            a, b, weight = action[len('pool:'):].split('|')
            made = tool['result'].get('candidate_ids', [])
            check(float(weight) in (.25, .5, .75), 'pool_registered_weight')
            check(len(made) == 1 and a in materialized and b in materialized and made[0] in materialized,
                  'pool_sources_materialized')
            if len(made) == 1 and all(k in materialized for k in (a, b, made[0])):
                expected = float(weight) * materialized[a][0] + (1-float(weight)) * materialized[b][0]
                check(np.array_equal(materialized[made[0]][0], expected), 'pool_numeric_recipe')
    elif policy == 'r20_iterative_input':
        check(False, 'r20_materialized_archive_missing')
    else:
        notes.append({'rule': 'legacy_bundle_only_default_and_selected_arrays'})

    audit = json.loads((directory / 'audit.json').read_text())
    for event in audit.get('outcome_reads', []):
        check(event['available_at'] <= event['cutoff'] <= record['origin_unix'],
              'broker_outcome_read_maturity', str(event))
    full_diagnostics = [entry for entry in audit.get('tool_audit', []) if 'complete_diagnostic' in entry]
    totals = Counter()
    previous_units, previous_numeric = 0, 0
    cohort, cohort_ids = None, None
    for entry in full_diagnostics:
        full = entry['complete_diagnostic']
        totals['retrospective_logical_units'] += full.get('logical_inference_units', 0)
        totals['retrospective_numeric_inferences'] += full.get('numeric_inferences', 0)
        totals['cache_hits'] += full.get('cache_hit_count', 0)
        totals['cache_misses'] += full.get('cache_miss_count', 0)
        if full.get('diagnostic_kind') != 'paired_retrospective_frozen_recompute':
            continue
        current_cohort, episodes = full.get('cohort_id'), full.get('episodes', [])
        if cohort is None:
            cohort, cohort_ids = current_cohort, episodes
        check(current_cohort == cohort and episodes == cohort_ids and full.get('cohort_locked'), 'fixed_diagnostic_cohort')
        units, numeric = full['logical_inference_units'], full['numeric_inferences']
        check(full['cumulative_logical_inference_units'] == previous_units + units, 'cumulative_diagnostic_units')
        check(full['cumulative_numeric_inferences'] == previous_numeric + numeric, 'cumulative_numeric_inference')
        previous_units, previous_numeric = full['cumulative_logical_inference_units'], full['cumulative_numeric_inferences']
        jobs = len(episodes) * len(full['candidates'])
        check(full['cache_hit_count'] + full['cache_miss_count'] == jobs, 'cache_total_pairs')
        check(0 <= numeric <= units <= full['cache_miss_count'], 'incremental_inference_bounds')
        check(full['cache_hit_count'] + units + full.get('unattempted_count', 0) == jobs, 'attempt_cache_accounting')
        rows = full.get('rows', [])
        pairs = [(r['candidate_id'], r['episode_id']) for r in rows]
        check(len(pairs) == len(set(pairs)) == full['cache_hit_count'] + units, 'cached_row_accounting')
        success = {(r['candidate_id'], r['episode_id']) for r in rows if r.get('status') == 'success'}
        paired = [episode for episode in episodes if all((candidate, episode) in success for candidate in full['candidates'])]
        check(full.get('paired_episode_ids') == paired and full.get('sample_count') == len(paired), 'common_successful_cohort')
        for value in full['candidates'].values():
            check(value.get('paired_episode_ids') == paired and value.get('sample_count') == len(paired),
                  'candidate_uses_common_cohort')
        check(bool(full.get('failures')) == any(r.get('status') == 'failure' for r in rows), 'cached_failure_disclosure')
    check(totals['retrospective_logical_units'] <= costs.get('historical_evaluations', 0),
          'retrospective_units_in_total_budget')

    iterations, revisions = trace.get('iterations', []), trace.get('revisions', [])
    verified_revisions, incomplete_revisions = 0, 0
    for index, iteration in enumerate(iterations):
        check(iteration.get('index') == index, 'iteration_index')
        check(iteration.get('cohort_id') == cohort and iteration.get('cohort_ids') == cohort_ids, 'iteration_fixed_cohort')
        matches = [tool for _, tool in diagnostic_tools if tool['action'] == iteration.get('diagnostic_action')]
        check(len(matches) == 1, 'iteration_diagnostic_exists')
        if len(matches) == 1:
            observation = matches[0].get('result', {}).get('observation', {})
            check(set(iteration['candidate_ids']) == set(observation.get('candidates', {})), 'iteration_candidate_set')
            check(iteration['incremental_historical_units'] == observation.get('logical_inference_units'), 'iteration_unit_count')
    for revision in revisions:
        positions = [i for i, tool in enumerate(tools) if tool.get('action') == revision.get('action')]
        check(len(positions) == 1, 'revision_tool_exists')
        if len(positions) != 1:
            continue
        i = positions[0]
        tool = tools[i]
        new_id = revision['new_candidate_id']
        check(tool.get('execution_mode') == 'llm_selected' and tool.get('protocol_required') is False, 'optional_revision_attribution')
        check(new_id in tool.get('result', {}).get('candidate_ids', []), 'revision_materialized')
        next_tool = tools[i+1] if i+1 < len(tools) else {}
        observation = next_tool.get('result', {}).get('observation', {})
        successful = (next_tool.get('action', '').startswith('diagnose:')
                      and observation.get('cohort_id') == cohort
                      and observation.get('episodes') == cohort_ids
                      and set(observation.get('candidates', {})) == set(revision['from_candidate_ids'] + [new_id]))
        if successful:
            verified_revisions += 1
        else:
            incomplete_revisions += 1
            safe_default = bool(trace.get('fallback')) and record['agent']['identity'] == record['base']['identity']
            check(safe_default, 'revision_requires_same_cohort_diagnosis_or_default_fallback', new_id)
            notes.append({'rule': 'incomplete_revision_reverted_to_default', 'candidate_id': new_id})
    if trace.get('fallback'):
        check(record['agent']['identity'] == record['base']['identity'], 'fallback_returns_default')
    return {'region': record['region'], 'origin_unix': record['origin_unix'],
            'trace_path': str(directory / 'trace.json'), 'violations': findings, 'notes': notes,
            'costs': costs, 'max_api_request_bytes': max_bytes, 'api_errors': api_errors,
            'tool_errors': tool_errors, 'fallback': trace.get('fallback'),
            'iterations': len(iterations), 'revisions': len(revisions),
            'verified_same_cohort_revisions': verified_revisions, 'incomplete_revisions': incomplete_revisions,
            'materialized_candidate_count': len(materialized),
            'unknown_actual_cost_events': estimated_events,
            'estimated_upper_bound_costs': dict(estimated_costs), **dict(totals)}


def audit(path):
    manifest, report, rows = read_arm(Path(path))
    audited = [audit_origin(manifest, row) for row in rows.values()]
    violations = sum(len(row['violations']) for row in audited)
    counts = {key: sum(row.get(key, 0) for row in audited) for key in (
        'api_errors', 'tool_errors', 'iterations', 'revisions', 'verified_same_cohort_revisions',
        'incomplete_revisions', 'materialized_candidate_count', 'retrospective_logical_units',
        'retrospective_numeric_inferences', 'cache_hits', 'cache_misses', 'unknown_actual_cost_events')}
    counts['estimated_upper_bound_costs'] = dict(sum(
        (Counter(row['estimated_upper_bound_costs']) for row in audited), Counter()))
    fallback = sum(bool(row['fallback']) for row in audited)
    report_checks = {'fallback_count': fallback == report['fallback_count'],
                     'api_failure_count': counts['api_errors'] == report['api_failure_count'],
                     'api_calls': sum(row['costs']['api_calls'] for row in audited) == report['api_calls']}
    for report_key, count_key in (('iteration_count', 'iterations'), ('revision_count', 'revisions')):
        if report_key in report:
            report_checks[report_key] = report[report_key] == counts[count_key]
    if not all(report_checks.values()):
        violations += sum(not value for value in report_checks.values())
    return {'status': 'PASS' if not violations else 'FAIL', 'path': str(Path(path).resolve()),
            'manifest_id': digest(manifest), 'policy': manifest['configuration'].get('policy', 'r18'),
            'origins': len(audited), 'violation_count': violations, 'report_checks': report_checks,
            'max_api_request_bytes': max(row['max_api_request_bytes'] for row in audited),
            'fallback_count': fallback, 'totals': counts, 'per_origin': audited,
            'labels_read': False, 'audit_script_sha256': sha_file(__file__),
            'limitations': ['Audit verifies persisted execution evidence and accounting, not predictive benefit.',
                           'Reserved upper bounds after missing tool results are separated from observed retrospective inference counts.',
                           'Legacy bundles do not contain every materialized candidate.',
                           'Error paths are reported; safe default fallback does not imply the failed revision completed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_origin'}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
