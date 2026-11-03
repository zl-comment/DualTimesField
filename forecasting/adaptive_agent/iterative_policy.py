"""R20 optional input revision on one fixed, fully mature historical cohort.

The first comparison follows R19. After seeing diagnostics Kev may commit,
pool, inspect additional evidence, or request another registered Chronos input
configuration. Every newly generated model is diagnosed on the same cohort
before it becomes selectable. Historical inference uses the host's incremental
cache. Revision is optional; the protocol does not force a different forecast.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
from pathlib import Path
import time

from forecasting.agent_forecasting.execution import call_with_deadline

from .contracts import BudgetLimits
from .controller import SKILL_IDS, clone, _costs, _request_state
from .review import validate_artifact

RELEASE = 'r20-iterative-input-agent-v1'
SKILL_ROOT = Path(__file__).parent / 'skills_r20'


def compact_request_state(context, state, limits):
    """Keep current decision evidence; full observations remain in the trace.

    Only superseded repeated observation versions and per-episode diagnostic
    rows are omitted. The latest aggregate diagnostic, including failed rows'
    summaries/support, remains intact rather than silently trimmed to fit.
    """
    request = _request_state(context, state, limits)
    latest = {}
    for observation in state['observations']:
        action = observation['action']
        family = ('paired_diagnostic' if action.startswith('diagnose:') or action == 'paired_diagnostics'
                  else 'analog_cohort' if action.startswith('retrieve_analogs:')
                  else 'review_proposal' if action == 'review_proposal'
                  else action)
        current = clone(observation)
        if family == 'paired_diagnostic' and isinstance(current.get('observation'), dict):
            current['observation'].pop('rows', None)
        latest[family] = current
    request['observations'] = list(latest.values())
    request['action_history'] = list(state['executed_actions'])
    request['compression'] = {'policy': 'latest_observation_per_family',
                              'full_observation_count': len(state['observations']),
                              'sent_observation_count': len(latest),
                              'complete_observations_retained_in_trace': True,
                              'latest_aggregate_diagnostic_retained': True}
    return request


def load_skills():
    result = {}
    for name in SKILL_IDS:
        files = {filename: (SKILL_ROOT / name / filename).read_text()
                 for filename in ('instructions.txt', 'schema.json', 'examples.jsonl', 'version')}
        schema = json.loads(files['schema.json'])
        if schema.get('skill_id') != name or files['version'].strip() != '3.0.0':
            raise ValueError('R20 requires its registered version 3.0.0 skills')
        for line in files['examples.jsonl'].splitlines():
            if line.strip():
                json.loads(line)
        result[name] = {'version': files['version'].strip(), 'instructions': files['instructions.txt'],
                        'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    return result


def run(context, host, client, limits=None):
    limits = limits or BudgetLimits()
    started = time.monotonic()
    deadline = started + limits.origin_deadline_seconds
    skills = load_skills()
    default = validate_artifact(context, call_with_deadline(
        lambda: host.default(context), deadline - time.monotonic()))
    state = {'default_candidate_id': default.candidate_id, 'candidate_ids': [default.candidate_id],
             'observations': [], 'executed_actions': [], 'reviewed_candidate_ids': [],
             'costs': {'api_calls': 0, 'tool_calls': 0, 'candidates': 1,
                       'episodes': 0, 'historical_evaluations': 0},
             'deadline_monotonic': deadline, 'remaining_seconds': deadline - time.monotonic(),
             'phase': 'initial_state'}
    trace = {'release': RELEASE, 'context_id': context.identity(),
             'default_candidate_id': default.candidate_id,
             'skills': {k: {f: v[f] for f in ('version', 'sha256')} for k, v in skills.items()},
             'protocol': {'id': 'fixed_cohort_iterative_input_comparison_v1',
                          'mandatory_evidence_acquisition': True,
                          'mandatory_alternative_forecast': True,
                          'paired_historical_comparison_when_supported': True,
                          'optional_input_revision_after_comparison': True,
                          'cohort_policy': 'first_retrieval_fixed_for_this_origin',
                          'no_mature_cohort_revision': 'disabled_requires_paired_comparison',
                          'external_event_route': 'not_used_in_this_numerical_policy',
                          'default_remains_legal': True,
                          'tool_coverage_is_not_llm_gain': True},
             'calls': [], 'tools': [], 'reviews': [], 'fallback': None,
             'iterations': [], 'revisions': [], 'optional_evidence_actions': [],
             'resource_accounting': {'costs_basis': 'observed_or_reserved_upper_bound_on_exception',
                                     'unknown_actual_cost_events': []}}

    def finish(artifact, reason):
        validate_artifact(context, artifact)
        trace.update(selected_candidate_id=artifact.candidate_id,
                     selected_forecast_id=artifact.identity(), stop_reason=reason,
                     materialized_candidate_count=len(state['candidate_ids']),
                     costs=clone(state['costs']), final_state=clone(state),
                     elapsed_seconds=time.monotonic() - started)
        return artifact, trace

    def fallback(reason, detail=''):
        trace['fallback'] = {'reason': reason, 'detail': detail[:500], 'candidate_id': default.candidate_id}
        return finish(default, reason)

    def artifact(candidate_id):
        if candidate_id not in state['candidate_ids']:
            raise ValueError('Candidate was not materialized by this protocol')
        value = validate_artifact(context, host.artifact(context, candidate_id))
        if value.candidate_id != candidate_id:
            raise ValueError('Artifact registry returned another candidate')
        return value

    def prompt(*names):
        return '\n\n'.join(skills[name]['instructions'] for name in names)

    def offered_actions():
        state['remaining_seconds'] = deadline - time.monotonic()
        if state['remaining_seconds'] <= 0:
            raise TimeoutError('Origin deadline exhausted')
        actions = {}
        if state['costs']['tool_calls'] >= limits.tool_calls:
            return actions
        for key, spec in host.actions(context, clone(state)).items():
            if key in state['executed_actions']:
                continue
            spec = {'description': spec} if isinstance(spec, str) else spec
            costs = _costs(spec.get('costs'))
            if any(state['costs'][k] + v > getattr(limits, k) for k, v in costs.items()):
                continue
            actions[key] = {**spec, 'costs': costs}
        return actions

    def choose(criteria, instruction, *, phase):
        if not criteria:
            raise ValueError('No legal actions remain in phase ' + phase)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or state['costs']['api_calls'] >= limits.api_calls:
            raise TimeoutError('Decision budget exhausted')
        state['phase'] = phase
        request_state = compact_request_state(context, state, limits)
        request_state.update(phase=phase, protocol=trace['protocol'])
        body = {'model': 'kev-latest', 'state': request_state, 'questions': {
            'action': {'type': 'choice', 'instructions': instruction, 'criteria': criteria}}}
        request_bytes = len(json.dumps(body, allow_nan=False).encode())
        if request_bytes > 24000:
            raise ValueError('Decision request exceeds 24000-byte budget')
        bounded = copy(client)
        bounded.timeout = min(limits.api_timeout_seconds, remaining)
        state['costs']['api_calls'] += 1
        try:
            action, record = call_with_deadline(
                lambda: bounded.choose(clone(request_state), instruction, clone(criteria)), bounded.timeout)
        except Exception as exc:
            trace['calls'].append({'phase': phase, 'record': clone(getattr(exc, 'record', {
                'request': body, 'error': type(exc).__name__ + ': ' + str(exc)[:500]}))})
            raise
        trace['calls'].append({'phase': phase, 'action': action, 'execution_mode': 'llm_selected',
                               'record': clone(record), 'state': request_state,
                               'request_bytes': request_bytes,
                               'available_actions': list(criteria)})
        if time.monotonic() >= deadline:
            raise TimeoutError('Origin deadline elapsed during API decision')
        if action not in criteria:
            raise ValueError('Kev selected an unavailable phase action')
        return action

    def execute(action, *, mode, mandatory):
        available = offered_actions()
        if action not in available:
            raise ValueError('Tool unavailable or exceeds remaining budget: ' + action)
        reserved = available[action]['costs']
        state['executed_actions'].append(action)
        state['costs']['tool_calls'] += 1
        entry = {'action': action, 'phase': state['phase'], 'execution_mode': mode,
                 'protocol_required': mandatory, 'reserved_costs': reserved,
                 'costs_before': clone(state['costs'])}
        trace['tools'].append(entry)
        returned = {}
        tool_exception = None
        def execute_host():
            returned['result'] = host.execute(context, action, clone(state))
            return returned['result']
        try:
            result = clone(call_with_deadline(execute_host, deadline - time.monotonic()))
        except Exception as exc:
            if 'result' in returned:
                # The deadline wrapper can raise after a host has returned a
                # partial diagnostic; retain its exact attempted-unit charge.
                result, tool_exception = clone(returned['result']), exc
            else:
                for key, value in reserved.items():
                    state['costs'][key] += value
                entry.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:500],
                             actual_costs_status='unknown_no_tool_result',
                             charged_costs=reserved, charged_costs_basis='estimated_upper_bound',
                             costs_after=clone(state['costs']))
                trace['resource_accounting']['unknown_actual_cost_events'].append({
                    'action': action, 'reserved_upper_bound_charged': reserved})
                raise
        entry['result'] = result
        actual = _costs(result.get('costs', reserved))
        # An interrupted diagnostic may return its partial result. Charge those
        # attempts before evaluating the clock, so deadline fallback cannot
        # erase work that actually occurred.
        for key, value in actual.items():
            state['costs'][key] += value
        entry.update(actual_costs_status='host_reported', charged_costs=actual,
                     charged_costs_basis='observed', costs_after=clone(state['costs']))
        if any(actual[k] > reserved[k] for k in actual):
            raise ValueError('Tool exceeded resource reservation')
        ids = result.get('candidate_ids', [])
        if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
            raise ValueError('Invalid candidate registry update')
        added = [key for key in dict.fromkeys(ids) if key not in state['candidate_ids']]
        if len(added) > actual['candidates']:
            raise ValueError('Tool generated unreserved candidates')
        for key in added:
            value = validate_artifact(context, host.artifact(context, key))
            if value.candidate_id != key:
                raise ValueError('Artifact identity differs from selected tool output')
        state['candidate_ids'].extend(added)
        observation = {'action': action, 'status': result.get('status', 'ok'),
                       'execution_mode': mode, 'protocol_required': mandatory,
                       'observation': result.get('observation', {})}
        state['observations'].append(observation)
        entry['costs_after'] = clone(state['costs'])
        if time.monotonic() >= deadline or tool_exception is not None:
            entry['status'] = 'deadline_exceeded_after_result'
            raise tool_exception or TimeoutError('Origin deadline elapsed during tool execution')
        return result

    try:
        initial = clone(call_with_deadline(lambda: host.initial_observation(context), deadline - time.monotonic()))
        eligible = initial.get('eligible_analog_count')
        if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 0:
            raise ValueError('Host must report its visible eligible_analog_count explicitly')
        trace['initial_observation'] = initial
        state['observations'].append({'action': 'initial_state', 'status': 'ok', 'observation': initial})
        state['phase'] = 'acquire_evidence'
        if eligible > 0:
            strategies = {'retrieve_analogs:' + name for name in ('recent', 'seasonal', 'state', 'diverse')}
            choices = {key: value['description'] for key, value in offered_actions().items() if key in strategies}
            action = choose(choices, prompt('plan_forecast', 'assess_state', 'retrieve_analogs'),
                            phase='select_historical_evidence')
            result = execute(action, mode='llm_selected', mandatory=True)
            observed_episodes = result.get('observation', {}).get('support_count', 0)
            cohort_ids = [item['episode_id'] for item in result.get('observation', {}).get('items', [])]
            if len(cohort_ids) != len(set(cohort_ids)) or len(cohort_ids) != observed_episodes:
                raise ValueError('Retrieved cohort IDs and support disagree')
        else:
            observed_episodes = 0
            cohort_ids = []
            # One forced action is not a useful LLM decision. Record its honest
            # attribution and the lack of matched LightGBM historical support.
            execute('inspect_recent_fit', mode='protocol_required', mandatory=True)
        state['phase'] = 'request_alternative'
        # Even the initial required candidate must leave room for a submission
        # and a mandatory review if diagnosis discovers sparse/failed support.
        if limits.api_calls - state['costs']['api_calls'] < 3:
            raise ValueError('Insufficient API budget for alternate forecast, final selection and review')
        required_tools = 2 if observed_episodes else 1
        if limits.tool_calls - state['costs']['tool_calls'] < required_tools:
            raise ValueError('Insufficient tool budget for alternate forecast and its diagnosis')
        if state['costs']['historical_evaluations'] + 2 * observed_episodes > limits.historical_evaluations:
            raise ValueError('Insufficient historical budget for the initial paired comparison')
        choices = {key: value['description'] for key, value in offered_actions().items()
                   if key.startswith('forecast:chronos:')}
        action = choose(choices, prompt('compose_forecast_request', 'assess_state'),
                        phase='select_alternative_input')
        requested = action[len('forecast:'):]
        result = execute(action, mode='llm_selected', mandatory=True)
        if requested not in result.get('candidate_ids', []) or requested == default.candidate_id:
            raise ValueError('Required alternate forecast was not generated')
        artifact(requested)

        model_candidate_ids = [default.candidate_id, requested]
        diagnosed_candidate_ids = set()
        fixed_cohort_identity = None
        paired = False
        low_support = observed_episodes < 3
        diagnostic_failures = False

        def diagnose_current():
            nonlocal paired, low_support, diagnostic_failures, fixed_cohort_identity
            state['phase'] = 'paired_diagnostics'
            target_ids = set(model_candidate_ids)
            diagnostics = [key for key in offered_actions()
                           if key.startswith('diagnose:') and set(key[len('diagnose:'):].split('|')) == target_ids]
            if len(diagnostics) != 1:
                raise ValueError('Paired diagnostic unavailable within the historical budget')
            result = execute(diagnostics[0], mode='protocol_required', mandatory=True)
            observation = result.get('observation', {})
            if set(observation.get('episodes', [])) != set(cohort_ids):
                raise ValueError('Diagnostic changed the fixed historical cohort')
            if set(observation.get('candidates', {})) != target_ids:
                raise ValueError('Diagnostic omitted a generated model candidate')
            paired_ids = observation.get('paired_episode_ids')
            support = observation.get('sample_count')
            if (not isinstance(paired_ids, list) or len(paired_ids) != len(set(paired_ids))
                    or not set(paired_ids).issubset(cohort_ids)
                    or isinstance(support, bool) or not isinstance(support, int)
                    or support != len(paired_ids)):
                raise ValueError('Diagnostic common successful episode support is invalid')
            for candidate in observation['candidates'].values():
                if (candidate.get('sample_count') != support
                        or candidate.get('paired_episode_ids') != paired_ids):
                    raise ValueError('Candidate diagnostics do not use the same successful episode intersection')
            cohort_identity = observation.get('cohort_id')
            if not cohort_identity:
                raise ValueError('Incremental host must identify its immutable cohort')
            if fixed_cohort_identity is not None and cohort_identity != fixed_cohort_identity:
                raise ValueError('Diagnostic cohort identity changed during revision')
            fixed_cohort_identity = cohort_identity
            paired = True
            diagnostic_failures = bool(observation.get('failures'))
            low_support = observation.get('sample_count', 0) < 3
            diagnosed_candidate_ids.update(target_ids)
            trace['iterations'].append({'index': len(trace['iterations']),
                'candidate_ids': list(model_candidate_ids), 'cohort_ids': list(cohort_ids),
                'cohort_id': cohort_identity, 'paired_support': observation.get('sample_count', 0),
                'diagnostic_action': diagnostics[0],
                'incremental_historical_units': result.get('costs', {}).get('historical_evaluations', 0)})
            trace['paired_diagnostic_executed'] = True
            trace['paired_diagnostic_has_failures'] = diagnostic_failures

        if observed_episodes:
            diagnose_current()
        else:
            state['observations'].append({'action': 'paired_diagnostics', 'status': 'unsupported',
                'execution_mode': 'protocol_required', 'protocol_required': True,
                'observation': {'reason': 'No fully mature common historical episodes.',
                                'lightgbm_chronos_paired_error_available': False,
                                'recent_fit_scope': 'Chronos-only 48-to-24 hindcast; not a matched LightGBM comparison.'}})
        trace['paired_diagnostic_executed'] = paired
        trace['paired_diagnostic_has_failures'] = diagnostic_failures
        trace['cohort_ids'] = list(cohort_ids)

        while state['costs']['api_calls'] < limits.api_calls:
            state['phase'] = 'iterate_or_commit'
            remaining_api = limits.api_calls - state['costs']['api_calls']
            remaining_tools = limits.tool_calls - state['costs']['tool_calls']
            # Diagnostics for a newly requested model are mandatory before its
            # next decision. Prior diagnosed candidates remain valid fallbacks.
            legal_candidates = [key for key in state['candidate_ids']
                                if not paired or key not in model_candidate_ids or key in diagnosed_candidate_ids]
            criteria = {'commit:' + key: 'Submit existing candidate ' + key for key in legal_candidates}
            available = offered_actions()
            # A pool requires one independent review decision after selection.
            pools = ({key: spec for key, spec in available.items() if key.startswith('pool:')}
                     if remaining_api >= 2 else {})
            criteria.update({key: spec['description'] for key, spec in pools.items()})
            revisions = {}
            # Reserve the action itself, a final selection and its review; also
            # reserve both forecast and incremental diagnosis tool executions.
            if (observed_episodes and remaining_api >= 3 and remaining_tools >= 2
                    and state['costs']['historical_evaluations'] + observed_episodes <= limits.historical_evaluations):
                revisions = {key: spec for key, spec in available.items() if key.startswith('forecast:chronos:')}
                criteria.update({key: 'Optionally revise inputs: ' + spec['description']
                                 for key, spec in revisions.items()})
            inspections = ({key: spec for key, spec in available.items()
                            if key in ('inspect_feedback', 'inspect_state')}
                           if remaining_api >= 3 else {})
            criteria.update({key: spec['description'] for key, spec in inspections.items()})
            action = choose(criteria, prompt('compare_candidates', 'compose_forecast_request', 'review_and_commit'),
                            phase='iterate_or_commit')
            if action in revisions:
                before_ids = list(model_candidate_ids)
                result = execute(action, mode='llm_selected', mandatory=False)
                revised_id = action[len('forecast:'):]
                if revised_id not in result.get('candidate_ids', []) or revised_id in model_candidate_ids:
                    raise ValueError('Input revision did not materialize a distinct requested model candidate')
                model_candidate_ids.append(revised_id)
                trace['revisions'].append({'action': action, 'from_candidate_ids': before_ids,
                    'new_candidate_id': revised_id, 'execution_mode': 'llm_selected', 'protocol_required': False})
                diagnose_current()
                continue
            if action in inspections:
                execute(action, mode='llm_selected', mandatory=False)
                trace['optional_evidence_actions'].append(action)
                continue
            if action.startswith('pool:'):
                result = execute(action, mode='llm_selected', mandatory=False)
                ids = result.get('candidate_ids', [])
                if len(ids) != 1:
                    raise ValueError('Selected pool did not produce exactly one candidate')
                selected = artifact(ids[0])
            else:
                selected = artifact(action[len('commit:'):])
            needs_review = low_support or diagnostic_failures or selected.model_id == 'quantile_pool'
            if not needs_review:
                trace['review_skipped'] = 'not_required'
                return finish(selected, 'agent_commit')
            if state['costs']['api_calls'] >= limits.api_calls:
                return fallback('required_review_budget_exhausted',
                                'A low-support, failed-diagnostic or pooled proposal requires approval.')
            review_state = {'action': 'review_proposal', 'status': 'ok', 'observation': {
                'proposed_candidate_id': selected.candidate_id, 'low_historical_support': low_support,
                'diagnostic_failures': diagnostic_failures,
                'quantile_pool': selected.model_id == 'quantile_pool'}}
            state['observations'].append(review_state)
            review = {'approve': 'Approve the proposed candidate and submit.'}
            if limits.api_calls - state['costs']['api_calls'] >= 3:
                review['continue'] = 'Reconsider or revise inputs within the remaining budget.'
            if selected.candidate_id != default.candidate_id:
                review['default'] = 'Decline this proposal and submit the original default.'
            outcome = choose(review, prompt('review_and_commit'), phase='review_prediction')
            trace['reviews'].append({'candidate_id': selected.candidate_id, 'outcome': outcome,
                                    'execution_mode': 'llm_selected', 'protocol_triggered': True})
            if outcome == 'approve':
                state['reviewed_candidate_ids'].append(selected.candidate_id)
                return finish(selected, 'reviewed_commit')
            if outcome == 'default':
                return finish(default, 'review_abstention')
        return fallback('api_budget_exhausted')
    except Exception as exc:
        return fallback('failed_decision', type(exc).__name__ + ': ' + str(exc))
