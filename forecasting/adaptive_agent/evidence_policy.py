"""R19 numerical evidence protocol with honest mandatory/LLM attribution.

Unlike the R18 free tool loop, this protocol requires one comparison before
submission. Kev selects a retrieval strategy and an actual alternate forecast;
the host evaluates both models on exactly the same mature historical episodes.
Kev may still select the default. Tool use forced by this protocol is never
presented as evidence of autonomous exploration or predictive improvement.
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

RELEASE = 'r19-evidence-first-agent-v1'
SKILL_ROOT = Path(__file__).parent / 'skills_r19'


def load_skills():
    result = {}
    for name in SKILL_IDS:
        files = {filename: (SKILL_ROOT / name / filename).read_text()
                 for filename in ('instructions.txt', 'schema.json', 'examples.jsonl', 'version')}
        schema = json.loads(files['schema.json'])
        if schema.get('skill_id') != name or files['version'].strip() != '2.0.0':
            raise ValueError('R19 requires its registered version 2.0.0 skills')
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
             'protocol': {'id': 'numerical_evidence_then_comparison_v1',
                          'mandatory_evidence_acquisition': True,
                          'mandatory_alternative_forecast': True,
                          'paired_historical_comparison_when_supported': True,
                          'external_event_route': 'not_used_in_this_numerical_policy',
                          'default_remains_legal': True,
                          'tool_coverage_is_not_llm_gain': True},
             'calls': [], 'tools': [], 'reviews': [], 'fallback': None}

    def finish(artifact, reason):
        validate_artifact(context, artifact)
        trace.update(selected_candidate_id=artifact.candidate_id,
                     selected_forecast_id=artifact.identity(), stop_reason=reason,
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
        request_state = _request_state(context, state, limits)
        request_state.update(phase=phase, protocol=trace['protocol'])
        body = {'model': 'kev-latest', 'state': request_state, 'questions': {
            'action': {'type': 'choice', 'instructions': instruction, 'criteria': criteria}}}
        if len(json.dumps(body, allow_nan=False).encode()) > 24000:
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
        try:
            result = clone(call_with_deadline(
                lambda: host.execute(context, action, clone(state)), deadline - time.monotonic()))
        except Exception as exc:
            entry.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:500])
            raise
        entry['result'] = result
        if time.monotonic() >= deadline:
            raise TimeoutError('Origin deadline elapsed during tool execution')
        actual = _costs(result.get('costs', reserved))
        if any(actual[k] > reserved[k] for k in actual):
            raise ValueError('Tool exceeded resource reservation')
        ids = result.get('candidate_ids', [])
        if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
            raise ValueError('Invalid candidate registry update')
        added = [key for key in dict.fromkeys(ids) if key not in state['candidate_ids']]
        if len(added) > reserved['candidates']:
            raise ValueError('Tool generated unreserved candidates')
        for key in added:
            value = validate_artifact(context, host.artifact(context, key))
            if value.candidate_id != key:
                raise ValueError('Artifact identity differs from selected tool output')
        state['candidate_ids'].extend(added)
        state['costs']['candidates'] = len(state['candidate_ids'])
        for key in ('episodes', 'historical_evaluations'):
            state['costs'][key] += actual[key]
        observation = {'action': action, 'status': result.get('status', 'ok'),
                       'execution_mode': mode, 'protocol_required': mandatory,
                       'observation': result.get('observation', {})}
        state['observations'].append(observation)
        entry['costs_after'] = clone(state['costs'])
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
        else:
            observed_episodes = 0
            # One forced action is not a useful LLM decision. Record its honest
            # attribution and the lack of matched LightGBM historical support.
            execute('inspect_recent_fit', mode='protocol_required', mandatory=True)
        state['phase'] = 'request_alternative'
        choices = {key: value['description'] for key, value in offered_actions().items()
                   if key.startswith('forecast:chronos:')}
        action = choose(choices, prompt('compose_forecast_request', 'assess_state'),
                        phase='select_alternative_input')
        requested = action[len('forecast:'):]
        result = execute(action, mode='llm_selected', mandatory=True)
        if requested not in result.get('candidate_ids', []) or requested == default.candidate_id:
            raise ValueError('Required alternate forecast was not generated')
        artifact(requested)

        state['phase'] = 'paired_diagnostics'
        paired = False
        low_support = observed_episodes < 3
        diagnostic_failures = False
        if observed_episodes:
            target_ids = {default.candidate_id, requested}
            diagnostics = [key for key in offered_actions()
                           if key.startswith('diagnose:') and set(key[len('diagnose:'):].split('|')) == target_ids]
            if len(diagnostics) != 1:
                raise ValueError('Paired diagnostic unavailable within the historical budget')
            result = execute(diagnostics[0], mode='protocol_required', mandatory=True)
            observation = result.get('observation', {})
            paired = True
            diagnostic_failures = bool(observation.get('failures'))
            low_support = observation.get('sample_count', 0) < 3
        else:
            state['observations'].append({'action': 'paired_diagnostics', 'status': 'unsupported',
                'execution_mode': 'protocol_required', 'protocol_required': True,
                'observation': {'reason': 'No fully mature common historical episodes.',
                                'lightgbm_chronos_paired_error_available': False,
                                'recent_fit_scope': 'Chronos-only 48-to-24 hindcast; not a matched LightGBM comparison.'}})
        trace['paired_diagnostic_executed'] = paired
        trace['paired_diagnostic_has_failures'] = diagnostic_failures

        while state['costs']['api_calls'] < limits.api_calls:
            state['phase'] = 'select_prediction'
            criteria = {'commit:' + key: 'Submit existing candidate ' + key for key in state['candidate_ids']}
            pools = {key: spec for key, spec in offered_actions().items() if key.startswith('pool:')}
            criteria.update({key: spec['description'] for key, spec in pools.items()})
            action = choose(criteria, prompt('compare_candidates', 'review_and_commit'), phase='select_prediction')
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
            review = {'approve': 'Approve the proposed candidate and submit.',
                      'continue': 'Reconsider the final candidate selection without inventing new facts.'}
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
