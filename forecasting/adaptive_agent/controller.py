"""R18 bounded typed-choice agent. All effects are mediated by a local ToolHost.

ToolHost protocol:
  default(context) -> ForecastArtifact
  actions(context, state) -> {id: {description, kind, costs}}
  execute(context, action_id, state) -> {status, observation, candidate_ids, costs}
  artifact(context, candidate_id) -> ForecastArtifact

Costs use candidates, episodes and historical_evaluations. Advertised costs are
upper bounds reserved before execution; returned costs are actual usage. Hosts
must honor state['deadline_monotonic'] during expensive tool operations.
"""
from __future__ import annotations

from copy import copy
import hashlib
import json
from pathlib import Path
import time

from forecasting.agent_forecasting.execution import call_with_deadline

from .contracts import BudgetLimits, plain
from .review import review_criteria, validate_artifact

RELEASE = 'r18-complete-agent-v1'
SKILL_ROOT = Path(__file__).parent / 'skills'
SKILL_IDS = ('plan_forecast', 'assess_state', 'interpret_evidence',
             'retrieve_analogs', 'compose_forecast_request',
             'compare_candidates', 'review_and_commit')
COST_FIELDS = ('candidates', 'episodes', 'historical_evaluations')


def clone(value):
    """Immutable trace snapshots and a fail-closed finite JSON boundary."""
    return json.loads(json.dumps(plain(value), allow_nan=False, sort_keys=True))


def load_skills():
    skills = {}
    for name in SKILL_IDS:
        directory = SKILL_ROOT / name
        files = {p: (directory / p).read_text() for p in
                 ('instructions.txt', 'schema.json', 'examples.jsonl', 'version')}
        schema = json.loads(files['schema.json'])
        if schema.get('skill_id') != name:
            raise ValueError('Skill schema does not match its directory')
        for line in files['examples.jsonl'].splitlines():
            if line.strip():
                json.loads(line)
        skills[name] = {
            'version': files['version'].strip(), 'instructions': files['instructions.txt'],
            'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        }
    return skills


def load_skill(name):
    if name not in SKILL_IDS:
        raise ValueError('Unknown project skill')
    return load_skills()[name]['instructions']


def _costs(value):
    value = value or {}
    if set(value) - set(COST_FIELDS):
        raise ValueError('Unknown tool cost dimension')
    result = {}
    for key in COST_FIELDS:
        number = value.get(key, 0)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError('Tool costs must be nonnegative integers')
        result[key] = number
    return result


def _request_state(context, state, limits):
    # Raw model_inputs, provider paths and full arrays never enter the API state.
    return clone({
        'task': 'Choose evidence, input configuration and a frozen forecast; do not generate prices.',
        'context_id': context.identity(), 'region': context.region,
        'origin_unix': int(context.origin_unix), 'cutoff': int(context.cutoff),
        'memory_watermark': int(context.memory_watermark),
        'default_candidate_id': state['default_candidate_id'],
        'candidate_ids': state['candidate_ids'],
        'observations': state['observations'],
        'reviewed_candidate_ids': state['reviewed_candidate_ids'],
        'remaining_budget': {key: getattr(limits, key) - state['costs'][key]
                             for key in ('api_calls', 'tool_calls', *COST_FIELDS)},
        'remaining_seconds': max(0.0, state['deadline_monotonic'] - time.monotonic()),
        'future_actuals_available': False,
    })


def run(context, host, client, limits=None):
    """Return a validated artifact and complete decision trace for one origin.

    Default failure is raised (there is no legal forecast to fall back to).
    Subsequent tool/API/decision failures preserve the valid frozen default.
    No confidence or hindsight-loss gate overrides an admissible Kev choice.
    """
    limits = limits or BudgetLimits()
    started = time.monotonic()
    if limits.api_calls < 1 or limits.tool_calls < 0 or limits.candidates < 1:
        raise ValueError('Invalid agent budget')
    skills = load_skills()
    deadline = started + limits.origin_deadline_seconds
    default = validate_artifact(context, call_with_deadline(
        lambda: host.default(context), deadline - time.monotonic()))
    state = {
        'default_candidate_id': default.candidate_id,
        'candidate_ids': [default.candidate_id], 'observations': [],
        'executed_actions': [], 'reviewed_candidate_ids': [],
        'costs': dict(api_calls=0, tool_calls=0, candidates=1,
                      episodes=0, historical_evaluations=0),
        'deadline_monotonic': deadline,
        'remaining_seconds': max(0.0, limits.origin_deadline_seconds - (time.monotonic() - started)),
    }
    trace = {
        'release': RELEASE, 'context_id': context.identity(),
        'skills': {k: {f: v[f] for f in ('version', 'sha256')} for k, v in skills.items()},
        'default_candidate_id': default.candidate_id,
        'calls': [], 'tools': [], 'reviews': [], 'fallback': None,
    }
    instructions = '\n\n'.join(skills[name]['instructions'] for name in SKILL_IDS)

    def finish(artifact, reason):
        validate_artifact(context, artifact)
        trace.update(selected_candidate_id=artifact.candidate_id,
                     selected_forecast_id=artifact.identity(), stop_reason=reason,
                     costs=clone(state['costs']), final_state=clone(state),
                     elapsed_seconds=time.monotonic() - started)
        return artifact, trace

    def fallback(reason, detail=''):
        trace['fallback'] = {'reason': reason, 'detail': detail[:500],
                             'candidate_id': default.candidate_id}
        return finish(default, reason)

    def choose(criteria, prompt, request_state):
        remaining = state['deadline_monotonic'] - time.monotonic()
        if remaining <= 0 or state['costs']['api_calls'] >= limits.api_calls:
            raise TimeoutError('Decision budget exhausted')
        body = {'model': 'kev-latest', 'state': request_state, 'questions': {
            'action': {'type': 'choice', 'instructions': prompt, 'criteria': criteria}}}
        if len(json.dumps(body, allow_nan=False).encode()) > 24000:
            raise ValueError('Decision request exceeds 24000-byte evidence budget')
        # A per-call shallow copy avoids mutating a shared transport timeout.
        bounded_client = copy(client)
        bounded_client.timeout = min(float(limits.api_timeout_seconds), remaining)
        state['costs']['api_calls'] += 1
        try:
            action, record = call_with_deadline(
                lambda: bounded_client.choose(clone(request_state), prompt, clone(criteria)),
                bounded_client.timeout)
        except Exception as exc:
            trace['calls'].append(clone(getattr(exc, 'record', {
                'request': body, 'error': type(exc).__name__ + ': ' + str(exc)[:500]})))
            raise
        trace['calls'].append({'action': action, 'record': clone(record),
                               'state': clone(request_state), 'available_actions': list(criteria)})
        if time.monotonic() > state['deadline_monotonic']:
            raise TimeoutError('Origin deadline elapsed during decision')
        if action not in criteria:
            raise ValueError('Kev returned an unavailable action')
        return action

    try:
        if hasattr(host, 'initial_observation'):
            initial = clone(call_with_deadline(
                lambda: host.initial_observation(context), deadline - time.monotonic()))
            state['observations'].append({'action': 'initial_state', 'status': 'ok',
                                           'observation': initial})
            trace['initial_observation'] = initial
        while state['costs']['api_calls'] < limits.api_calls:
            state['remaining_seconds'] = state['deadline_monotonic'] - time.monotonic()
            if state['remaining_seconds'] <= 0:
                return fallback('origin_deadline')
            criteria = {'commit_default': 'Submit the original frozen default and stop.',
                        'abstain': 'Abstain and return the original frozen default.'}
            for candidate_id in state['candidate_ids']:
                criteria['commit:' + candidate_id] = 'Submit the existing candidate ' + candidate_id
                if (candidate_id not in state['reviewed_candidate_ids']
                        and state['costs']['api_calls'] + 1 < limits.api_calls):
                    criteria['review:' + candidate_id] = 'Independently review candidate ' + candidate_id
            offered = {}
            if (state['costs']['api_calls'] + 1 < limits.api_calls
                    and state['costs']['tool_calls'] < limits.tool_calls):
                for action_id, spec in host.actions(context, clone(state)).items():
                    if action_id in state['executed_actions']:
                        continue
                    if action_id in criteria or action_id.startswith(('commit:', 'review:')):
                        raise ValueError('Tool action shadows a reserved submission action')
                    spec = {'description': spec} if isinstance(spec, str) else spec
                    costs = _costs(spec.get('costs'))
                    if any(state['costs'][k] + costs[k] > getattr(limits, k) for k in COST_FIELDS):
                        continue
                    offered[action_id] = {**spec, 'costs': costs}
                    criteria[action_id] = str(spec['description'])
            action = choose(criteria, instructions, _request_state(context, state, limits))
            if action in ('commit_default', 'abstain'):
                return finish(default, action)
            if action.startswith('commit:'):
                artifact = host.artifact(context, action[len('commit:'):])
                if artifact.candidate_id != action[len('commit:'):]:
                    raise ValueError('Artifact registry returned a different candidate')
                return finish(artifact, 'agent_commit')
            if action.startswith('review:'):
                candidate_id = action[len('review:'):]
                artifact = validate_artifact(context, host.artifact(context, candidate_id))
                if artifact.candidate_id != candidate_id:
                    raise ValueError('Artifact registry returned a different candidate')
                review_state = _request_state(context, state, limits)
                review_state['proposed_candidate_id'] = candidate_id
                outcome = choose(review_criteria(candidate_id),
                                 skills['review_and_commit']['instructions'], review_state)
                review_record = {'candidate_id': candidate_id, 'outcome': outcome}
                state['reviewed_candidate_ids'].append(candidate_id)
                state['observations'].append({'action': action, 'status': 'ok',
                                               'observation': review_record})
                trace['reviews'].append(review_record)
                if outcome == 'approve':
                    return finish(artifact, 'reviewed_commit')
                if outcome == 'default':
                    return finish(default, 'review_abstention')
                continue
            advertised = offered[action]['costs']
            state['costs']['tool_calls'] += 1
            state['executed_actions'].append(action)
            before = clone(state)
            try:
                result = clone(call_with_deadline(
                    lambda: host.execute(context, action, clone(state)),
                    deadline - time.monotonic()))
            except Exception as exc:
                trace['tools'].append({'action': action, 'status': 'error',
                                       'reserved_costs': advertised,
                                       'error': type(exc).__name__ + ': ' + str(exc)[:500]})
                raise
            tool_record = {'action': action, 'result': result,
                           'reserved_costs': advertised, 'costs_before': before['costs']}
            trace['tools'].append(tool_record)
            actual = _costs(result.get('costs', advertised))
            if any(actual[k] > advertised[k] for k in COST_FIELDS):
                raise ValueError('Tool exceeded its advertised resource reservation')
            candidate_ids = result.get('candidate_ids', [])
            if not isinstance(candidate_ids, list) or not all(isinstance(c, str) for c in candidate_ids):
                raise ValueError('Malformed candidate registry update')
            additions = [c for c in dict.fromkeys(candidate_ids) if c not in state['candidate_ids']]
            if len(additions) > advertised['candidates']:
                raise ValueError('Tool materialized unreserved candidates')
            for candidate_id in additions:
                new_artifact = validate_artifact(context, host.artifact(context, candidate_id))
                if new_artifact.candidate_id != candidate_id:
                    raise ValueError('Artifact registry returned a different candidate')
            state['candidate_ids'].extend(additions)
            state['costs']['candidates'] = len(state['candidate_ids'])
            for key in ('episodes', 'historical_evaluations'):
                state['costs'][key] += actual[key]
            state['observations'].append({'action': action, 'status': result.get('status', 'ok'),
                                           'observation': result.get('observation', {})})
            tool_record['costs_after'] = clone(state['costs'])
            if time.monotonic() > state['deadline_monotonic']:
                return fallback('origin_deadline')
        return fallback('api_budget_exhausted')
    except Exception as exc:
        return fallback('failed_decision', type(exc).__name__ + ': ' + str(exc))
