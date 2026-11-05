"""R21 two-stage Kev policy for one frozen LightGBM and admitted input operations."""
from __future__ import annotations

from copy import copy
import hashlib
import json
from pathlib import Path
import time

from forecasting.agent_forecasting.execution import call_with_deadline

from .contracts import BudgetLimits
from .controller import SKILL_IDS, clone, _costs, _request_state
from .decision_evidence import build_decision_evidence
from .iterative_policy import compact_request_state
from .review import validate_artifact


RELEASE = 'r21-single-lightgbm-input-agent-v2'
SKILL_ROOT = Path(__file__).parent / 'skills_r21'


def load_skills():
    result = {}
    for name in SKILL_IDS:
        files = {filename: (SKILL_ROOT / name / filename).read_text()
                 for filename in ('instructions.txt', 'schema.json', 'examples.jsonl', 'version')}
        schema = json.loads(files['schema.json'])
        if schema.get('skill_id') != name or schema.get('policy') != 'r21_single_lightgbm_input' \
                or files['version'].strip() != '5.0.0':
            raise ValueError('R21 requires its registered version 5.0.0 skills')
        for line in files['examples.jsonl'].splitlines():
            if line.strip():
                json.loads(line)
        result[name] = {
            'version': files['version'].strip(),
            'instructions': files['instructions.txt'],
            'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        }
    return result


def run(context, host, client, limits=None):
    """Let Kev decide whether to experiment, then which admitted operation to run.

    The two decisions are separate API calls.  A nonidentity operation cannot be
    committed until it has been recomputed on the first locked mature cohort and
    compared on the common successful episode intersection.
    """
    limits = limits or BudgetLimits()
    started = time.monotonic()
    deadline = started + limits.origin_deadline_seconds
    skills = load_skills()
    default = validate_artifact(context, call_with_deadline(
        lambda: host.default(context), deadline - time.monotonic()))
    state = {
        'default_candidate_id': default.candidate_id,
        'candidate_ids': [default.candidate_id],
        'observations': [], 'executed_actions': [], 'reviewed_candidate_ids': [],
        'costs': {'api_calls': 0, 'tool_calls': 0, 'candidates': 1,
                  'episodes': 0, 'historical_evaluations': 0},
        'deadline_monotonic': deadline, 'remaining_seconds': deadline - time.monotonic(),
        'phase': 'initial_state',
    }
    trace = {
        'release': RELEASE, 'context_id': context.identity(),
        'default_candidate_id': default.candidate_id,
        'skills': {key: {field: value[field] for field in ('version', 'sha256')}
                   for key, value in skills.items()},
        'protocol': {
            'id': 'single_frozen_lightgbm_evidence_admission_v2',
            'predictor_families': ['lightgbm'],
            'chronos_runtime': False,
            'stage_a': 'decide whether paired evidence warrants changing the canonical input',
            'stage_b': 'choose one already diagnosed registered input operation',
            'operation_admission': ('deterministic paired evidence gate before Kev; '
                                    'Kev cannot override a rejected operation'),
            'cohort_policy': 'first_retrieval_fixed_for_this_origin',
            'new_operation_requires_incremental_paired_diagnosis': True,
            'no_mature_cohort_policy': 'submit identity',
            'default_remains_legal': True,
            'operation_semantics': 'training-mean counterfactual; not missingness or column deletion',
        },
        'calls': [], 'tools': [], 'reviews': [], 'fallback': None,
        'iterations': [], 'revisions': [], 'optional_evidence_actions': [],
        'stage_a_decisions': [], 'stage_b_decisions': [],
        'resource_accounting': {'costs_basis': 'observed_or_reserved_upper_bound_on_exception',
                                'unknown_actual_cost_events': []},
    }

    def finish(artifact, reason):
        validate_artifact(context, artifact)
        trace.update(selected_candidate_id=artifact.candidate_id,
                     selected_forecast_id=artifact.identity(), stop_reason=reason,
                     materialized_candidate_count=len(state['candidate_ids']),
                     costs=clone(state['costs']), final_state=clone(state),
                     elapsed_seconds=time.monotonic() - started)
        return artifact, trace

    def fallback(reason, detail=''):
        trace['fallback'] = {'reason': reason, 'detail': detail[:500],
                             'candidate_id': default.candidate_id}
        return finish(default, reason)

    def artifact(candidate_id):
        if candidate_id not in state['candidate_ids']:
            raise ValueError('Candidate was not materialized by R21')
        value = validate_artifact(context, host.artifact(context, candidate_id))
        if value.candidate_id != candidate_id or value.model_id != 'lightgbm':
            raise ValueError('R21 artifact is not from the one frozen LightGBM')
        return value

    def prompt(*names):
        return '\n\n'.join(skills[name]['instructions'] for name in names)

    def offered_actions():
        state['remaining_seconds'] = deadline - time.monotonic()
        if state['remaining_seconds'] <= 0:
            raise TimeoutError('Origin deadline exhausted')
        if state['costs']['tool_calls'] >= limits.tool_calls:
            return {}
        result = {}
        for key, spec in host.actions(context, clone(state)).items():
            if key in state['executed_actions']:
                continue
            spec = {'description': spec} if isinstance(spec, str) else spec
            costs = _costs(spec.get('costs'))
            if any(state['costs'][name] + value > getattr(limits, name)
                   for name, value in costs.items()):
                continue
            result[key] = {**spec, 'costs': costs}
        return result

    def choose(criteria, instruction, *, phase, stage):
        if not criteria:
            raise ValueError('No legal choices remain in R21 phase ' + phase)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or state['costs']['api_calls'] >= limits.api_calls:
            raise TimeoutError('R21 decision budget exhausted')
        state['phase'] = phase
        request_state = compact_request_state(context, state, limits)
        request_state.update(phase=phase, protocol=trace['protocol'])
        bounded = copy(client)
        bounded.timeout = min(limits.api_timeout_seconds, remaining)
        state['costs']['api_calls'] += 1
        try:
            action, record = call_with_deadline(
                lambda: bounded.choose(clone(request_state), instruction, clone(criteria)),
                bounded.timeout)
        except Exception as exc:
            trace['calls'].append({'phase': phase, 'stage': stage,
                'record': clone(getattr(exc, 'record', {
                    'error': type(exc).__name__ + ': ' + str(exc)[:500]}))})
            raise
        trace['calls'].append({'phase': phase, 'stage': stage, 'action': action,
                               'execution_mode': 'llm_selected', 'record': clone(record),
                               'state': request_state, 'available_actions': list(criteria)})
        if action not in criteria:
            raise ValueError('Kev selected an unavailable R21 choice')
        if stage == 'A':
            trace['stage_a_decisions'].append({'phase': phase, 'action': action})
        elif stage == 'B':
            trace['stage_b_decisions'].append({'phase': phase, 'action': action})
        return action

    def execute(action, *, mandatory):
        available = offered_actions()
        if action not in available:
            raise ValueError('R21 tool unavailable or exceeds budget: ' + action)
        reserved = available[action]['costs']
        state['executed_actions'].append(action)
        state['costs']['tool_calls'] += 1
        entry = {'action': action, 'phase': state['phase'],
                 'execution_mode': 'llm_selected' if not mandatory else 'protocol_required',
                 'protocol_required': mandatory, 'reserved_costs': reserved}
        trace['tools'].append(entry)
        try:
            result = clone(call_with_deadline(
                lambda: host.execute(context, action, clone(state)),
                deadline - time.monotonic()))
        except Exception as exc:
            for name, value in reserved.items():
                state['costs'][name] += value
            entry.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:500],
                         charged_costs=reserved, charged_costs_basis='estimated_upper_bound')
            trace['resource_accounting']['unknown_actual_cost_events'].append({
                'action': action, 'reserved_upper_bound_charged': reserved})
            raise
        actual = _costs(result.get('costs', reserved))
        if any(actual[name] > reserved[name] for name in actual):
            raise ValueError('R21 tool exceeded resource reservation')
        for name, value in actual.items():
            state['costs'][name] += value
        entry.update(status='ok', result=result, charged_costs=actual,
                     charged_costs_basis='observed', costs_after=clone(state['costs']))
        ids = result.get('candidate_ids', [])
        if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
            raise ValueError('Invalid R21 candidate registry update')
        added = [key for key in dict.fromkeys(ids) if key not in state['candidate_ids']]
        if len(added) > actual['candidates']:
            raise ValueError('R21 generated unreserved candidates')
        for key in added:
            value = validate_artifact(context, host.artifact(context, key))
            if value.candidate_id != key or value.model_id != 'lightgbm':
                raise ValueError('R21 tool returned an invalid candidate artifact')
        state['candidate_ids'].extend(added)
        state['observations'].append({
            'action': action, 'status': result.get('status', 'ok'),
            'execution_mode': entry['execution_mode'], 'protocol_required': mandatory,
            'observation': result.get('observation', {})})
        return result

    def diagnose_current(cohort_ids):
        state['phase'] = 'paired_diagnostics'
        target_ids = set(state['candidate_ids'])
        actions = [key for key in offered_actions()
                   if key.startswith('diagnose:')
                   and set(key[len('diagnose:'):].split('|')) == target_ids]
        if len(actions) != 1:
            raise ValueError('R21 paired diagnostic unavailable')
        result = execute(actions[0], mandatory=True)
        observation = result.get('observation', {})
        paired_ids = observation.get('paired_episode_ids', [])
        if (set(observation.get('episodes', [])) != set(cohort_ids)
                or set(observation.get('candidates', {})) != target_ids
                or not set(paired_ids).issubset(cohort_ids)):
            raise ValueError('R21 diagnostic changed cohort or candidate support')
        for row in observation['candidates'].values():
            if row.get('paired_episode_ids') != paired_ids:
                raise ValueError('R21 diagnostic is not paired on common episodes')
        trace['iterations'].append({
            'index': len(trace['iterations']), 'candidate_ids': list(state['candidate_ids']),
            'cohort_ids': list(cohort_ids), 'cohort_id': observation.get('cohort_id'),
            'paired_support': len(paired_ids), 'diagnostic_action': actions[0],
            'incremental_historical_units': result.get('costs', {}).get('historical_evaluations', 0),
        })
        return observation

    try:
        initial = clone(call_with_deadline(
            lambda: host.initial_observation(context), deadline - time.monotonic()))
        trace['initial_observation'] = initial
        state['observations'].append({'action': 'initial_state', 'status': 'ok',
                                      'observation': initial})
        eligible = initial.get('eligible_analog_count')
        if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 0:
            raise ValueError('R21 host must report eligible mature episodes')
        if eligible == 0:
            trace['no_mature_cohort'] = True
            return finish(default, 'identity_no_mature_cohort')

        strategies = {'retrieve_analogs:' + name for name in ('recent', 'seasonal', 'state', 'diverse')}
        available = offered_actions()
        choices = {key: available[key]['description'] for key in strategies if key in available}
        retrieval = choose(choices, prompt('plan_forecast', 'retrieve_analogs'),
                           phase='select_historical_cohort', stage='evidence')
        result = execute(retrieval, mandatory=False)
        cohort_ids = [item['episode_id'] for item in result['observation'].get('items', [])]
        if not cohort_ids or len(cohort_ids) != len(set(cohort_ids)):
            return fallback('invalid_or_empty_mature_cohort')

        # Materialize the small, development-admitted operation set under the
        # protocol, then diagnose every candidate once on the same locked
        # cohort.  Kev receives comparative evidence before its gate decision;
        # it is not asked whether to buy evidence it has never seen.
        operations = sorted(key for key in offered_actions()
                            if key.startswith('forecast:')
                            and key != 'forecast:' + default.candidate_id)
        if not operations:
            return finish(default, 'identity_no_admitted_operation')
        for action in operations:
            before = list(state['candidate_ids'])
            result = execute(action, mandatory=True)
            new_id = action[len('forecast:'):]
            if new_id not in result.get('candidate_ids', []) or new_id in before:
                raise ValueError('R21 protocol did not materialize its admitted candidate')
            trace['revisions'].append({'action': action, 'from_candidate_ids': before,
                                       'new_candidate_id': new_id,
                                       'execution_mode': 'protocol_required'})
        latest_diagnostic = diagnose_current(cohort_ids)
        support = latest_diagnostic.get('sample_count', 0)
        failures = latest_diagnostic.get('failures', [])
        if support < 3 or failures:
            return finish(default, 'identity_insufficient_paired_support')

        numerical_state = initial.get('numerical_state', {})
        decision_evidence = build_decision_evidence(
            latest_diagnostic, default.candidate_id, numerical_state)
        trace['decision_admission'] = clone(decision_evidence)
        state['observations'].append({
            'action': 'input_operation_admission', 'status': 'ok',
            'execution_mode': 'protocol_required',
            'observation': clone(decision_evidence),
        })
        eligible = decision_evidence['eligible_candidate_ids']
        if not eligible:
            return finish(default, 'identity_no_operation_passed_evidence_gate')

        gate = choose({
            'use_identity': 'Keep the complete canonical input and submit the default forecast.',
            'use_operated_input': ('At least one protocol-admitted operation has stable paired historical '
                                   'benefit and changes the current forecast; continue to choose only among '
                                   'the admitted operation IDs recorded in input_operation_admission.'),
        }, prompt('plan_forecast', 'assess_state', 'compare_candidates'),
            phase='decide_whether_to_change_input', stage='A')
        if gate == 'use_identity':
            return finish(default, 'agent_commit_identity')
        criteria = {'commit:' + key: 'Submit evidence-admitted diagnosed candidate ' + key
                    for key in eligible}
        selected = choose(criteria, prompt('compose_forecast_request', 'compare_candidates',
                                            'review_and_commit'),
                          phase='select_input_operation', stage='B')
        return finish(artifact(selected[len('commit:'):]), 'agent_commit')
    except TimeoutError as exc:
        return fallback('timeout', str(exc))
    except Exception as exc:
        return fallback('protocol_error', type(exc).__name__ + ': ' + str(exc))
