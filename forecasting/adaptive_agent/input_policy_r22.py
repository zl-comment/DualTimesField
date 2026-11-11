"""R22 state-aligned input selection with a fail-closed risk review."""
from __future__ import annotations

from copy import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from forecasting.agent_forecasting.execution import call_with_deadline

from .contracts import BudgetLimits, digest
from .controller import SKILL_IDS, clone, _costs, _request_state
from .r22_evidence_report import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MATERIAL_QS_LOSS_THRESHOLD,
    SCHEMA as EVIDENCE_SCHEMA,
    build_evidence_report,
)
from .r22_risk_review import (
    APPROVE_SELECTED,
    PACKET_SCHEMA,
    REVERT_TO_IDENTITY,
    RISK_THRESHOLDS,
    build_review_packet,
    fail_closed_decision,
    registered_reason_codes,
    validate_review_action,
)
from .iterative_policy import compact_request_state
from .review import validate_artifact


RELEASE = 'r22-state-aligned-input-agent-v1'
SKILL_ROOT = Path(__file__).parent / 'skills_r22'
POLICY_ID = 'r22_state_aligned_input'


def _source_fingerprint():
    root = Path(__file__).parent
    paths = [Path(__file__), root / 'r22_evidence_report.py',
             root / 'r22_risk_review.py', root / 'iterative_runtime.py']
    payload = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return digest(payload)


def manifest_contract():
    return {
        'policy': POLICY_ID,
        'release': RELEASE,
        'evidence_schema': EVIDENCE_SCHEMA,
        'risk_packet_schema': PACKET_SCHEMA,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'material_qs_loss_threshold': MATERIAL_QS_LOSS_THRESHOLD,
        'risk_thresholds': asdict(RISK_THRESHOLDS),
        'risk_reason_codes': registered_reason_codes(),
        'source_fingerprint': _source_fingerprint(),
    }


def load_skills():
    result = {}
    for name in SKILL_IDS:
        files = {filename: (SKILL_ROOT / name / filename).read_text()
                 for filename in ('instructions.txt', 'schema.json', 'examples.jsonl', 'version')}
        schema = json.loads(files['schema.json'])
        if schema.get('skill_id') != name or schema.get('policy') != POLICY_ID \
                or files['version'].strip() != '6.0.0':
            raise ValueError('R22 requires its registered version 6.0.0 skills')
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
    """Select an admitted input operation, then independently review its risk.

    The selection decisions and risk review are separate API calls. A nonidentity operation cannot be
    committed until it has been recomputed on the first locked mature cohort and
    compared on the common successful episode intersection. The reviewer can only
    approve the selected operation or revert to identity.
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
            'id': 'single_frozen_lightgbm_state_aligned_risk_review_v1',
            'predictor_families': ['lightgbm'],
            'chronos_runtime': False,
            'stage_a': 'decide whether structured evidence warrants changing the canonical input',
            'stage_b': 'choose one already diagnosed registered input operation',
            'stage_c': 'approve the selected operation or revert to identity',
            'operation_admission': ('deterministic structured evidence gate before Kev; '
                                    'Kev cannot override a rejected operation'),
            'risk_review': ('fresh bounded call with no Stage A/B rationale; '
                            'hard vetoes and reviewer failures commit identity'),
            'cohort_policy': 'first_retrieval_fixed_for_this_origin',
            'new_operation_requires_incremental_paired_diagnosis': True,
            'no_mature_cohort_policy': 'submit identity',
            'default_remains_legal': True,
            'operation_semantics': 'training-mean counterfactual; not missingness or column deletion',
        },
        'calls': [], 'tools': [], 'reviews': [], 'fallback': None,
        'iterations': [], 'revisions': [], 'optional_evidence_actions': [],
        'stage_a_decisions': [], 'stage_b_decisions': [], 'stage_c_decisions': [],
        'risk_vetoes': [],
        'resource_accounting': {'costs_basis': 'observed_or_reserved_upper_bound_on_exception',
                                'unknown_actual_cost_events': []},
    }

    def finish(artifact, reason):
        validate_artifact(context, artifact)
        trace.setdefault('final_commit_reason', reason)
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

    def choose(criteria, instruction, *, phase, stage, request_state=None):
        if not criteria:
            raise ValueError('No legal choices remain in R21 phase ' + phase)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or state['costs']['api_calls'] >= limits.api_calls:
            raise TimeoutError('R21 decision budget exhausted')
        state['phase'] = phase
        request_state = (compact_request_state(context, state, limits)
                         if request_state is None else clone(request_state))
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
        elif stage == 'C':
            trace['stage_c_decisions'].append({'phase': phase, 'action': action})
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
        retrieval_observation = clone(result['observation'])
        cohort_ids = [item['episode_id'] for item in retrieval_observation.get('items', [])]
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
        registry_specs = {key: asdict(value) for key, value in host.registry.specs.items()}
        report_provenance = {
            'predictor_fingerprint': digest(host.registry.model_identities),
            'operation_admission_fingerprint': digest({
                'release': host.registry.release_id, 'specs': registry_specs}),
            'skill_fingerprint': digest(trace['skills']),
            'source_fingerprint': _source_fingerprint(),
        }
        evidence_report = build_evidence_report(
            context, latest_diagnostic, retrieval_observation, numerical_state,
            default.candidate_id, report_provenance)
        trace['evidence_report'] = clone(evidence_report)
        trace['evidence_report_sha256'] = digest(evidence_report)
        state['observations'].append({
            'action': 'r22_structured_evidence_report', 'status': 'ok',
            'execution_mode': 'protocol_required',
            'observation': clone(evidence_report),
        })
        eligible = evidence_report['eligible_candidate_ids']
        if not eligible:
            return finish(default, 'identity_no_operation_passed_structured_evidence')

        def selection_request_state(*, stage_a_authorized):
            request = _request_state(context, state, limits)
            request.update({
                'observations': [{
                    'action': 'r22_structured_evidence_report', 'status': 'ok',
                    'execution_mode': 'protocol_required',
                    'observation': clone(evidence_report),
                }],
                'action_history': list(state['executed_actions']),
                'stage_a_authorized': stage_a_authorized,
                'compression': {
                    'policy': 'validated_structured_report_only',
                    'raw_retrieval_and_diagnostics_retained_in_trace': True,
                    'raw_retrieval_and_diagnostics_sent_to_kev': False,
                },
            })
            return request

        gate = choose({
            'use_identity': 'Keep the complete canonical input and submit the default forecast.',
            'use_operated_input': ('At least one protocol-admitted operation has stable paired historical '
                                   'benefit and changes the current forecast; continue to choose only among '
                                   'the admitted operation IDs recorded in input_operation_admission.'),
        }, prompt('plan_forecast', 'assess_state', 'compare_candidates'),
            phase='decide_whether_to_change_input', stage='A',
            request_state=selection_request_state(stage_a_authorized=False))
        if gate == 'use_identity':
            return finish(default, 'agent_commit_identity')
        criteria = {'commit:' + key: 'Submit evidence-admitted diagnosed candidate ' + key
                    for key in eligible}
        selected = choose(criteria, prompt('compose_forecast_request', 'compare_candidates',
                                            'review_and_commit'),
                          phase='select_input_operation', stage='B',
                          request_state=selection_request_state(stage_a_authorized=True))
        selected_candidate_id = selected[len('commit:'):]
        trace['stage_b_selected_candidate_id'] = selected_candidate_id
        expected_provenance = clone(evidence_report['provenance'])
        review_packet = build_review_packet(
            evidence_report, selected_candidate_id, default.candidate_id,
            expected_provenance)
        trace['risk_review_packet_sha256'] = review_packet['packet_sha256']
        trace['risk_vetoes'] = list(review_packet['hard_vetoes'])
        if not review_packet['review_allowed']:
            review_decision = validate_review_action(
                review_packet, {'action': APPROVE_SELECTED})
            trace['risk_review_decision'] = clone(review_decision)
            trace['final_commit_reason'] = 'hard_veto:' + review_decision['reason_code']
            return finish(default, 'risk_hard_veto_revert_identity')

        review_state = _request_state(context, state, limits)
        review_state.update({
            'observations': [{
                'action': 'risk_review_packet', 'status': 'ok',
                'execution_mode': 'protocol_required',
                'observation': clone(review_packet),
            }],
            'action_history': [],
            'selected_candidate_id': selected_candidate_id,
            'review_independence': ('fresh risk-only request; no Stage A/B rationale; '
                                    'no candidate switching or new evidence'),
        })
        try:
            review_action = choose({
                APPROVE_SELECTED: 'Approve only the already selected candidate.',
                REVERT_TO_IDENTITY: 'Reject the selected candidate and commit identity.',
            }, prompt('assess_state', 'compare_candidates', 'review_and_commit'),
                phase='independent_risk_review', stage='C', request_state=review_state)
            response = {'action': review_action}
            if review_action == REVERT_TO_IDENTITY:
                response['reason_code'] = 'risk_reviewer_uncertain'
            review_decision = validate_review_action(review_packet, response)
        except TimeoutError:
            review_decision = fail_closed_decision(review_packet, 'risk_review_timeout')
        except Exception:
            review_decision = fail_closed_decision(review_packet, 'risk_review_exception')
        trace['risk_review_decision'] = clone(review_decision)
        if trace['stage_c_decisions']:
            trace['stage_c_decisions'][-1].update({
                'selected_candidate_id': selected_candidate_id,
                'final_candidate_id': review_decision['final_candidate_id'],
                'reason_code': review_decision['reason_code'],
            })
        if review_decision['final_candidate_id'] == default.candidate_id:
            trace['final_commit_reason'] = 'risk_review_revert'
            return finish(default, 'risk_review_revert_identity')
        trace['final_commit_reason'] = 'risk_review_approve'
        return finish(artifact(selected_candidate_id), 'risk_review_approved_commit')
    except TimeoutError as exc:
        return fallback('timeout', str(exc))
    except Exception as exc:
        return fallback('protocol_error', type(exc).__name__ + ': ' + str(exc))
