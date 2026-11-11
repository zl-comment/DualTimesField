"""Behavioral admission and replay checks for the isolated R23 policy."""
from copy import deepcopy
import json

import pytest

from forecasting.adaptive_agent import input_policy_r22, input_policy_r23
from forecasting.adaptive_agent.r23_evidence_report import apply_admission
from forecasting.adaptive_agent.replay import main, policy_host_factory, policy_module
from forecasting.adaptive_agent.iterative_runtime import R22IterativeToolHost
from test_adaptive_agent_input_policy_r22 import Client, Host, context, DEFAULT, OPERATED


def evidence():
    client = Client(['retrieve_analogs:recent', 'use_identity'])
    _, trace = input_policy_r22.run(context(), Host(), client)
    return trace['evidence_report']


def test_negative_six_hour_bucket_can_pass_positive_half_days():
    report = evidence()
    effect = report['operation_effects'][OPERATED]
    effect['horizon_qs_gain']['hours_1_6']['mean'] = -0.1
    effect['eligible'] = False
    effect['rejection_reasons'] = ['nonpositive_horizon_bucket_qs_gain']
    report['eligible_candidate_ids'] = []
    before = deepcopy(report)
    revised = apply_admission(report)
    assert revised['eligible_candidate_ids'] == [OPERATED]
    assert report == before
    assert revised['operation_effects'][OPERATED]['horizon_qs_gain'] == effect['horizon_qs_gain']


@pytest.mark.parametrize('bucket', ['hours_1_6', 'hours_13_18'])
def test_nonpositive_half_day_still_rejected(bucket):
    report = evidence()
    report['operation_effects'][OPERATED]['horizon_qs_gain'][bucket]['mean'] = -0.5
    revised = apply_admission(report)
    assert not revised['eligible_candidate_ids']
    assert any('12_qs_gain' in reason for reason in
               revised['operation_effects'][OPERATED]['rejection_reasons'])


@pytest.mark.parametrize('updates,reason', [
    ({'available': False}, 'similarity_evidence_unavailable'),
    ({'mean_qs_gain': -0.1}, 'nonpositive_similarity_weighted_qs_gain'),
    ({'mean_qs_gain': 0.0}, 'nonpositive_similarity_weighted_qs_gain'),
    ({'effective_support': 4.99}, 'insufficient_similarity_effective_support'),
])
def test_similarity_evidence_has_actual_admission_authority(updates, reason):
    report = evidence()
    report['operation_effects'][OPERATED]['similarity_weighted'].update(updates)
    revised = apply_admission(report)
    assert not revised['eligible_candidate_ids']
    assert reason in revised['operation_effects'][OPERATED]['rejection_reasons']


def test_effective_support_boundary_and_nonfinite_rejection():
    report = evidence()
    report['operation_effects'][OPERATED]['similarity_weighted']['effective_support'] = 5.0
    assert apply_admission(report)['eligible_candidate_ids'] == [OPERATED]
    report['operation_effects'][OPERATED]['similarity_weighted']['mean_qs_gain'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        apply_admission(report)


class ConflictingHorizonHost(Host):
    def execute(self, ctx, action, state):
        result = super().execute(ctx, action, state)
        if action.startswith('diagnose:'):
            paired = result['observation']['candidates'][OPERATED]['paired_vs_default']
            paired['horizon_qs_gain']['hours_1_6']['mean'] = -0.1
        return result


def test_r23_opens_review_for_candidate_that_r22_rejects():
    ctx = context()
    selected22, trace22 = input_policy_r22.run(
        ctx, ConflictingHorizonHost(), Client(['retrieve_analogs:recent']))
    assert selected22.candidate_id == DEFAULT
    assert not trace22['stage_a_decisions']
    client = Client(['retrieve_analogs:recent', 'use_operated_input',
                     'commit:' + OPERATED, 'approve_selected'])
    selected23, trace23 = input_policy_r23.run(ctx, ConflictingHorizonHost(), client)
    assert selected23.candidate_id == OPERATED
    assert trace23['fallback'] is None
    assert trace23['protocol']['admission_contract']['schema'] == 'r23-state-aware-admission-v1'
    review = client.calls[-1][2]
    assert review['action_history'] == []
    assert review['admission_contract']['six_hour_buckets'] == 'review_evidence_not_individual_hard_vetoes'


@pytest.mark.parametrize('action', ['approve_selected', 'revert_to_identity'])
def test_reviewer_retains_bounded_choice(action):
    client = Client(['retrieve_analogs:recent', 'use_operated_input',
                     'commit:' + OPERATED, action])
    selected, trace = input_policy_r23.run(context(), Host(), client)
    assert selected.candidate_id == (OPERATED if action == 'approve_selected' else DEFAULT)
    assert trace['fallback'] is None


def test_new_admission_cannot_override_risk_veto():
    client = Client(['retrieve_analogs:recent', 'use_operated_input', 'commit:' + OPERATED])
    selected, trace = input_policy_r23.run(context(), ConflictingHorizonHost(width_delta=0.2), client)
    assert selected.candidate_id == DEFAULT
    assert trace['final_commit_reason'] == 'hard_veto:risk_width_inflation'
    assert not trace['stage_c_decisions']


def test_r23_timeout_is_fail_closed():
    client = Client(['retrieve_analogs:recent', 'use_operated_input', 'commit:' + OPERATED],
                    fail_review=True)
    selected, trace = input_policy_r23.run(context(), Host(), client)
    assert selected.candidate_id == DEFAULT
    assert trace['risk_review_decision']['reason_code'] == 'risk_review_timeout'


def test_replay_registration_contract_and_plan_only(tmp_path, capsys):
    assert policy_module('r23_state_aware_input') is input_policy_r23
    assert policy_host_factory('r23_state_aware_input') is R22IterativeToolHost
    contract = input_policy_r23.manifest_contract()
    assert contract['admission_contract']['schema'] == 'r23-state-aware-admission-v1'
    assert contract['risk_thresholds'] == input_policy_r22.manifest_contract()['risk_thresholds']
    assert all(skill['version'] == '7.0.0' for skill in input_policy_r23.load_skills().values())
    output = tmp_path / 'plan'
    assert main(['--policy', 'r23_state_aware_input', '--operator-admission', 'unused.json',
                 '--split', 'selection', '--output', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['configuration']['policy'] == 'r23_state_aware_input'
    assert not output.exists()
