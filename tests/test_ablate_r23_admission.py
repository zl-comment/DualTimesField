"""Synthetic tests for the pre-registered R23 admission ablation."""
import json

import numpy as np
import pytest

from forecasting.adaptive_agent.replay import bundle_name, digest, sha_file
from scripts.ablate_r23_admission import (
    IDENTITY_ID, VARIANTS, adoption_decision, eligible_under, rejection_reasons,
    select_commit_for, analyze,
)
from test_analyze_r22_blocked_cv import (
    KEEP24, KEEP48, REGIONS, input_declarations, quantiles, write_cache, write_json,
)


def base_effect(**overrides):
    """An operation effect that passes every V0 condition."""
    effect = {
        'candidate_id': KEEP24,
        'common_support': 8,
        'paired_qs_gain': {'mean_qs_gain': 0.5, 'median_qs_gain': 0.4,
                           'win_fraction': 0.75},
        'horizon_qs_gain': {bucket: {'mean': 0.3} for bucket in
                            ('hours_1_6', 'hours_7_12', 'hours_13_18', 'hours_19_24')},
        'similarity_weighted': {'available': True, 'effective_support': 7,
                                'mean_qs_gain': 0.6},
        'current_forecast': {'current_disagreement': {'median_absolute_difference': 1.0}},
    }
    effect.update(overrides)
    return effect


def report_for(effects, eligible_ids=None):
    eligible = sorted(cid for cid, effect in effects.items() if effect.pop('eligible', True)) \
        if eligible_ids is None else eligible_ids
    return {'operation_effects': effects,
            'cohort_quality': {'coverage_grade': 'sufficient'},
            'eligible_candidate_ids': eligible}


def test_v0_reproduces_saved_eligibility_and_reasons():
    blocked_half_day = base_effect()
    blocked_half_day['horizon_qs_gain'] = {
        'hours_1_6': {'mean': 0.3}, 'hours_7_12': {'mean': -0.4},
        'hours_13_18': {'mean': 0.3}, 'hours_19_24': {'mean': 0.3}}
    report = report_for({KEEP24: base_effect(), KEEP48: blocked_half_day},
                        eligible_ids=[KEEP24])
    rejections = eligible_under(report, VARIANTS['V0_baseline'])
    assert not rejections[KEEP24]
    assert rejections[KEEP48] == ['nonpositive_first12_qs_gain']
    assert eligible_under(report, VARIANTS['V1_drop_half_day'])[KEEP48] == []
    assert 'nonpositive_first12_qs_gain' in eligible_under(
        report, VARIANTS['V2_support_3'])[KEEP48]


def test_variant_condition_differences():
    low_support = base_effect(common_support=4)
    low_support['similarity_weighted'] = {'available': True, 'effective_support': 4,
                                          'mean_qs_gain': 0.6}
    reasons_v0 = rejection_reasons(low_support, 'sufficient', VARIANTS['V0_baseline'])
    reasons_v2 = rejection_reasons(low_support, 'sufficient', VARIANTS['V2_support_3'])
    assert set(reasons_v0) == {'insufficient_common_support',
                               'insufficient_similarity_effective_support'}
    assert reasons_v2 == []

    zero_change = base_effect()
    zero_change['current_forecast'] = {
        'current_disagreement': {'median_absolute_difference': 0.0}}
    assert rejection_reasons(zero_change, 'sufficient',
                             VARIANTS['V0_baseline']) == ['zero_current_forecast_change']
    assert rejection_reasons(zero_change, 'sufficient',
                             VARIANTS['V3_drop_current_change']) == []

    weak_win = base_effect()
    weak_win['paired_qs_gain'] = {'mean_qs_gain': 0.5, 'median_qs_gain': -0.1,
                                  'win_fraction': 0.4}
    assert set(rejection_reasons(weak_win, 'sufficient', VARIANTS['V0_baseline'])) == {
        'nonpositive_median_qs_gain', 'weak_win_fraction'}
    assert rejection_reasons(weak_win, 'sufficient', VARIANTS['V4_state_gate_only']) == []

    unavailable = base_effect()
    unavailable['similarity_weighted'] = {'available': False}
    for name in VARIANTS:
        assert 'similarity_evidence_unavailable' in rejection_reasons(
            unavailable, 'sufficient', VARIANTS[name])
    assert rejection_reasons(base_effect(), 'insufficient',
                             VARIANTS['V0_baseline']) == ['cohort_quality_not_sufficient']


def test_select_commit_tie_break_order():
    weaker_similarity = base_effect()
    weaker_similarity['similarity_weighted'] = {'available': True, 'effective_support': 7,
                                                'mean_qs_gain': 0.2}
    effects = {KEEP24: weaker_similarity, KEEP48: base_effect()}
    report = report_for(effects)
    rejections = {cid: [] for cid in effects}
    assert select_commit_for(report, rejections) == KEEP48

    equal = base_effect()
    equal['paired_qs_gain'] = {'mean_qs_gain': 0.1, 'median_qs_gain': 0.4,
                               'win_fraction': 0.75}
    effects = {KEEP24: base_effect(), KEEP48: equal}
    assert select_commit_for(report_for(effects), {cid: [] for cid in effects}) == KEEP24
    assert select_commit_for(report_for({}), {}) is None
    assert select_commit_for(report_for({KEEP24: base_effect()}), {KEEP24: []}) == KEEP24


def test_adoption_decision_rule():
    def block(eligible, qs, mae, coverage=0.0):
        return {'eligible_nonempty_origins': eligible,
                'deltas_vs_identity': {'qs': qs, 'mae': mae, 'coverage90': coverage}}

    variants = {'V0_baseline': {r: block(10, -0.01, -0.01) for r in REGIONS}}
    variants.update({name: {r: block(10, -0.01, -0.01) for r in REGIONS}
                     for name in ('V1_drop_half_day', 'V2_support_3',
                                  'V4_state_gate_only')})
    variants['V3_drop_current_change'] = {r: block(11, -0.02, -0.01) for r in REGIONS}
    variants['V3_drop_current_change']['NSW1'] = block(12, -0.02, -0.01)
    variants['V1_drop_half_day'] = {r: block(30, -0.03, -0.02) for r in REGIONS}
    decision = adoption_decision(variants)
    # V3 qualifies with the fewest newly eligible origins; V1 also qualifies.
    assert decision['action'] == 'adopt_variant_into_r24a_contract'
    assert decision['variant'] == 'V3_drop_current_change'
    assert set(decision['qualifying_variants']) == {'V3_drop_current_change',
                                                    'V1_drop_half_day'}

    variants['V1_drop_half_day']['QLD1'] = block(30, 0.01, -0.02)
    variants['V3_drop_current_change']['QLD1'] = block(12, -0.01, 0.02)
    decision = adoption_decision(variants)
    assert decision['action'] == 'keep_r23_admission_proceed_r24a'
    assert decision['qualifying_variants'] == []


def write_r23_root(root, cache, split, origins):
    """Three origins per region: report-bearing, no-report, and hard-vetoed."""
    inputs = input_declarations(cache, split)
    for region in REGIONS:
        targets = np.load(cache / 'data_cache' / region / split / 'target_time_unix.npy')
        actual = np.load(cache / 'data_cache' / region / split / 'actual.npy')
        schedule = [(int(origin), region, index) for index, origin in enumerate(origins)]
        manifest = {'configuration': {'policy': 'r23_state_aware_input', 'split': split,
                                      'regions': [region],
                                      'origins_per_region': str(len(origins)),
                                      'seed': 2026},
                    'inputs': {region: inputs[region]},
                    'expected_origins': len(schedule),
                    'schedule_sha256': digest(schedule)}
        manifest_id = digest(manifest)
        region_root = root / region
        (region_root / 'origins').mkdir(parents=True)
        write_json(region_root / 'run_manifest.json', manifest)
        for index, origin in enumerate(origins):
            identity = quantiles(actual[index] + 2)
            keep24 = quantiles(actual[index])
            keep48 = quantiles(actual[index] - 3)
            # index 0 carries a report admitting keep24 (V0) plus a half-day-blocked
            # keep48; index 1 has no report (no mature cohort); index 2 selected
            # keep24 and was reverted by a material-downside hard veto.
            if index == 0:
                blocked = base_effect()
                blocked['candidate_id'] = KEEP48
                blocked['horizon_qs_gain'] = {
                    'hours_1_6': {'mean': 0.3}, 'hours_7_12': {'mean': -0.4},
                    'hours_13_18': {'mean': 0.3}, 'hours_19_24': {'mean': 0.3}}
                effect24 = base_effect()
                effect24['similarity_weighted'] = {'available': True,
                                                   'effective_support': 7,
                                                   'mean_qs_gain': 0.6}
                evidence = report_for({KEEP24: effect24, KEEP48: blocked},
                                      eligible_ids=[KEEP24])
                committed_id, agent, stop = IDENTITY_ID, identity, 'agent_commit_identity'
            elif index == 1:
                evidence, committed_id, agent, stop = {}, IDENTITY_ID, identity, \
                    'identity_no_mature_cohort'
            else:
                evidence, committed_id, agent, stop = {}, IDENTITY_ID, identity, \
                    'risk_hard_veto_revert_identity'
            directory = region_root / 'origins' / bundle_name(region, origin)
            directory.mkdir()
            ids = [IDENTITY_ID, KEEP24, KEEP48] if index != 1 else [IDENTITY_ID]
            forecasts = {IDENTITY_ID: identity, KEEP24: keep24, KEEP48: keep48}
            with (directory / 'predictions.npz').open('wb') as stream:
                np.savez_compressed(stream, base=identity, agent=agent,
                                    target_time_unix=targets[index])
            with (directory / 'materialized_candidates.npz').open('wb') as stream:
                np.savez_compressed(stream, candidate_ids=np.asarray(ids),
                                    forecast_ids=np.asarray([f'id:{name}' for name in ids]),
                                    quantiles=np.stack([forecasts[name] for name in ids]),
                                    target_time_unix=targets[index])
            trace = {'selected_candidate_id': committed_id,
                     'selected_forecast_id': f'synth:{region}:{index}:agent',
                     'stop_reason': stop, 'evidence_report': evidence,
                     'stage_b_selected_candidate_id': KEEP24 if index == 2 else None,
                     'risk_vetoes': ['risk_material_downside'] if index == 2 else [],
                     'final_commit_reason': 'hard_veto:risk_material_downside'
                     if index == 2 else stop}
            write_json(directory / 'trace.json', trace)
            write_json(directory / 'audit.json', {'synthetic': True})
            write_json(directory / 'record.json', {
                'manifest_id': manifest_id, 'region': region, 'origin_unix': int(origin),
                'index': index, 'context_id': f'{region}:{origin}',
                'base': {'candidate_id': IDENTITY_ID,
                         'identity': f'synth:{region}:{index}:base'},
                'agent': {'candidate_id': committed_id,
                          'identity': trace['selected_forecast_id']},
                'files': {name: sha_file(directory / name) for name in
                          ('predictions.npz', 'trace.json', 'audit.json',
                           'materialized_candidates.npz')}})
        write_json(region_root / 'report.json',
                   {'status': 'COMPLETE', 'manifest_id': manifest_id,
                    'origins': len(schedule), 'expected_origins': len(schedule)})
        write_json(region_root / 'progress.json', {'status': 'COMPLETE',
                                                   'completed': len(schedule),
                                                   'total': len(schedule)})
        write_json(region_root / 'predictions_complete.json',
                   {'manifest_id': manifest_id, 'completed': len(schedule),
                    'expected': len(schedule)})


@pytest.fixture
def synthetic_r23(tmp_path):
    split = 'calibration'
    origins = np.arange(3, dtype=np.int64) * 86400 + 1656597600
    cache = tmp_path / 'aligned'
    write_cache(cache, split, origins)
    root = tmp_path / 'r23'
    write_r23_root(root, cache, split, origins)
    return root, cache


def test_analyze_runs_ablation_with_guards(synthetic_r23):
    root, cache = synthetic_r23
    result = analyze(root, cache)
    assert result['status'] == 'COMPLETE_ADMISSION_ABLATION'
    for region in REGIONS:
        block = result['variants']['V0_baseline'][region]
        assert block['eligible_nonempty_origins'] == 1
        assert block['committed_nonidentity_origins'] == 1
        assert block['committed_candidate_counts'][KEEP24] == 1
        assert block['deltas_vs_identity']['qs'] < 0
        v1 = result['variants']['V1_drop_half_day'][region]
        assert v1['eligible_nonempty_origins'] == 1  # keep48 loses the tie-break
        assert result['v0_binding_reasons'][region]['nonpositive_first12_qs_gain'] == 1
        veto = result['veto_audit'][region]
        assert veto['origins'] == 1
        assert veto['reason_codes'] == {'risk_material_downside': 1}
        assert veto['deltas_vs_identity']['qs'] < 0  # vetoed keep24 was the better forecast
    assert result['adoption_decision']['action'] in (
        'adopt_variant_into_r24a_contract', 'keep_r23_admission_proceed_r24a')


def test_v0_reproduction_guard_aborts_on_tampering(synthetic_r23, tmp_path):
    root, cache = synthetic_r23
    region = REGIONS[0]
    origin = sorted((root / region / 'origins').iterdir())[0]
    trace = json.loads((origin / 'trace.json').read_text())
    trace['evidence_report']['eligible_candidate_ids'] = [KEEP24, KEEP48]
    (origin / 'trace.json').write_text(json.dumps(trace))
    record_path = origin / 'record.json'
    record = json.loads(record_path.read_text())
    record['files']['trace.json'] = sha_file(origin / 'trace.json')
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='V0 reproduction mismatch'):
        analyze(root, cache)
