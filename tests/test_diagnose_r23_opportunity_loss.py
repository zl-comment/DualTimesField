"""Synthetic tests for the R23 opportunity-loss diagnosis."""
import json

import numpy as np
import pytest

from forecasting.adaptive_agent.replay import bundle_name, digest, sha_file
from scripts.diagnose_r23_opportunity_loss import (
    IDENTITY_ID, analyze, branch_decision, load_origin_row, losses, origin_diagnosis,
    region_summary,
)
from test_analyze_r22_blocked_cv import (
    KEEP24, KEEP48, REGIONS, input_declarations, quantiles, write_cache, write_json,
)


def write_r23_root(root, cache, split, origins):
    inputs = input_declarations(cache, split)
    for region in REGIONS:
        targets = np.load(cache / 'data_cache' / region / split / 'target_time_unix.npy')
        actual = np.load(cache / 'data_cache' / region / split / 'actual.npy')
        schedule = [(int(origin), region, index) for index, origin in enumerate(origins)]
        manifest = {
            'configuration': {'policy': 'r23_state_aware_input', 'split': split,
                              'regions': [region], 'origins_per_region': str(len(origins)),
                              'seed': 2026},
            'inputs': {region: inputs[region]}, 'expected_origins': len(schedule),
            'schedule_sha256': digest(schedule),
        }
        manifest_id = digest(manifest)
        region_root = root / region
        (region_root / 'origins').mkdir(parents=True)
        write_json(region_root / 'run_manifest.json', manifest)
        for index, origin in enumerate(origins):
            # index 0: identity only, no mature cohort (early stop, zero headroom).
            # index 1: three candidates, nothing eligible (admission loss case).
            # index 2: keep24 eligible, agent commits identity (selection loss case).
            # index 3: keep24 eligible and committed (realized gain case).
            identity = quantiles(actual[index] + 2)
            keep24 = quantiles(actual[index])
            keep48 = quantiles(actual[index] - 3)
            if index == 0:
                candidates, eligible = {IDENTITY_ID: identity}, []
                committed_id, agent, stop = IDENTITY_ID, identity, 'identity_no_mature_cohort'
            elif index == 1:
                candidates, eligible = {IDENTITY_ID: identity, KEEP24: keep24, KEEP48: keep48}, []
                committed_id, agent, stop = IDENTITY_ID, identity, \
                    'identity_no_operation_passed_structured_evidence'
            elif index == 2:
                candidates, eligible = {IDENTITY_ID: identity, KEEP24: keep24, KEEP48: keep48}, [KEEP24]
                committed_id, agent, stop = IDENTITY_ID, identity, 'agent_commit_identity'
            else:
                candidates, eligible = {IDENTITY_ID: identity, KEEP24: keep24, KEEP48: keep48}, [KEEP24]
                committed_id, agent, stop = KEEP24, keep24, 'risk_review_approved_commit'
            directory = region_root / 'origins' / bundle_name(region, origin)
            directory.mkdir()
            ids = list(candidates)
            with (directory / 'predictions.npz').open('wb') as stream:
                np.savez_compressed(stream, base=identity, agent=agent,
                                    target_time_unix=targets[index])
            with (directory / 'materialized_candidates.npz').open('wb') as stream:
                np.savez_compressed(stream, candidate_ids=np.asarray(ids),
                                    forecast_ids=np.asarray([f'id:{name}' for name in ids]),
                                    quantiles=np.stack([candidates[name] for name in ids]),
                                    target_time_unix=targets[index])
            trace = {'selected_candidate_id': committed_id,
                     'selected_forecast_id': f'synth:{region}:{index}:agent',
                     'stop_reason': stop,
                     'evidence_report': {'eligible_candidate_ids': eligible}}
            write_json(directory / 'trace.json', trace)
            write_json(directory / 'audit.json', {'synthetic': True})
            record = {
                'manifest_id': manifest_id, 'region': region, 'origin_unix': int(origin),
                'index': index, 'context_id': f'{region}:{origin}',
                'base': {'candidate_id': IDENTITY_ID, 'identity': f'synth:{region}:{index}:base'},
                'agent': {'candidate_id': committed_id,
                          'identity': trace['selected_forecast_id']},
                'files': {name: sha_file(directory / name) for name in
                          ('predictions.npz', 'trace.json', 'audit.json',
                           'materialized_candidates.npz')},
            }
            write_json(directory / 'record.json', record)
        write_json(region_root / 'report.json', {'status': 'COMPLETE',
                                                 'manifest_id': manifest_id,
                                                 'origins': len(schedule),
                                                 'expected_origins': len(schedule)})
        write_json(region_root / 'progress.json', {'status': 'COMPLETE',
                                                   'completed': len(schedule),
                                                   'total': len(schedule)})
        write_json(region_root / 'predictions_complete.json',
                   {'manifest_id': manifest_id, 'completed': len(schedule),
                    'expected': len(schedule)})


@pytest.fixture
def synthetic_r23(tmp_path):
    split = 'calibration'
    origins = np.arange(4, dtype=np.int64) * 86400 + 1656597600
    cache = tmp_path / 'aligned'
    write_cache(cache, split, origins)
    root = tmp_path / 'r23'
    write_r23_root(root, cache, split, origins)
    return root, cache


def test_analyze_decomposes_synthetic_run(synthetic_r23):
    root, cache = synthetic_r23
    result = analyze(root, cache)
    assert result['status'] == 'COMPLETE_OPPORTUNITY_LOSS_DIAGNOSIS'
    assert result['branch_decision']['action'] in (
        'parameterize_operations_first', 'proceed_with_r24a_loop', 'admission_contract_first')
    for region in REGIONS:
        block = result['regions'][region]
        assert block['origins'] == 4
        assert block['early_stop_origins'] == 2
        assert block['eligible_nonempty_origins'] == 2
        assert block['committed_nonidentity_origins'] == 1
        # keep24 is the post-hoc materialized oracle wherever it exists (indices 1-3)
        assert block['oracle_materialized_pick_counts'][KEEP24] == 3
        assert block['oracle_materialized_pick_counts'][IDENTITY_ID] == 1
        # index 0 (identity==identity), 1 (eligible empty -> identity), 3 (keep24)
        assert block['committed_matches_oracle_eligible'] == 3
        qs = block['qs']
        # index 0 contributes zero everywhere; keep24 beats identity on 1-3
        assert qs['generation_headroom_identity_minus_oracle_materialized'] > 0
        # admission loss only on index 1 (keep24 materialized, nothing eligible)
        assert qs['admission_loss_oracle_eligible_minus_oracle_materialized'] > 0
        # selection loss only on index 2 (eligible keep24, committed identity)
        assert qs['selection_risk_loss_committed_minus_oracle_eligible'] > 0
        # telescoping identity: identity - committed = G - A - S
        realized = qs['identity'] - qs['committed']
        assert realized == pytest.approx(
            qs['generation_headroom_identity_minus_oracle_materialized']
            - qs['admission_loss_oracle_eligible_minus_oracle_materialized']
            - qs['selection_risk_loss_committed_minus_oracle_eligible'])
        assert set(losses(block)) == {'G_generation_headroom', 'A_admission_loss',
                                      'S_selection_risk_loss'}
    assert result['region_equal']['losses']['G_generation_headroom'] == pytest.approx(
        float(np.mean([result['regions'][region]['losses']['G_generation_headroom']
                       for region in REGIONS])))


def test_origin_diagnosis_sign_conventions():
    actual = np.full(24, 50.0)
    identity = quantiles(actual + 2)
    keep24 = quantiles(actual)
    keep48 = quantiles(actual - 3)
    committed = keep24
    diagnosis = origin_diagnosis(actual, {IDENTITY_ID: identity, KEEP24: keep24,
                                          KEEP48: keep48}, [KEEP24], committed)
    qs = diagnosis['qs']
    assert diagnosis['oracle_materialized_id_qs'] == KEEP24
    assert diagnosis['oracle_eligible_id_qs'] == KEEP24
    assert qs['generation_headroom_identity_minus_oracle_materialized'] > 0
    assert qs['admission_loss_oracle_eligible_minus_oracle_materialized'] == 0
    assert qs['selection_risk_loss_committed_minus_oracle_eligible'] == 0
    assert qs['oracle_materialized'] <= qs['identity']
    assert qs['oracle_eligible'] >= qs['oracle_materialized']
    assert qs['committed'] >= qs['oracle_eligible']


def test_origin_diagnosis_identity_only_legal_set():
    actual = np.full(24, 50.0)
    identity = quantiles(actual + 2)
    keep24 = quantiles(actual)
    diagnosis = origin_diagnosis(actual, {IDENTITY_ID: identity, KEEP24: keep24}, [], identity)
    qs = diagnosis['qs']
    assert diagnosis['oracle_eligible_id_qs'] == IDENTITY_ID
    assert qs['admission_loss_oracle_eligible_minus_oracle_materialized'] == pytest.approx(
        qs['identity'] - qs['oracle_materialized'])
    assert qs['selection_risk_loss_committed_minus_oracle_eligible'] == 0


def test_origin_diagnosis_rejects_invalid_sets():
    actual = np.full(24, 50.0)
    identity = quantiles(actual)
    with pytest.raises(ValueError, match='Identity candidate missing'):
        origin_diagnosis(actual, {'other': identity}, [], identity)
    with pytest.raises(ValueError, match='absent from materialized'):
        origin_diagnosis(actual, {IDENTITY_ID: identity}, [KEEP24], identity)
    with pytest.raises(ValueError, match='finite actuals'):
        origin_diagnosis(np.full(24, np.nan), {IDENTITY_ID: identity}, [], identity)


def test_load_origin_row_rejects_early_stop_with_eligible(tmp_path):
    actual = np.full(24, 50.0)
    identity = quantiles(actual + 2)
    keep24 = quantiles(actual)
    directory = tmp_path / bundle_name('NSW1', 1700000000)
    directory.mkdir()
    targets = 1700000000 + np.arange(24) * 3600
    with (directory / 'materialized_candidates.npz').open('wb') as stream:
        np.savez_compressed(stream,
                            candidate_ids=np.asarray([IDENTITY_ID, KEEP24]),
                            forecast_ids=np.asarray(['a', 'b']),
                            quantiles=np.stack([identity, keep24]),
                            target_time_unix=targets)
    row = {'arrays': {'target_time_unix': targets, 'agent': identity},
           'record': {'agent': {'candidate_id': IDENTITY_ID}},
           'trace': {'stop_reason': 'identity_no_operation_passed_structured_evidence',
                     'evidence_report': {'eligible_candidate_ids': [KEEP24]}}}
    with pytest.raises(ValueError, match='cannot have eligible'):
        load_origin_row(tmp_path, 'NSW1', 1700000000, actual, row)


def test_branch_decision_rule():
    def losses_for(G, A, S):
        return {'G_generation_headroom': G, 'A_admission_loss': A,
                'S_selection_risk_loss': S}
    assert branch_decision(losses_for(0.01, 0.0, 0.0))[0] == 'parameterize_operations_first'
    assert branch_decision(losses_for(0.10, 0.02, 0.05))[0] == 'proceed_with_r24a_loop'
    assert branch_decision(losses_for(0.10, 0.05, 0.02))[0] == 'admission_contract_first'
    assert branch_decision(losses_for(0.05, 0.01, 0.01))[0] == 'admission_contract_first'


def test_region_summary_counts():
    def diagnosis(value):
        fields = ('identity', 'oracle_materialized', 'oracle_eligible', 'committed',
                  'generation_headroom_identity_minus_oracle_materialized',
                  'admission_loss_oracle_eligible_minus_oracle_materialized',
                  'selection_risk_loss_committed_minus_oracle_eligible')
        return {'oracle_materialized_id_qs': KEEP24, 'oracle_eligible_id_qs': KEEP24,
                'qs': dict.fromkeys(fields, value), 'mae': dict.fromkeys(fields, value)}
    rows = [
        {'stop_reason': 'agent_commit_identity', 'eligible': [KEEP24],
         'committed_id': IDENTITY_ID, 'diagnosis': diagnosis(1.0)},
        {'stop_reason': 'risk_review_approved_commit', 'eligible': [KEEP24],
         'committed_id': KEEP24, 'diagnosis': diagnosis(3.0)},
    ]
    summary = region_summary(rows)
    assert summary['origins'] == 2
    assert summary['early_stop_origins'] == 0
    # row 1 commits identity while keep24 is the eligible oracle (selection loss);
    # row 2 commits the eligible oracle exactly.
    assert summary['committed_matches_oracle_eligible'] == 1
    assert summary['stop_reasons'] == {'agent_commit_identity': 1,
                                       'risk_review_approved_commit': 1}
    assert summary['qs']['identity'] == pytest.approx(2.0)
