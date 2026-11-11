from copy import deepcopy
import json

import numpy as np
import pytest

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.iterative_runtime import IterativeToolHost, R22IterativeToolHost
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.r22_evidence_report import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    build_evidence_report,
    validate_evidence_report,
)
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.state import summarize_state


ORIGIN = 1_700_000_000
DEFAULT = 'lightgbm:full:72'
OPERATED = 'chronos:price_only:24'


def context(origin=ORIGIN):
    price = np.concatenate([np.arange(24.), np.arange(24.) + 10, np.arange(24.) + 20])
    snapshot = Snapshot('NSW1', origin, origin + 3600 * np.arange(24), price,
                        np.concatenate([np.full(24, -.5), np.full(24, .1), np.full(24, .4)]),
                        np.zeros((72, 10)), np.zeros((24, 10)), 'r22-fixture')
    return ForecastContext(snapshot, {'future_exogenous': np.ones((24, 2))},
                           {'future_exogenous_staleness_hours': 1.5})


class Adapter:
    def __init__(self, shift):
        self.shift = shift
        self.identity = {'sha256': f'fixture-{shift}', 'train_end_unix': 100,
                         'selection_end_unix': 200}

    def predict(self, ctx, profile, window):
        return np.tile(np.arange(5.) + self.shift, (24, 1))


def episodes(count=8):
    return [{'episode_id': f'day-{day}', 'context': context(ORIGIN - day * 86400),
             'actual': np.full(24, 10. + day),
             'outcome_available_at': ORIGIN - (day - 1) * 86400,
             'data_role': 'holdout', 'state_tags': ['stable']}
            for day in range(1, count + 1)]


def runtime(kind=R22IterativeToolHost, count=8):
    registry = ForecastRegistry({'lightgbm': Adapter(0), 'chronos': Adapter(1)}, 'r22-fixture')
    return kind(registry, EvidenceStore(), AnalogStore(episodes(count)), FeedbackLedger('r22'))


def diagnose(host, count=8):
    ctx = context()
    host.default(ctx)
    retrieval = host.execute(ctx, 'retrieve_analogs:diverse', {'costs': {'historical_evaluations': 0}})
    host.execute(ctx, 'forecast:' + OPERATED, {'costs': {'historical_evaluations': 0}})
    action = next(key for key in host.actions(ctx, {'costs': {'historical_evaluations': 0}})
                  if key.startswith('diagnose:'))
    diagnostic = host.execute(ctx, action, {'costs': {'historical_evaluations': 0}})['observation']
    assert diagnostic['sample_count'] == count
    return ctx, retrieval['observation'], diagnostic


def provenance():
    return {'predictor_fingerprint': 'predictor-sha',
            'operation_admission_fingerprint': 'admission-sha',
            'skill_fingerprint': 'skill-sha', 'source_fingerprint': 'source-sha'}


def report(count=8):
    ctx, retrieval, diagnostic = diagnose(runtime(count=count), count=count)
    return build_evidence_report(ctx, diagnostic, retrieval, summarize_state(ctx),
                                 DEFAULT, provenance()), diagnostic


def test_r22_host_adds_four_buckets_and_bounded_paired_risk_only():
    _, _, diagnostic = diagnose(runtime())
    paired = diagnostic['candidates'][OPERATED]['paired_vs_default']
    assert diagnostic['r22_aggregate_schema'] == 'r22-paired-diagnostic-v1'
    assert set(paired['horizon_qs_gain']) == {
        'hours_1_6', 'hours_7_12', 'hours_13_18', 'hours_19_24'}
    assert all(row['mean'] > 0 for row in paired['horizon_qs_gain'].values())
    assert paired['coverage90_delta']['mean'] == 0
    assert paired['width90_delta']['mean'] == 0
    assert paired['downside']['material_loss_fraction'] == 0
    assert paired['bootstrap_mean_qs_gain']['seed'] == BOOTSTRAP_SEED
    assert paired['bootstrap_mean_qs_gain']['replicates'] == BOOTSTRAP_REPLICATES
    assert paired['bootstrap_mean_qs_gain']['role'].endswith('not_significance_test')
    assert 'rows' not in diagnostic
    serialized = json.dumps(diagnostic, allow_nan=False)
    assert '"actual"' not in serialized and '"target"' not in serialized


def test_r21_host_observation_remains_unextended():
    _, _, diagnostic = diagnose(runtime(IterativeToolHost))
    assert 'r22_aggregate_schema' not in diagnostic
    metrics = diagnostic['candidates'][DEFAULT]['metrics']
    assert not any(key.startswith('qs_bucket_') for key in metrics)
    paired = diagnostic['candidates'][OPERATED]['paired_vs_default']
    assert 'horizon_qs_gain' not in paired and 'downside' not in paired


def test_report_has_frozen_sections_risk_fields_and_current_only_state():
    result, _ = report()
    assert set(result) == {'schema', 'provenance', 'current_regime', 'cohort_quality',
                           'operation_effects', 'uncertainty_downside',
                           'eligible_candidate_ids'}
    assert result['provenance']['context_id']
    assert result['provenance']['cohort_id']
    assert result['cohort_quality']['common_paired_support'] == 8
    assert result['cohort_quality']['coverage_grade'] == 'sufficient'
    assert result['cohort_quality']['data_quality_valid']
    assert result['current_regime']['data_quality']['critical_failure_count'] == 0
    effect = result['operation_effects'][OPERATED]
    assert effect['common_support'] == 8
    assert effect['current_forecast']['identity_candidate_id'] == DEFAULT
    assert effect['current_forecast']['selected_candidate_id'] == OPERATED
    assert set(effect['current_forecast']['current_disagreement']) == {
        'median_absolute_difference', 'q05_absolute_difference',
        'q95_absolute_difference', 'width90_absolute_difference'}
    assert effect['downside']['material_loss_fraction'] == 0
    assert effect['downside']['lower_tail_mean_qs_gain'] > 0
    assert effect['coverage90_delta']['mean'] == 0
    assert effect['width90_delta']['mean'] == 0
    assert result['eligible_candidate_ids'] == [OPERATED]
    serialized = json.dumps(result, sort_keys=True, allow_nan=False)
    assert '"actual"' not in serialized and '"labels"' not in serialized


def test_small_sample_bootstrap_is_fixed_seed_supporting_evidence():
    first, _ = report(count=4)
    second, _ = report(count=4)
    assert first == second
    uncertainty = first['uncertainty_downside'][OPERATED]
    bootstrap = uncertainty['bootstrap_mean_qs_gain']
    assert bootstrap['seed'] == 20261111
    assert bootstrap['role'] == 'small_sample_supporting_evidence_not_significance_test'
    assert first['cohort_quality']['coverage_grade'] == 'weak'
    assert first['eligible_candidate_ids'] == []


@pytest.mark.parametrize('mutation', ('nan', 'label', 'schema', 'candidate', 'bootstrap'))
def test_strict_validator_rejects_nonfinite_leakage_and_schema_drift(mutation):
    result, _ = report()
    corrupted = deepcopy(result)
    if mutation == 'nan':
        corrupted['operation_effects'][OPERATED]['coverage90_delta']['mean'] = float('nan')
    elif mutation == 'label':
        corrupted['current_regime']['actual'] = [1]
    elif mutation == 'schema':
        corrupted['extra'] = True
    elif mutation == 'candidate':
        corrupted['eligible_candidate_ids'] = ['not-materialized']
    else:
        corrupted['provenance']['bootstrap_seed'] += 1
    with pytest.raises(ValueError):
        validate_evidence_report(corrupted)


def test_builder_rejects_audit_rows_and_provenance_mismatch():
    ctx, retrieval, diagnostic = diagnose(runtime())
    diagnostic['rows'] = []
    with pytest.raises(ValueError, match='bounded API-visible'):
        build_evidence_report(ctx, diagnostic, retrieval, summarize_state(ctx),
                              DEFAULT, provenance())
    diagnostic.pop('rows')
    bad = provenance()
    bad['unexpected'] = 'typo'
    with pytest.raises(ValueError, match='provenance input'):
        build_evidence_report(ctx, diagnostic, retrieval, summarize_state(ctx), DEFAULT, bad)
