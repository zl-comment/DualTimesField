"""R19 initial evidence visibility without extra actions or future information."""
from dataclasses import replace
import json

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.runtime import ToolHost

ORIGIN = 1700000000
DEFAULT = 'lightgbm:full:72'
ALTERNATIVE = 'chronos:price_only:24'


def context(origin=ORIGIN, watermark=None, region='NSW1'):
    return ForecastContext(Snapshot(region, origin, origin + np.arange(24) * 3600,
                                    np.arange(72.), np.zeros(72), np.zeros((72, 10)),
                                    np.zeros((24, 10)), 'fixture'), memory_watermark=watermark)


class FrozenFixture:
    identity = {'sha256': 'fixture', 'train_end_unix': 100, 'selection_end_unix': 200}

    def predict(self, ctx, profile, window):
        return np.tile(np.arange(5.), (24, 1))


def episode(name, origin, region='NSW1'):
    return {'episode_id': name, 'context': context(origin, region=region),
            'actual': np.ones(24), 'outcome_available_at': origin + 86400}


def notice(name, published):
    return {'evidence_id': name, 'source_id': name, 'version': '1', 'availability_basis': 'verified',
            'published_at': published, 'region': 'NSW1', 'payload': {'text': 'A planned outage.'}}


def host(scope='full_evidence', episodes=(), notices=()):
    registry = ForecastRegistry({'lightgbm': FrozenFixture(), 'chronos': FrozenFixture()}, 'r19-fixture')
    return ToolHost(registry, EvidenceStore(notices), AnalogStore(episodes), FeedbackLedger('arm'),
                    information_scope=scope)


def feedback(runtime, origin, candidate=DEFAULT):
    ctx = context(origin)
    artifact = runtime.registry.forecast(ctx, candidate)
    key = runtime.feedback.append_forecast(ctx, artifact)
    runtime.feedback.append_outcome(key, np.ones(24), origin + 86400)
    return key


def test_initial_state_includes_selected_default_feedback_without_counterfactual_claim():
    runtime = host()
    feedback(runtime, ORIGIN - 3 * 86400)
    feedback(runtime, ORIGIN - 2 * 86400, ALTERNATIVE)
    observed = runtime.initial_observation(context())
    state = observed['numerical_state']
    assert set(state['mature_feedback']) == {DEFAULT, ALTERNATIVE}
    default = state['default_mature_feedback']
    assert default['candidate_id'] == DEFAULT
    assert default['sample_count'] == 1 and default['status'] == 'available'
    assert default['metrics']['mae'] == 1.
    assert default['independent_model_comparison'] is False
    assert default['observation_policy'] == 'selected_forecasts_only'
    assert state['feedback_scope']['mature_selected_forecast_count'] == 2
    assert state['feedback_scope']['independent_model_comparison'] is False
    serialized = json.dumps(observed, allow_nan=False)
    assert '"actual"' not in serialized and 'warehouse_fingerprint' not in serialized
    assert 'ledger_fingerprint' not in serialized


def test_initial_counts_and_feedback_use_current_region_cutoff_and_watermark():
    runtime = host(episodes=[episode('old', ORIGIN - 3 * 86400),
                             episode('too-late', ORIGIN - 86400),
                             episode('other-region', ORIGIN - 3 * 86400, 'QLD1')],
                   notices=[notice('visible', ORIGIN - 1), notice('future', ORIGIN + 1)])
    feedback(runtime, ORIGIN - 3 * 86400)
    feedback(runtime, ORIGIN - 86400)
    ctx = context(watermark=ORIGIN - 86400)
    observed = runtime.initial_observation(ctx)
    assert observed['eligible_analog_count'] == 1
    assert observed['available_evidence_count'] == 1
    assert observed['numerical_state']['default_mature_feedback']['sample_count'] == 1
    assert observed['numerical_state']['feedback_scope']['memory_watermark'] == ORIGIN - 86400


def test_initial_future_insertions_do_not_change_observation_or_invent_default_support():
    baseline = host(episodes=[episode('old', ORIGIN - 3 * 86400)], notices=[notice('visible', ORIGIN - 1)])
    altered = host(episodes=[episode('old', ORIGIN - 3 * 86400), episode('future', ORIGIN + 86400)],
                   notices=[notice('visible', ORIGIN - 1), notice('future', ORIGIN + 1)])
    feedback(altered, ORIGIN + 86400)
    feedback(baseline, ORIGIN - 2 * 86400, ALTERNATIVE)
    feedback(altered, ORIGIN - 2 * 86400, ALTERNATIVE)
    one, two = baseline.initial_observation(context()), altered.initial_observation(context())
    assert one == two
    default = one['numerical_state']['default_mature_feedback']
    assert default['sample_count'] == 0 and default['metrics'] == {}
    assert default['status'] == 'no_mature_selected_forecasts_in_recent_window'


def test_common_input_initial_observation_does_not_read_disabled_stores():
    runtime = host('common_input')
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled external store accessed')
    runtime.feedback.snapshot = forbidden
    runtime.evidence.search = forbidden
    runtime.analogs.eligible_count = forbidden
    observed = runtime.initial_observation(context())
    assert observed['available_evidence_count'] == observed['eligible_analog_count'] == 0
    assert observed['numerical_state']['mature_feedback'] == {}
    assert observed['numerical_state']['default_mature_feedback']['status'] == 'disabled_common_input'


def test_initial_feedback_summary_uses_latest_eight_and_reports_total_separately():
    runtime = host()
    for day in range(1, 11):
        feedback(runtime, ORIGIN - day * 86400, DEFAULT if day > 8 else ALTERNATIVE)
    observed = runtime.initial_observation(context())['numerical_state']
    assert observed['feedback_scope']['summary_window'] == 'latest_8_mature_selected_forecasts'
    assert observed['feedback_scope']['mature_selected_forecast_count'] == 8
    assert observed['feedback_scope']['total_mature_selected_forecast_count'] == 10
    assert set(observed['mature_feedback']) == {ALTERNATIVE}
    assert observed['default_mature_feedback']['sample_count'] == 0
    assert observed['default_mature_feedback']['status'] == 'no_mature_selected_forecasts_in_recent_window'
