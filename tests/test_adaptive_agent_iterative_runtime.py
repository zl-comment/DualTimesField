"""Incremental inference accounting, immutable cohorts and genuinely paired losses."""
from dataclasses import replace

import numpy as np
import pytest

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.iterative_runtime import IterativeToolHost

ORIGIN = 1700000000
DEFAULT = 'lightgbm:full:72'
ALTERNATIVE = 'chronos:price_only:24'


def context(origin=ORIGIN, watermark=None, region='NSW1'):
    return ForecastContext(Snapshot(region, origin, origin + 3600 * np.arange(24),
                                    np.arange(72.), np.zeros(72), np.zeros((72, 10)),
                                    np.zeros((24, 10)), 'fixture'), memory_watermark=watermark)


class CountedAdapter:
    def __init__(self, shift=0, failures=(), timeout_origin=None):
        self.identity = {'sha256': 'fixture-' + str(shift), 'train_end_unix': 100,
                         'selection_end_unix': 200}
        self.shift, self.failures, self.timeout_origin = shift, set(failures), timeout_origin
        self.calls = []

    def predict(self, ctx, profile, window):
        self.calls.append((ctx.origin_unix, profile, window))
        if ctx.origin_unix == self.timeout_origin:
            raise TimeoutError('injected deadline')
        if ctx.origin_unix in self.failures:
            raise RuntimeError('injected historical candidate failure')
        return np.tile(np.arange(5.) + self.shift, (24, 1))


def episodes(count=8):
    return [{'episode_id': f'day-{day}', 'context': context(ORIGIN - day * 86400),
             'actual': np.full(24, day * 10.), 'outcome_available_at': ORIGIN - (day - 1) * 86400,
             'data_role': 'holdout'} for day in range(1, count + 1)]


def host(rows=None, default_failures=(), alternative_failures=()):
    default, alternative = CountedAdapter(0, default_failures), CountedAdapter(100, alternative_failures)
    registry = ForecastRegistry({'lightgbm': default, 'chronos': alternative}, 'r20-fixture')
    runtime = IterativeToolHost(registry, EvidenceStore(), AnalogStore(episodes() if rows is None else rows),
                                FeedbackLedger('r20-arm'))
    return runtime, default, alternative


def state(used=0):
    return {'costs': {'historical_evaluations': used}}


def prepare(runtime, ctx=None):
    ctx = context() if ctx is None else ctx
    runtime.default(ctx)
    result = runtime.execute(ctx, 'retrieve_analogs:diverse', state())
    assert result['observation']['cohort_locked']
    return ctx


def diagnose(runtime, ctx, used=0):
    actions = runtime.actions(ctx, state(used))
    action = next(key for key in actions if key.startswith('diagnose:'))
    return runtime.execute(ctx, action, state(used))


def test_incremental_cache_counts_real_model_calls_and_compares_all_candidates():
    runtime, default, alternative = host()
    ctx = prepare(runtime)
    first = diagnose(runtime, ctx)
    assert first['costs']['historical_evaluations'] == 8
    assert first['observation']['numeric_inferences'] == 8
    assert len(default.calls) == 9  # one current default, eight historical forecasts
    runtime.execute(ctx, 'forecast:' + ALTERNATIVE, state(8))
    second = diagnose(runtime, ctx, 8)
    observation = second['observation']
    assert second['costs']['historical_evaluations'] == 8
    assert observation['cache_hit_count'] == 8
    assert observation['cumulative_logical_inference_units'] == 16
    assert set(observation['candidates']) == {DEFAULT, ALTERNATIVE}
    paired = observation['candidates'][ALTERNATIVE]['paired_vs_default']
    assert paired['default_candidate_id'] == DEFAULT
    assert paired['mean_qs_gain'] < 0 and paired['win_fraction'] == 3 / 8
    assert paired['win_count'] + paired['tie_count'] + paired['loss_count'] == 8
    assert len(default.calls) == len(alternative.calls) == 9
    assert observation['current_forecast_comparison']['pairs'][0]['median_absolute_difference'] == 100
    repeated = diagnose(runtime, ctx, 16)
    assert repeated['costs']['historical_evaluations'] == 0
    assert repeated['observation']['cache_hit_count'] == 16
    assert repeated['observation']['numeric_inferences'] == 0
    assert len(default.calls) == len(alternative.calls) == 9
    assert 'rows' not in repeated['observation']
    assert len([item for item in runtime.audit if 'complete_diagnostic' in item][-1]['complete_diagnostic']['rows']) == 16


def test_first_cohort_is_immutable_including_empty_selection():
    runtime, _, _ = host()
    ctx = prepare(runtime)
    assert not any(key.startswith('retrieve_analogs:') for key in runtime.actions(ctx, state()))
    with pytest.raises(ValueError):
        runtime.execute(ctx, 'retrieve_analogs:recent', state())
    runtime.episode_ids.reverse()
    with pytest.raises(ValueError, match='immutable'):
        runtime.actions(ctx, state())
    empty, _, _ = host(rows=[])
    ctx = prepare(empty)
    actions = empty.actions(ctx, state())
    assert not any(key.startswith(('retrieve_analogs:', 'diagnose:')) for key in actions)


def test_diagnostic_cost_cannot_exceed_remaining_budget_even_if_state_is_forged():
    runtime, default, alternative = host()
    ctx = prepare(runtime)
    with pytest.raises(ValueError, match='budget'):
        diagnose(runtime, ctx, used=33)
    assert len(default.calls) == 1
    assert runtime._charged_diagnostic_units == 0
    diagnose(runtime, ctx)
    # Fill all five allowed candidates: 5 x 8 = 40, never recompute old pairs.
    for index, name in enumerate(('chronos:price_only:24', 'chronos:price_only:48',
                                 'chronos:calendar:24', 'chronos:calendar:48'), 1):
        runtime.execute(ctx, 'forecast:' + name, state(index * 8))
        result = diagnose(runtime, ctx, index * 8)
        assert result['costs']['historical_evaluations'] == 8
    assert runtime._charged_diagnostic_units == 40
    assert diagnose(runtime, ctx, 40)['costs']['historical_evaluations'] == 0
    with pytest.raises(ValueError):
        runtime.execute(ctx, 'forecast:chronos:base:24', state(40))
    assert len(default.calls) + len(alternative.calls) == 45


def test_failed_pairs_are_charged_once_and_metrics_use_common_success_intersection():
    runtime, default, alternative = host(default_failures=[ORIGIN - 86400],
                                        alternative_failures=[ORIGIN - 2 * 86400])
    ctx = prepare(runtime)
    first = diagnose(runtime, ctx)
    assert first['costs']['historical_evaluations'] == 8
    assert first['observation']['sample_count'] == 7
    runtime.execute(ctx, 'forecast:' + ALTERNATIVE, state(8))
    result = diagnose(runtime, ctx, 8)['observation']
    expected = {f'day-{day}' for day in range(3, 9)}
    assert set(result['paired_episode_ids']) == expected
    assert result['cohort_size'] == 8 and result['sample_count'] == 6
    assert result['paired_support_fraction'] == .75
    assert len(result['failures']) == 2
    for candidate in (DEFAULT, ALTERNATIVE):
        summary = result['candidates'][candidate]
        assert set(summary['paired_episode_ids']) == expected
        assert summary['successful_count'] == 7 and summary['sample_count'] == 6
        assert summary['missing_count'] == 1
    assert result['candidates'][DEFAULT]['metrics']['mae'] == np.mean(np.arange(3, 9) * 10 - 2)
    assert result['candidates'][ALTERNATIVE]['metrics']['mae'] == np.mean(102 - np.arange(3, 9) * 10)
    default.failures.clear()
    alternative.failures.clear()
    repeated = diagnose(runtime, ctx, 16)['observation']
    assert len(repeated['failures']) == 2 and repeated['numeric_inferences'] == 0
    assert repeated['sample_count'] == 6  # recovery does not silently retry failures


def test_disjoint_success_sets_produce_no_comparable_metric():
    rows = episodes(2)
    runtime, _, _ = host(rows=rows, default_failures=[ORIGIN - 86400],
                         alternative_failures=[ORIGIN - 2 * 86400])
    ctx = prepare(runtime)
    runtime.execute(ctx, 'forecast:' + ALTERNATIVE, state())
    result = diagnose(runtime, ctx)['observation']
    assert result['sample_count'] == 0 and not result['comparison_available']
    assert all(item['metrics'] is None for item in result['candidates'].values())
    assert all(item['successful_count'] == 1 for item in result['candidates'].values())


def test_maturity_watermark_and_context_identity_guard_before_any_historical_inference():
    rows = episodes(3) + [{'episode_id': 'future', 'context': context(ORIGIN + 86400),
                           'actual': np.full(24, 999999.), 'outcome_available_at': ORIGIN + 2 * 86400}]
    runtime, default, _ = host(rows=rows)
    ctx = prepare(runtime, context(watermark=ORIGIN - 86400))
    result = diagnose(runtime, ctx)['observation']
    assert set(result['episodes']) == {'day-2', 'day-3'}
    assert len(default.calls) == 3
    with pytest.raises(ValueError, match='another origin'):
        diagnose(runtime, context(ORIGIN + 3600))
    assert len(default.calls) == 3


def test_cache_is_bound_to_release_model_inputs_and_original_cohort():
    for mutation in ('release', 'model', 'input', 'cohort'):
        runtime, _, alternative = host()
        ctx = prepare(runtime)
        diagnose(runtime, ctx)
        if mutation == 'release':
            runtime.registry.release_id = 'changed'
        elif mutation == 'model':
            alternative.identity['sha256'] = 'changed'
        elif mutation == 'input':
            runtime.artifacts[DEFAULT] = replace(runtime.artifacts[DEFAULT], input_hash='changed')
        else:
            runtime._cohort[0]['actual'] = np.zeros(24)
        with pytest.raises(ValueError):
            diagnose(runtime, ctx)


def test_timeout_stops_new_attempts_and_keeps_first_failure_charged():
    runtime, default, _ = host()
    ctx = prepare(runtime)
    first_origin = runtime._cohort[0]['context'].origin_unix
    default.timeout_origin = first_origin
    result = diagnose(runtime, ctx)
    assert result['costs']['historical_evaluations'] == 1
    assert result['observation']['unattempted_count'] == 7
    assert result['observation']['failures'][0]['reason'] == 'TimeoutError'
    default.timeout_origin = None
    resumed = diagnose(runtime, ctx, 1)
    assert resumed['costs']['historical_evaluations'] == 7
    assert len(resumed['observation']['failures']) == 1
    assert resumed['observation']['sample_count'] == 7
    assert sum(origin == first_origin for origin, _, _ in default.calls) == 1
