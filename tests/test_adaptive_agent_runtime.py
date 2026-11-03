"""Full real ToolHost/controller paths with deterministic frozen adapter fixtures.

No fake tool observation substitutes for evidence/analogs/diagnostics. Only the
numeric model and decision service are injected, so tests are inexpensive and
do not claim predictive gains or real historical news coverage.
"""
import dataclasses
import json
import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import BudgetLimits, ForecastContext
from forecasting.adaptive_agent.controller import run
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.runtime import ToolHost


DEFAULT = 'lightgbm:full:72'
ALTERNATIVE = 'chronos:price_only:24'
ORIGIN = 1700000000


def context(origin=ORIGIN, watermark=None):
    snapshot = Snapshot('NSW1', origin, origin + 3600 * np.arange(24),
                        np.arange(72.), np.arange(72.) / 20,
                        np.zeros((72, 10)), np.zeros((24, 10)), 'fixture-preprocessing')
    return ForecastContext(snapshot, metadata={'split': 'selection'}, memory_watermark=watermark)


class FrozenFixture:
    def __init__(self, name):
        self.identity = {'sha256': 'fixture-' + name, 'train_end_unix': 100,
                         'selection_end_unix': 200}
        self.name = name
        self.calls = []

    def predict(self, ctx, profile, window):
        self.calls.append((ctx.identity(), profile, window))
        return np.tile(np.arange(5.), (24, 1)) + ctx.snapshot.price[-window:].mean()


class Choices:
    timeout = 30

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def choose(self, state, instructions, criteria):
        self.requests.append({'state': state, 'criteria': criteria})
        action = next(self.actions)
        if callable(action):
            action = action(state, criteria)
        if isinstance(action, Exception):
            raise action
        return action, {'selected_probability': .4, 'synthetic_decision': True}


def notice(name='visible', published=ORIGIN - 3600, text='A planned outage of 300 MW affects NSW1.'):
    return {'evidence_id': name, 'source_id': 'synthetic-notice', 'version': '1',
            'availability_basis': 'verified', 'published_at': published,
            'region': 'NSW1', 'payload': {'text': text}}


def episode(name, origin, available=None, actual=50):
    return {'episode_id': name, 'context': context(origin),
            'outcome_available_at': available if available is not None else origin + 86400,
            'actual': np.full(24, actual), 'state_tags': ['outage'], 'data_role': 'selection'}


def host(scope='full_evidence', episodes=None, notices=None):
    registry = ForecastRegistry({'lightgbm': FrozenFixture('lightgbm'),
                                 'chronos': FrozenFixture('chronos')}, 'r18-complete-agent-v1')
    samples = episodes if episodes is not None else [episode('old', ORIGIN - 3 * 86400)]
    documents = notices if notices is not None else [notice()]
    return ToolHost(registry, EvidenceStore(documents), AnalogStore(samples),
                    FeedbackLedger('synthetic-r18-arm'), information_scope=scope)


class RuntimeTests(unittest.TestCase):
    def test_full_evidence_to_review_commit_changes_forecast(self):
        ctx, runtime = context(), host()
        client = Choices(['search_evidence', 'read_evidence:visible',
                          'interpret_event:visible:outage', 'retrieve_analogs:event',
                          'forecast:' + ALTERNATIVE, 'diagnose:' + DEFAULT + '|' + ALTERNATIVE,
                          'review:' + ALTERNATIVE, 'approve'])
        selected, trace = run(ctx, runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(trace['release'], 'r18-complete-agent-v1')
        self.assertEqual(selected.candidate_id, ALTERNATIVE)
        self.assertFalse(np.array_equal(selected.quantiles, runtime.artifact(ctx, DEFAULT).quantiles))
        self.assertEqual(trace['costs'], {'api_calls': 8, 'tool_calls': 6, 'candidates': 2,
                                         'episodes': 1, 'historical_evaluations': 2})
        self.assertEqual(runtime.event_facts[0]['evidence_id'], 'visible')
        self.assertEqual(runtime.episode_ids, ['old'])
        diagnostic = trace['tools'][5]['result']['observation']
        self.assertEqual(diagnostic['diagnostic_kind'], 'retrospective_frozen_recompute')
        self.assertEqual(diagnostic['candidates'][ALTERNATIVE]['sample_count'], 1)
        self.assertEqual(diagnostic['candidates'][ALTERNATIVE]['data_roles'], ['model_selection_exposed'])
        self.assertEqual(trace['stop_reason'], 'reviewed_commit')

    def test_read_required_before_interpretation_and_cannot_reinterpret(self):
        runtime, ctx = host(), context()
        runtime.default(ctx)
        state = {'costs': {'historical_evaluations': 0}}
        runtime.execute(ctx, 'search_evidence', state)
        offered = runtime.actions(ctx, state)
        self.assertIn('read_evidence:visible', offered)
        self.assertFalse(any(k.startswith('interpret_event:') for k in offered))
        with self.assertRaises(ValueError):
            runtime.execute(ctx, 'interpret_event:visible:outage', state)
        read = runtime.execute(ctx, 'read_evidence:visible', state)
        quote = read['observation']['quoted_candidates'][0]['quote_span']['text']
        self.assertIn('300 MW', quote)
        runtime.execute(ctx, 'interpret_event:visible:outage', state)
        self.assertFalse(any(k.startswith('interpret_event:visible:') for k in runtime.actions(ctx, state)))

    def test_late_document_quote_visible_before_typed_interpretation(self):
        text = 'Background information. ' * 100 + 'A planned outage of 250 MW affects NSW1.'
        runtime, ctx = host(notices=[notice(text=text)]), context()
        runtime.default(ctx)
        state = {'costs': {'historical_evaluations': 0}}
        searched = runtime.execute(ctx, 'search_evidence', state)
        self.assertNotIn('250 MW', searched['observation']['items'][0]['payload']['text'])
        read = runtime.execute(ctx, 'read_evidence:visible', state)
        self.assertIn('250 MW', read['observation']['quoted_candidates'][0]['quote_span']['text'])

    def test_future_evidence_and_outcomes_not_in_any_request(self):
        sentinel = 91234567.0
        runtime = host(episodes=[episode('old', ORIGIN - 3 * 86400),
                                 episode('future-episode', ORIGIN - 3600, actual=sentinel)],
                       notices=[notice(), notice('future-notice', ORIGIN + 1, 'future_secret_text')])
        client = Choices(['search_evidence', 'retrieve_analogs:diverse', 'commit_default'])
        _, trace = run(context(), runtime, client)
        serialized = json.dumps(client.requests)
        for forbidden in ('future-episode', 'future-notice', 'future_secret_text', str(sentinel), '"actual"'):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(runtime.episode_ids, ['old'])
        self.assertEqual(trace['costs']['episodes'], 1)
        with self.assertRaises(ValueError):
            runtime.analogs.get_episode(context(), 'future-episode')

    def test_cross_context_host_reuse_rejected(self):
        runtime = host()
        run(context(), runtime, Choices(['commit_default']))
        with self.assertRaisesRegex(ValueError, 'another origin'):
            run(context(ORIGIN + 3600), runtime, Choices(['commit_default']))

    def test_common_input_disables_external_tools(self):
        runtime, client = host(scope='common_input'), Choices(['commit_default'])
        _, trace = run(context(), runtime, client)
        criteria = client.requests[0]['criteria']
        self.assertNotIn('search_evidence', criteria)
        self.assertNotIn('inspect_feedback', criteria)
        self.assertFalse(any(key.startswith('retrieve_analogs:') for key in criteria))
        self.assertEqual(trace['initial_observation']['evidence_status'], 'disabled_common_input')

    def test_registered_pool_changes_output_without_editing_source_forecasts(self):
        runtime, ctx = host(), context()
        def commit_pool(state, criteria):
            return next(key for key in criteria if key.startswith('commit:pool:'))
        client = Choices(['forecast:' + ALTERNATIVE,
                          f'pool:{DEFAULT}|{ALTERNATIVE}|0.25', commit_pool])
        output, trace = run(ctx, runtime, client)
        self.assertIsNone(trace['fallback'])
        expected = .25 * runtime.artifact(ctx, DEFAULT).quantiles + .75 * runtime.artifact(ctx, ALTERNATIVE).quantiles
        np.testing.assert_array_equal(output.quantiles, expected)
        self.assertEqual(output.model_id, 'quantile_pool')
        self.assertEqual(trace['costs']['candidates'], 3)

    def test_runtime_failure_preserves_default(self):
        runtime = host()
        def fail(*args):
            raise RuntimeError('synthetic numeric outage')
        runtime.registry.adapters['chronos'].predict = fail
        output, trace = run(context(), runtime, Choices(['forecast:' + ALTERNATIVE]))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('synthetic numeric outage', trace['fallback']['detail'])
        self.assertEqual(trace['tools'][0]['status'], 'error')
        self.assertEqual(runtime.registry.stats['numeric_inferences'], 2)

    def test_candidate_and_historical_budgets_filter_dynamic_host_actions(self):
        runtime = host()
        client = Choices(['retrieve_analogs:diverse', 'commit_default'])
        run(context(), runtime, client, BudgetLimits(candidates=1, historical_evaluations=1))
        self.assertFalse(any(k.startswith('forecast:') for k in client.requests[0]['criteria']))
        self.assertNotIn('inspect_recent_fit', client.requests[0]['criteria'])
        self.assertIn('diagnose:' + DEFAULT, client.requests[1]['criteria'])

    def test_feedback_maturity_and_watermark_observed_by_real_runtime(self):
        runtime, ctx = host(), context(watermark=ORIGIN - 86400)
        old = context(ORIGIN - 3 * 86400)
        prediction = runtime.registry.forecast(old, DEFAULT)
        key = runtime.feedback.append_forecast(old, prediction)
        runtime.feedback.append_outcome(key, np.ones(24), old.origin_unix + 86400)
        later = context(ORIGIN - 86400)
        later_prediction = runtime.registry.forecast(later, DEFAULT)
        later_key = runtime.feedback.append_forecast(later, later_prediction)
        runtime.feedback.append_outcome(later_key, np.ones(24), ORIGIN)
        _, trace = run(ctx, runtime, Choices(['inspect_feedback', 'commit_default']))
        rows = trace['tools'][0]['result']['observation']['rows']
        self.assertEqual([r['forecast_id'] for r in rows], [key])
        self.assertNotIn('actual', rows[0])

    def test_context_inputs_are_immutable_and_labels_rejected(self):
        ctx = context()
        with self.assertRaises(ValueError):
            ctx.snapshot.price[0] = 123
        with self.assertRaises(TypeError):
            ctx.metadata['split'] = 'test'
        for key in ('actual', 'target', 'labels'):
            with self.assertRaises(ValueError):
                dataclasses.replace(ctx, model_inputs={key: np.ones(24)})
            with self.assertRaises(ValueError):
                dataclasses.replace(ctx, metadata={key: [1]})

    def test_future_ledger_completion_does_not_change_visible_feedback_fingerprint(self):
        runtime, ctx = host(), context()
        runtime.default(ctx)
        state = {'costs': {'historical_evaluations': 0}}
        before = runtime.execute(ctx, 'inspect_feedback', state)['observation']
        future = context(ORIGIN + 86400)
        prediction = runtime.registry.forecast(future, DEFAULT)
        key = runtime.feedback.append_forecast(future, prediction)
        runtime.feedback.append_outcome(key, np.ones(24), future.origin_unix + 86400)
        after = runtime.execute(ctx, 'inspect_feedback', state)['observation']
        self.assertEqual(before, after)

    def test_future_warehouse_and_completion_changes_leave_api_states_identical(self):
        normal = host()
        with_future = host(episodes=[episode('old', ORIGIN - 3 * 86400),
                                     episode('future-episode', ORIGIN + 3600, actual=12345678.)],
                           notices=[notice(), notice('future-notice', ORIGIN + 1, 'unknown future outage')])
        future = context(ORIGIN + 86400)
        prediction = with_future.registry.forecast(future, DEFAULT)
        key = with_future.feedback.append_forecast(future, prediction)
        with_future.feedback.append_outcome(key, np.ones(24), future.origin_unix + 86400)
        all_requests = []
        for runtime in (normal, with_future):
            client = Choices(['search_evidence', 'retrieve_analogs:diverse',
                              'inspect_feedback', 'commit_default'])
            output, trace = run(context(), runtime, client)
            self.assertIsNone(trace['fallback'])
            for request in client.requests:
                request['state'].pop('remaining_seconds')
            all_requests.append(client.requests)
        self.assertEqual(all_requests[0], all_requests[1])
        self.assertNotEqual(normal.evidence.fingerprint, with_future.evidence.fingerprint)
        self.assertNotEqual(normal.analogs.fingerprint, with_future.analogs.fingerprint)


if __name__ == '__main__':
    unittest.main()
