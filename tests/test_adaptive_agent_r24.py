"""R24a incremental host behavior and bounded loop policy checks."""
import numpy as np
import pytest

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent import input_policy_r24
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.controller import SKILL_IDS
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.iterative_runtime import R24IterativeToolHost
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.replay import policy_host_factory, policy_module
from test_adaptive_agent_input_policy_r22 import Client, DEFAULT, Host, OPERATED, context
from test_adaptive_agent_iterative_runtime import (
    ALTERNATIVE, DEFAULT as R20_DEFAULT, ORIGIN, CountedAdapter, state,
)


def snapshot_at(origin, demand_start=10., demand_end=30.):
    return Snapshot('NSW1', origin, origin + 3600 * np.arange(24),
                    np.linspace(demand_start, demand_end, 72), np.linspace(-1, 1, 72),
                    np.ones((72, 10)), np.ones((24, 10)), 'preprocessing')


def episode(day, demand_start=10., demand_end=30.):
    return {'episode_id': f'day-{day}', 'context': ForecastContext(snapshot_at(
        ORIGIN - day * 86400, demand_start, demand_end)),
        'actual': np.full(24, day * 10.), 'outcome_available_at': ORIGIN - (day - 1) * 86400,
        'data_role': 'holdout'}


def r24_host(rows):
    default, alternative = CountedAdapter(0), CountedAdapter(100)
    registry = ForecastRegistry({'lightgbm': default, 'chronos': alternative}, 'r24-fixture')
    runtime = R24IterativeToolHost(registry, EvidenceStore(), AnalogStore(rows),
                                   FeedbackLedger('r24-arm'))
    return runtime, default, alternative


def completion_action(runtime, ctx, used):
    return next(key for key in runtime.actions(ctx, state(used))
                if key.startswith('diagnose:'))


def test_diagnose_one_is_incremental_and_completion_reuses_cache():
    runtime, _, _ = r24_host([episode(day) for day in range(1, 9)])
    ctx = ForecastContext(snapshot_at(ORIGIN))
    runtime.default(ctx)
    runtime.execute(ctx, 'retrieve_analogs:diverse', state())
    one = runtime.execute(ctx, 'diagnose_one:' + R20_DEFAULT, state())
    assert one['costs']['historical_evaluations'] == 8
    assert one['observation']['candidates'][R20_DEFAULT]['newly_successful_count'] == 8
    runtime.execute(ctx, 'forecast:' + ALTERNATIVE, state(8))
    two = runtime.execute(ctx, 'diagnose_one:' + ALTERNATIVE, state(8))
    assert two['costs']['historical_evaluations'] == 8
    assert 'paired_vs_default' in two['observation']['candidates'][ALTERNATIVE]
    full = runtime.execute(ctx, completion_action(runtime, ctx, 16), state(16))
    assert full['costs']['historical_evaluations'] == 0
    assert full['observation']['cache_hit_count'] == 16
    assert full['observation']['r22_aggregate_schema'] == 'r22-paired-diagnostic-v1'


class StubAnalogStore(AnalogStore):
    """Deterministic augmentation: the state retrieval always offers four
    extra standard-profile episodes that the first retrieval never returned."""

    def __init__(self, rows, extras):
        super().__init__(rows)
        self._extras = {row['episode_id']: row for row in extras}

    def retrieve(self, context, strategy='diverse', limit=8, event_tags=()):
        if strategy == 'state':
            return {'status': 'available',
                    'items': [{'episode_id': key, 'origin_unix': row['context'].origin_unix,
                               'outcome_available_at': row['outcome_available_at'],
                               'state_tags': ['stable'], 'distance': 0.5, 'metrics': {},
                               'data_role': 'holdout'} for key, row in self._extras.items()],
                    'warehouse_fingerprint': 'stub'}
        return super().retrieve(context, strategy=strategy, limit=limit, event_tags=event_tags)

    def get_episode(self, context, key):
        if key in self._extras:
            return dict(self._extras[key])
        return super().get_episode(context, key)


def r24_stub_host():
    rows = [episode(day) for day in range(1, 9)]
    extras = [episode(day) for day in range(20, 24)]
    default, alternative = CountedAdapter(0), CountedAdapter(100)
    registry = ForecastRegistry({'lightgbm': default, 'chronos': alternative}, 'r24-fixture')
    runtime = R24IterativeToolHost(registry, EvidenceStore(), StubAnalogStore(rows, extras),
                                   FeedbackLedger('r24-arm'))
    return runtime, default


def test_augment_appends_episodes_and_preserves_diagnoses():
    runtime, _ = r24_stub_host()
    ctx = ForecastContext(snapshot_at(ORIGIN))
    runtime.default(ctx)
    runtime.execute(ctx, 'retrieve_analogs:diverse', state())
    one = runtime.execute(ctx, 'diagnose_one:' + R20_DEFAULT, state())
    assert one['costs']['historical_evaluations'] == 8
    aug = runtime.execute(ctx, 'augment:state', state(8))
    assert aug['observation']['status'] == 'cohort_appended'
    new_ids = aug['observation']['new_episode_ids']
    assert len(new_ids) == 4
    assert aug['observation']['cohort_size'] == 12
    two = runtime.execute(ctx, 'diagnose_one:' + R20_DEFAULT, state(8))
    # Only the appended episodes are charged; prior diagnoses are cache hits.
    assert two['costs']['historical_evaluations'] == 4
    assert two['observation']['candidates'][R20_DEFAULT]['cache_hit_count'] == 8
    runtime.execute(ctx, 'forecast:' + ALTERNATIVE, state(12))
    full = runtime.execute(ctx, completion_action(runtime, ctx, 12), state(12))
    assert full['costs']['historical_evaluations'] == 12  # only the alternative
    assert full['observation']['cache_hit_count'] == 12  # default's pairs all hit
    assert full['observation']['cohort_size'] == 12
    assert len(full['observation']['episodes']) == 12


def test_augment_without_new_episodes_fails_loudly():
    runtime, _, _ = r24_host([episode(day) for day in range(1, 9)])
    ctx = ForecastContext(snapshot_at(ORIGIN))
    runtime.default(ctx)
    runtime.execute(ctx, 'retrieve_analogs:diverse', state())
    with pytest.raises(ValueError, match='no new unique mature episodes'):
        runtime.execute(ctx, 'augment:state', state())


class LoopHost(Host):
    """R22 fake host extended with loop actions and an augmentable cohort."""

    def __init__(self, *, unproductive=False, **kwargs):
        super().__init__(**kwargs)
        self.cohort = []
        self.items = []
        self.cohort_id = 'locked'
        self.augment_available = True
        self.unproductive = unproductive

    def actions(self, ctx, state):
        result = super().actions(ctx, state)
        if self.locked and len(self.artifacts) > 1:
            for candidate_id in (DEFAULT, OPERATED):
                result['diagnose_one:' + candidate_id] = {
                    'description': 'diagnose ' + candidate_id,
                    'kind': 'compare_candidates', 'costs': {'historical_evaluations': 8}}
            if self.augment_available:
                result['augment:state'] = {
                    'description': 'append mature episodes',
                    'kind': 'retrieve_analogs', 'costs': {'episodes': 8}}
        return result

    def execute(self, ctx, action, state):
        if action.startswith('diagnose_one:'):
            candidate_id = action[len('diagnose_one:'):]
            successful = 0 if self.unproductive else 8
            return {'status': 'available', 'candidate_ids': [],
                    'costs': {'historical_evaluations': 8},
                    'observation': {
                        'status': 'available', 'cohort_id': self.cohort_id,
                        'cohort_size': len(self.cohort),
                        'newly_diagnosed_count': successful,
                        'candidates': {
                            candidate_id: {'newly_successful_count': successful,
                                           'sample_count': successful}}}}
        if action.startswith('augment:'):
            self.augment_available = False
            self.cohort_id = 'locked2'
            self.cohort = (self.cohort or []) + ['e8', 'e9']
            self.items = self.items + [
                {'episode_id': name,
                 'origin_unix': ctx.origin_unix - (19 + offset) * 86400,
                 'outcome_available_at': ctx.origin_unix - (18 + offset) * 86400,
                 'state_tags': ['stable'], 'distance': 0.9, 'metrics': {},
                 'data_role': 'model_selection_exposed'}
                for offset, name in enumerate(('e8', 'e9'))]
            return {'status': 'ok', 'candidate_ids': [], 'costs': {'episodes': 8},
                    'observation': {
                         'status': 'cohort_appended', 'new_episode_ids': ['e8', 'e9'],
                         'cohort_size': len(self.cohort), 'cohort_id': self.cohort_id,
                         'retrieval_observation': {
                             'items': list(self.items), 'cohort_id': self.cohort_id,
                             'cohort_locked': True, 'cohort_size': len(self.cohort),
                             'strategy': 'recent+state',
                             'warehouse_fingerprint': 'warehouse2'}}}
        result = super().execute(ctx, action, state)
        if action.startswith('retrieve_analogs:'):
            self.items = list(result['observation']['items'])
            self.cohort = [item['episode_id'] for item in self.items]
        elif action.startswith('diagnose:') and len(self.cohort) > 8:
            observation = result['observation']
            observation['episodes'] = list(self.cohort)
            observation['paired_episode_ids'] = list(self.cohort)
            observation['cohort_id'] = self.cohort_id
            observation['cohort_size'] = len(self.cohort)
            for block in observation['candidates'].values():
                block['paired_episode_ids'] = list(self.cohort)
        return result


def test_loop_round_then_agent_stop_commits_operated():
    client = Client(['retrieve_analogs:recent', 'diagnose_one:' + OPERATED,
                     'stop_and_evaluate', 'use_operated_input', 'commit:' + OPERATED,
                     'approve_selected'])
    selected, trace = input_policy_r24.run(context(), LoopHost(), client)
    assert selected.candidate_id == OPERATED
    assert trace['fallback'] is None
    assert trace['loop']['rounds'][0]['action'] == 'diagnose_one:' + OPERATED
    assert trace['loop']['rounds'][0]['productive'] is True
    assert trace['loop']['stop_reason'] == 'agent_stop'
    assert trace['stage_l_decisions'][0]['action'] == 'diagnose_one:' + OPERATED
    loop_state = client.calls[1][2]
    assert loop_state['phase'] == 'iterative_loop'
    assert loop_state['loop']['feedback_scope'] == (
        'historical gains, support and current candidate '
        'disagreement only; no risk signals, no stage rationale')


def test_loop_caps_at_three_rounds():
    client = Client(['retrieve_analogs:recent'] + ['diagnose_one:' + OPERATED] * 3
                    + ['use_operated_input', 'commit:' + OPERATED, 'approve_selected'])
    selected, trace = input_policy_r24.run(context(), LoopHost(), client)
    assert selected.candidate_id == OPERATED
    assert trace['loop']['stop_reason'] == 'max_rounds_reached'
    assert len(trace['loop']['rounds']) == 3


def test_unproductive_round_ends_loop():
    host = LoopHost(unproductive=True)
    client = Client(['retrieve_analogs:recent', 'diagnose_one:' + OPERATED,
                     'use_operated_input', 'commit:' + OPERATED, 'approve_selected'])
    selected, trace = input_policy_r24.run(context(), host, client)
    assert trace['loop']['stop_reason'] == 'unproductive_round'
    assert len(trace['loop']['rounds']) == 1
    assert selected.candidate_id == OPERATED  # completion still supports the commit


def test_augment_round_extends_cohort_for_report():
    client = Client(['retrieve_analogs:recent', 'augment:state',
                     'diagnose_one:' + OPERATED, 'stop_and_evaluate',
                     'use_operated_input', 'commit:' + OPERATED, 'approve_selected'])
    selected, trace = input_policy_r24.run(context(), LoopHost(), client)
    assert selected.candidate_id == OPERATED
    assert trace['loop']['rounds'][0]['action'] == 'augment:state'
    assert trace['loop']['rounds'][0]['cohort_size'] == 10
    assert len(trace['iterations'][-1]['cohort_ids']) == 10
    assert len(trace['evidence_report']['provenance']['episode_ids']) == 10


def test_stage_a_identity_after_loop():
    client = Client(['retrieve_analogs:recent', 'stop_and_evaluate', 'use_identity'])
    selected, trace = input_policy_r24.run(context(), LoopHost(), client)
    assert selected.candidate_id == DEFAULT
    assert trace['stop_reason'] == 'agent_commit_identity'
    assert trace['loop']['rounds'][0]['action'] == 'stop_and_evaluate'


class ExplodingClient(Client):
    def choose(self, request_state, instruction, criteria):
        if request_state['phase'] == 'iterative_loop':
            raise ValueError('boom')
        return super().choose(request_state, instruction, criteria)


def test_protocol_errors_fail_loudly_without_identity_fallback():
    client = ExplodingClient(['retrieve_analogs:recent', 'use_operated_input',
                              'commit:' + OPERATED, 'approve_selected'])
    with pytest.raises(RuntimeError, match='R24a protocol error'):
        input_policy_r24.run(context(), LoopHost(), client)


def test_policy_and_skills_registration():
    assert policy_module('r24_iterative_loop_input') is input_policy_r24
    assert policy_host_factory('r24_iterative_loop_input') is R24IterativeToolHost
    skills = input_policy_r24.load_skills()
    assert set(skills) == set(SKILL_IDS)
    assert all(entry['version'] == '8.0.0' for entry in skills.values())
    contract = input_policy_r24.manifest_contract()
    assert contract['policy'] == 'r24_iterative_loop_input'
    assert contract['admission_contract']['schema'] == 'r23-state-aware-admission-v1'
    assert contract['max_loop_rounds'] == 3
