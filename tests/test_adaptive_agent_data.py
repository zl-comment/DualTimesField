"""Causal data-boundary and durable feedback tests, independent of model quality."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext, ForecastArtifact
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.events import EventResolver
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.state import summarize_state

ORIGIN = 1672531200


def context(origin=ORIGIN, region='NSW1', watermark=None, price=None):
    snapshot = Snapshot(region, origin, origin + np.arange(24) * 3600,
                        np.arange(72, dtype=float) if price is None else price,
                        np.zeros(72), np.zeros((72, 10)), np.zeros((24, 10)), 'frozen')
    return ForecastContext(snapshot, memory_watermark=watermark)


def evidence(key, published, **kwargs):
    return {'evidence_id': key, 'source_id': 'fixture', 'version': key,
            'published_at': published, 'availability_basis': 'verified', 'region': 'NSW1',
            'payload': {'text': 'Planned outage of 400 MW in NSW1.'}, **kwargs}


def artifact(ctx):
    return ForecastArtifact('model:base:72', ctx.identity(), np.tile(np.arange(5.), (24, 1)),
                            ctx.snapshot.target_time_unix, 'frozen', 'input_hash', {'nested': {'x': 1}})


def episode(ctx, key, **kwargs):
    return {'episode_id': key, 'context': ctx, 'actual': np.ones(24),
            'outcome_available_at': ctx.origin_unix + 86400, **kwargs}


def test_evidence_excludes_future_and_proxy_before_ranking_and_replacement():
    ctx = context()
    store = EvidenceStore([evidence('old', ORIGIN - 100),
                           evidence('future', ORIGIN + 1, supersedes=['old']),
                           evidence('proxy', ORIGIN - 20, availability_basis='archive_proxy'),
                           evidence('wrong-region', ORIGIN - 10, region='QLD1')])
    assert [r['evidence_id'] for r in store.search(ctx, 'outage')['items']] == ['old']
    with pytest.raises(ValueError):
        store.read(ctx, 'future')
    later = context(ORIGIN + 2)
    assert [r['evidence_id'] for r in store.search(later)['items']] == ['future']


def test_evidence_does_not_confuse_validity_with_publication(tmp_path):
    record = evidence('planned', ORIGIN - 1, valid_from=ORIGIN + 86400, valid_to=ORIGIN + 172800)
    path = tmp_path / 'warehouse.jsonl'
    path.write_text(json.dumps(record) + '\n')
    store = EvidenceStore.from_jsonl(path)
    assert store.read(context(), 'planned')['valid_from'] > ORIGIN
    record['payload']['text'] = 'mutated'
    assert store.read(context(), 'planned')['payload']['text'] != 'mutated'
    assert EvidenceStore().search(context())['status'] == 'real_data_unavailable'


def test_evidence_rejects_conflicting_immutable_ids_and_hashes():
    with pytest.raises(ValueError, match='Conflicting'):
        EvidenceStore([evidence('same', ORIGIN - 2), evidence('same', ORIGIN - 1)])
    with pytest.raises(ValueError, match='hash mismatch'):
        EvidenceStore([evidence('bad', ORIGIN, content_hash='fabricated')])
    with pytest.raises(ValueError):
        EvidenceStore([evidence('self', ORIGIN, supersedes=['self'])])


def test_event_facts_are_quotes_with_typed_choice_and_no_price_effect():
    row = EvidenceStore([evidence('e', ORIGIN - 1)]).read(context(), 'e')
    resolver = EventResolver()
    options = resolver.resolve(row)
    assert 'outage' in options['choices']
    fact = resolver.resolve(row, 'outage')['facts'][0]
    span = fact['quote_span']
    assert row['payload']['text'][span['start']:span['end']] == span['text']
    assert fact['explicit_quantities'][0]['value_literal'] == '400'
    assert fact['numerical_price_effect'] is None
    assert resolver.resolve(row, 'unknown')['facts'] == []
    with pytest.raises(ValueError):
        resolver.resolve(row, 'increase_price_100')
    row['availability_basis'] = 'archive_proxy'
    assert resolver.resolve(row)['status'] == 'unavailable'


def test_analogs_filter_maturity_watermark_region_and_never_send_actual():
    previous = context(ORIGIN - 2 * 86400)
    good = episode(previous, 'good')
    delayed = episode(context(ORIGIN - 86400), 'delayed', outcome_available_at=ORIGIN + 1)
    other = episode(context(ORIGIN - 86400, 'QLD1'), 'other')
    store = AnalogStore([good, delayed, other])
    result = store.retrieve(context())
    assert [r['episode_id'] for r in result['items']] == ['good']
    assert 'actual' not in result['items'][0] and 'context' not in result['items'][0]
    assert store.get_episode(context(), 'good')['actual'].shape == (24,)
    with pytest.raises(ValueError):
        store.get_episode(context(), 'delayed')
    assert store.retrieve(context(watermark=ORIGIN - 86401))['items'] == []


def test_analog_diversity_rejects_overlapping_outcomes_and_event_filter():
    rows = [episode(context(ORIGIN - 86400 * 3 + offset * 3600), str(offset),
                    state_tags=['outage' if offset % 2 else 'quiet']) for offset in (0, 1, 24, 25)]
    store = AnalogStore(rows)
    picked = store.retrieve(context(), 'diverse')['items']
    assert len(picked) == 2
    assert abs(picked[0]['origin_unix'] - picked[1]['origin_unix']) >= 86400
    assert all('outage' in r['state_tags'] for r in store.retrieve(context(), 'event', event_tags=['outage'])['items'])
    assert store.retrieve(context(), 'event', event_tags=['nonexistent'])['items'] == []
    for strategy in ('recent', 'seasonal', 'state'):
        assert store.retrieve(context(), strategy)['items']
    with pytest.raises(ValueError):
        store.retrieve(context(), limit=9)


def test_analog_records_immutable_and_require_full_maturity():
    row = episode(context(ORIGIN - 86400), 'record', state_tags=['quiet'])
    store = AnalogStore([row])
    row['actual'][:] = 999
    row['state_tags'].append('changed')
    assert store.get_episode(context(), 'record')['actual'][0] == 1
    assert store.retrieve(context())['items'][0]['state_tags'] == ['quiet']
    with pytest.raises(ValueError, match='mature'):
        AnalogStore([episode(context(), 'bad', outcome_available_at=ORIGIN + 23 * 3600)])


def test_ledger_maturity_persistence_arm_isolation_and_immutability(tmp_path):
    past = context(ORIGIN - 86400)
    ledger = FeedbackLedger('agent', tmp_path / 'feedback.jsonl')
    forecast_id = ledger.append_forecast(past, artifact(past))
    assert ledger.append_forecast(past, artifact(past)) == forecast_id
    with pytest.raises(ValueError, match='mature'):
        ledger.append_outcome(forecast_id, np.ones(24), ORIGIN - 1)
    ledger.append_outcome(forecast_id, np.ones(24), ORIGIN)
    assert len(ledger.snapshot(context())) == 1
    assert ledger.snapshot(context(watermark=ORIGIN - 1)) == []
    restored = FeedbackLedger('agent', tmp_path / 'feedback.jsonl')
    assert restored.snapshot(context()) == ledger.snapshot(context())
    assert restored.fingerprint == ledger.fingerprint
    assert 'actual' not in ledger.snapshot(context())[0]
    with pytest.raises(ValueError, match='Cross-arm'):
        FeedbackLedger('fixed', tmp_path / 'feedback.jsonl')
    with pytest.raises(ValueError, match='revise'):
        ledger.append_outcome(forecast_id, np.zeros(24), ORIGIN)
    assert ledger.snapshot(context(region='QLD1')) == []


def test_ledger_completion_order_cannot_leak_future_feedback():
    ledger = FeedbackLedger('arm')
    early, late = context(ORIGIN - 2 * 86400), context(ORIGIN - 86400)
    early_id = ledger.append_forecast(early, artifact(early))
    late_id = ledger.append_forecast(late, artifact(late))
    ledger.append_outcome(late_id, np.ones(24), ORIGIN)
    ledger.append_outcome(early_id, np.ones(24), ORIGIN - 86400)
    snap = ledger.snapshot(context(watermark=ORIGIN - 1))
    assert [r['forecast_id'] for r in snap] == [early_id]


def test_ledger_rejects_tampering_and_stale_writers(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    first, stale = FeedbackLedger('arm', path), FeedbackLedger('arm', path)
    ctx = context()
    first.append_forecast(ctx, artifact(ctx))
    with pytest.raises(RuntimeError, match='Stale'):
        stale.append_forecast(ctx, artifact(ctx))
    event = json.loads(path.read_text())
    event['payload']['quantiles'][0][0] = -999
    path.write_text(json.dumps(event) + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        FeedbackLedger('arm', path)


def test_state_excludes_future_feedback_and_reports_units():
    ctx = context()
    future = {'outcome_available_at': ORIGIN + 1, 'region': 'NSW1', 'model_id': 'future',
              'metrics': {'qs': -999, 'mae': -999, 'coverage_90': 1}}
    state = summarize_state(ctx, [future])
    assert state['mature_feedback'] == {}
    assert state['history_hours'] == 72
    assert state['days_oldest_first'][-1]['demand_std_mean'] == 0
    assert 'price_level_change' in state['state_tags']


def test_evidence_reports_conflicting_versions_without_guessing_and_rejects_cycles():
    store = EvidenceStore([evidence('a', ORIGIN - 2, chain_id='one'),
                           evidence('b', ORIGIN - 1, chain_id='one'),
                           evidence('resolution', ORIGIN + 1, chain_id='one', supersedes=['a', 'b'])])
    rows = store.search(context())['items']
    assert len(rows) == 2
    assert all(row['version_status'] == 'conflict_unresolved' for row in rows)
    assert [row['evidence_id'] for row in store.search(context(ORIGIN + 2))['items']] == ['resolution']
    assert store.read(context(ORIGIN + 2), 'a')['version_status'] == 'superseded'
    with pytest.raises(ValueError, match='Cyclic'):
        EvidenceStore([evidence('a', ORIGIN - 2, supersedes=['b']),
                       evidence('b', ORIGIN - 1, supersedes=['a'])])


def test_t004_proxy_cannot_upgrade_with_early_receipt_and_missing_verified_clock():
    store = EvidenceStore([evidence('proxy', ORIGIN - 100, first_seen_at=ORIGIN - 50,
                                    availability_basis='archive_proxy'),
                           evidence('verified', ORIGIN - 100)])
    assert [row['evidence_id'] for row in store.search(context())['items']] == ['verified']
    with pytest.raises(ValueError, match='publication time'):
        EvidenceStore([evidence('missing-clock', None)])
    with pytest.raises(ValueError, match='precede'):
        EvidenceStore([evidence('reversed', ORIGIN, first_seen_at=ORIGIN - 1)])


def test_t008_t009_cancellation_late_receipt_and_future_effective_notice():
    original = evidence('original', ORIGIN - 100, valid_from=ORIGIN + 86400,
                        payload={'text': 'Planned outage of 400 MW.'})
    cancellation = evidence('cancel', ORIGIN - 1, first_seen_at=ORIGIN + 50,
                            supersedes=['original'], payload={'text': 'The outage is withdrawn.'})
    store = EvidenceStore([original, cancellation])
    visible = store.search(context())['items']
    assert [row['evidence_id'] for row in visible] == ['original']
    assert visible[0]['valid_from'] > ORIGIN
    later = store.search(context(ORIGIN + 51))['items']
    assert [row['evidence_id'] for row in later] == ['cancel']
    assert EventResolver().resolve(later[0], 'outage')['facts'][0]['status'] == 'withdrawn'
    with pytest.raises(ValueError, match='unavailable'):
        store.read(context(), 'cancel')


def test_t029_t030_partial_and_late_outcome_release():
    ledger = FeedbackLedger('agent')
    past = context(ORIGIN - 86400)
    fid = ledger.append_forecast(past, artifact(past))
    with pytest.raises(ValueError, match='24-hour'):
        ledger.append_outcome(fid, np.ones(23), ORIGIN)
    ledger.append_outcome(fid, np.ones(24), ORIGIN + 100)
    assert ledger.snapshot(context()) == []
    assert ledger.snapshot(context(ORIGIN + 99)) == []
    assert len(ledger.snapshot(context(ORIGIN + 100))) == 1


def test_t031_two_arms_share_observations_without_sharing_actions(tmp_path):
    left = FeedbackLedger('left', tmp_path / 'left.jsonl')
    right = FeedbackLedger('right', tmp_path / 'right.jsonl')
    past = context(ORIGIN - 86400)
    a = artifact(past)
    b = replace(a, candidate_id='different-model', quantiles=a.quantiles + 10)
    a_id, b_id = left.append_forecast(past, a), right.append_forecast(past, b)
    left.append_outcome(a_id, np.ones(24), ORIGIN)
    right.append_outcome(b_id, np.ones(24), ORIGIN)
    assert left.snapshot(context())[0]['forecast_id'] == a_id
    assert right.snapshot(context())[0]['forecast_id'] == b_id
    assert left.snapshot(context())[0]['metrics'] != right.snapshot(context())[0]['metrics']
    with pytest.raises(ValueError, match='Unknown forecast'):
        left.append_outcome(b_id, np.ones(24), ORIGIN)


def test_t032_random_outcome_completion_order_has_same_watermark_snapshot():
    contexts = [context(ORIGIN - day * 86400) for day in range(1, 7)]
    results = []
    for seed in range(5):
        ledger = FeedbackLedger('arm')
        ids = [ledger.append_forecast(ctx, artifact(ctx)) for ctx in contexts]
        for i in np.random.default_rng(seed).permutation(len(ids)):
            ledger.append_outcome(ids[i], np.ones(24), contexts[i].origin_unix + 86400)
        results.append(ledger.snapshot(context(watermark=ORIGIN - 86400)))
    assert all(result == results[0] for result in results)
    assert len(results[0]) == 5


@pytest.mark.parametrize('mutation', ['premature', 'partial', 'false_metric', 'bad_axis', 'bad_identity', 'extra_field'])
def test_rehashed_ledger_still_validates_causal_schema(tmp_path, mutation):
    import hashlib
    path = tmp_path / 'feedback.jsonl'
    ledger = FeedbackLedger('arm', path)
    past = context(ORIGIN - 86400)
    fid = ledger.append_forecast(past, artifact(past))
    ledger.append_outcome(fid, np.ones(24), ORIGIN)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == 'premature':
        rows[1]['payload']['outcome_available_at'] = ORIGIN - 1
    elif mutation == 'partial':
        rows[1]['payload']['actual'].pop()
    elif mutation == 'false_metric':
        rows[1]['payload']['metrics']['qs'] = -999
    elif mutation == 'bad_axis':
        rows[0]['payload']['target_time_unix'][0] -= 3600
    elif mutation == 'bad_identity':
        rows[0]['payload']['forecast_id'] = 'fabricated'
    else:
        rows[1]['payload']['future_private_actions'] = ['choose_winner']
    previous = None
    for row in rows:
        row['previous_hash'] = previous
        body = {key: value for key, value in row.items() if key != 'event_hash'}
        row['event_hash'] = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        previous = row['event_hash']
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        FeedbackLedger('arm', path)


def test_first_seen_archive_fetch_is_explicit_distinct_from_actual_availability():
    store = EvidenceStore([evidence('delayed-availability', ORIGIN - 2, first_seen_at=ORIGIN + 100),
                           evidence('archival-copy', ORIGIN - 2, first_seen_at=ORIGIN + 100,
                                    first_seen_role='archive_fetch')])
    items = store.search(context())['items']
    assert [row['evidence_id'] for row in items] == ['archival-copy']
    assert items[0]['first_seen_role'] == 'archive_fetch'
    with pytest.raises(ValueError, match='verified publication'):
        EvidenceStore([evidence('proxy', ORIGIN - 2, availability_basis='archive_proxy',
                                first_seen_at=ORIGIN + 100, first_seen_role='archive_fetch')])


@pytest.mark.parametrize('prices', [np.ones(72), np.ones(72) * -100, np.r_[np.zeros(71), 10000]])
def test_t005_constant_negative_and_spike_states_remain_finite_without_event_claims(prices):
    state = summarize_state(context(price=prices))
    json.dumps(state, allow_nan=False)
    assert state['nonfinite_count'] == 0
    assert 'facts' not in state and 'event_type' not in state
    with pytest.raises(ValueError, match='snapshot'):
        context(price=np.r_[prices[:-1], np.nan])


def test_t006_missing_sources_and_duplicate_reposts_not_independent_evidence():
    assert EvidenceStore().search(context())['status'] == 'real_data_unavailable'
    store = EvidenceStore([evidence('original', ORIGIN - 100, source_id='publisher'),
                           evidence('repost', ORIGIN - 50, source_id='republisher')])
    items = store.search(context())['items']
    assert len(items) == 1
    assert {items[0]['evidence_id'], *items[0]['duplicate_evidence_ids']} == {'original', 'repost'}
