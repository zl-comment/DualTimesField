"""Lean checks for the R24b candidate ledger."""
import numpy as np
import pytest

from forecasting.adaptive_agent import input_policy_r24b
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.iterative_runtime import R24IterativeToolHost
from forecasting.adaptive_agent.memory import FeedbackLedger, demand_state_bucket
from forecasting.adaptive_agent.replay import policy_host_factory, policy_module
from test_adaptive_agent_input_policy_r22 import (
    Client, DEFAULT, OPERATED, forecast as make_forecast,
)
from test_adaptive_agent_r24 import LoopHost, snapshot_at

ORIGIN = 1_700_000_000


def ledger_context(origin, demand=None):
    demand = (np.linspace(-1, 1, 72) if demand is None else demand)
    return ForecastContext(snapshot_at(origin), {'history_values': np.column_stack(
        [np.linspace(-1, 1, 72), demand])}, {})


def test_demand_state_bucket_sign():
    assert demand_state_bucket(ledger_context(ORIGIN)) == 'demand_warming'
    assert demand_state_bucket(ledger_context(ORIGIN, np.linspace(1, -1, 72))) == 'demand_cooling'


def test_ledger_round_trips_state_bucket(tmp_path):
    ledger = FeedbackLedger('r24b-test', tmp_path / 'feedback.jsonl')
    ctx = ledger_context(ORIGIN)
    key = ledger.append_forecast(ctx, make_forecast(ctx, OPERATED, 10),
                                 state_bucket='demand_warming')
    ledger.append_outcome(key, np.full(24, 10.0), ORIGIN + 24 * 3600)
    rows = ledger.snapshot(ledger_context(ORIGIN + 2 * 86400))
    assert len(rows) == 1
    assert rows[0]['state_bucket'] == 'demand_warming'
    assert rows[0]['candidate_id'] == OPERATED


def test_old_ledger_format_still_validates(tmp_path):
    ledger = FeedbackLedger('r24b-test', tmp_path / 'feedback.jsonl')
    ctx = ledger_context(ORIGIN)
    key = ledger.append_forecast(ctx, make_forecast(ctx, DEFAULT, 10))
    ledger.append_outcome(key, np.full(24, 10.0), ORIGIN + 24 * 3600)
    reloaded = FeedbackLedger('r24b-test', tmp_path / 'feedback.jsonl')
    assert len(reloaded.snapshot(ledger_context(ORIGIN + 2 * 86400))) == 1


class LedgerHost(LoopHost):
    """LoopHost plus a real feedback ledger matured from an earlier origin."""

    def __init__(self):
        super().__init__()
        self.feedback = FeedbackLedger('r24b-policy-test')

    def prime_ledger(self, earlier):
        for candidate_id, level in ((DEFAULT, 12), (OPERATED, 10)):
            art = make_forecast(earlier, candidate_id, level)
            key = self.feedback.append_forecast(earlier, art,
                                                state_bucket='demand_warming')
            self.feedback.append_outcome(key, np.full(24, 10.0),
                                         earlier.origin_unix + 24 * 3600)


def test_stage_a_sees_candidate_ledger_with_bucket_counts():
    ctx = ledger_context(ORIGIN)
    host = LedgerHost()
    host.prime_ledger(ledger_context(ORIGIN - 2 * 86400))
    client = Client(['retrieve_analogs:recent', 'stop_and_evaluate', 'use_identity'])
    selected, trace = input_policy_r24b.run(ctx, host, client)
    assert selected.candidate_id == DEFAULT
    ledger = trace['candidate_ledger']
    assert ledger['matured_candidate_rows'] == 2
    assert ledger['candidates'][OPERATED]['sample_count'] == 1
    assert ledger['candidates'][OPERATED]['state_bucket_counts'] == {'demand_warming': 1}
    assert ledger['candidates'][OPERATED]['mean_qs_minus_identity_mean_qs'] < 0
    stage_a_state = client.calls[-1][2]
    actions = [row['action'] for row in stage_a_state['observations']]
    assert actions[0] == 'r24b_candidate_ledger'
    assert actions[1] == 'r22_structured_evidence_report'


def test_registration():
    assert policy_module('r24b_candidate_ledger_input') is input_policy_r24b
    assert policy_host_factory('r24b_candidate_ledger_input') is R24IterativeToolHost
    contract = input_policy_r24b.manifest_contract()
    assert contract['candidate_ledger_schema'] == 'r24b-candidate-ledger-v1'
    assert all(entry['version'] == '8.1.0'
               for entry in input_policy_r24b.load_skills().values())
