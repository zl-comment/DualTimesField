"""Canonical cache broker: label release, historical reservoir and host isolation."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from forecasting.adaptive_agent.context import ContextRepository
from forecasting.adaptive_agent.analogs import AnalogStore
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.memory import FeedbackLedger
from forecasting.adaptive_agent.runtime import ToolHost
from forecasting.aligned_inputs import INPUT_KEYS

ORIGIN = 1672531200


def cache(directory, count=73):
    metadata = {'region': 'NSW1', 'target_std': 2., 'target_mean': 50.,
                'preprocessing_hash': 'fixture-transform', 'split': 'test'}
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    origins = ORIGIN + np.arange(count) * 3600
    np.save(directory / 'origin_unix.npy', origins)
    np.save(directory / 'target_time_unix.npy', origins[:, None] + np.arange(24) * 3600)
    shapes = {'history_values': (72, 2), 'history_calendar': (72, 10),
              'future_calendar': (24, 10)}
    for key in INPUT_KEYS:
        np.save(directory / f'input__{key}.npy', np.zeros((count,) + shapes.get(key, (1,)), dtype=np.float32))
    np.save(directory / 'actual.npy', np.broadcast_to(np.arange(count)[:, None], (count, 24)))
    return ContextRepository(directory)


def test_first_origin_has_no_eligible_history_or_label_reads(tmp_path, monkeypatch):
    repo = cache(tmp_path)
    ctx = repo.context(0)
    original = np.load
    def guarded(path, *args, **kwargs):
        if Path(path).name == 'actual.npy':
            raise AssertionError('First origin must not read any target outcomes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    assert repo.episodes(ctx) == []
    assert repo.outcome_reads == []
    assert ctx.model_inputs.keys() == set(INPUT_KEYS)
    assert 'actual' not in ctx.metadata


def test_mature_actual_rejects_future_before_opening_file(tmp_path, monkeypatch):
    repo = cache(tmp_path)
    original = np.load
    opened = []
    def watched(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', watched)
    with pytest.raises(ValueError, match='fully matured'):
        repo.mature_actual(0, ORIGIN + 86400 - 1)
    assert opened == []
    actual = repo.mature_actual(0, ORIGIN + 86400)
    assert opened == ['actual.npy']
    np.testing.assert_array_equal(actual, np.zeros(24))
    assert repo.outcome_reads == [{'index': 0, 'available_at': ORIGIN + 86400, 'cutoff': ORIGIN + 86400}]
    with pytest.raises(ValueError):
        repo.mature_actual(-1, ORIGIN + 86400)


def test_reservoir_recent16_plus_uniform_older16_spans_current_split(tmp_path):
    repo = cache(tmp_path, 24 * 100 + 1)
    current = repo.context(24 * 100)
    rows = repo.episodes(current)
    timestamps = np.array([row['context'].origin_unix for row in rows])
    assert len(rows) == 32 and len(set(timestamps)) == 32
    np.testing.assert_array_equal(timestamps[:16], current.origin_unix - np.arange(1, 17) * 86400)
    assert timestamps.max() - timestamps.min() == 99 * 86400
    assert timestamps.min() == ORIGIN
    assert np.min(np.diff(np.sort(timestamps))) >= 86400
    assert all(row['outcome_available_at'] <= current.cutoff for row in rows)
    assert all(row['retrieval_scope'] == 'current_split_only' for row in rows)
    assert all(row['data_role'] == 'test' for row in rows)
    assert len(repo.episodes(current, limit=8)) == 8
    assert len(repo.episodes(current, limit=1)) == 1
    with pytest.raises(ValueError):
        repo.episodes(current, limit=33)


def test_reservoir_ignores_target_values_and_obeys_watermark(tmp_path):
    repo = cache(tmp_path, 24 * 40 + 1)
    current = repo.context(24 * 40)
    ids = [row['episode_id'] for row in repo.episodes(current)]
    np.save(tmp_path / 'actual.npy', np.full((24 * 40 + 1, 24), 999.))
    assert [row['episode_id'] for row in repo.episodes(current)] == ids
    delayed = replace(current, memory_watermark=ORIGIN + 10 * 86400)
    rows = repo.episodes(delayed)
    assert len(rows) == 10
    assert max(row['outcome_available_at'] for row in rows) <= delayed.memory_watermark
    assert max(row['context'].origin_unix for row in rows) == ORIGIN + 9 * 86400


def test_tool_host_has_no_repository_or_evaluation_accessor(tmp_path):
    repo = cache(tmp_path)
    ctx = repo.context(72)
    host = ToolHost(object(), EvidenceStore(), AnalogStore(repo.episodes(ctx)), FeedbackLedger('arm'))
    assert not hasattr(host, 'evaluation_actual')
    assert not hasattr(host, 'mature_actual')
    assert all(not isinstance(value, ContextRepository) for value in vars(host).values())
    assert all(getattr(value, '__self__', None) is not repo for value in vars(host).values())
    summaries = host.analogs.retrieve(ctx)['items']
    assert all('actual' not in row and 'context' not in row for row in summaries)
    np.testing.assert_array_equal(repo.evaluation_actual([0, 1])[:, 0], [0, 1])
