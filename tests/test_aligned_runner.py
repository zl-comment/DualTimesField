"""Integration checks at the model boundary, including label-poisoning isolation."""
import copy

import numpy as np
import pytest
import torch

from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.aligned_runner import _forward, fit_neural


def split(start, seed):
    rng = np.random.default_rng(seed)
    n, history, horizon = 12, 8, 3
    shapes = [(history, 2), (history, 2), (horizon, 2), (1,), (horizon, 2),
              (horizon, 1), (horizon, 3)]
    inputs = {k: rng.normal(size=(n, *s)).astype('float32') for k, s in zip(INPUT_KEYS, shapes)}
    y = rng.normal(size=(n, horizon)).astype('float32')
    origins = start + np.arange(n) * 3600
    times = origins[:, None] + np.arange(horizon)[None, :] * 3600
    return AlignedSplit(inputs=inputs, actual=y * 10 + 20, target=y,
                        origin_unix=origins, target_time_unix=times,
                        metadata={'preprocessing_hash': 'same-training-scaler',
                                  'input_hash': 'fixture-input', 'sample_hash': 'fixture-samples',
                                  'target_mean': 20., 'target_std': 10.,
                                  'region': 'NSW1', 'protocol_id': 'fixture'})


@pytest.mark.parametrize('kind', ['mlp', 'linear'])
def test_evaluation_label_poisoning_cannot_change_trained_prediction(kind):
    torch.set_num_threads(2)
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    config = {'training': {'weight_decay': 0., 'epochs': 1}}
    q, _, meta = fit_neural(kind, train, selection, evaluation, config, epochs=1, batch_size=6)
    poison = copy.deepcopy(evaluation)
    poison.actual[:] = 1e15
    poison.target[:] = -1e15
    q2, _, meta2 = fit_neural(kind, train, selection, poison, config, epochs=1, batch_size=6)
    np.testing.assert_array_equal(q, q2)
    assert meta == meta2
    assert np.isfinite(q).all() and q.shape == (12, 3, 5)


def test_model_forward_never_receives_supervision():
    class BoundarySpy(torch.nn.Module):
        def forward(self, values):
            assert set(values) == set(INPUT_KEYS)
            return values['history_values']
    sample = split(0, 1)
    batch = {k: torch.from_numpy(v) for k, v in sample.inputs.items()}
    batch['target_price'] = torch.tensor(float('nan'))
    batch['target_quantile'] = torch.tensor(float('nan'))
    _forward(BoundarySpy(), 'mlp', batch)


def test_overlapping_or_unrealized_training_targets_are_rejected():
    train, selection, evaluation = split(0, 1), split(100, 2), split(200000, 3)
    with pytest.raises(ValueError, match='chronologically available'):
        fit_neural('linear', train, selection, evaluation, {'training': {}}, epochs=1)


def test_input_scaler_mismatch_is_rejected():
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    evaluation.metadata['preprocessing_hash'] = 'different'
    with pytest.raises(ValueError, match='preprocessing'):
        fit_neural('linear', train, selection, evaluation, {'training': {}}, epochs=1)


def test_wrong_model_quantile_levels_cannot_be_mislabeled_by_common_scorer():
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    config = {'training': {}, 'model': {'quantiles': [.1,.2,.5,.8,.9]}}
    with pytest.raises(ValueError, match='quantiles differ'):
        fit_neural('linear', train, selection, evaluation, config, epochs=1)


def test_model_price_column_convention_is_explicit():
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    config = {'training': {}, 'data': {'history_columns': ['demand','price'], 'target_column': 'price'}}
    with pytest.raises(ValueError, match='column zero'):
        fit_neural('linear', train, selection, evaluation, config, epochs=1)


def test_preloaded_batches_preserve_values_and_shuffle_order():
    from forecasting.aligned_runner import AlignedTensorBatches, _loader
    train = split(0, 1)
    expected = _loader(train, 5, shuffle=True, seed=2028)
    actual = AlignedTensorBatches(train, 5, 'cpu', shuffle=True, seed=2028, include_targets=True)
    for _ in range(3):
        for reference, candidate in zip(expected, actual, strict=True):
            for key in (*INPUT_KEYS, 'target_price', 'target_quantile', 'target_price_raw'):
                torch.testing.assert_close(reference[key], candidate[key], rtol=0, atol=0)
        assert actual.tensors['target_price'] is actual.tensors['target_quantile']


def test_preload_option_does_not_change_training_or_predictions():
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    config = {'training': {'weight_decay': 0.}}
    q, _, meta = fit_neural('linear', train, selection, evaluation, config,
                           epochs=2, batch_size=5, preload_batches=False)
    q2, _, meta2 = fit_neural('linear', train, selection, evaluation, config,
                             epochs=2, batch_size=5, preload_batches=True)
    np.testing.assert_array_equal(q, q2)
    assert meta['selection_curve'] == meta2['selection_curve']
    assert meta2['preloaded_batches'] is True


def test_best_checkpoint_and_progress_distinguish_budget_exhaustion(tmp_path):
    import json
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    checkpoint, progress = tmp_path / 'cell' / 'best.pt', tmp_path / 'cell' / 'progress.jsonl'
    q, _, meta = fit_neural('linear', train, selection, evaluation, {'training': {}},
                           epochs=2, patience=4, batch_size=6,
                           checkpoint_path=checkpoint, progress_path=progress,
                           preload_batches=True)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['best_epoch'] == meta['best_epoch']
    assert saved['best_selection_value'] == min(meta['selection_curve'])
    assert saved['model_state'] and saved['optimizer_resume_supported'] is False
    assert saved['input_schema']['history_values']['shape'] == [8, 2]
    assert saved['forecast_horizon'] == 3
    assert saved['train_metadata']['preprocessing_hash'] == train.metadata['preprocessing_hash']
    assert 'evaluation_metadata' not in saved
    from forecasting import general_baselines
    from forecasting.aligned_runner import _native_batch, _predict
    example = {key: torch.zeros((1, *value['shape'])) for key, value in saved['input_schema'].items()}
    restored = general_baselines.build('linear', _native_batch(example))
    restored.load_state_dict(saved['model_state'])
    restored_q, _ = _predict(restored, 'linear', evaluation, torch.device('cpu'), 6)
    np.testing.assert_array_equal(q, restored_q)
    records = [json.loads(row) for row in progress.read_text().splitlines()]
    assert [row['event'] for row in records] == ['start', 'epoch', 'epoch', 'complete']
    assert records[-1]['stop_reason'] == meta['stop_reason'] == 'epoch_budget_exhausted'
    assert meta['budget_cap_reached'] and not meta['early_stopped']
    assert not checkpoint.with_name('best.pt.tmp').exists()


def test_early_stop_keeps_first_best_checkpoint(tmp_path, monkeypatch):
    import json
    import forecasting.aligned_runner as runner
    train, selection, evaluation = split(0, 1), split(100000, 2), split(200000, 3)
    predict = runner._predict
    def flat_selection(model, kind, split_data, device, batch_size, *, loader=None):
        if split_data is selection:
            return np.repeat(split_data.actual[..., None], 5, axis=-1), None
        return predict(model, kind, split_data, device, batch_size, loader=loader)
    monkeypatch.setattr(runner, '_predict', flat_selection)
    checkpoint, progress = tmp_path / 'best.pt', tmp_path / 'progress.jsonl'
    _, _, meta = runner.fit_neural('linear', train, selection, evaluation, {'training': {}},
                                   epochs=10, patience=2, batch_size=6,
                                   checkpoint_path=checkpoint, progress_path=progress)
    assert meta['epochs_executed'] == 3 and meta['best_epoch'] == 0
    assert meta['early_stopped'] and not meta['budget_cap_reached']
    assert torch.load(checkpoint, weights_only=True)['best_epoch'] == 0
    assert json.loads(progress.read_text().splitlines()[-1])['stop_reason'] == 'early_stopping'


def test_preload_gpu_budget_reserves_memory_without_allocating(monkeypatch):
    from forecasting.aligned_runner import _preload_allowed
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10 * 1024 ** 3, 12 * 1024 ** 3))
    assert _preload_allowed(torch.device('cuda'), 'auto', 1024 ** 3)
    assert not _preload_allowed(torch.device('cuda'), 'auto', 4 * 1024 ** 3)
    with pytest.raises(MemoryError, match='memory'):
        _preload_allowed(torch.device('cuda'), True, 4 * 1024 ** 3)
