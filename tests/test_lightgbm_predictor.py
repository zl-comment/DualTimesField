"""The diagnostic reader works independently after removal of Full V2."""
import copy
import json

import numpy as np
import pytest

from forecasting.aligned_baselines import fit_predict
from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.lightgbm.predictor import FrozenLightGBMPredictor


def split(start):
    rng = np.random.default_rng(start)
    shapes = [(4, 2), (4, 2), (2, 2), (1,), (2, 2), (2, 1), (2, 3)]
    inputs = {k: rng.normal(size=(48, *s)).astype('float32') for k, s in zip(INPUT_KEYS, shapes)}
    target = inputs['future_exogenous'][..., 0] * 2
    origins = (start + np.arange(48)) * 3600
    times = origins[:, None] + np.arange(2) * 3600
    meta = dict(region='synthetic', protocol_id='fixture', preprocessing_hash='same',
                input_hash=str(start), sample_hash=str(start), target_mean=10., target_std=3.)
    return AlignedSplit(inputs, target * 3 + 10, origins, times, meta, target)


@pytest.fixture
def saved(tmp_path):
    train, selection, evaluation = (split(s) for s in (0, 100, 200))
    result = fit_predict('lightgbm', train, selection, evaluation, seed=2026,
                         params=dict(n_estimators=5, early_stopping_rounds=2,
                                     num_leaves=5, min_child_samples=3, n_jobs=1),
                         model_directory=tmp_path)
    # Production manifests without this optional field use canonical 72/24 shapes.
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['input_shapes'] = {k: list(v.shape[1:]) for k, v in train.inputs.items()}
    path.write_text(json.dumps(manifest))
    return tmp_path, evaluation, result


def test_saved_prediction_matches_native_and_does_not_read_labels(saved):
    directory, evaluation, native = saved
    reader = FrozenLightGBMPredictor(directory, region='synthetic', seed=2026, threads=1)
    expected = np.sort(native.quantiles, axis=-1)
    np.testing.assert_array_equal(reader.predict(evaluation), expected)
    poisoned = copy.deepcopy(evaluation)
    poisoned.actual[:] = np.nan
    poisoned.target[:] = np.nan
    np.testing.assert_array_equal(reader.predict(poisoned), expected)


def test_reader_rejects_wrong_identity_or_changed_model(saved):
    directory, evaluation, _ = saved
    with pytest.raises(ValueError, match='seed mismatch'):
        FrozenLightGBMPredictor(directory, seed=2027, threads=1)
    reader = FrozenLightGBMPredictor(directory, threads=1)
    path = directory / reader.entries[(0, 0)]['path']
    path.write_text(path.read_text() + '\nmodified\n')
    with pytest.raises(ValueError, match='changed after loading'):
        reader.predict(evaluation)
