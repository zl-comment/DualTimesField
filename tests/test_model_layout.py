"""Keep nested model configs executable and equivalent to their matrix settings."""
import copy
import json
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from forecasting.aligned_experiment import validate_config
from forecasting.datasets import load_forecast_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('full', 'no_fields', 'mlp', 'linear', 'dlinear', 'patchtst',
          'gatets', 'timexer', 'pricefm', 'lightgbm', 'xgboost', 'lasso')


@pytest.mark.parametrize('folder', MODELS)
def test_model_config_preserves_common_protocol_and_effective_budget(folder):
    path = ROOT / 'configs' / folder / 'aligned.yaml'
    config = load_forecast_config(path)
    assert Path(config['project_root']) == ROOT
    validate_config(config)
    kind, = config['experiment']['models']
    source = json.loads((ROOT / 'forecasting' / folder / 'source.json').read_text())
    assert source['experiment_model_id'] == kind
    filename = ('aemo_forecast_aligned_20261013_extension.yaml'
                if folder in ('timexer', 'pricefm') else 'aemo_forecast_aligned_20261012.yaml')
    matrix = yaml.safe_load((ROOT / 'configs/experiments' / filename).read_text())
    for key in ('data', 'forecast_protocol', 'aligned_protocol', 'future_exogenous',
                'ctf_exogenous', 'quantile_exogenous', 'predispatch_exogenous',
                'interconnector_exogenous', 'origin_context'):
        assert config.get(key) == matrix.get(key), key
    training = dict(config['training'])
    training['output_directory'] = matrix['training']['output_directory']
    effective_rate = matrix['experiment']['neural']['learning_rates'].get(kind)
    if effective_rate is not None:
        assert config['experiment']['neural']['learning_rates'][kind] == effective_rate
        assert config['training']['learning_rate'] == effective_rate
        training['learning_rate'] = matrix['training']['learning_rate']
    assert training == matrix['training']
    expected = copy.deepcopy(matrix['model'])
    if kind == 'no_fields':
        expected['field_forecast'] = False  # Existing runner applies this switch.
    elif kind != 'full':
        expected = {'quantiles': expected['quantiles']}
        assert 'loss' not in config  # Full's auxiliary losses are not used here.
    assert config['model'] == expected
    for group in ('baseline_parameters',):
        assert config['experiment'][group].get(kind) == matrix['experiment'][group].get(kind)
    assert config['experiment']['neural']['learning_rates'].get(kind) == matrix['experiment']['neural']['learning_rates'].get(kind)
    if kind in matrix.get('model_baselines', {}):
        assert config['model_baselines'][kind] == matrix['model_baselines'][kind]


def test_single_model_scope_does_not_allow_unknown_models_or_incomplete_seeds():
    config = load_forecast_config(ROOT / 'configs/lightgbm/aligned.yaml')
    for key, value in [('models', ['unknown']), ('models', ['lightgbm', 'xgboost']), ('seeds', [2026])]:
        bad = copy.deepcopy(config)
        bad['experiment'][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    with pytest.raises(ValueError):
        validate_config(config, extension=True)


def test_nested_legacy_config_resolves_project_data_root():
    config = load_forecast_config(ROOT / 'configs/legacy/aemo_forecast_ic_calibrator.yaml')
    assert Path(config['project_root']) == ROOT


@pytest.mark.parametrize('folder', ['mlp', 'linear', 'dlinear', 'patchtst', 'gatets'])
def test_extracted_neural_parameters_match_previously_implicit_defaults(folder):
    from forecasting.aligned_inputs import INPUT_KEYS
    from forecasting.aligned_runner import _native_batch
    from forecasting.general_baselines import build
    from forecasting.gatets.model import build_gatets
    from forecasting.mlp.model import PlainMLP

    config = load_forecast_config(ROOT / 'configs' / folder / 'aligned.yaml')
    kind, = config['experiment']['models']
    shapes = [(72, 2), (72, 10), (24, 10), (1,), (24, 11), (24, 1), (24, 3)]
    example = {key: torch.zeros(2, *shape) for key, shape in zip(INPUT_KEYS, shapes)}
    def make(parameters):
        torch.manual_seed(2026)
        if kind == 'mlp':
            return PlainMLP(example, 24, **parameters)
        if kind == 'gatets_adapter':
            return build_gatets(kind, _native_batch(example), **parameters)
        return build(kind, _native_batch(example), **parameters)
    old = make({}).state_dict()
    new = make(config['model_baselines'][kind]).state_dict()
    assert old.keys() == new.keys()
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)


def test_single_split_cache_reader_does_not_open_other_periods(tmp_path):
    from forecasting.aligned_experiment import save_cache, sha
    from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
    from forecasting.common.cache import load_split

    values = np.ones((2, 2), dtype='float32')
    sample = AlignedSplit({key: values.copy() for key in INPUT_KEYS}, values,
                          np.array([0, 3600]), np.array([[0, 3600], [3600, 7200]]),
                          {'region': 'NSW1'}, values)
    cache = tmp_path / 'data_cache/NSW1'
    save_cache(cache, {'selection': sample, 'test': sample})
    manifest = {'cache_hashes': {'NSW1': sha(cache / 'hashes.json')}}
    (cache / 'test/actual.npy').write_bytes(b'inaccessible test labels')
    loaded = load_split(tmp_path, manifest, 'NSW1', 'selection')
    np.testing.assert_array_equal(loaded.actual, sample.actual)
    with pytest.raises(ValueError, match='changed'):
        load_split(tmp_path, manifest, 'NSW1', 'test')
