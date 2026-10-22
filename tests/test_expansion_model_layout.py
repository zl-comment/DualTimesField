"""Layout and registry contracts for the 2026-10-22 expansion baselines."""
import json
from pathlib import Path

import pytest
import yaml

from forecasting.aligned_expansion import validate_expansion_config
from forecasting.common.baseline_registry import (EXPANSION_MODELS, REGISTRY,
                                                  expansion_availability,
                                                  expansion_model_ids)
from forecasting.datasets import load_forecast_config

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = tuple(spec.model_id for spec in EXPANSION_MODELS)
MATRIX = 'aemo_forecast_aligned_expansion_20261022.yaml'
PARENT_MATRIX = 'aemo_forecast_aligned_20261012.yaml'
SHARED_SECTIONS = ('data', 'forecast_protocol', 'aligned_protocol', 'future_exogenous',
                   'ctf_exogenous', 'quantile_exogenous', 'predispatch_exogenous',
                   'interconnector_exogenous', 'origin_context')


def test_registry_is_fixed_source_and_consistent():
    assert expansion_model_ids() == ('lightgbm_cp', 'lightgbm_qra', 'xgboost_cp', 'ddnn_jsu', 'nhits_qra')
    for spec in EXPANSION_MODELS:
        assert spec.output_table == 'Q'
        assert (ROOT / 'forecasting' / spec.model_id / 'model.py').is_file()
        assert spec.resource in ('cpu', 'gpu')
    availability = expansion_availability()
    assert all(entry['implemented'] for entry in availability.values())
    # Tree-calibration models reuse frozen bases; the two native networks do not.
    assert REGISTRY['lightgbm_cp'].base_model == 'lightgbm'
    assert REGISTRY['xgboost_cp'].base_model == 'xgboost'
    assert REGISTRY['ddnn_jsu'].base_model is None and REGISTRY['ddnn_jsu'].native_training
    assert REGISTRY['nhits_qra'].requires_calibration and REGISTRY['nhits_qra'].native_training


def test_expansion_matrix_config_declares_every_registered_model():
    config = load_forecast_config(ROOT / 'configs/experiments' / MATRIX)
    assert validate_expansion_config(config) == list(expansion_model_ids())


def test_unknown_or_partial_expansion_matrices_are_rejected():
    partial = yaml.safe_load((ROOT / 'configs/experiments' / MATRIX).read_text())
    partial['experiment']['models'] = partial['experiment']['models'][:2]
    bad = yaml.safe_load((ROOT / 'configs/experiments' / MATRIX).read_text())
    bad['experiment']['models'] = ['lightgbm', 'lightgbm_cp']
    with pytest.raises(ValueError):
        validate_expansion_config(partial)
    with pytest.raises(ValueError):
        validate_expansion_config(bad)
    no_schema = yaml.safe_load((ROOT / 'configs/experiments' / MATRIX).read_text())
    no_schema.pop('expansion')
    with pytest.raises(ValueError):
        validate_expansion_config(no_schema)


def test_expansion_shared_sections_match_parent_protocol():
    parent = yaml.safe_load((ROOT / 'configs/experiments' / PARENT_MATRIX).read_text())
    matrix = yaml.safe_load((ROOT / 'configs/experiments' / MATRIX).read_text())
    for key in SHARED_SECTIONS:
        assert matrix[key] == parent[key], key
    assert matrix['model'] == {'quantiles': parent['model']['quantiles']}
    for folder in FOLDERS:
        config = load_forecast_config(ROOT / 'configs' / folder / 'aligned.yaml')
        for key in SHARED_SECTIONS:
            assert config[key] == parent[key], (folder, key)
        training = dict(config['training'])
        training['output_directory'] = parent['training']['output_directory']
        if folder in ('ddnn_jsu', 'nhits_qra'):
            training['learning_rate'] = parent['training']['learning_rate']
            rate = matrix['experiment']['neural']['learning_rates'][folder]
            assert config['training']['learning_rate'] == rate
            assert config['experiment']['neural']['learning_rates'][folder] == rate
        assert training == parent['training'], folder
        source = json.loads((ROOT / 'forecasting' / folder / 'source.json').read_text())
        assert source['experiment_model_id'] == folder
        assert (ROOT / 'forecasting' / folder / 'README.md').is_file()
        kind, = config['experiment']['models']
        assert kind == folder
        assert validate_expansion_config(config) == [folder]
        for group, section in (('baseline_parameters', 'experiment'), ('model_baselines', None)):
            declared = (config['experiment'] if group == 'baseline_parameters' else config).get(group, {})
            expected = matrix['experiment'].get(group, {}) if group == 'baseline_parameters' \
                else matrix.get('model_baselines', {})
            assert declared.get(folder) == expected.get(folder), (folder, group)
