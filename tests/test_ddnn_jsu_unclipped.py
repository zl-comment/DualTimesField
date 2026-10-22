"""Regression for the formal JSU tail clipping and its selection-time impact."""
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import johnsonsu
import torch
import yaml

from forecasting.ddnn_jsu.model import jsu_negative_log_likelihood, jsu_quantiles
from forecasting.aligned_expansion import validate_expansion_config
from forecasting.aligned_extension import assert_shared_protocol


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_positive_and_negative_tails_match_scipy_without_plateaus(dtype):
    # +/-100 also overflows float32 sinh, but has finite float64 quantiles.
    mu = torch.tensor([[-20., 20., -100., 100.]], dtype=dtype)
    sigma = torch.tensor([[3., 4., 2., 2.]], dtype=dtype)
    xi = torch.tensor([[-5., 2., 1., -1.]], dtype=dtype)
    lam = torch.tensor([[.1, 2., 3., 4.]], dtype=dtype)
    levels = torch.tensor([.05, .1, .5, .9, .95], dtype=torch.float64)
    actual = jsu_quantiles(mu, sigma, xi, lam, levels)
    expected = johnsonsu.ppf(levels.numpy(),
        a=(-mu.double() / sigma.double()).numpy()[..., None],
        b=(1 / sigma.double()).numpy()[..., None],
        loc=xi.double().numpy()[..., None], scale=lam.double().numpy()[..., None])
    assert actual.dtype == torch.float64 and torch.isfinite(actual).all()
    assert torch.all(actual.diff(dim=-1) > 0)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-13, atol=1e-10)


def test_nll_matches_the_same_unbounded_scipy_density():
    target = torch.tensor([[-1e8, -20., 0., 300., 1e8]], dtype=torch.float64)
    mu = torch.tensor([[-20., .4, 2., -.2, 20.]], dtype=torch.float64)
    sigma = torch.tensor([[.2, 1.4, .8, 4., 3.]], dtype=torch.float64)
    xi = torch.tensor([[-5., 1., 7., 0., 2.]], dtype=torch.float64)
    lam = torch.tensor([[.1, 2., 3., 4., .5]], dtype=torch.float64)
    expected = -johnsonsu.logpdf(target.numpy(), a=(-mu / sigma).numpy(), b=(1 / sigma).numpy(),
                                  loc=xi.numpy(), scale=lam.numpy()).mean()
    actual = jsu_negative_log_likelihood(target, mu, sigma, xi, lam)
    assert float(actual) == pytest.approx(expected, rel=1e-12)


def test_float64_overflow_is_an_error_instead_of_a_capped_forecast():
    scalar = torch.ones(1, 1)
    with pytest.raises(FloatingPointError, match='refusing to clip'):
        jsu_quantiles(scalar * 1000, scalar, scalar, scalar, torch.tensor([.5]))


@pytest.mark.parametrize('levels', [[0., .5], [.5, 1.], [float('nan')], []])
def test_invalid_levels_are_rejected(levels):
    scalar = torch.ones(1, 1)
    with pytest.raises(ValueError, match='levels'):
        jsu_quantiles(scalar, scalar, scalar, scalar, torch.tensor(levels))


def test_invalid_scales_and_parameters_are_rejected():
    scalar = torch.ones(1, 1)
    for sigma, lam in [(scalar * 0, scalar), (scalar, -scalar)]:
        with pytest.raises(ValueError, match='positive'):
            jsu_quantiles(scalar, sigma, scalar, lam, torch.tensor([.5]))
    with pytest.raises(ValueError, match='finite'):
        jsu_quantiles(scalar * float('nan'), scalar, scalar, scalar, torch.tensor([.5]))


def test_tail_gradient_does_not_flatten_at_the_old_clamp_boundary():
    mu = torch.tensor([[20.]], requires_grad=True)
    scalar = torch.ones_like(mu)
    q = jsu_quantiles(mu, scalar, scalar * 0, scalar, torch.tensor([.5]))
    q.sum().backward()
    assert mu.grad.item() == pytest.approx(np.cosh(20), rel=1e-6)


def test_repair_configuration_preserves_formal_training_budget_and_information():
    root = Path(__file__).resolve().parents[1]
    repaired = yaml.safe_load((root / 'configs/experiments/aemo_forecast_ddnn_jsu_unclipped_20261022.yaml').read_text())
    matrix = yaml.safe_load((root / 'configs/experiments/aemo_forecast_aligned_expansion_20261022.yaml').read_text())
    assert validate_expansion_config(repaired) == ['ddnn_jsu']
    assert_shared_protocol(matrix, repaired)
    assert repaired['training'] == matrix['training']
    assert repaired['model_baselines']['ddnn_jsu'] == matrix['model_baselines']['ddnn_jsu']
    assert repaired['training']['learning_rate'] == .0003
    assert repaired['experiment']['neural']['learning_rates']['ddnn_jsu'] == .0003
    assert repaired['repair']['new_hyperparameter_search'] is False
    for name in ('ddnn_jsu', 'nhits_qra'):
        single = yaml.safe_load((root / 'configs' / name / 'aligned.yaml').read_text())
        assert single['training']['learning_rate'] == single['experiment']['neural']['learning_rates'][name]
        assert matrix['training']['learning_rate'] == matrix['experiment']['neural']['learning_rates'][name]
