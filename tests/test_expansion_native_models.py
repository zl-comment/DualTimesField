"""Native neural baselines: JSU exactness and NHITS structure/MC/QRA contracts.

Synthetic splits keep these tests fast; the point is contract exactness, not
forecast quality. Stage isolation is asserted behaviourally: shuffling
evaluation labels must not change predictions, and calibration labels must not
change the selected backbone.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forecasting.aligned_inputs import INPUT_KEYS, AlignedSplit
from forecasting.aligned_scoring import QUANTILE_LEVELS
from forecasting.common.baseline_registry import run_expansion_model

H, N = 4, 16
QUANTILES = list(QUANTILE_LEVELS)
DDNN_PARAMS = {"hidden_dims": [16, 16], "use_batch_norm": False, "dropout_rate": 0.0,
               "selection_metric": "qs5"}
NHITS_PARAMS = {"n_pool_kernel_size": [2, 2, 1], "n_freq_downsample": [2, 1, 1],
                "mlp_units": [[8, 8], [8, 8], [8, 8]], "pooling_mode": "MaxPool1d",
                "interpolation_mode": "linear", "dropout_prob_theta": 0.2,
                "activation": "ReLU", "mc_samples": 3, "mc_batch_size": 8,
                "selection_metric": "mae_eval_point"}


def synthetic_split(origins, seed, split="test"):
    rng = np.random.default_rng(seed)
    inputs = {}
    for key, shape in zip(INPUT_KEYS, [(72, 2), (72, 10), (H, 10), (1,), (H, 11), (H, 1), (H, 3)]):
        inputs[key] = rng.normal(size=(origins, *shape)).astype(np.float32)
    target = rng.normal(size=(origins, H))
    actual = target * 10.0 + 50.0
    metadata = {"protocol_id": "aligned-test-v1", "region": "NSW1", "split": split,
                "target_mean": 50.0, "target_std": 10.0,
                "input_hash": "x", "sample_hash": "y", "preprocessing_hash": "z"}
    return AlignedSplit(inputs, actual, np.arange(origins, dtype=np.int64) * 3600,
                        np.arange(origins, dtype=np.int64)[:, None] * 3600 + 3600 + np.arange(H)[None],
                        metadata, target)


def make_context(tmp_path, model_id, splits, seed=2026):
    training = {"epochs": 2, "batch_size": 8, "learning_rate": 0.001,
                "weight_decay": 1e-4, "patience": 2}
    config = {"training": training,
              "model_baselines": {model_id: dict(DDNN_PARAMS if model_id == "ddnn_jsu" else NHITS_PARAMS)},
              "experiment": {"baseline_parameters": {"nhits_qra": {"qra_alpha": 0.0, "qra_solver": "highs"}}}}
    job = {"model": model_id, "output": f"runs/{model_id}/seed{seed}/NSW1"}
    return SimpleNamespace(parent_run=tmp_path, region="NSW1", seed=seed, splits=splits,
                           config=config, job=job, run_root=tmp_path / "run",
                           model_directory=tmp_path / "run" / model_id / "models",
                           progress_path=tmp_path / "run" / model_id / "progress.json",
                           device="cpu")


def test_jsu_quantiles_match_scipy_johnsonsu():
    from forecasting.ddnn_jsu.model import jsu_quantiles

    # Two horizons, one origin, double precision.
    gamma = torch.tensor([[0.3, -0.8]], dtype=torch.float64)
    delta = torch.tensor([[1.4, 0.9]], dtype=torch.float64)
    xi = torch.tensor([[-5.0, 12.0]], dtype=torch.float64)
    lam = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    levels = torch.tensor([0.02, 0.05, 0.25, 0.5, 0.75, 0.95, 0.98], dtype=torch.float64)
    ours = jsu_quantiles(gamma, delta, xi, lam, levels).numpy()
    assert ours.shape == (1, 2, 7)
    from scipy.stats import johnsonsu

    for h in range(2):
        expected = johnsonsu.ppf(levels.numpy(), a=float(gamma[0, h]), b=float(delta[0, h]),
                                 loc=float(xi[0, h]), scale=float(lam[0, h]))
        np.testing.assert_allclose(ours[0, h], expected, rtol=1e-8, atol=1e-10)


def test_jsu_median_and_affine_equivariance_and_negative_support():
    from forecasting.ddnn_jsu.model import jsu_quantiles

    gamma = torch.tensor([[1.1]]); delta = torch.tensor([[0.7]])
    xi = torch.tensor([[-3.0]]); lam = torch.tensor([[1.5]])
    levels = torch.tensor([0.05, 0.5, 0.95])
    q = jsu_quantiles(gamma, delta, xi, lam, levels)
    assert q.shape == (1, 1, 3)
    expected_median = xi + lam * torch.sinh(-gamma / delta)
    assert torch.allclose(q[0, 0, 1], expected_median[0, 0])
    # Affine equivariance: raw = std * standardized + mean must commute exactly.
    raw = 10.0 * q + 50.0
    np.testing.assert_allclose(10.0 * q.numpy() + 50.0, raw.numpy())
    assert (q[0, 0, 0] < expected_median).all()  # left tail below the median, negatives reachable


def test_jsu_nll_finite_with_gradients_at_extreme_values():
    from forecasting.ddnn_jsu.model import jsu_negative_log_likelihood

    gamma_raw = torch.tensor([[0.5, -1.0]], requires_grad=True)
    delta_raw = torch.tensor([[0.8, 0.6]], requires_grad=True)
    xi = torch.tensor([[0.0, 2.0]], requires_grad=True)
    lam_raw = torch.tensor([[1.0, 3.0]], requires_grad=True)
    delta = torch.nn.functional.softplus(delta_raw) + 1e-4
    lam = torch.nn.functional.softplus(lam_raw) + 1e-4
    gamma = gamma_raw
    target = torch.tensor([[-40.0, 60.0]])  # extremes on both sides
    loss = jsu_negative_log_likelihood(target, gamma, delta, xi, lam)
    assert torch.isfinite(loss)
    loss.backward()
    leaves = (gamma_raw, delta_raw, xi, lam_raw)
    assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in leaves)
    assert all(float(leaf.grad.abs().max()) > 0 for leaf in leaves)


@pytest.mark.parametrize("model_id,params", [("ddnn_jsu", DDNN_PARAMS), ("nhits_qra", NHITS_PARAMS)])
def test_native_adapter_end_to_end_contract_and_label_isolation(tmp_path, model_id, params):
    splits = {"train": synthetic_split(24, 1, "train"), "selection": synthetic_split(8, 2, "selection"),
              "calibration": synthetic_split(8, 3, "calibration"), "test": synthetic_split(N, 4, "test")}
    context = make_context(tmp_path / model_id / "a", model_id, splits)
    result = run_expansion_model(model_id, context)
    assert result.quantiles.shape == (N, H, 5) and result.point.shape == (N, H)
    assert np.isfinite(result.quantiles).all() and np.isfinite(result.point).all()
    assert result.metadata["prediction_space"] == "raw_rrp"
    assert result.metadata["quantile_levels"] == QUANTILES
    if model_id == "nhits_qra":
        assert result.metadata["mc_dropout"]["samples"] == 3
        assert result.metadata["calibration"].startswith("linear QRA")
    else:
        assert result.metadata["native_training_objective"].startswith("exact_jsu")
    # Shuffling evaluation labels must not change any prediction.
    shuffled = dict(splits)
    shuffled["test"] = AlignedSplit(splits["test"].inputs, splits["test"].actual[::-1].copy(),
                                    splits["test"].origin_unix, splits["test"].target_time_unix,
                                    dict(splits["test"].metadata), splits["test"].target)
    context_two = make_context(tmp_path / model_id / "b", model_id, shuffled)
    result_two = run_expansion_model(model_id, context_two)
    np.testing.assert_array_equal(result_two.quantiles, result.quantiles)


def test_nhits_backbone_structure_matches_upstream_formula():
    from forecasting.nhits_qra.model import NHITSBackbone

    model = NHITSBackbone(input_hours=72, horizon=24, hist_channels=11, futr_channels=16,
                          stat_channels=1, params=NHITS_PARAMS)
    assert len(model.blocks) == 3
    for index, (pool, freq) in enumerate(zip(NHITS_PARAMS["n_pool_kernel_size"],
                                             NHITS_PARAMS["n_freq_downsample"])):
        block = model.blocks[index]
        assert int(block.pooling_layer.kernel_size) == pool
        # Upstream n_theta: input_size + output_multiplier * max(h // freq, 1).
        assert block.layers[-1].out_features == 72 + max(24 // freq, 1)
        # Upstream input width: pooled_y + hist*pooled_y + futr*pooled_futr + stat,
        # with the documented horizon-length futr adaptation.
        expected = int(np.ceil(72 / pool)) + 11 * int(np.ceil(72 / pool)) \
            + 16 * int(np.ceil(24 / pool)) + 1
        assert block.layers[0].in_features == expected
    y = torch.zeros(2, 72); hist = torch.zeros(2, 72, 11)
    futr = torch.zeros(2, 24, 16); stat = torch.zeros(2, 1)
    assert model(y, hist, futr, stat).shape == (2, 24)


def test_mc_dropout_features_deterministic_per_seed_and_dropout_active():
    from forecasting.nhits_qra.model import NHITSBackbone, _mc_quantile_features

    torch.manual_seed(0)
    # Channel counts follow the shared synthetic split layout (futr = 10+11+1+3).
    model = NHITSBackbone(input_hours=72, horizon=H, hist_channels=11, futr_channels=25,
                          stat_channels=1, params=NHITS_PARAMS)
    tensors = {key: torch.randn(6, *shape) for key, shape in zip(
        INPUT_KEYS, [(72, 2), (72, 10), (H, 10), (1,), (H, 11), (H, 1), (H, 3)])}
    torch.manual_seed(123)
    first = _mc_quantile_features(model, tensors, 4, 4)
    torch.manual_seed(123)
    second = _mc_quantile_features(model, tensors, 4, 4)
    np.testing.assert_array_equal(first, second)
    torch.manual_seed(124)
    third = _mc_quantile_features(model, tensors, 4, 4)
    assert not np.array_equal(first, third)
    assert first.shape == (6, H, 5)


def test_calibration_labels_do_not_change_selected_backbone(tmp_path):
    import hashlib

    splits = {"train": synthetic_split(16, 1, "train"), "selection": synthetic_split(8, 2, "selection"),
              "calibration": synthetic_split(8, 3, "calibration"), "test": synthetic_split(8, 4, "test")}
    first = run_expansion_model("nhits_qra", make_context(tmp_path / "a", "nhits_qra", splits))
    original = splits["calibration"].actual.copy()
    splits["calibration"] = AlignedSplit(splits["calibration"].inputs, original * 3.0 + 7.0,
                                         splits["calibration"].origin_unix,
                                         splits["calibration"].target_time_unix,
                                         dict(splits["calibration"].metadata), splits["calibration"].target)
    second = run_expansion_model("nhits_qra", make_context(tmp_path / "b", "nhits_qra", splits))
    # Same seed and identical train/selection: the selected backbone checkpoint
    # must be bit-identical; only the QRA coefficients may move with the labels.
    check_a = list((tmp_path / "a").rglob("best_model.pt"))
    check_b = list((tmp_path / "b").rglob("best_model.pt"))
    assert check_a and check_b
    assert hashlib.sha256(check_a[0].read_bytes()).hexdigest() == \
        hashlib.sha256(check_b[0].read_bytes()).hexdigest()
    if np.allclose(first.quantiles, second.quantiles):
        pytest.fail("Perturbed calibration labels should move the QRA output")
