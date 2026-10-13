"""Canonical input, quantile and official TensorFlow parity guarantees."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from forecasting.pricefm.model import (
    FUTURE_KEYS, HISTORY_KEYS, INPUT_KEYS, UPSTREAM_SOURCE_SHA256, build_pricefm,
)

VENDOR = Path(__file__).resolve().parents[1] / "forecasting/pricefm/vendor/model.py"


def _batch():
    torch.manual_seed(17)
    return {
        "history_values": torch.randn(3, 72, 2),
        "history_calendar": torch.randn(3, 72, 10),
        "future_calendar": torch.randn(3, 24, 10),
        "future_exogenous": torch.randn(3, 24, 11),
        "ctf_exogenous": torch.randn(3, 24, 2),
        "quantile_exogenous": torch.randn(3, 24, 3),
        "origin_context": torch.randn(3, 1),
    }


def test_source_snapshot_is_exact_and_no_upstream_license_is_invented():
    assert hashlib.sha256(VENDOR.read_bytes()).hexdigest() == UPSTREAM_SOURCE_SHA256
    assert "No license file" in VENDOR.with_name("LICENSE_STATUS.md").read_text()


def test_all_inputs_and_parameters_receive_gradients_and_labels_are_ignored():
    torch.set_num_threads(2)
    batch = {key: value.requires_grad_() for key, value in _batch().items()}
    model = build_pricefm(batch)
    forecast = model(batch)
    assert forecast.shape == (3, 24, 5)
    assert torch.isfinite(forecast).all()
    assert torch.all(forecast.diff(dim=-1) >= 0)
    # Pinball, the actual task loss, must train every input branch and head.
    target = torch.randn(3, 24, 1)
    errors = target - forecast
    q = torch.tensor(model.quantiles)
    torch.maximum(q * errors, (q - 1) * errors).mean().backward()
    for key in INPUT_KEYS:
        assert batch[key].grad is not None, key
        assert torch.isfinite(batch[key].grad).all(), key
        assert batch[key].grad.abs().sum() > 0, key
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    poisoned = dict(batch, target_price=torch.full((3, 24, 1), float("nan")),
                    target_quantile=torch.full((3, 24, 1), float("nan")))
    torch.testing.assert_close(model(poisoned), forecast)
    clone = build_pricefm(batch)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(batch), forecast)


def test_each_information_block_changes_predictions():
    batch = _batch()
    model = build_pricefm(batch).eval()
    original = model(batch)
    for key in INPUT_KEYS:
        altered = dict(batch)
        altered[key] = batch[key] + 0.75
        assert not torch.allclose(original, model(altered)), key


def test_negative_prices_and_empty_optional_feature_blocks():
    batch = _batch()
    for key in ("future_exogenous", "ctf_exogenous", "quantile_exogenous"):
        batch[key] = torch.empty(3, 24, 0)
    batch["origin_context"] = torch.empty(3, 0)
    model = build_pricefm(batch)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.quantile_heads[model.anchor_index].bias.fill_(-10)
    # Absolute value constrains increments, never the anchor price itself.
    torch.testing.assert_close(model(batch), torch.full((3, 24, 5), -10.0))


def test_missing_or_native_adapted_blocks_are_rejected():
    batch = _batch()
    model = build_pricefm(batch)
    batch.pop("quantile_exogenous")
    with pytest.raises(ValueError, match="canonical input"):
        build_pricefm(batch)
    with pytest.raises(ValueError, match="quantile_exogenous"):
        model(batch)
    batch = _batch()
    batch["ctf_exogenous"] = torch.cat([batch["ctf_exogenous"], batch["quantile_exogenous"]], -1)
    with pytest.raises(ValueError, match="ctf_exogenous"):
        model(batch)


def test_shared_quantile_levels_cannot_be_silently_overridden():
    with pytest.raises(ValueError, match="shared five levels"):
        build_pricefm(_batch(), quantiles=(0.01, 0.1, 0.5, 0.9, 0.99))


def test_graph_forward_matches_independent_numpy_equations():
    model = build_pricefm(_batch(), emb_dim=8, num_experts=3)
    rng = np.random.default_rng(53)
    lag = rng.normal(size=(3, 2, 72, model.history_dim)).astype("float32")
    lead = rng.normal(size=(3, 2, 24, model.future_dim)).astype("float32")
    gate = np.array([[1., 0.], [1., .3], [.5, 1.]], dtype="float32")
    def dense(value, layer):
        return value @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
    encoded_input = dense(lag.reshape(3, 2, -1), model.b1_project)
    encoded_input += dense(lead.reshape(3, 2, -1), model.b2_project)
    logits = dense(encoded_input, model.enc_gate)
    mixture = np.exp(logits - logits.max(-1, keepdims=True))
    mixture /= mixture.sum(-1, keepdims=True)
    encoded = np.zeros_like(encoded_input)
    for i, expert in enumerate(model.enc_experts):
        projected = dense(encoded_input, expert)
        encoded += projected / (1 + np.exp(-projected)) * mixture[..., i:i + 1]
    pooled = (encoded * gate[..., None]).sum(1) / (gate.sum(1, keepdims=True) + 1e-6)
    raw_heads = [dense(pooled, layer) for layer in model.quantile_heads]
    expected = np.stack([
        raw_heads[2] - np.abs(raw_heads[1]) - np.abs(raw_heads[0]),
        raw_heads[2] - np.abs(raw_heads[1]), raw_heads[2],
        raw_heads[2] + np.abs(raw_heads[3]),
        raw_heads[2] + np.abs(raw_heads[3]) + np.abs(raw_heads[4]),
    ], -1)
    actual = model.forward_graph(torch.tensor(lag), torch.tensor(lead), torch.tensor(gate))
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-5, rtol=2e-5)


def test_official_tensorflow_forward_and_backward_parity():
    tf = pytest.importorskip("tensorflow", reason="Optional isolated upstream parity environment")
    torch.set_num_threads(2)
    tf.keras.utils.set_random_seed(13)
    spec = importlib.util.spec_from_file_location("pricefm_official_validation", VENDOR)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    batch = _batch()
    model = build_pricefm(batch, emb_dim=8, num_experts=3)
    keras_model = official.build_graph_gated_quantile_model(
        (72, model.history_dim), (24, model.future_dim), 24,
        model.quantiles, emb_dim=8, num_experts=3,
    )
    model.load_official_keras_weights(keras_model)
    # Multiple nodes and nonuniform masks ensure pooling is also verified.
    rng = np.random.default_rng(41)
    lag = rng.normal(size=(3, 2, 72, model.history_dim)).astype("float32")
    lead = rng.normal(size=(3, 2, 24, model.future_dim)).astype("float32")
    gates = np.array([[1, 0], [1, .25], [.5, 1]], dtype="float32")
    lag_tf, lead_tf = tf.Variable(lag), tf.Variable(lead)
    with tf.GradientTape() as tape:
        expected = keras_model({"X_lag_all": lag_tf, "X_lead_all": lead_tf,
                                "graph_gate": tf.constant(gates)})
        objective = tf.reduce_mean(tf.square(expected))
    tf_grads = tape.gradient(objective, [lag_tf, lead_tf] + keras_model.trainable_variables)
    lag_pt, lead_pt = torch.tensor(lag, requires_grad=True), torch.tensor(lead, requires_grad=True)
    actual = model.forward_graph(lag_pt, lead_pt, torch.tensor(gates))
    actual.square().mean().backward()
    np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(lag_pt.grad.numpy(), tf_grads[0].numpy(), atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(lead_pt.grad.numpy(), tf_grads[1].numpy(), atol=2e-6, rtol=2e-4)
    grad_by_id = {id(var): grad for var, grad in zip(keras_model.trainable_variables, tf_grads[2:])}
    layers = [(model.b1_project, "b1_project"), (model.b2_project, "b2_project"),
              (model.enc_gate, "enc_gate")]
    layers += [(layer, f"enc_expert{i}") for i, layer in enumerate(model.enc_experts)]
    for i, (q, layer) in enumerate(zip(model.quantiles, model.quantile_heads)):
        suffix = "label" if i == model.anchor_index else "pre_project"
        layers.append((layer, f"y_q{round(q * 100):02}_{suffix}"))
    for layer, name in layers:
        kernel, bias = keras_model.get_layer(name).trainable_variables
        np.testing.assert_allclose(layer.weight.grad.numpy(), grad_by_id[id(kernel)].numpy().T,
                                   atol=2e-5, rtol=2e-4, err_msg=name)
        np.testing.assert_allclose(layer.bias.grad.numpy(), grad_by_id[id(bias)].numpy(),
                                   atol=2e-5, rtol=2e-4, err_msg=name)
    # Also compare the exact single-node canonical adaptation with official TF.
    history = torch.cat([batch[k] for k in HISTORY_KEYS], -1).numpy()[:, None]
    context = batch["origin_context"][:, None].expand(-1, 24, -1)
    future = torch.cat([batch[k] for k in FUTURE_KEYS] + [context], -1).numpy()[:, None]
    expected = keras_model({"X_lag_all": history, "X_lead_all": future,
                            "graph_gate": np.ones((3, 1), dtype="float32")})
    np.testing.assert_allclose(model(batch).detach().numpy(), expected.numpy(), atol=2e-5, rtol=2e-5)
