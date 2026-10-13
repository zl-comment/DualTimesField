"""PriceFM's pinned 2026 MoE architecture under the shared AEMO contract.

This is a PyTorch port, trained from scratch, with one graph node per region.
It does not reproduce the original multi-country foundation-model pretraining.
See ``vendor/pricefm/PROVENANCE.md`` for the source and explicit adaptations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_REPOSITORY = "https://github.com/runyao-yu/PriceFM"
UPSTREAM_COMMIT = "c72d1228bde80417d5cc782521328e02ab5401c3"
UPSTREAM_SOURCE_SHA256 = "c5ae12e0a9ae85535cad30e41f84c3b58fe2265c464249f2705802ece74f03c2"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
HISTORY_KEYS = ("history_values", "history_calendar")
FUTURE_KEYS = ("future_calendar", "future_exogenous", "ctf_exogenous", "quantile_exogenous")
INPUT_KEYS = HISTORY_KEYS + FUTURE_KEYS + ("origin_context",)


class PriceFMQuantileForecaster(nn.Module):
    """Two projections, dense MoE, graph pooling and hierarchical quantiles.

    The canonical batch has seven separate blocks. In particular, callers must
    not first append quantile_exogenous into ctf_exogenous via native_batch().
    No fitted scaling, clipping, pretrained weights or target access occurs here.
    """

    def __init__(self, input_shapes: Mapping[str, Sequence[int]], *,
                 emb_dim: int = 32, num_experts: int = 4,
                 quantiles: Sequence[float] = QUANTILES):
        super().__init__()
        self.input_shapes = {key: tuple(input_shapes[key]) for key in INPUT_KEYS}
        self.quantiles = tuple(float(q) for q in quantiles)
        if (not self.quantiles or any(q <= 0 or q >= 1 for q in self.quantiles)
                or any(a >= b for a, b in zip(self.quantiles, self.quantiles[1:]))
                or 0.5 not in self.quantiles):
            raise ValueError("PriceFM requires increasing quantiles in (0, 1), including q50")
        if emb_dim <= 0 or num_experts <= 0:
            raise ValueError("PriceFM embedding size and number of experts must be positive")
        for key in HISTORY_KEYS + FUTURE_KEYS:
            shape = self.input_shapes[key]
            if len(shape) != 2 or shape[0] <= 0 or shape[1] < 0:
                raise ValueError(f"PriceFM expects {key} shape [time, features]")
        self.input_hours = self.input_shapes["history_values"][0]
        self.output_hours = self.input_shapes["future_calendar"][0]
        if any(self.input_shapes[key][0] != self.input_hours for key in HISTORY_KEYS):
            raise ValueError("PriceFM historical time dimensions must agree")
        if any(self.input_shapes[key][0] != self.output_hours for key in FUTURE_KEYS):
            raise ValueError("PriceFM future time dimensions must agree")
        if len(self.input_shapes["origin_context"]) != 1:
            raise ValueError("PriceFM origin_context must have shape [batch, features]")
        self.history_dim = sum(self.input_shapes[key][1] for key in HISTORY_KEYS)
        self.future_dim = (sum(self.input_shapes[key][1] for key in FUTURE_KEYS)
                           + self.input_shapes["origin_context"][0])
        self.anchor_index = self.quantiles.index(0.5)
        self.num_experts = int(num_experts)
        self.emb_dim = int(emb_dim)
        self.pool_eps = 1e-6  # Preserve upstream denominator even with one node.
        self.b1_project = nn.Linear(self.input_hours * self.history_dim, emb_dim)
        self.b2_project = nn.Linear(self.output_hours * self.future_dim, emb_dim)
        self.enc_gate = nn.Linear(emb_dim, num_experts)
        self.enc_experts = nn.ModuleList(nn.Linear(emb_dim, emb_dim) for _ in range(num_experts))
        self.quantile_heads = nn.ModuleList(nn.Linear(emb_dim, self.output_hours)
                                            for _ in self.quantiles)
        # Keras Dense defaults, rather than PyTorch's distinct initialization.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.provenance = {
            "repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT,
            "source_sha256": UPSTREAM_SOURCE_SHA256,
            "license": "upstream does not declare a license",
            "architecture_version": "2026 MoE",
            "implementation": "PyTorch port of pinned official TensorFlow model",
            "adaptation": "single-region single-node graph; canonical AEMO input dimensions",
            "graph_nodes": 1, "graph_gate": [1.0],
            "pretrained": False, "training": "from scratch on shared training split",
            "input_blocks": list(INPUT_KEYS),
            "history_order": list(HISTORY_KEYS),
            "future_order": list(FUTURE_KEYS) + ["origin_context broadcast across horizon"],
            "emb_dim": emb_dim, "num_experts": num_experts,
            "quantiles": list(self.quantiles), "pool_epsilon": self.pool_eps,
        }

    def forward_graph(self, history: torch.Tensor, future: torch.Tensor,
                      graph_gate: torch.Tensor) -> torch.Tensor:
        """Official graph-shaped forward, useful for upstream equivalence checks.

        history/future: [batch, nodes, time, features], gate: [batch, nodes].
        Public aligned forward below always supplies exactly one node and gate=1.
        """
        if (history.ndim != 4 or future.ndim != 4
                or history.shape[:2] != future.shape[:2]
                or tuple(history.shape[2:]) != (self.input_hours, self.history_dim)
                or tuple(future.shape[2:]) != (self.output_hours, self.future_dim)
                or graph_gate.shape != history.shape[:2]):
            raise ValueError("PriceFM graph inputs do not match configured dimensions")
        x = self.b1_project(history.flatten(2)) + self.b2_project(future.flatten(2))
        weights = self.enc_gate(x).softmax(dim=-1)
        # Preserve official expert ordering and addition semantics.
        encoded = torch.stack([F.silu(expert(x)) * weights[..., i:i + 1]
                               for i, expert in enumerate(self.enc_experts)], dim=0).sum(dim=0)
        pooled = ((encoded * graph_gate.unsqueeze(-1)).sum(dim=1)
                  / (graph_gate.sum(dim=1, keepdim=True) + self.pool_eps))
        anchor = self.quantile_heads[self.anchor_index](pooled)
        outputs = {self.anchor_index: anchor}
        previous = anchor
        for i in range(self.anchor_index + 1, len(self.quantiles)):
            previous = previous + self.quantile_heads[i](pooled).abs()
            outputs[i] = previous
        previous = anchor
        for i in range(self.anchor_index - 1, -1, -1):
            previous = previous - self.quantile_heads[i](pooled).abs()
            outputs[i] = previous
        return torch.stack([outputs[i] for i in range(len(self.quantiles))], dim=-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        size = batch["history_values"].shape[0]
        for key, shape in self.input_shapes.items():
            if key not in batch or tuple(batch[key].shape) != (size, *shape):
                raise ValueError(f"PriceFM canonical input {key} must have shape {(size, *shape)}")
        history = torch.cat([batch[key] for key in HISTORY_KEYS], dim=-1)
        context = batch["origin_context"][:, None, :].expand(-1, self.output_hours, -1)
        future = torch.cat([batch[key] for key in FUTURE_KEYS] + [context], dim=-1)
        return self.forward_graph(history.unsqueeze(1), future.unsqueeze(1),
                                  history.new_ones((size, 1)))

    def load_official_keras_weights(self, keras_model) -> None:
        """Copy Dense kernels (transposed) and biases for numerical validation.

        No TensorFlow import is required by the training implementation. This
        method is not used by the experiment runner or for pretrained weights.
        """
        layers = [(self.b1_project, "b1_project"), (self.b2_project, "b2_project"),
                  (self.enc_gate, "enc_gate")]
        layers += [(layer, f"enc_expert{i}") for i, layer in enumerate(self.enc_experts)]
        for i, (q, layer) in enumerate(zip(self.quantiles, self.quantile_heads)):
            suffix = "label" if i == self.anchor_index else "pre_project"
            layers.append((layer, f"y_q{round(q * 100):02}_{suffix}"))
        with torch.no_grad():
            for target, name in layers:
                kernel, bias = keras_model.get_layer(name).get_weights()
                target.weight.copy_(torch.as_tensor(kernel.T.copy(), device=target.weight.device))
                target.bias.copy_(torch.as_tensor(bias.copy(), device=target.bias.device))


def build_pricefm(example: Mapping[str, torch.Tensor], *, emb_dim: int = 32,
                  num_experts: int = 4,
                  quantiles: Sequence[float] = QUANTILES) -> PriceFMQuantileForecaster:
    """Construct from canonical batched inputs; target fields are never read."""
    if tuple(float(q) for q in quantiles) != QUANTILES:
        raise ValueError("Aligned PriceFM quantiles must match the shared five levels")
    missing = set(INPUT_KEYS) - example.keys()
    if missing:
        raise ValueError(f"PriceFM requires all canonical input blocks: {sorted(missing)}")
    return PriceFMQuantileForecaster(
        {key: tuple(example[key].shape[1:]) for key in INPUT_KEYS},
        emb_dim=emb_dim, num_experts=num_experts, quantiles=quantiles,
    )
