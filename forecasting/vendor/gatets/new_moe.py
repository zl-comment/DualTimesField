import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFBRouter(nn.Module):
    """
    Wrap any gating network that returns *unnormalised* logits.
    Adds a non-trainable per-expert bias that is updated on-the-fly.
    """

    def __init__(self, base_gate: nn.Module, num_experts: int, eta: float = 0.1):
        super().__init__()
        self.base_gate = base_gate  # e.g. GatingNetwork
        self.register_buffer("bias", torch.zeros(num_experts))  # no grad!
        self.eta = eta

    @torch.no_grad()
    def update_bias(self, expert_counts: torch.Tensor):
        mean = expert_counts.float().mean()
        if mean == 0:
            return
        violation = (expert_counts.float() - mean) / mean
        self.bias.sub_(self.eta * violation)  # b ← b − η·ε

    def forward(self, x: torch.Tensor, top_k: int):
        logits = self.base_gate(x)  # (B, S, E) *raw* scores
        biased_logits = logits + self.bias  # broadcast
        probs = torch.softmax(biased_logits, dim=-1)
        topk_vals, topk_idx = probs.topk(top_k, dim=-1)
        # Hard-routing mask exactly like your current code
        mask = torch.zeros_like(probs).scatter_(-1, topk_idx, 1.0)
        probs = (probs * mask) / (mask.sum(dim=-1, keepdim=True) + 1e-9)
        return probs, topk_idx  # return idx so we can count tokens


class ExpertTransformerDecoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1
    ):
        """
        A mini transformer decoder block used as an expert.
        d_model: input and output dimension.
        n_heads: number of attention heads for self-attention.
        dim_feedforward: hidden dimension of the internal feed-forward network.
        dropout: dropout probability.
        """
        super(ExpertTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Use GELU instead of ReLU for smoother activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        # Feed-forward sub-layer
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        """
        Simple gating network that uses a linear projection and softmax to create a
        probability distribution over experts.
        """
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs


# class KroneckerGatingNetwork(nn.Module):
#     """
#     Attention-inspired gating layer that uses a Kronecker (outer) product
#     and initialises all learnable tensors from 𝒩(0, 0.02²).
#     """

#     def __init__(self, d_model: int, num_experts: int, init_std: float = 0.02):
#         super().__init__()
#         self.d_model = d_model
#         self.num_experts = num_experts
#         self.init_std = init_std

#         # Learnable expert queries  (E, D)
#         self.expert_queries = nn.Parameter(torch.empty(num_experts, d_model))

#         # Optional projection for keys  (D → D)
#         self.key_proj = nn.Linear(d_model, d_model)

#         # One projection vector per expert for the D²-dim Kronecker features
#         self.out_proj = nn.Parameter(
#             torch.empty(num_experts, d_model * d_model))

#         self.scale = d_model ** -0.5      # √d term

#         self.reset_parameters()

#     def reset_parameters(self):
#         """Initialise all parameters with 𝒩(0, init_std²)."""
#         nn.init.normal_(self.expert_queries, mean=0.0, std=self.init_std)
#         nn.init.normal_(self.out_proj,      mean=0.0, std=self.init_std)
#         nn.init.normal_(self.key_proj.weight, mean=0.0, std=self.init_std)
#         nn.init.zeros_(self.key_proj.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x : (batch, seq_len, d_model) token embeddings
#         returns (batch, seq_len, num_experts) gating probabilities
#         """
#         B, S, D = x.shape
#         keys = self.key_proj(x)                          # (B, S, D)

#         # Kronecker product k ⊗ q  → (B, S, E, D²)
#         kron = torch.einsum('bsd,ed->bsed', keys, self.expert_queries)
#         kron = kron.reshape(B, S, self.num_experts, D * D)

#         # Project Kronecker features to scalar scores
#         scores = (kron * self.out_proj.unsqueeze(0).unsqueeze(0)).sum(-1)
#         scores = scores/self.scale                     # (B, S, E)

#         return F.softmax(scores, dim=-1)


class KroneckerGatingNetwork(nn.Module):
    """
    Factorises W ∈ ℝ^{E×D} as (A ⊗ B) with
        A ∈ ℝ^{e_a × d_a},  B ∈ ℝ^{e_b × d_b}.
    Returns UNnormalised logits (B,S,E) → wrap with LFBRouter.
    """

    def __init__(
        self, d_model: int, num_experts: int, d_a: int = 8, e_a: int | None = None
    ):
        super().__init__()

        # --- embed dim factors --------------------------------------------------
        assert d_model % d_a == 0, "d_a must divide d_model"
        d_b = d_model // d_a
        self.d_a, self.d_b = d_a, d_b

        # --- expert factors -----------------------------------------------------
        if e_a is None:
            e_a = int(num_experts**0.5)
            while num_experts % e_a:
                e_a -= 1
        e_b = num_experts // e_a
        self.e_a, self.e_b = e_a, e_b

        # --- learnable parameters ----------------------------------------------
        self.A = nn.Parameter(torch.randn(e_a, d_a) * d_a**-0.5)
        self.B = nn.Parameter(torch.randn(e_b, d_b) * d_b**-0.5)

        self.proj = (
            nn.Identity()
            if d_a * d_b == d_model
            else nn.Linear(d_model, d_a * d_b, bias=False)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise all parameters with 𝒩(0, init_std²)."""
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        # nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.key_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, S, d_model)  →  logits : (B, S, num_experts)
        """
        B, S, _ = x.shape
        x = self.proj(x).view(B, S, self.d_a, self.d_b)  # (B,S,a,d)

        # First factor  (A):   ea, bsad -> bsed
        tmp = torch.einsum("ea,bsad->bsed", self.A, x)  # (B,S,e_a,d_b)

        # Second factor (B):   fd, bsed -> bsef
        logits = torch.einsum("fd,bsed->bsef", self.B, tmp)  # (B,S,e_a,e_b)

        return logits.reshape(B, S, -1)  # (B,S,E)


class AttentionMoEGating(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_experts: int):
        """
        Gating mechanism for a neural network mixture-of-experts using an attention-like approach.
        It is inspired by the DummyInputHMMTransitions algorithm but computes the gating
        weights via dot-product attention.

        Args:
            input_dim: Dimension of the raw input features.
            model_dim: Dimension to which inputs are projected for computing attention.
            num_experts: Number of experts (discrete states).
        """
        super(AttentionMoEGating, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.model_dim = model_dim

        # Linear projection: projects raw input (input_dim) to model_dim.
        self.W_input = nn.Linear(input_dim, model_dim, bias=False)

        # Learned expert key embeddings (one per expert); shape: (num_experts, model_dim)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, model_dim))

        # Learned fixed "markov" contribution (simulating the part from the previous state).
        # This is of shape (num_experts,).
        self.markov_contrib = nn.Parameter(torch.randn(num_experts))

        # Additional bias term (per expert); shape: (num_experts,).

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given an array of covariates X of shape (T, input_dim), compute gating (transition)
        probabilities for each time step using a dot-product attention mechanism.

        Returns a tensor of shape (T, num_experts), where each row is a probability distribution
        (i.e. the entries sum to 1) over experts.
        """
        # Project the input covariates: shape goes from (T, input_dim) to (T, model_dim)
        X_proj = self.W_input(X)

        # Compute the dot-product scores between inputs and expert keys:
        # expert_keys.t() has shape (model_dim, num_experts), so the result is (T, num_experts).
        # Scale the scores by sqrt(model_dim) (as in standard attention mechanisms).
        scale = self.model_dim**0.5
        attention_scores = torch.matmul(X_proj, self.expert_keys.t()) / scale

        # Combine the attention scores with the fixed "markov" contribution and bias.
        # Both self.markov_contrib and self.bias have shape (num_experts,), so we unsqueeze them to add along the time dimension.
        psi = (
            attention_scores
            + self.markov_contrib.unsqueeze(0)  # + self.bias.unsqueeze(0)
        )

        # Apply softmax over the expert dimension (last dimension) to produce a probability distribution.
        gating_weights = F.softmax(psi, dim=-1)
        return gating_weights


class MoETransformerDecoderFFN(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        expert_n_heads: int,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        model_dim: input/output dimension.
        ffn_dim: hidden dimension for each expert.
        num_experts: total experts.
        top_k: number of experts selected per token.
        expert_n_heads: number of heads for the transformer-decoder experts.
        dropout: dropout probability.
        """
        super(MoETransformerDecoderFFN, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Initialize transformer-decoder experts
        self.experts = nn.ModuleList(
            [
                ExpertTransformerDecoder(model_dim, expert_n_heads, ffn_dim, dropout)
                for _ in range(num_experts)
            ]
        )
        print(gating)
        if gating == "classic":
            self.gating = GatingNetwork(model_dim, num_experts)
        elif gating == "paper":
            self.gating = AttentionMoEGating(model_dim, ffn_dim, num_experts)
        elif gating == "kronecker":
            self.gating = KroneckerGatingNetwork(model_dim, num_experts)
        elif gating == "lfb_classic":
            raw_gate = GatingNetwork(model_dim, num_experts)
            self.gating = LFBRouter(raw_gate, num_experts)
        elif gating == "lfb_kronecker":
            raw_gate = KroneckerGatingNetwork(model_dim, num_experts)
            self.gating = LFBRouter(raw_gate, num_experts)

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple:
        """
        Args
        ----
        x           : (batch, seq_len, model_dim)
        return_gate : if True, also return routing probabilities and selected indices.

        Returns
        -------
          • Tensor (batch, seq_len, model_dim)                        – when return_gate=False
          • Tuple(moe_out, gating_probs, topk_idx)                    – when return_gate=True
        """
        B, S, _ = x.shape

        # ──────────────────────────── 1. ROUTING ────────────────────────────
        if isinstance(self.gating, LFBRouter):
            # LFBRouter: returns already-masked, renormalised probabilities *and* indices
            gating_probs, topk_idx = self.gating(x, self.top_k)  # (B,S,E)  (B,S,K)
        else:
            # Legacy (aux-loss or plain) gating – expects raw *logits*
            # (B,S,E)
            logits = self.gating(x)
            gating_probs = torch.softmax(logits, dim=-1)  # (B,S,E)

            # Hard Top-K selection
            if self.top_k < self.num_experts:
                topk_vals, topk_idx = gating_probs.topk(
                    self.top_k, dim=-1
                )  # idx: (B,S,K)
                mask = torch.zeros_like(gating_probs).scatter_(-1, topk_idx, 1.0)
                gating_probs = (gating_probs * mask) / (
                    mask.sum(dim=-1, keepdim=True) + 1e-9
                )
            else:  # dense routing
                topk_idx = (
                    torch.arange(self.num_experts, device=x.device)
                    .view(1, 1, -1)
                    .expand(B, S, -1)
                )  # (B,S,E)

        # ──────────────────────────── 2. EXPERT PASS ────────────────────────────
        # Stack expert outputs → (B, S, D, E)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        # Weighted sum using routing probabilities
        moe_out = (expert_outputs * gating_probs.unsqueeze(-2)).sum(dim=-1)  # (B,S,D)

        # ─────────────────────── 3. LFB bias update (no grad) ───────────────────────
        if self.training and isinstance(self.gating, LFBRouter):
            with torch.no_grad():
                # collapse (B,S,K) → (E,) utilisation counts
                counts = torch.bincount(
                    topk_idx.reshape(-1), minlength=self.num_experts
                )
                # b ← b − η·ε
                self.gating.update_bias(counts)

        if return_gate:
            return moe_out, gating_probs, topk_idx
        return moe_out


class MoETransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        expert_n_heads: int,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        A Transformer layer that applies self-attention followed by the MoE module.
        """
        super(MoETransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.moe_decoder = MoETransformerDecoderFFN(
            model_dim=d_model,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_n_heads=expert_n_heads,
            dropout=dropout,
            gating=gating,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_gate=False) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        if return_gate:
            out, gating, topk = self.moe_decoder(x, return_gate=True)
            return self.norm2(x + self.dropout(out)), gating, topk
        else:
            decoder_output = self.moe_decoder(x)
            x = self.norm2(x + self.dropout(decoder_output))
            return x


class MoETransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 24,
        target_size: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        use_decoder_expert: bool = True,
        expert_n_heads: int = None,
        dropout: float = 0.1,
        gating: str = "attention",
    ):
        """
        input_size: input window length (default 24).
        target_size: forecast horizon (default 1 for one-step ahead).
        d_model: model embedding dimension.
        n_heads: self-attention heads.
        num_layers: number of Transformer layers.
        ffn_dim: hidden dimension for expert feed-forward networks.
        num_experts: number of MoE experts.
        top_k: how many experts to activate per token.
        use_decoder_expert: if True, uses transformer-decoder experts.
        expert_n_heads: number of attention heads for each expert (defaults to n_heads if None).
        dropout: dropout probability.
        """
        super(MoETransformerForecaster, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.d_model = d_model

        # Project the univariate input to d_model dimensions.
        self.value_embed = nn.Linear(1, d_model)
        # Learnable positional embeddings.
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_size, d_model))

        self.name = "MOE" + gating

        expert_n_heads = expert_n_heads if expert_n_heads is not None else n_heads
        if use_decoder_expert:
            self.layers = nn.ModuleList(
                [
                    MoETransformerDecoderLayer(
                        d_model,
                        n_heads,
                        ffn_dim,
                        num_experts,
                        top_k,
                        expert_n_heads,
                        dropout,
                        gating,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "Only transformer-decoder experts are implemented in this example."
            )

        # Output layer to generate forecasts.
        self.output_layer = nn.Linear(d_model, target_size)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        # x: shape (batch, input_size) or (batch, input_size, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        if seq_len > self.input_size:
            raise ValueError(
                f"Input length {seq_len} exceeds model input_size {self.input_size}."
            )

        # Embed the input and add positional encoding.
        x_embed = self.value_embed(x)
        x_embed = x_embed + self.pos_embedding[:, :seq_len, :]

        out = x_embed
        all_gates, all_topk = [], []
        for layer in self.layers:
            if return_gate:
                out, gating, topk = layer(out, return_gate=True)
                # list of (batch, seq_len, num_experts)
                all_gates.append(gating)
                all_topk.append(topk)  # list of (batch, seq_len, top_k)
            else:
                out = layer(out)

        final = out[:, -1, :]
        forecast = self.output_layer(final).squeeze(-1)

        if return_gate:
            return forecast, all_gates, all_topk
        return forecast


def count_active_params(model: torch.nn.Module, top_k: int) -> int:
    """
    Estimate the number of parameters actively used during inference when only top_k experts per MoE module are computed.

    The function computes:
      active_params = non-MoE params + sum_over_MoE_modules[(top_k / num_experts_module) * expert_params_module]

    Args:
        model (nn.Module): The forecasting model.
        top_k (int): The number of experts activated per token.

    Returns:
        int: Estimated active parameter count.
    """
    # Total parameters in the model (all are always stored).
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    moe_expert_params_total = 0  # Sum of parameters for all MoE experts.
    active_expert_params = (
        # Sum of parameters from experts that would be computed in a forward pass.
        0
    )

    # Iterate over all modules to find MoE modules.
    for module in model.modules():
        if isinstance(module, MoETransformerDecoderFFN):
            num_experts = len(module.experts)
            # Count parameters for experts in this MoE module.
            expert_params_module = sum(
                sum(p.numel() for p in expert.parameters() if p.requires_grad)
                for expert in module.experts
            )
            moe_expert_params_total += expert_params_module
            # Only top_k experts are used at inference.
            active_expert_params += (top_k / num_experts) * expert_params_module

    # All other parameters (e.g. input projection, positional embeddings, self-attention outside MoE, gating network, etc.)
    non_moe_params = total_params - moe_expert_params_total

    # The total "active" parameters is the non-MoE parameters plus the fraction of MoE expert parameters.
    active_total = non_moe_params + int(active_expert_params)
    return active_total


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import torch.optim as optim
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    from utils import (
        calculate_metrics,
        calculate_metrics_by_batch,
        create_dataset,
        get_dataset,
    )

    device = "cpu"
    time_series = get_dataset("sunspot")
    scaler = MinMaxScaler()
    series = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    window_size = 64
    horizon = 48

    X_train, y_train, X_val, y_val, X_test, y_test = create_dataset(
        window_size=window_size, horizon=horizon, series=series
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Instantiate the model.
    model = MoETransformerForecaster(
        input_size=window_size,
        target_size=horizon,
        d_model=16,
        n_heads=2,
        num_layers=1,
        ffn_dim=16,
        num_experts=16,
        top_k=2,
        use_decoder_expert=True,
        expert_n_heads=2,  # number of heads for the expert blocks
        dropout=0.1,
        gating="kronecker",
    )

    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    best_val_mse = float("inf")
    best_model_state = None
    best_epoch = -1
    pairs_count_list = []
    # Simple training loop.
    for epoch in range(20):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        batch_size = 64
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= X_train.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, gates, chosen = model(X_val, True)
            chosen = np.asarray(chosen).squeeze(0)

            all_pairs = [
                tuple(sorted(pair.tolist()))
                for sample in chosen  # sample: shape (64,2)
                for pair in sample  # pair:   shape (2,)
            ]

            # 3. Count them
            pair_counts = Counter(all_pairs)
            print(len(pair_counts))
            pairs_count_list.append(len(pair_counts))
            # 4. See your dict
            print(pair_counts)
            val_loss = criterion(val_pred, y_val).item()
        print(f"Epoch {epoch + 1}: Train MSE={epoch_loss:.4f}, Val MSE={val_loss:.4f}")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_pred, gates, chosen = model(X_val, True)
        val_loss = criterion(val_pred, y_val).item()

    result_dict = {}
    result_dict["val_pred"] = val_pred
    result_dict["gates"] = gates
    result_dict["chosen"] = chosen
    result_dict["y_val"] = y_val

    epochs = np.arange(1, len(pairs_count_list) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, pairs_count_list, marker="o", label="Unique pairs (val)")
    plt.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"Best Val @ epoch {best_epoch}",
    )
    plt.scatter([best_epoch], [pairs_count_list[best_epoch - 1]], color="red", zorder=5)
    plt.xlabel("Epoch")
    plt.ylabel("Unique top-2 pairs across val")
    plt.title("Unique expert pairs per epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("unique_pairs_per_epoch.png", dpi=300, bbox_inches="tight")
    plt.close()

    import pickle

    # with open("results.pkl", "wb") as f:
    #     pickle.dump(result_dict, f)
    # ------------------------------------------------------------
    # Unique expert-pair stats per window (seq_len positions each)
    # When return_gate=True, `chosen` holds the top-k expert indices per token.
    # We compute, for each window, how many unique top-2 pairs appear across
    # its seq_len positions, then report mean/median and save a box plot.
    # ------------------------------------------------------------
    import numpy as np
    from matplotlib import pyplot as plt

    try:
        chosen_arr = np.asarray(chosen)
        # Handle shape either (N, S, K) or (L, N, S, K) where L=n_layers
        if chosen_arr.ndim == 4:
            # take the last layer's routing indices
            chosen_np = chosen_arr[-1]
        elif chosen_arr.ndim == 3:
            chosen_np = chosen_arr
        else:
            raise ValueError(f"Unexpected 'chosen' shape: {chosen_arr.shape}")

        N, S, K = chosen_np.shape
        if K < 2:
            print("Warning: top_k < 2; unique pair count may be trivial.")

        # Count unique pairs per window
        unique_pairs_per_window = []
        for i in range(N):
            pairs = [tuple(sorted(chosen_np[i, t, :2].tolist())) for t in range(S)]
            unique_pairs_per_window.append(len(set(pairs)))

        unique_pairs_per_window = np.asarray(unique_pairs_per_window)
        mean_pairs = float(unique_pairs_per_window.mean())
        median_pairs = float(np.median(unique_pairs_per_window))
        print(
            f"Unique top-2 expert pairs per window: mean={mean_pairs:.3f}, median={median_pairs:.3f}"
        )

        # Box plot
        plt.figure(figsize=(4, 5))
        plt.boxplot(unique_pairs_per_window, vert=True, patch_artist=True)
        plt.ylabel("Unique top-2 pairs per window")
        plt.title("Distribution of unique expert pairs")
        plt.savefig("unique_pairs_boxplot.png", dpi=600, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Could not compute unique-pair stats: {e}")

    y_val = y_val.detach().numpy()
    val_pred = val_pred.detach().numpy()

    print(calculate_metrics(y_val, val_pred, scaler, horizon))

    plt.plot(val_pred, color="black")
    plt.plot(y_val, color="green")
    plt.show()
