import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable
#from .data import graph_mask

@register_keras_serializable()
class AbsActivation(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super().get_config()
    


def _q_to_norm(q):
    """Convert quantile to [0,1] scale."""
    q = float(q)
    return q / 100.0 if q > 1.0 else q

def _q_to_pct_int(q):
    """Convert quantile to int percent for layer naming."""
    qn = _q_to_norm(q)
    return int(round(qn * 100))

def HierarchicalQuantileHead(
    shared_rep,      # (B, D)
    quantiles,       # e.g. [0.1,0.5,0.9] or [10,50,90]
    output_dim,      # y_dim
    prefix="y"
):
    # --- normalize & sort
    q_norm = [_q_to_norm(q) for q in quantiles]
    q_pct  = [_q_to_pct_int(q) for q in quantiles]  # for naming + dict keys

    # sorted by q value
    sorted_items = sorted(zip(q_norm, q_pct), key=lambda t: t[0])  # [(qn, pct), ...]
    sorted_qn    = [t[0] for t in sorted_items]
    sorted_pct   = [t[1] for t in sorted_items]

    # anchor index: closest to 0.5
    anchor_i = min(range(len(sorted_qn)), key=lambda i: abs(sorted_qn[i] - 0.5))
    anchor_pct = sorted_pct[anchor_i]

    # --- anchor (median-like)
    out_anchor = layers.Dense(output_dim, activation="linear", name=f"{prefix}_q{anchor_pct:02}_label")(shared_rep)

    outputs = {anchor_pct: out_anchor}

    # --- upper quantiles: add positive steps
    prev = out_anchor
    for pct in sorted_pct[anchor_i + 1:]:
        pre  = layers.Dense(output_dim, activation="linear", name=f"{prefix}_q{pct:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{pct:02}_step")(pre)  # >= 0
        o    = layers.Add(name=f"{prefix}_q{pct:02}_label")([prev, step])
        outputs[pct] = o
        prev = o

    # --- lower quantiles: subtract positive steps
    prev = out_anchor
    for pct in reversed(sorted_pct[:anchor_i]):
        pre  = layers.Dense(output_dim, activation="linear", name=f"{prefix}_q{pct:02}_pre_project")(shared_rep)
        step = AbsActivation(name=f"{prefix}_q{pct:02}_step")(pre)  # >= 0
        o    = layers.Subtract(name=f"{prefix}_q{pct:02}_label")([prev, step])
        outputs[pct] = o
        prev = o

    # --- return in the ORIGINAL quantiles order as (B, y_dim, Q)
    expanded = []
    for pct in q_pct:
        t = outputs[pct]  # (B, y_dim)
        t = ExpandDimsLast(name=f"{prefix}_q{pct:02}_expand")(t)  # (B, y_dim, 1)
        expanded.append(t)

    if len(expanded) == 1:
        return expanded[0]  # (B, y_dim, 1)

    return layers.Concatenate(axis=-1, name=f"{prefix}_quantiles_concat")(expanded)  # (B, y_dim, Q)



def build_graph_gated_quantile_model(
    x1_shape,
    x2_shape,
    y_dim,
    quantiles,
    emb_dim=32,
    num_experts=4,
):
    Q = len(quantiles)

    X1_all = layers.Input(shape=(None,) + tuple(x1_shape), name="X_lag_all")  # (B, n_in, 24, 3)
    X2_all = layers.Input(shape=(None,) + tuple(x2_shape), name="X_lead_all")  # (B, n_in, 24, 2)
    gate   = layers.Input(shape=(None,), name="graph_gate")                       # (B, n_in)

    # Projection
    b1 = layers.TimeDistributed(layers.Flatten(), name="b1_flatten")(X1_all)      # (B, n_in, 72)
    b1 = layers.TimeDistributed(layers.Dense(emb_dim, activation="linear"),
                                name="b1_project")(b1)                            # (B, n_in, D)

    b2 = layers.TimeDistributed(layers.Flatten(), name="b2_flatten")(X2_all)      # (B, n_in, 96)
    b2 = layers.TimeDistributed(layers.Dense(emb_dim, activation="linear"),
                                name="b2_project")(b2)                            # (B, n_in, D)

    x = layers.Add(name="branch_add")([b1, b2])                                   # (B, n_in, D)

    # MoE encoder
    enc_gate = layers.TimeDistributed(layers.Dense(num_experts, activation="softmax"), name="enc_gate")(x)  # (B, n_in, E)
    enc_experts = []
    for e in range(num_experts):
        h = layers.TimeDistributed(layers.Dense(emb_dim, activation="swish"), name=f"enc_expert{e}")(x)  # (B, n_in, D)
        w = SliceIndexLayer(index=e, axis=2, keepdims=True, name=f"enc_w{e}")(enc_gate)  # (B, n_in, 1)
        enc_experts.append(layers.Multiply()([h, w]))
    x = layers.Add(name="enc_moe")(enc_experts)  # (B, n_in, D)

    # Graph constraint
    w = ExpandDimsLast(name="gate_expand")(gate)                 # (B, n_in, 1)
    pooled = WeightedAvgPool(name="weighted_avg_pool")([x, w])   # (B, D)

    #out = layers.Dense(y_dim * Q, activation="linear", name="out_dense")(pooled) 
    #out = ReshapeQuantiles(y_dim=y_dim, q=Q, name="y_quantiles")(out)
    out = HierarchicalQuantileHead(
        shared_rep=pooled,     # (B, D)
        quantiles=quantiles,
        output_dim=y_dim,
        prefix="y"
    )


    return Model(
        inputs={"X_lag_all": X1_all, "X_lead_all": X2_all, "graph_gate": gate},
        outputs=out,
    )



@register_keras_serializable()
class SliceIndexLayer(layers.Layer):
    def __init__(self, index, axis=-1, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
        self.axis = int(axis)
        self.keepdims = bool(keepdims)

    def call(self, inputs):
        x = tf.gather(inputs, indices=self.index, axis=self.axis)
        if self.keepdims:
            x = tf.expand_dims(x, axis=self.axis)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "index": self.index,
            "axis": self.axis,
            "keepdims": self.keepdims,
        }
    

@register_keras_serializable()
class ExpandDimsLast(layers.Layer):
    # (B, N) -> (B, N, 1)
    def call(self, x):
        return tf.expand_dims(x, axis=-1)


@register_keras_serializable()
class WeightedAvgPool(layers.Layer):
    # x: (B, N, D), w: (B, N, 1) -> (B, D)
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, inputs):
        x, w = inputs
        num = tf.reduce_sum(x * w, axis=1)               # (B, D)
        den = tf.reduce_sum(w, axis=1) + self.eps        # (B, 1)
        return num / den

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}

@register_keras_serializable()
class ReshapeQuantiles(layers.Layer):
    # (B, y_dim*Q) -> (B, y_dim, Q)
    def __init__(self, y_dim, q, **kwargs):
        super().__init__(**kwargs)
        self.y_dim = int(y_dim)
        self.q = int(q)

    def call(self, x):
        return tf.reshape(x, (-1, self.y_dim, self.q))

    def get_config(self):
        return {**super().get_config(), "y_dim": self.y_dim, "q": self.q}
    


@register_keras_serializable()
class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, reduction="sum_over_batch_size", name="quantile_loss"):
        super().__init__(reduction=reduction, name=name)
        qs = np.asarray(quantiles, dtype="float32")
        if qs.max() > 1.0:
            qs = qs / 100.0
        self.quantiles = qs.tolist()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 2:
            y_true = tf.expand_dims(y_true, axis=-1)  # (B, y_dim, 1)

        qs = tf.constant(self.quantiles, tf.float32)   # (Q,)
        qs = tf.reshape(qs, (1, 1, -1))                # (1,1,Q)

        e = y_true - y_pred                            # (B, y_dim, Q)
        loss = tf.maximum(qs * e, (qs - 1.0) * e)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()   # includes name + reduction
        cfg.update({"quantiles": self.quantiles})
        return cfg
    
    
def save_best_ckpt(path):
    return ModelCheckpoint(path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1)


def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "ExpandDimsLast": ExpandDimsLast,
            "WeightedAvgPool": WeightedAvgPool,
            "ReshapeQuantiles": ReshapeQuantiles,
            "QuantileLoss": QuantileLoss,
        },
    )


def transfer_moe_weights(src_model, dst_model, num_experts):
    # Encoder: gate + experts
    dst_model.get_layer("enc_gate").set_weights(src_model.get_layer("enc_gate").get_weights())

    for e in range(num_experts):
        dst_model.get_layer(f"enc_expert{e}").set_weights(src_model.get_layer(f"enc_expert{e}").get_weights())
        

def exclude_target_country(countries, target_country):
    return [c for c in countries if c != target_country]