"""Model factory and compatibility exports; implementations live in model packages."""
from torch import nn
from .common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball
from .dlinear.model import MovingAverage, DLinear
from .patchtst.model import RevIN, PatchTSTLayer, PatchTST
from .linear.model import KnownLinear, LinearMSE, Linear
from .itransformer.model import ITransformerLayer, ITransformer
from .informer.model import ProbAttention, FullAttention, AttentionLayer, DataEmbedding, DistilConv, InformerEncoderLayer, InformerDecoderLayer, Informer


def build(kind: str, example: dict, **parameters) -> nn.Module:
    """Builds a model from one collated example batch."""
    input_hours = example["history_values"].shape[1]
    output_hours = example["future_calendar"].shape[1]
    future_dim = _known_future(example).shape[-1]
    history_exog_dim = example["history_values"].shape[-1] - 1
    common = dict(input_hours=input_hours, output_hours=output_hours, future_dim=future_dim,
                  history_exog_dim=history_exog_dim)
    if common.keys() & parameters.keys():
        raise ValueError('Input dimensions are determined by the shared batch')
    common.update(parameters)
    if kind == "dlinear":
        return DLinear(**common)
    # Attribution variants of DLinear and PatchTST.
    if kind == "dlinear_noadapter":
        return DLinear(**common, adapter=False)
    if kind == "linear":
        return Linear(**common)
    if kind == "known_linear":
        return KnownLinear(**common)
    if kind == "linear_mse":
        return LinearMSE(**common)
    if kind == "patchtst":
        return PatchTST(**common)
    if kind == "patchtst_noadapter":
        return PatchTST(**common, adapter=False)
    if kind == "itransformer":
        return ITransformer(**common)
    if kind == "informer":
        context = example.get("origin_context")
        return Informer(**common, calendar_dim=example["future_calendar"].shape[-1],
                        context_dim=0 if context is None else context.shape[-1])
    raise ValueError(kind)

# Chosen from {1e-4, 5e-4, 1e-3} by the lowest validation pinball loss on raw
# NSW1 prices, seed 2026, then fixed for every region, seed, and price treatment.
LEARNING_RATES = {"dlinear": 1e-3, "patchtst": 1e-4, "itransformer": 1e-3, "informer": 1e-3}
# Attribution variants use the learning rate of the model they modify.
LEARNING_RATES |= {"dlinear_noadapter": 1e-3, "linear": 1e-3, "known_linear": 1e-3, "patchtst_noadapter": 1e-4,
                   "linear_mse": 1e-3}
