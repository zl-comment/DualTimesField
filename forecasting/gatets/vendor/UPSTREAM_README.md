# GateTS: Versatile and Efficient Forecasting via Attention-Inspired routed Mixture-of-Experts

new_moe.py - contains main code for GateTS.
Class MoETransformerForecaster in mixture of experts model where we can change gating.
if gating == 'kronecker', then this is GateTS.
if gating == 'classic', this mixture of experts forecasting model with classic one layer gating.
if gating == 'paper', this mixture of experts with HMM gating described in paper.

default parameters:
```
d_model=48,
n_heads=4,
num_layers=1,
ffn_dim=64,
num_experts=16,
top_k=2,
use_decoder_expert=True,
expert_n_heads= 2,
dropout=0.1,
```

lstm_forecast.py - contains Lstm model for time series forecasting.

patchtst.py - contains patchTST model for time series forecasting.

modelling.py - contains train, val and test of all models with saving their results.

experiments.py - have all experiments described in paper.

utils - contains code for create from tsf file - dataset, also calculating metrics, calculating parameters.

to run all experminets and get csv with results:

```
uv run experiments.py
```
## Install uv

To install uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
and install dependencies:

```
uv sync
```


# Datasets

Datasets can be downloaded here:

saugeen -- https://zenodo.org/records/4656058

solar -- https://zenodo.org/records/4656027

wind -- https://zenodo.org/records/4656032

sunspot -- https://zenodo.org/records/4654722

# Licence
CC-BY 4.0
