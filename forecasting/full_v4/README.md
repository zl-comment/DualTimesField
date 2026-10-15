# Full V4

Local implementation of trainable historical dual fields, a current-evidence forecaster,
separate sparse null/candidate selection, and time-OOF LightGBM residual correction.
The original baseline data boundary and score implementation are reused unchanged.

Plan: `docs/full_v4_design_20261014.md`. Configuration: `configs/full_v4/experiment.yaml`.
V3's audited calendar availability policy and native tree/checkpoint engine are reused;
V3 fixed-basis fields, distance retrieval and forced priors are not used.

```
.venv/bin/python -m forecasting.full_v4.experiment prepare --run-root outputs/forecasting/full_v4_20261014
.venv/bin/python -m forecasting.full_v4.experiment launch --run-root outputs/forecasting/full_v4_20261014
.venv/bin/python -m forecasting.full_v4.experiment report --run-root outputs/forecasting/full_v4_20261014
```

Only frozen workers train. Three region processes use separate GPUs; each completes
prefix-only early stopping, prefix refits, OOF prediction and the final front ends,
then runs its five tree contrasts. The launcher automatically audits/evaluates all
15 cells and archives the aggregate report after completion. No test score changes
hyperparameters or stops/selects variants. Interrupted fits restart that fit; complete
checkpoint/feature stages are reused only after identity and checksum verification.

Raw-price q50 scores remain retrospective single-seed evidence. Additional historical
information and OOF training coverage differ from the original full-training baseline;
the matched-row direct-tree control separates this training-coverage difference.
