# Mechanism-state evidence constraints

Question: does Anchor overcorrect when OOF residual evidence is inconsistent
across chronological blocks within the same supply/demand/transmission state?
This is a falsifiable reliability hypothesis, not a proven cause of QLD errors.
See the [problem and fixed protocol](../../docs/v5_mechanism_state_20261023.md).

The [first-stage evidence and verification](../../logs/forecasting/v5_mechanism_state_20261023/README.md)
are archived. The frozen-output diagnosis provides mixed support for the
hypothesis; the real-data three-seed integration smoke passed exact replay.
Full confirmation has not run, and no causal or performance improvement is claimed.

The policy uses existing vintage-aligned inputs only. Every OOF fold is checked
against its actual public features, including net load's original soft saturation.
Training-only residual evidence separately bounds location, lower scale and upper
scale. No current labels, sign constraint, additional neural weights or new loss.
All-three-zero gates restore Anchor to B; the unchanged history branch still acts.

Three matched arms: `bounded_reference`, `constant_evidence`, `state_evidence`.
The constant control has exactly the same three mean training gates as the state
policy. This controls average attenuation, not realized trained correction size.

Diagnose the existing frozen outputs without neural retraining:

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose_mechanism_state.py \
  --output outputs/forecasting/v5_mechanism_state_diagnosis_20261023
```

Run the explicitly labelled three-seed integration smoke (9 fits, 2 epochs):

```bash
PYTHONPATH=. .venv/bin/python -m forecasting.full_v5_mechanism_state.runner \
  --development --smoke --devices cuda:3 cuda:6 cuda:7 \
  --output outputs/forecasting/v5_mechanism_state_smoke_20261023
PYTHONPATH=. .venv/bin/python scripts/verify_mechanism_state.py \
  outputs/forecasting/v5_mechanism_state_smoke_20261023 --replay
```

Omit `--smoke` and use a new output for the full 54-fit, 20-epoch development
confirmation. The bounded reference must replay its previous trajectory exactly.
All policies and selections freeze before evaluation; resume verifies provenance.
The old acceptance gates remain and mechanism CSVs report harm AND benefit.
No automatic main-table update; all prior failures remain archived.
