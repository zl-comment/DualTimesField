# Anchor development and latest Full_joint

The current Full_joint uses the balanced objective and the asymmetric location/
lower-scale/upper-scale head from `confirmation_model.AsymmetricScaleModel`.
Its three-neural-seed aligned evaluation entry point is `aligned_final`, with
the [frozen training, selection and test protocol](../../docs/v5_latest_aligned_20261023.md).
The tree seed stays 2026; neural seeds are 2026/2027/2028. Main results use the
validation-selected guarded output, with independently selected raw output and
same-checkpoint interventions reported separately. All nine fits freeze before
the common 2023–2024 test cache can be opened.

```bash
.venv/bin/python -m forecasting.full_v5_anchor_balance.aligned_final \
  --devices cuda:3 cuda:6 cuda:7 --output outputs/forecasting/NEW_RUN
```

The output must be new; use `--resume` only for that same frozen run. The four
development candidates did not pass all predeclared gates; evaluating the newest
structure does not establish a stable improvement.

## Historical development experiments

See [the design and preregistered interpretation](../../docs/v5_anchor_balance_20261022.md)
and [the two-round ZCode discussion](../../logs/forecasting/v5_anchor_balance_20261022/discussion.md).
The development entry points below reuse the original joint model and authenticated
OOF bundles. They remain restricted to development years; only `aligned_final`
implements the separate frozen 2023–2024 evaluation described above.

From the repository root, run the complete screening comparison:

```bash
.venv/bin/python -m forecasting.full_v5_anchor_balance.experiment \
  --region all --fold all --profile screen --seed 2026 --tree-seed 2026 \
  --development --output outputs/forecasting/v5_anchor_balance_screen_20261022_r1
```

The output directory must not exist. `--fold 2021` means train 2018–2019,
select 2020, evaluate 2021; `--fold 2022` means train 2018–2020, select 2021,
evaluate 2022. `screen` uses rotating daily origins and 3 epochs. For full
hourly coverage and 20 epochs, use `--profile full` and a fresh output directory.
`--seed` controls neural training; `--tree-seed` checks archived tree provenance.

The three variants are `legacy`, `balanced`, and `identity_anchor`. Every
variant records both the original Full-QS checkpoint selection (`raw`) and the
shared-alpha selection with explicit baseline fallback (`guarded`). Compare
like selection rules when attributing changes to the objective.

Outputs include `manifest.json` (config, source/data/sample hashes, OOF mapping),
`source/`, per-variant training logs and checkpoints, `selection_complete.json`,
prediction NPZ files, and `report.json` / `report.md`. All selection finishes
before evaluation scoring begins. `report.json` also includes quarters,
Australian seasons, forecast-price groups, coverage and group sample counts.

The selection guard is not a future-performance guarantee. A fallback to B
is reported as fallback, not as evidence of a learned improvement. Screening
results do not establish convergence or statistical significance.

The completed first screen is archived in
[the result summary](../../logs/forecasting/v5_anchor_balance_20261022/summary.md)
and [the interpretation](../../logs/forecasting/v5_anchor_balance_20261022/interpretation.md).
It improved on guarded legacy training in both years, but did not pass all
preregistered checks. Those historical results are preserved separately from the
current asymmetric Full_joint evaluation.

The subsequent [architecture protocol](../../docs/v5_anchor_architecture_20261022.md)
compares balanced and identity controls with location-only calibration, separate
location/scale heads, staged training, and a wider calibration head. All six
use six epochs (staged: three Anchor + three history) and the same screen
sampling. Unlike the first screen, identity also uses the balanced objective.

```bash
.venv/bin/python -m forecasting.full_v5_anchor_balance.architecture_experiment \
  --development --workers 3 --seed 2026 --tree-seed 2026 \
  --output outputs/forecasting/v5_anchor_architecture_screen_20261022_r1
```

Three region workers train independently; evaluation waits for all 36 fits.
The staged history phase verifies that both the shared encoder and calibration
head, and the Anchor predictions on a fixed probe, remain exactly unchanged.
Summarize a completed run with
`logs/forecasting/v5_anchor_architecture_20261022/summarize_run.py RUN_DIRECTORY`.
The completed screen's [summary](../../logs/forecasting/v5_anchor_architecture_20261022/summary.md)
and [interpretation](../../logs/forecasting/v5_anchor_architecture_20261022/interpretation.md)
report all six variants. No candidate passed every preregistered check; location-only
with selection-time shrinkage improved both annual macro metrics but retained a
TAS1/2021 MAE regression. This does not establish architecture as the sole cause.
