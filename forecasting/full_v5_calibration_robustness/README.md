# Two isolated calibration robustness trials

This package keeps the existing asymmetric `Full_joint` architecture and the
historical four-arm experiment intact. See the
[frozen protocol](../../docs/v5_calibration_robustness_20261023.md) and the
[prediction-only diagnosis](../../logs/forecasting/v5_calibration_diagnosis_20261023/README.md).

Three logical arms use two physical training trajectories:

- `reference`: replay the previous asymmetric model and original aggregate guard.
- `multiwindow`: select a checkpoint and shared alpha on four chronological
  quarters from exactly the same reference trajectory, without retraining.
- `bounded_calibration`: smoothly bound location to ±30 and both tail scales
  to [0.9, 1.1]; retain the original loss and aggregate selection rule.

All arms retain the original development folds, hourly samples, fixed tree seed
2026, neural seeds 2026/2027/2028, and 20 epochs. There are 36 physical fits and
54 logical selections. All selections freeze before evaluation. The reference
must exactly reproduce the old training losses, selected parameters, and outputs.
No combined multiwindow-plus-bounds arm is trained in this experiment.

```bash
.venv/bin/python -m forecasting.full_v5_calibration_robustness.runner \
  --development --devices cuda:3 cuda:6 cuda:7 \
  --output outputs/forecasting/v5_calibration_robustness_20261023_r2
```

Use a new output directory, or `--resume` with the identical frozen configuration.
The first run was stopped before evaluation because a NumPy string in checkpoint
metadata could not be loaded with `weights_only=True`; the second run restarts
from scratch with Python string keys. The failed run is retained separately.

After r2 completed all 36 trajectories, a globally scoped count incorrectly
treated region-local fit IDs as unique. The corrected runner identifies physical
fits by their region/fold/seed checkpoint paths. It resumes the already frozen
weights without retraining or changing selection. See the separate
[orchestration repair record](../../docs/v5_calibration_orchestration_fix_20261023.md).

After completion, independently audit every saved prediction and sample-replay
the new checkpoints through the original CPU history interface:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_calibration_robustness.py \
  outputs/forecasting/v5_calibration_robustness_20261023_r2 --replay
```

The old macro, regional, spring, and individual-seed acceptance gates remain;
candidates must also be noninferior to the current reference on both macro
metrics. Fallback is recorded as fallback, and failures never automatically
replace the 2023–2024 main comparison.
