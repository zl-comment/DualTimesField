# Full V4 implementation and run validation

Branch: `research/full-v4-trainable-dual-field-20261014`.

Target: five declared contrasts × NSW1/QLD1/TAS1 × seed 2026. This is a retrospective experiment with additional historical information and unchanged canonical forecast samples, original baseline inputs and scoring.

## Verified before formal launch

- **117 tests passed**: 14 V4 tests plus all 103 V3 regression tests covering the reused calendar memory, residual-tree engine and report contracts.
- First NSW1 smoke: five successful model cells, two neural epochs maximum, 120 trees per cell, three scored splits per cell.
- Three-region smoke: **15/15 completed**, 27 small frontend fits, **1,800 saved tree files**, 45 prediction artifacts. Every complete tree checkpoint and prediction checksum audited. Train limit 256 origins, evaluation/OOF limit 128 origins, one OOF fold, one historical-pretraining epoch, at most two joint epochs and two tree rounds. These outputs are excluded from all formal rankings.
- Final frontend preprocessing asserted exactly equal to all seven canonical blocks in all regions. Fold-internal preprocessing is rebuilt from raw archives, including shortfall thresholds and nonlinear compression.
- Verified exact empty-memory fallback, strict null history independence, historical/current gradient paths, signed events, spline basis, explicit fusion formula, query-label isolation, full-label publication boundaries, finite outputs, checkpoint reload and original-price residual decoding.
- The unsorted-quantile crossing diagnostic was exercised on real smoke checkpoints.

Smoke roots: `outputs/forecasting/full_v4_smoke_20261014` and `outputs/forecasting/full_v4_smoke_all_20261014`.

## Formal execution

Frozen root: `outputs/forecasting/full_v4_20261014`. GPU assignment: NSW1 → 5, QLD1 → 6, TAS1 → 7. Each region runs three frontend variants with three expanding OOF folds (internal selection + prefix refit) and final fitting: 21 frontend fits per region, 63 in total. Each fitted frontend uses prefix-local historical pretraining. Trees run at most two cells per region concurrently, four CPU threads per cell.

Formal configuration retains all eligible training origins, 20 historical pretraining epochs, at most 100 joint epochs with patience 10, and the original 2,000-round/patience-50 tree budget. No test score changes hyperparameters, selection epochs or variants.

The launcher automatically audits the complete 15-cell matrix, scores the common five quantiles, writes `report.md`, `summary.json`, `results.csv`, and archives these small reports here after completion. Native frontend diagnostics are also archived in `frontend_diagnostics.json`. Full models and predictions remain in the ignored output directory.

Formal launch: 2026-10-14T08:43:23.010472+00:00 (UTC); PID 2331175.

Frozen source SHA256: `77ccb942f786b5a9e73742dc6e6688412dfb7c636a04652961ae15898ae88c23`.

Frozen config SHA256: `03c164a05dd1a431167902c617a0bf064b0421b16bb253070e91bb4a5b3d86ae`.

Formal run completed: **15/15 cells, 0 failures**, at **2026-10-15 01:02:51 Asia/Shanghai**. All 63 frontend fits completed. The complete-grid report passed its checkpoint, prediction and alignment audits. Archived reports match the frozen run; the CSV uses LF line endings.
