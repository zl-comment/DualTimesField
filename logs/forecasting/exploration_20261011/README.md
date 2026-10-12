# Forecasting exploration execution

## Completed run and interpretation limits

All 1,143 jobs and the final report completed successfully. The consolidated
[statistical report](reports/report.md) covers static, quarterly rolling, and
capped650 protocols, including validation-selected floors and paired block
bootstrap comparisons. [Event results](reports/events_report.md) are separate.
The complete machine-readable report is stored as `reports/report.json.gz`;
read it with `gzip.open(path, "rt")` and `json.load`, or decompress with
`gzip -dk reports/report.json.gz`. Intermediate JSON files, process logs,
PID/lock files, and model/prediction outputs remain local; the committed manifest,
completion status, final reports, and figures document this frozen run.

Post-run implementation review identified the following limitations. These
results preserve the configurations actually executed; no corrective retraining
is included in this run.

- **PatchTST requires correction and rerunning before model-quality claims.**
  Its RevIN restores statistics from standardized asinh-price history, but this
  exploration runner supervises the output in standardized raw-price space.
  This imposes an inappropriate cross-space normalization constraint. The size
  of its effect has not been measured in a corrected experiment.
- PatchTST and GateTS use an additive linear external-input adapter; their
  nonlinear backbones see price history only. Equal feature availability does
  not establish equally effective nonlinear fusion of those features.
- Neural baselines use fixed learning rates and 30 epochs without systematic
  validation-set hyperparameter search. All three TAS1 static GateTS-adapter
  fits select epoch 30, so sufficient training has not been established.
- Full uses a separate transformed-space MSE point head, whereas XGBoost and
  neural baselines report the raw-price quantile median as their point forecast.
  XGBoost also fits and early-stops each of 24 horizons independently.
- Timestamp checks use archive run/LASTCHANGED metadata as availability proxies;
  actual public-release and transmission delays have not been fully audited.
- XGBoost leads the macro-average MAE and approximate CRPS in this configuration
  matrix, not every regional metric. This is not evidence of universal SOTA.
  Test periods have already been inspected; follow-up development must use
  training/validation data and disclose the exploratory status of test results.

The original plan was audited by three agents and corrected before execution.
The frozen matrix is in `manifest.json`: 1,143 jobs, 127 configurations,
NSW1/QLD1/TAS1 and seeds 2026/2027/2028 in every cell.

| Protocol | Configurations | Fits |
|---|---:|---:|
| Static, full ablations and controls | 35 | 315 |
| Eight quarterly refits, 11 models | 88 | 792 |
| Capped-price sensitivity | 4 | 36 |

Run from the repository root, using the existing environment:

```bash
.venv/bin/python -u -m forecasting.exploration_runner \
  --manifest logs/forecasting/exploration_20261011/manifest.json \
  --gpus 6 7 --workers-per-gpu 4
```

The same command resumes completed jobs after checking artifact identities and
hashes. The scheduler refuses source, configuration, and manifest changes;
new scientific versions require a separate output root. Do not regenerate the
manifest during an active run. Data hashes are checked before scheduling.
The source snapshot and environment record live in the ignored output tree.

Progress: `manifest.status.json`; supervisor log: `supervisor.log`; per-job logs:
`manifest_logs/`. Each status row includes its region, seed (in its id), GPU,
pid, command, timestamps, exit status, and log path. Failed or blocked jobs never
count as completed. An interrupted job retrains from its initial seed; successful
jobs are reused only when all recorded artifact hashes match.

Outputs live at `outputs/forecasting/exploration_20261011/runs/`. Each price run
has validation/test predictions, metrics, and run metadata; event classifiers
save probabilities without inventing price forecasts. Models/checkpoints are
retained. The final report runs automatically after all jobs succeed:

```bash
.venv/bin/python -m forecasting.exploration_report \
  --manifest logs/forecasting/exploration_20261011/manifest.json
```

When every static job has succeeded, a static-only report can also be generated
with `--protocol static`. Reports contain per-seed metrics, region mean ± sample
SD, 168-hour paired block-bootstrap comparisons (19,999 replicates), Holm
correction and 24-hour block sensitivity, independent validation-chosen floors,
event probabilities, routing diagnostics, deterministic cases and fit costs.
Static, rolling, capped, and ensemble scores are labeled separately. Neither
2023–2024 nor the previously studied 2025 data is claimed as a new blind test.

Validation before launch: all unit/integration tests; every config/model/loss
shape; all three-region/three-seed cells; source-publication time checks; real
NSW1 one-epoch full-model and GateTS-adapter smoke runs. Smoke runs are excluded
from scientific results. Completion and experimental findings must be read from
status and reports, never inferred from the existence of this runbook.
