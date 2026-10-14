# Full V3 implementation and launch validation

Branch: `research/full-v3-conditional-analogs-20261013`.

Full repository regression before final audit additions: **330 passed, 1 skipped, 3 subtests passed**.
Final V3 suites: fields 23, memory 17, model 12, priors 8, experiment 21, report 22 = **103 passed**.
Five real NSW1 smoke fits: 64 training origins, 32 selection origins, 24 horizons, five quantiles, two rounds, 120 saved/reloaded trees per variant. All predictions finite with shape [32,24,5]. Smoke outputs are excluded from formal results.

Independent reviews covered memory publication timing, curve decomposition equations, five prior formulas, residual decoding, frozen execution, recovery checks, output/reference hashes and report completeness.

Formal run: `outputs/forecasting/full_v3_20261013`; 15 jobs, 6 CPU workers, 4 threads per worker; seed 2026; three regions; no truncation; original 2000-round/50-patience budget.

Frozen source SHA256: `1c3f2de8ac5417cf9a3a7173ec849a41ad02c458a981055222c0927cc4f447bf`.
Frozen configuration SHA256: `dfbda2afd3bbbd34bd833cfb5c477231bbdf5802ae3dc75c2ae50b00f58c4966`.

| Region | Train | Selection | Calibration | Test | Unavailable-future violations |
|---|---:|---:|---:|---:|---:|
| NSW1 | 61273 | 4321 | 4393 | 17521 | 0 |
| QLD1 | 61273 | 4321 | 4393 | 17521 | 0 |
| TAS1 | 61273 | 4321 | 4393 | 17521 | 0 |

Original baseline inputs, models and outputs were not modified. V3 uses additional historical outcome memory; comparisons with the original baseline are not a same-information structural claim. Tree weights stay fixed while the as-of history library grows. All evaluation periods are retrospective.

Formal run completed: **15/15 jobs, 0 failures**. The completed report, results CSV and summary JSON are archived alongside this note. Full prediction artifacts and fitted models remain in the ignored run directory. Post-run diagnostics are archived in `logs/forecasting/full_v3_diagnosis_20261014`.
