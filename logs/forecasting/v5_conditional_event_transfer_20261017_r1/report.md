# Conditional historical event transfer

Frozen July-August encoder; October-only gate fitting; November2022 retrospective development scores, seed2026. No December assessment or final test.

| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |
|---|---:|---:|---:|---:|---:|
| B | 33.206288 | 37.412038 | 65.566701 | 45.395009 | 30.421851 |
| C_roll | 34.132323 | 34.856338 | 66.010817 | 44.999826 | 30.961751 |
| H_recent | 41.253983 | 46.957856 | 66.993527 | 51.735122 | 36.798154 |
| H_previous_global | 48.518704 | 56.172955 | 68.381336 | 57.690998 | 42.165495 |
| T_only | 37.569933 | 43.670693 | 66.206971 | 49.149199 | 34.503500 |
| TE_unshifted | 39.857517 | 44.542399 | 66.181383 | 50.193767 | 35.560931 |
| TE_full | 38.003231 | 44.668246 | 67.049564 | 49.907014 | 35.239675 |
| TE_fixed_pulses | 37.912030 | 45.217920 | 66.220963 | 49.783638 | 35.157693 |
| TE_event_off | 37.402378 | 42.316700 | 66.157981 | 48.625687 | 33.979623 |
| TE_time_shift_off | 38.333812 | 44.061652 | 66.760464 | 49.718643 | 35.129669 |

Incremental-value gate: **False**. All-region non-regression: **False**.
State-aligned event transport is a conditional forecasting hypothesis, not identification of marginal generators or a causal market mechanism.

## NSW1
- Assessment origins: 696; unique actual spike hours: 4.
- Full gate fit: 720 mature origins, parameters 60, selected null=False.
- Full gate activation: {'gate_trend': {'mean': 0.5068313528945823, 'positive_fraction': 1.0}, 'gate_event': {'mean': 0.0317380175632597, 'positive_fraction': 0.23820641762452108}, 'mean_abs_shift': 10.966439095443137}.
- Shared historical candidates across geometry/timing controls: True; all memory maturity violations zero.
- Wall seconds: 24.0.

## QLD1
- Assessment origins: 696; unique actual spike hours: 18.
- Full gate fit: 720 mature origins, parameters 60, selected null=False.
- Full gate activation: {'gate_trend': {'mean': 0.5992488001321518, 'positive_fraction': 1.0}, 'gate_event': {'mean': 0.09821186783078237, 'positive_fraction': 0.3331537356321839}, 'mean_abs_shift': 18.343601405870245}.
- Shared historical candidates across geometry/timing controls: True; all memory maturity violations zero.
- Wall seconds: 22.1.

## TAS1
- Assessment origins: 696; unique actual spike hours: 6.
- Full gate fit: 720 mature origins, parameters 60, selected null=False.
- Full gate activation: {'gate_trend': {'mean': 0.09300647665953206, 'positive_fraction': 0.39960488505747127}, 'gate_event': {'mean': 0.07324119595813236, 'positive_fraction': 0.390205938697318}, 'mean_abs_shift': 2.9836436546340557}.
- Shared historical candidates across geometry/timing controls: True; all memory maturity violations zero.
- Wall seconds: 21.4.

T_only and TE_unshifted refit their own gates. T_only retains event-related inputs but removes the direct event term; it is not an event-information-free ablation. TE_event_off and TE_time_shift_off reuse the exact full model gates and features. Candidate identities, distances and weights are shared across timing and geometry controls.
All quantiles shift together, retain q50 point scoring, and corrections stay within +/-150. Actual first-publication timestamp audits remain limited to existing archive proxies. Results cannot be compared numerically to the prior December table as though the evaluation period were unchanged.
