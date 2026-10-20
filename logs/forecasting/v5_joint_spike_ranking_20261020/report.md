# V5 joint final test: spike-window ranking vs all aligned baselines

Frozen 2023-2024 final test, seed 2026, rearranged-q50 MAE (AUD/MWh),
equal weight per origin-hour cell, identical cells across all models.
Bias = signed mean error (negative = under-forecast).
Retrospective, single seed, no significance tests; extreme bands hold
few origins, so sub-percent differences between close models are noise-level.

## moderate spikes: window max in [300, 1000)

### NSW1 — n=1969 origins / 47256 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | time_shift_off_same_head | 41.630 | -26.6 |
| 2 | Full_joint ** | 42.035 | -26.7 |
| 3 | event_off_same_head | 42.364 | -28.3 |
| 4 | Anchor_joint | 42.433 | -27.2 |
| 5 | B_retrained | 43.564 | -29.7 |
| 6 | lightgbm ** | 43.581 | -29.7 |
| 7 | xgboost ** | 44.984 | -31.0 |
| 8 | patchtst | 57.162 | -20.2 |
| 9 | gatets_adapter | 59.744 | -33.5 |
| 10 | timexer | 61.597 | -23.2 |
| 11 | mlp | 65.724 | -52.7 |
| 12 | full | 66.949 | -29.9 |
| 13 | dlinear | 67.063 | -30.2 |
| 14 | linear | 67.292 | -30.4 |
| 15 | pricefm | 67.991 | -40.3 |
| 16 | no_fields | 69.089 | -28.6 |
| 17 | lasso_arx72_cp | 94.003 | -83.7 |

### QLD1 — n=2789 origins / 66936 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | time_shift_off_same_head | 38.873 | -25.6 |
| 2 | event_off_same_head | 39.052 | -26.5 |
| 3 | Full_joint ** | 39.101 | -25.7 |
| 4 | lightgbm ** | 39.133 | -25.1 |
| 5 | Anchor_joint | 39.174 | -26.7 |
| 6 | B_retrained | 39.302 | -25.3 |
| 7 | xgboost ** | 40.596 | -25.2 |
| 8 | patchtst | 51.325 | -22.3 |
| 9 | timexer | 51.822 | -15.1 |
| 10 | no_fields | 56.270 | -13.6 |
| 11 | gatets_adapter | 56.302 | -22.6 |
| 12 | dlinear | 56.910 | -25.6 |
| 13 | linear | 57.394 | -28.3 |
| 14 | full | 58.519 | -31.7 |
| 15 | mlp | 60.871 | -50.6 |
| 16 | pricefm | 68.586 | -59.6 |
| 17 | lasso_arx72_cp | 83.867 | -58.6 |

### TAS1 — n=539 origins / 12936 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | patchtst | 56.162 | -26.6 |
| 2 | timexer | 56.474 | -22.5 |
| 3 | time_shift_off_same_head | 59.206 | -41.6 |
| 4 | Full_joint ** | 59.310 | -41.7 |
| 5 | linear | 59.327 | -18.3 |
| 6 | event_off_same_head | 59.345 | -41.8 |
| 7 | Anchor_joint | 59.410 | -41.9 |
| 8 | lasso_arx72_cp | 60.995 | -25.0 |
| 9 | dlinear | 62.254 | -20.3 |
| 10 | gatets_adapter | 66.890 | -33.8 |
| 11 | lightgbm ** | 69.062 | -56.3 |
| 12 | B_retrained | 69.442 | -56.7 |
| 13 | xgboost ** | 70.858 | -57.8 |
| 14 | mlp | 72.008 | -55.5 |
| 15 | pricefm | 73.233 | -56.4 |
| 16 | no_fields | 85.899 | -75.9 |
| 17 | full | 86.503 | -74.1 |

## extreme spikes: window max >= 1000

### NSW1 — n=1269 origins / 30456 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | B_retrained | 312.829 | -298.2 |
| 2 | lightgbm ** | 312.888 | -298.3 |
| 3 | time_shift_off_same_head | 313.417 | -298.7 |
| 4 | Anchor_joint | 313.742 | -298.5 |
| 5 | Full_joint ** | 313.787 | -298.9 |
| 6 | event_off_same_head | 314.270 | -300.6 |
| 7 | xgboost ** | 314.437 | -299.3 |
| 8 | gatets_adapter | 320.378 | -277.4 |
| 9 | patchtst | 320.831 | -274.4 |
| 10 | mlp | 327.470 | -307.8 |
| 11 | timexer | 329.399 | -287.8 |
| 12 | dlinear | 329.950 | -259.7 |
| 13 | linear | 330.594 | -261.1 |
| 14 | full | 332.775 | -268.4 |
| 15 | no_fields | 335.671 | -258.6 |
| 16 | pricefm | 337.108 | -287.4 |
| 17 | lasso_arx72_cp | 354.208 | -339.1 |

### QLD1 — n=1345 origins / 32280 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | no_fields | 206.048 | -153.2 |
| 2 | lightgbm ** | 210.152 | -195.7 |
| 3 | time_shift_off_same_head | 210.351 | -197.0 |
| 4 | B_retrained | 210.374 | -195.9 |
| 5 | Full_joint ** | 210.736 | -197.1 |
| 6 | event_off_same_head | 211.521 | -198.7 |
| 7 | Anchor_joint | 211.689 | -199.3 |
| 8 | full | 212.040 | -173.7 |
| 9 | xgboost ** | 212.153 | -195.9 |
| 10 | patchtst | 212.864 | -181.8 |
| 11 | gatets_adapter | 214.629 | -172.5 |
| 12 | linear | 215.859 | -168.5 |
| 13 | dlinear | 216.122 | -164.3 |
| 14 | mlp | 223.726 | -209.1 |
| 15 | timexer | 226.024 | -181.5 |
| 16 | pricefm | 229.339 | -211.1 |
| 17 | lasso_arx72_cp | 253.942 | -226.4 |

### TAS1 — n=362 origins / 8688 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | event_off_same_head | 208.052 | -177.1 |
| 2 | Full_joint ** | 208.059 | -177.0 |
| 3 | time_shift_off_same_head | 208.080 | -177.0 |
| 4 | Anchor_joint | 208.096 | -177.2 |
| 5 | patchtst | 209.467 | -173.3 |
| 6 | timexer | 210.984 | -166.9 |
| 7 | lightgbm ** | 213.047 | -188.2 |
| 8 | linear | 213.070 | -153.5 |
| 9 | B_retrained | 213.218 | -188.6 |
| 10 | lasso_arx72_cp | 213.761 | -157.9 |
| 11 | xgboost ** | 213.873 | -188.4 |
| 12 | dlinear | 214.613 | -152.0 |
| 13 | gatets_adapter | 217.251 | -179.0 |
| 14 | mlp | 218.180 | -191.4 |
| 15 | full | 218.240 | -195.1 |
| 16 | no_fields | 219.443 | -199.0 |
| 17 | pricefm | 221.332 | -193.5 |

## sustained extreme: window mean >= 300

### NSW1 — n=406 origins / 9744 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | time_shift_off_same_head | 731.199 | -717.0 |
| 2 | Anchor_joint | 731.451 | -716.8 |
| 3 | Full_joint ** | 731.639 | -717.8 |
| 4 | event_off_same_head | 733.177 | -721.2 |
| 5 | B_retrained | 734.364 | -724.3 |
| 6 | lightgbm ** | 734.455 | -724.4 |
| 7 | patchtst | 734.636 | -674.6 |
| 8 | gatets_adapter | 735.519 | -688.8 |
| 9 | xgboost ** | 737.064 | -727.0 |
| 10 | full | 747.347 | -658.4 |
| 11 | mlp | 747.573 | -726.9 |
| 12 | dlinear | 747.677 | -652.6 |
| 13 | linear | 749.628 | -656.6 |
| 14 | timexer | 751.206 | -710.1 |
| 15 | no_fields | 755.793 | -647.6 |
| 16 | pricefm | 764.570 | -693.2 |
| 17 | lasso_arx72_cp | 799.297 | -793.3 |

### QLD1 — n=383 origins / 9192 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | no_fields | 432.666 | -376.3 |
| 2 | full | 442.722 | -398.8 |
| 3 | patchtst | 453.535 | -424.4 |
| 4 | dlinear | 454.127 | -407.0 |
| 5 | linear | 454.620 | -412.3 |
| 6 | gatets_adapter | 455.088 | -412.3 |
| 7 | time_shift_off_same_head | 462.895 | -451.3 |
| 8 | lightgbm ** | 463.382 | -452.1 |
| 9 | Full_joint ** | 463.471 | -451.5 |
| 10 | B_retrained | 463.767 | -452.5 |
| 11 | event_off_same_head | 464.837 | -453.9 |
| 12 | Anchor_joint | 465.160 | -455.1 |
| 13 | xgboost ** | 466.281 | -452.9 |
| 14 | mlp | 473.899 | -460.2 |
| 15 | timexer | 475.501 | -439.0 |
| 16 | pricefm | 480.036 | -457.0 |
| 17 | lasso_arx72_cp | 513.711 | -500.0 |

### TAS1 — n=143 origins / 3432 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | timexer | 365.742 | -326.9 |
| 2 | patchtst | 367.372 | -332.8 |
| 3 | Full_joint ** | 372.586 | -342.8 |
| 4 | event_off_same_head | 372.626 | -343.0 |
| 5 | time_shift_off_same_head | 372.633 | -342.8 |
| 6 | Anchor_joint | 372.722 | -343.2 |
| 7 | linear | 375.927 | -293.8 |
| 8 | lasso_arx72_cp | 377.145 | -307.7 |
| 9 | dlinear | 381.903 | -294.6 |
| 10 | lightgbm ** | 384.362 | -360.9 |
| 11 | B_retrained | 384.811 | -361.7 |
| 12 | gatets_adapter | 385.070 | -327.1 |
| 13 | xgboost ** | 385.195 | -360.5 |
| 14 | mlp | 394.016 | -371.0 |
| 15 | pricefm | 399.323 | -375.3 |
| 16 | no_fields | 399.689 | -375.6 |
| 17 | full | 399.839 | -376.4 |

## calm complement: window max < 300 (contrast)

### NSW1 — n=14283 origins / 342792 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | B_retrained | 20.607 | -5.0 |
| 2 | lightgbm ** | 20.614 | -4.9 |
| 3 | xgboost ** | 21.242 | -4.0 |
| 4 | event_off_same_head | 22.137 | -10.7 |
| 5 | time_shift_off_same_head | 22.155 | -10.5 |
| 6 | Full_joint ** | 22.175 | -10.5 |
| 7 | Anchor_joint | 22.320 | -10.7 |
| 8 | gatets_adapter | 28.338 | -6.1 |
| 9 | patchtst | 29.245 | -6.7 |
| 10 | mlp | 30.310 | -18.1 |
| 11 | dlinear | 30.707 | -6.5 |
| 12 | linear | 30.852 | -6.5 |
| 13 | timexer | 32.333 | 1.3 |
| 14 | full | 33.016 | -12.3 |
| 15 | pricefm | 33.142 | -13.6 |
| 16 | no_fields | 33.327 | -10.0 |
| 17 | lasso_arx72_cp | 42.479 | -11.2 |

### QLD1 — n=13387 origins / 321288 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | lightgbm ** | 20.726 | -2.9 |
| 2 | B_retrained | 20.780 | -2.9 |
| 3 | event_off_same_head | 21.637 | -7.3 |
| 4 | time_shift_off_same_head | 21.657 | -7.1 |
| 5 | Anchor_joint | 21.692 | -7.4 |
| 6 | Full_joint ** | 21.713 | -7.1 |
| 7 | xgboost ** | 21.854 | -2.0 |
| 8 | patchtst | 29.319 | -7.3 |
| 9 | dlinear | 30.616 | -2.3 |
| 10 | linear | 30.648 | -4.0 |
| 11 | no_fields | 30.972 | -1.3 |
| 12 | gatets_adapter | 31.167 | -0.8 |
| 13 | timexer | 32.603 | 3.3 |
| 14 | mlp | 33.141 | -23.6 |
| 15 | full | 33.335 | -15.9 |
| 16 | pricefm | 39.365 | -32.5 |
| 17 | lasso_arx72_cp | 50.894 | -4.6 |

### TAS1 — n=16620 origins / 398880 cells

| rank | model | MAE | bias |
|---:|---|---:|---:|
| 1 | B_retrained | 23.365 | -6.0 |
| 2 | lightgbm ** | 23.382 | -6.0 |
| 3 | event_off_same_head | 23.536 | -4.9 |
| 4 | Full_joint ** | 23.537 | -4.9 |
| 5 | time_shift_off_same_head | 23.553 | -4.9 |
| 6 | Anchor_joint | 23.557 | -4.9 |
| 7 | xgboost ** | 23.710 | -6.0 |
| 8 | patchtst | 27.821 | -2.8 |
| 9 | mlp | 28.749 | -8.4 |
| 10 | full | 28.854 | -11.2 |
| 11 | dlinear | 29.092 | 2.2 |
| 12 | linear | 29.192 | 0.5 |
| 13 | timexer | 29.255 | 1.7 |
| 14 | no_fields | 29.550 | -17.6 |
| 15 | lasso_arx72_cp | 29.817 | 2.0 |
| 16 | gatets_adapter | 29.923 | -4.7 |
| 17 | pricefm | 30.367 | -10.8 |
