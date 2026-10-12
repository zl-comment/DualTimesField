# Forecasting exploration results

> Interpretation notice: These are the original frozen-run results. A post-run audit found a PatchTST RevIN/target-space adaptation defect; its scores require corrected retraining before model-quality claims. Other fusion, tuning, and availability limitations are documented in [the run README](../README.md#completed-run-and-interpretation-limits). No corrective retraining is included here.

retrospective exploration; neither 2023-24 nor 2025 is an unseen holdout

Every table reports three seeds independently; macro results average regions within each seed before computing sample standard deviation.

CRPS~ is twice the equal-weight mean of five nonuniform quantile pinball losses, not an exact CRPS integral. Intervals, tails, monthly and horizon details are retained in JSON.

## static

Rows retain manifest order; no automatic performance ranking or success claim is made.

### Main comparisons

| Variant | MAE | Window RMSE | CRPS~ | CRPS~ negative | CRPS~ spike | Coverage 90 | AIS 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 38.923 ± 0.296 | 75.955 ± 0.269 | 25.936 ± 0.017 | 18.114 ± 0.750 | 795.998 ± 5.392 | 0.884 ± 0.007 | 369.817 ± 1.907 |
| full+floor | 38.967 ± 0.279 | 75.978 ± 0.281 | 26.001 ± 0.035 | 19.874 ± 1.019 | 795.943 ± 5.400 | 0.856 ± 0.007 | 372.630 ± 3.081 |
| pd_calibrator | 39.510 ± 0.101 | 76.866 ± 0.245 | 26.565 ± 0.287 | 20.629 ± 0.895 | 867.522 ± 15.554 | 0.866 ± 0.008 | 391.264 ± 5.121 |
| pd_calibrator+floor | 39.532 ± 0.171 | 76.875 ± 0.310 | 26.648 ± 0.279 | 22.221 ± 0.882 | 867.490 ± 15.564 | 0.841 ± 0.007 | 394.270 ± 5.294 |
| pd_nofields | 40.274 ± 0.591 | 77.826 ± 0.821 | 26.697 ± 0.271 | 19.517 ± 0.843 | 854.412 ± 10.882 | 0.871 ± 0.009 | 400.066 ± 8.066 |
| pd_nofields+floor | 40.374 ± 0.568 | 77.885 ± 0.804 | 26.674 ± 0.250 | 19.580 ± 0.910 | 854.411 ± 10.883 | 0.868 ± 0.007 | 399.644 ± 7.729 |
| xgboost | 37.968 ± 0.019 | 76.367 ± 0.033 | 25.527 ± 0.014 | 22.616 ± 0.018 | 821.036 ± 1.473 | 0.851 ± 0.000 | 397.826 ± 0.722 |
| xgboost+floor | 37.970 ± 0.019 | 76.369 ± 0.031 | 25.502 ± 0.014 | 22.443 ± 0.016 | 821.036 ± 1.473 | 0.850 ± 0.000 | 396.954 ± 0.724 |
| linear | 41.676 ± 0.073 | 80.307 ± 0.075 | 26.855 ± 0.196 | 21.384 ± 0.516 | 875.723 ± 10.485 | 0.844 ± 0.003 | 395.135 ± 4.642 |
| dlinear | 41.831 ± 0.335 | 80.467 ± 0.388 | 27.017 ± 0.202 | 21.163 ± 0.403 | 882.611 ± 12.842 | 0.848 ± 0.005 | 398.535 ± 4.558 |
| patchtst | 63.171 ± 1.386 | 99.151 ± 1.519 | 37.066 ± 0.583 | 25.452 ± 0.250 | 898.424 ± 8.876 | 0.909 ± 0.004 | 515.766 ± 7.391 |
| gatets | 48.548 ± 0.553 | 89.102 ± 0.676 | 30.638 ± 0.076 | 29.749 ± 0.383 | 928.778 ± 4.102 | 0.882 ± 0.003 | 458.465 ± 1.825 |
| gatets_adapter | 41.705 ± 0.043 | 80.464 ± 0.095 | 26.310 ± 0.045 | 21.512 ± 0.686 | 830.254 ± 1.746 | 0.902 ± 0.004 | 386.366 ± 1.453 |

### Ablations

| Variant | MAE | Window RMSE | CRPS~ | CRPS~ negative | CRPS~ spike | Coverage 90 | AIS 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_tail | 38.937 ± 0.496 | 76.022 ± 0.606 | 26.265 ± 0.124 | 20.216 ± 0.676 | 848.524 ± 2.921 | 0.862 ± 0.007 | 380.798 ± 3.901 |
| no_tail+floor | 38.978 ± 0.445 | 76.037 ± 0.569 | 26.374 ± 0.141 | 22.298 ± 0.349 | 848.505 ± 2.919 | 0.836 ± 0.007 | 384.778 ± 4.232 |
| concat | 39.304 ± 0.134 | 76.484 ± 0.223 | 26.616 ± 0.202 | 19.148 ± 0.323 | 826.631 ± 8.970 | 0.886 ± 0.004 | 384.985 ± 3.167 |
| concat+floor | 39.412 ± 0.148 | 76.582 ± 0.234 | 26.687 ± 0.172 | 20.994 ± 0.425 | 826.572 ± 8.975 | 0.859 ± 0.004 | 387.558 ± 2.125 |
| concat_no_tail | 39.470 ± 0.095 | 76.766 ± 0.091 | 27.087 ± 0.339 | 21.825 ± 0.533 | 881.053 ± 10.924 | 0.867 ± 0.005 | 398.774 ± 5.526 |
| no_ic | 39.319 ± 0.003 | 76.739 ± 0.165 | 25.799 ± 0.104 | 17.116 ± 0.644 | 800.993 ± 11.750 | 0.892 ± 0.007 | 372.998 ± 3.229 |
| no_fields | 40.018 ± 0.366 | 77.404 ± 0.504 | 26.688 ± 0.322 | 19.440 ± 1.036 | 811.043 ± 27.998 | 0.890 ± 0.011 | 393.888 ± 12.110 |
| uniform | 39.603 ± 0.230 | 76.814 ± 0.262 | 27.424 ± 0.093 | 18.857 ± 1.057 | 857.748 ± 8.201 | 0.898 ± 0.007 | 407.938 ± 2.854 |
| no_quantile_gate | 39.364 ± 0.335 | 76.726 ± 0.406 | 26.422 ± 0.399 | 18.502 ± 1.218 | 820.478 ± 25.493 | 0.887 ± 0.002 | 382.210 ± 9.482 |
| no_event | 39.740 ± 0.017 | 76.880 ± 0.083 | 27.017 ± 0.120 | 18.522 ± 0.689 | 838.768 ± 29.668 | 0.888 ± 0.005 | 408.280 ± 6.183 |
| raw_history | 39.583 ± 0.236 | 76.975 ± 0.155 | 26.019 ± 0.110 | 18.632 ± 1.200 | 793.867 ± 9.015 | 0.881 ± 0.010 | 369.145 ± 2.920 |
| no_pd | 42.873 ± 0.333 | 80.567 ± 0.274 | 27.945 ± 0.091 | 17.585 ± 1.092 | 831.061 ± 25.020 | 0.873 ± 0.011 | 398.814 ± 2.215 |
| no_scarcity | 39.307 ± 0.289 | 76.424 ± 0.363 | 26.307 ± 0.346 | 19.044 ± 0.318 | 817.741 ± 14.050 | 0.887 ± 0.007 | 385.588 ± 8.489 |
| gabor | 39.734 ± 0.185 | 77.252 ± 0.257 | 26.217 ± 0.147 | 19.070 ± 1.333 | 798.907 ± 13.972 | 0.883 ± 0.015 | 373.138 ± 4.301 |
| adaptive | 41.010 ± 1.254 | 79.629 ± 2.064 | 26.315 ± 0.448 | 18.028 ± 0.776 | 811.957 ± 24.204 | 0.878 ± 0.011 | 377.235 ± 8.532 |
| echo | 39.817 ± 0.742 | 78.082 ± 2.559 | 26.248 ± 0.126 | 18.437 ± 0.366 | 800.309 ± 9.211 | 0.868 ± 0.008 | 376.958 ± 4.817 |
| D_all_inputs | 39.592 ± 0.171 | 76.982 ± 0.303 | 26.990 ± 0.296 | 21.552 ± 0.518 | 876.593 ± 8.590 | 0.867 ± 0.005 | 397.573 ± 4.342 |
| C_predispatch | 39.385 ± 0.246 | 76.687 ± 0.272 | 26.664 ± 0.269 | 21.179 ± 1.158 | 869.348 ± 14.056 | 0.866 ± 0.006 | 393.153 ± 4.131 |
| B_scarcity | 44.509 ± 0.128 | 82.254 ± 0.269 | 27.756 ± 0.156 | 17.270 ± 0.301 | 834.617 ± 1.529 | 0.890 ± 0.005 | 394.067 ± 5.347 |
| A_history | 45.594 ± 0.253 | 83.615 ± 0.329 | 29.883 ± 0.031 | 21.025 ± 0.827 | 916.483 ± 0.399 | 0.886 ± 0.005 | 441.240 ± 1.402 |
| event_joint_dgf | 38.966 ± 0.271 | 76.134 ± 0.360 | 26.135 ± 0.205 | 17.399 ± 0.277 | 794.941 ± 11.761 | 0.871 ± 0.007 | 374.265 ± 4.309 |
| event_joint_raw | 39.246 ± 0.504 | 76.305 ± 0.555 | 26.114 ± 0.234 | 17.862 ± 1.038 | 800.312 ± 4.818 | 0.873 ± 0.001 | 372.321 ± 3.783 |
| event_joint_none | 39.482 ± 0.428 | 76.514 ± 0.457 | 25.932 ± 0.053 | 18.023 ± 0.323 | 792.979 ± 4.795 | 0.875 ± 0.005 | 369.132 ± 1.396 |
| event_joint_raw_no_extremes | 39.646 ± 0.113 | 76.962 ± 0.101 | 26.103 ± 0.140 | 17.934 ± 0.962 | 794.288 ± 1.854 | 0.871 ± 0.007 | 372.654 ± 5.295 |

### Frozen-event price forecasts

| Variant | MAE | Window RMSE | CRPS~ | CRPS~ negative | CRPS~ spike | Coverage 90 | AIS 90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| event_frozen_dgf | 38.923 ± 0.296 | 75.955 ± 0.269 | 25.936 ± 0.017 | 18.114 ± 0.750 | 795.998 ± 5.392 | 0.884 ± 0.007 | 369.817 ± 1.907 |
| event_frozen_raw | 38.923 ± 0.296 | 75.955 ± 0.269 | 25.936 ± 0.017 | 18.114 ± 0.750 | 795.998 ± 5.392 | 0.884 ± 0.007 | 369.817 ± 1.907 |
| event_frozen_none | 38.923 ± 0.296 | 75.955 ± 0.269 | 25.936 ± 0.017 | 18.114 ± 0.750 | 795.998 ± 5.392 | 0.884 ± 0.007 | 369.817 ± 1.907 |


### Regional seed means and standard deviations

| Variant | Region | MAE | CRPS~ | q05 negative pinball | q95 spike pinball |
|---|---|---:|---:|---:|---:|
| full | NSW1 | 45.572 ± 0.397 | 33.793 ± 0.017 | 10.565 ± 1.313 | 804.413 ± 31.715 |
| full | QLD1 | 39.990 ± 0.715 | 26.737 ± 0.406 | 3.630 ± 0.174 | 418.784 ± 15.327 |
| full | TAS1 | 31.207 ± 0.850 | 17.278 ± 0.427 | 5.887 ± 0.776 | 698.506 ± 9.813 |
| full+floor | NSW1 | 45.624 ± 0.381 | 33.946 ± 0.110 | 15.796 ± 4.331 | 804.413 ± 31.715 |
| full+floor | QLD1 | 40.104 ± 0.619 | 26.732 ± 0.402 | 4.742 ± 0.090 | 418.784 ± 15.327 |
| full+floor | TAS1 | 31.174 ± 0.862 | 17.326 ± 0.427 | 8.920 ± 0.575 | 698.506 ± 9.813 |
| no_tail | NSW1 | 45.566 ± 0.382 | 34.242 ± 0.137 | 14.503 ± 1.722 | 910.266 ± 28.848 |
| no_tail | QLD1 | 39.634 ± 0.351 | 26.859 ± 0.217 | 3.649 ± 0.113 | 507.116 ± 25.856 |
| no_tail | TAS1 | 31.610 ± 1.290 | 17.693 ± 0.627 | 6.777 ± 0.543 | 713.818 ± 9.350 |
| no_tail+floor | NSW1 | 45.577 ± 0.376 | 34.441 ± 0.158 | 21.487 ± 0.737 | 910.266 ± 28.848 |
| no_tail+floor | QLD1 | 39.807 ± 0.206 | 26.917 ± 0.179 | 4.930 ± 0.228 | 507.116 ± 25.856 |
| no_tail+floor | TAS1 | 31.549 ± 1.274 | 17.764 ± 0.625 | 9.596 ± 0.386 | 713.818 ± 9.350 |
| concat | NSW1 | 45.826 ± 0.405 | 34.358 ± 0.311 | 9.752 ± 0.913 | 916.184 ± 34.105 |
| concat | QLD1 | 40.095 ± 0.222 | 28.085 ± 0.212 | 4.192 ± 0.135 | 457.023 ± 22.798 |
| concat | TAS1 | 31.992 ± 0.547 | 17.404 ± 0.126 | 5.568 ± 0.273 | 697.815 ± 5.951 |
| concat+floor | NSW1 | 45.945 ± 0.379 | 34.502 ± 0.286 | 15.140 ± 4.118 | 916.184 ± 34.105 |
| concat+floor | QLD1 | 40.350 ± 0.199 | 28.120 ± 0.220 | 5.328 ± 0.217 | 457.023 ± 22.798 |
| concat+floor | TAS1 | 31.941 ± 0.569 | 17.440 ± 0.108 | 8.596 ± 0.300 | 697.815 ± 5.951 |
| concat_no_tail | NSW1 | 45.896 ± 0.286 | 35.165 ± 0.748 | 13.834 ± 0.827 | 1029.614 ± 25.101 |
| concat_no_tail | QLD1 | 40.553 ± 0.066 | 28.585 ± 0.705 | 4.703 ± 0.564 | 562.768 ± 21.874 |
| concat_no_tail | TAS1 | 31.960 ± 0.587 | 17.511 ± 0.117 | 6.906 ± 0.547 | 707.691 ± 4.891 |
| no_ic | NSW1 | 45.686 ± 0.276 | 33.288 ± 0.210 | 8.175 ± 2.210 | 819.302 ± 35.105 |
| no_ic | QLD1 | 40.609 ± 0.425 | 26.755 ± 0.323 | 3.833 ± 0.135 | 432.380 ± 52.898 |
| no_ic | TAS1 | 31.662 ± 0.204 | 17.353 ± 0.449 | 5.728 ± 0.403 | 703.677 ± 8.130 |
| no_fields | NSW1 | 46.429 ± 1.246 | 33.455 ± 0.426 | 7.973 ± 0.767 | 920.829 ± 109.742 |
| no_fields | QLD1 | 41.044 ± 1.114 | 29.240 ± 0.788 | 4.138 ± 0.407 | 408.820 ± 11.006 |
| no_fields | TAS1 | 32.580 ± 0.532 | 17.370 ± 0.187 | 6.393 ± 0.456 | 678.720 ± 3.337 |
| uniform | NSW1 | 45.710 ± 0.370 | 34.845 ± 0.635 | 10.749 ± 1.699 | 980.505 ± 41.686 |
| uniform | QLD1 | 41.208 ± 0.577 | 30.098 ± 0.149 | 3.866 ± 0.279 | 542.800 ± 4.022 |
| uniform | TAS1 | 31.890 ± 1.174 | 17.328 ± 0.243 | 5.641 ± 0.299 | 677.225 ± 5.336 |
| no_quantile_gate | NSW1 | 45.991 ± 0.361 | 34.592 ± 0.807 | 10.708 ± 1.550 | 930.635 ± 75.251 |
| no_quantile_gate | QLD1 | 40.321 ± 0.752 | 27.398 ± 0.660 | 3.909 ± 0.034 | 440.533 ± 54.911 |
| no_quantile_gate | TAS1 | 31.782 ± 1.136 | 17.276 ± 0.337 | 6.096 ± 0.457 | 685.522 ± 1.792 |
| no_event | NSW1 | 46.041 ± 0.592 | 34.506 ± 0.297 | 9.529 ± 0.321 | 968.855 ± 107.366 |
| no_event | QLD1 | 40.492 ± 0.100 | 29.322 ± 0.216 | 4.367 ± 0.237 | 471.093 ± 38.326 |
| no_event | TAS1 | 32.687 ± 0.626 | 17.223 ± 0.133 | 6.198 ± 1.127 | 674.817 ± 6.629 |
| raw_history | NSW1 | 45.630 ± 0.104 | 33.955 ± 0.106 | 11.795 ± 1.970 | 827.132 ± 32.263 |
| raw_history | QLD1 | 41.073 ± 0.268 | 26.904 ± 0.051 | 3.644 ± 0.130 | 404.918 ± 20.828 |
| raw_history | TAS1 | 32.045 ± 0.507 | 17.196 ± 0.205 | 5.787 ± 0.804 | 688.972 ± 2.671 |
| no_pd | NSW1 | 49.333 ± 0.231 | 36.083 ± 0.481 | 8.560 ± 1.276 | 924.144 ± 89.292 |
| no_pd | QLD1 | 44.754 ± 0.759 | 28.482 ± 0.588 | 4.512 ± 0.287 | 414.040 ± 20.318 |
| no_pd | TAS1 | 34.532 ± 0.567 | 19.270 ± 0.352 | 5.202 ± 0.657 | 721.270 ± 0.748 |
| no_scarcity | NSW1 | 46.078 ± 0.616 | 34.455 ± 0.076 | 10.729 ± 1.209 | 950.817 ± 26.533 |
| no_scarcity | QLD1 | 41.185 ± 1.192 | 27.445 ± 0.917 | 4.060 ± 0.366 | 399.455 ± 63.299 |
| no_scarcity | TAS1 | 30.658 ± 0.107 | 17.021 ± 0.198 | 5.926 ± 0.737 | 692.981 ± 1.865 |
| gabor | NSW1 | 45.468 ± 0.051 | 34.113 ± 0.052 | 11.393 ± 1.773 | 830.103 ± 31.112 |
| gabor | QLD1 | 41.587 ± 0.151 | 27.086 ± 0.214 | 3.833 ± 0.159 | 414.374 ± 25.933 |
| gabor | TAS1 | 32.147 ± 0.733 | 17.453 ± 0.437 | 5.822 ± 0.877 | 695.585 ± 8.442 |
| adaptive | NSW1 | 46.257 ± 0.444 | 33.993 ± 0.406 | 8.430 ± 1.077 | 816.911 ± 25.955 |
| adaptive | QLD1 | 45.490 ± 3.435 | 27.657 ± 0.704 | 4.080 ± 0.276 | 488.653 ± 75.602 |
| adaptive | TAS1 | 31.283 ± 0.147 | 17.296 ± 0.332 | 5.585 ± 0.232 | 690.404 ± 2.365 |
| echo | NSW1 | 45.603 ± 0.087 | 33.780 ± 0.278 | 10.388 ± 0.895 | 805.561 ± 29.514 |
| echo | QLD1 | 42.284 ± 2.343 | 27.328 ± 0.259 | 4.201 ± 0.087 | 433.490 ± 23.537 |
| echo | TAS1 | 31.562 ± 0.490 | 17.636 ± 0.164 | 6.298 ± 0.666 | 705.566 ± 8.154 |
| D_all_inputs | NSW1 | 45.895 ± 0.279 | 35.180 ± 0.742 | 14.266 ± 1.153 | 1025.801 ± 25.672 |
| D_all_inputs | QLD1 | 40.805 ± 0.405 | 28.253 ± 0.413 | 4.407 ± 0.176 | 549.551 ± 15.515 |
| D_all_inputs | TAS1 | 32.076 ± 0.514 | 17.537 ± 0.123 | 6.847 ± 0.498 | 706.620 ± 5.840 |
| C_predispatch | NSW1 | 45.974 ± 0.229 | 34.859 ± 0.456 | 12.117 ± 2.132 | 1029.428 ± 25.830 |
| C_predispatch | QLD1 | 40.489 ± 0.413 | 27.737 ± 0.384 | 4.783 ± 1.043 | 515.433 ± 31.206 |
| C_predispatch | TAS1 | 31.692 ± 0.495 | 17.397 ± 0.099 | 7.211 ± 0.686 | 708.634 ± 1.613 |
| B_scarcity | NSW1 | 51.690 ± 0.347 | 34.337 ± 0.262 | 6.539 ± 0.707 | 841.577 ± 8.060 |
| B_scarcity | QLD1 | 46.794 ± 0.549 | 29.960 ± 0.257 | 4.937 ± 0.577 | 487.443 ± 16.838 |
| B_scarcity | TAS1 | 35.043 ± 0.442 | 18.972 ± 0.249 | 5.990 ± 0.269 | 727.242 ± 7.581 |
| A_history | NSW1 | 53.612 ± 0.220 | 37.771 ± 0.172 | 8.074 ± 0.376 | 1066.746 ± 5.590 |
| A_history | QLD1 | 46.752 ± 0.232 | 32.165 ± 0.357 | 5.177 ± 0.862 | 639.171 ± 7.018 |
| A_history | TAS1 | 36.419 ± 0.766 | 19.713 ± 0.291 | 7.561 ± 0.631 | 734.191 ± 0.561 |
| pd_calibrator | NSW1 | 45.974 ± 0.225 | 34.828 ± 0.381 | 11.562 ± 2.271 | 1029.229 ± 23.492 |
| pd_calibrator | QLD1 | 40.543 ± 0.479 | 27.444 ± 0.621 | 4.530 ± 1.337 | 508.182 ± 39.494 |
| pd_calibrator | TAS1 | 32.014 ± 0.276 | 17.423 ± 0.136 | 7.175 ± 0.641 | 709.324 ± 1.727 |
| pd_calibrator+floor | NSW1 | 45.985 ± 0.207 | 34.955 ± 0.355 | 16.166 ± 4.575 | 1029.229 ± 23.492 |
| pd_calibrator+floor | QLD1 | 40.612 ± 0.636 | 27.507 ± 0.634 | 5.897 ± 1.499 | 508.182 ± 39.494 |
| pd_calibrator+floor | TAS1 | 31.998 ± 0.269 | 17.481 ± 0.131 | 9.706 ± 0.451 | 709.324 ± 1.727 |
| pd_nofields | NSW1 | 46.995 ± 0.946 | 33.614 ± 0.725 | 8.496 ± 1.774 | 1021.029 ± 47.795 |
| pd_nofields | QLD1 | 40.774 ± 0.655 | 28.328 ± 0.282 | 3.182 ± 0.176 | 464.137 ± 9.065 |
| pd_nofields | TAS1 | 33.055 ± 0.539 | 18.150 ± 0.122 | 8.402 ± 0.323 | 701.213 ± 10.615 |
| pd_nofields+floor | NSW1 | 47.003 ± 0.958 | 33.614 ± 0.725 | 8.496 ± 1.774 | 1021.029 ± 47.795 |
| pd_nofields+floor | QLD1 | 41.070 ± 0.620 | 28.209 ± 0.275 | 2.260 ± 0.088 | 464.137 ± 9.065 |
| pd_nofields+floor | TAS1 | 33.050 ± 0.540 | 18.200 ± 0.112 | 9.912 ± 1.501 | 701.213 ± 10.615 |
| event_joint_dgf | NSW1 | 45.640 ± 0.598 | 33.477 ± 0.336 | 8.751 ± 0.936 | 792.245 ± 4.113 |
| event_joint_dgf | QLD1 | 39.952 ± 0.862 | 27.355 ± 0.362 | 4.152 ± 0.254 | 427.866 ± 44.634 |
| event_joint_dgf | TAS1 | 31.307 ± 0.600 | 17.572 ± 0.045 | 5.878 ± 1.281 | 703.402 ± 8.101 |
| event_frozen_dgf | NSW1 | 45.572 ± 0.397 | 33.793 ± 0.017 | 10.565 ± 1.313 | 804.413 ± 31.715 |
| event_frozen_dgf | QLD1 | 39.990 ± 0.715 | 26.737 ± 0.406 | 3.630 ± 0.174 | 418.784 ± 15.327 |
| event_frozen_dgf | TAS1 | 31.207 ± 0.850 | 17.278 ± 0.427 | 5.887 ± 0.776 | 698.506 ± 9.813 |
| event_joint_raw | NSW1 | 45.923 ± 0.588 | 33.887 ± 0.372 | 9.832 ± 2.336 | 816.590 ± 27.401 |
| event_joint_raw | QLD1 | 39.760 ± 0.337 | 26.628 ± 0.082 | 3.524 ± 0.030 | 425.089 ± 28.870 |
| event_joint_raw | TAS1 | 32.056 ± 1.447 | 17.828 ± 0.271 | 5.733 ± 0.608 | 699.017 ± 10.716 |
| event_frozen_raw | NSW1 | 45.572 ± 0.397 | 33.793 ± 0.017 | 10.565 ± 1.313 | 804.413 ± 31.715 |
| event_frozen_raw | QLD1 | 39.990 ± 0.715 | 26.737 ± 0.406 | 3.630 ± 0.174 | 418.784 ± 15.327 |
| event_frozen_raw | TAS1 | 31.207 ± 0.850 | 17.278 ± 0.427 | 5.887 ± 0.776 | 698.506 ± 9.813 |
| event_joint_none | NSW1 | 45.844 ± 0.856 | 33.503 ± 0.124 | 9.970 ± 0.646 | 791.697 ± 9.638 |
| event_joint_none | QLD1 | 40.270 ± 0.103 | 26.815 ± 0.303 | 3.512 ± 0.030 | 420.345 ± 32.717 |
| event_joint_none | TAS1 | 32.333 ± 0.524 | 17.479 ± 0.187 | 5.809 ± 0.133 | 694.160 ± 10.751 |
| event_frozen_none | NSW1 | 45.572 ± 0.397 | 33.793 ± 0.017 | 10.565 ± 1.313 | 804.413 ± 31.715 |
| event_frozen_none | QLD1 | 39.990 ± 0.715 | 26.737 ± 0.406 | 3.630 ± 0.174 | 418.784 ± 15.327 |
| event_frozen_none | TAS1 | 31.207 ± 0.850 | 17.278 ± 0.427 | 5.887 ± 0.776 | 698.506 ± 9.813 |
| event_joint_raw_no_extremes | NSW1 | 46.090 ± 0.784 | 33.567 ± 0.241 | 9.312 ± 1.139 | 789.712 ± 11.770 |
| event_joint_raw_no_extremes | QLD1 | 40.915 ± 0.586 | 27.236 ± 0.231 | 4.133 ± 0.561 | 430.531 ± 10.344 |
| event_joint_raw_no_extremes | TAS1 | 31.933 ± 0.563 | 17.505 ± 0.270 | 5.838 ± 0.478 | 695.854 ± 3.302 |
| xgboost | NSW1 | 45.050 ± 0.030 | 32.837 ± 0.038 | 18.198 ± 0.071 | 912.108 ± 6.313 |
| xgboost | QLD1 | 39.936 ± 0.008 | 26.957 ± 0.015 | 5.239 ± 0.035 | 434.787 ± 1.657 |
| xgboost | TAS1 | 28.919 ± 0.037 | 16.787 ± 0.024 | 9.388 ± 0.032 | 668.381 ± 0.749 |
| xgboost+floor | NSW1 | 45.054 ± 0.026 | 32.837 ± 0.038 | 18.198 ± 0.071 | 912.108 ± 6.313 |
| xgboost+floor | QLD1 | 39.936 ± 0.007 | 26.881 ± 0.016 | 4.366 ± 0.035 | 434.787 ± 1.657 |
| xgboost+floor | TAS1 | 28.919 ± 0.037 | 16.787 ± 0.024 | 9.386 ± 0.034 | 668.381 ± 0.749 |
| linear | NSW1 | 48.629 ± 0.327 | 33.722 ± 0.130 | 15.136 ± 1.177 | 887.643 ± 26.637 |
| linear | QLD1 | 44.568 ± 0.654 | 28.911 ± 0.574 | 5.378 ± 1.023 | 580.358 ± 25.081 |
| linear | TAS1 | 31.832 ± 0.261 | 17.931 ± 0.117 | 8.389 ± 0.255 | 727.508 ± 2.326 |
| dlinear | NSW1 | 48.834 ± 0.725 | 33.996 ± 0.530 | 14.646 ± 0.476 | 911.527 ± 61.133 |
| dlinear | QLD1 | 44.888 ± 0.260 | 29.127 ± 0.173 | 5.022 ± 0.471 | 592.877 ± 2.742 |
| dlinear | TAS1 | 31.771 ± 0.229 | 17.927 ± 0.093 | 8.103 ± 0.227 | 728.366 ± 2.730 |
| patchtst | NSW1 | 79.774 ± 3.622 | 48.252 ± 1.719 | 10.080 ± 0.252 | 1000.955 ± 41.008 |
| patchtst | QLD1 | 74.879 ± 1.166 | 43.538 ± 0.344 | 9.460 ± 0.045 | 589.371 ± 1.487 |
| patchtst | TAS1 | 34.861 ± 0.114 | 19.407 ± 0.037 | 6.433 ± 0.218 | 728.163 ± 0.609 |
| gatets | NSW1 | 56.505 ± 0.239 | 39.216 ± 0.113 | 23.388 ± 1.220 | 1103.563 ± 7.033 |
| gatets | QLD1 | 52.478 ± 1.100 | 32.926 ± 0.124 | 6.365 ± 0.527 | 646.200 ± 7.516 |
| gatets | TAS1 | 36.661 ± 0.326 | 19.771 ± 0.052 | 10.719 ± 0.547 | 696.667 ± 11.237 |
| gatets_adapter | NSW1 | 48.266 ± 0.204 | 33.045 ± 0.120 | 13.249 ± 1.636 | 842.353 ± 2.267 |
| gatets_adapter | QLD1 | 44.403 ± 0.372 | 28.308 ± 0.104 | 4.182 ± 0.615 | 485.901 ± 1.816 |
| gatets_adapter | TAS1 | 32.447 ± 0.150 | 17.578 ± 0.107 | 6.517 ± 0.098 | 688.597 ± 3.746 |

### Paired differences against full

Candidate minus full; negative favors the candidate. Holm adjustment spans all comparisons, metrics and regions within this protocol and block length.

| Candidate | Block hours | Metric | Difference | 95% CI | Holm p |
|---|---:|---|---:|---|---:|
| full+floor | 168 | mae | 0.045 | [0.017, 0.072] | 0.5894 |
| full+floor | 168 | crps_quantile_approx | 0.065 | [0.019, 0.116] | 1 |
| no_tail | 168 | mae | 0.014 | [-0.062, 0.085] | 1 |
| no_tail | 168 | crps_quantile_approx | 0.329 | [0.006, 0.715] | 1 |
| no_tail+floor | 168 | mae | 0.055 | [-0.018, 0.122] | 1 |
| no_tail+floor | 168 | crps_quantile_approx | 0.438 | [0.112, 0.820] | 1 |
| concat | 168 | mae | 0.381 | [0.217, 0.560] | 0.0602 |
| concat | 168 | crps_quantile_approx | 0.680 | [0.352, 1.088] | 0.6782 |
| concat+floor | 168 | mae | 0.489 | [0.328, 0.664] | 0.0492 |
| concat+floor | 168 | crps_quantile_approx | 0.751 | [0.421, 1.158] | 0.3016 |
| concat_no_tail | 168 | mae | 0.547 | [0.372, 0.752] | 0.0492 |
| concat_no_tail | 168 | crps_quantile_approx | 1.151 | [0.555, 1.885] | 0.7456 |
| no_ic | 168 | mae | 0.396 | [0.126, 0.729] | 1 |
| no_ic | 168 | crps_quantile_approx | -0.137 | [-0.269, -0.000] | 1 |
| no_fields | 168 | mae | 1.095 | [0.834, 1.400] | 0.0492 |
| no_fields | 168 | crps_quantile_approx | 0.752 | [0.370, 1.181] | 0.1726 |
| uniform | 168 | mae | 0.680 | [0.454, 0.926] | 0.0492 |
| uniform | 168 | crps_quantile_approx | 1.488 | [0.931, 2.178] | 0.0602 |
| no_quantile_gate | 168 | mae | 0.441 | [0.322, 0.578] | 0.0492 |
| no_quantile_gate | 168 | crps_quantile_approx | 0.486 | [0.229, 0.798] | 0.8492 |
| no_event | 168 | mae | 0.817 | [0.521, 1.156] | 0.0492 |
| no_event | 168 | crps_quantile_approx | 1.081 | [0.590, 1.682] | 0.322 |
| raw_history | 168 | mae | 0.660 | [0.481, 0.848] | 0.0492 |
| raw_history | 168 | crps_quantile_approx | 0.083 | [-0.032, 0.188] | 1 |
| no_pd | 168 | mae | 3.950 | [3.496, 4.450] | 0.0492 |
| no_pd | 168 | crps_quantile_approx | 2.009 | [1.637, 2.433] | 0.0492 |
| no_scarcity | 168 | mae | 0.384 | [0.258, 0.507] | 0.0492 |
| no_scarcity | 168 | crps_quantile_approx | 0.371 | [0.052, 0.714] | 1 |
| gabor | 168 | mae | 0.811 | [0.617, 1.028] | 0.0492 |
| gabor | 168 | crps_quantile_approx | 0.281 | [0.170, 0.388] | 0.0492 |
| adaptive | 168 | mae | 2.087 | [1.553, 2.720] | 0.0492 |
| adaptive | 168 | crps_quantile_approx | 0.379 | [0.216, 0.572] | 0.0831 |
| echo | 168 | mae | 0.894 | [0.575, 1.284] | 0.0602 |
| echo | 168 | crps_quantile_approx | 0.312 | [0.163, 0.485] | 0.3825 |
| D_all_inputs | 168 | mae | 0.669 | [0.472, 0.899] | 0.0492 |
| D_all_inputs | 168 | crps_quantile_approx | 1.054 | [0.484, 1.753] | 0.9039 |
| C_predispatch | 168 | mae | 0.462 | [0.265, 0.681] | 0.0602 |
| C_predispatch | 168 | crps_quantile_approx | 0.728 | [0.186, 1.389] | 1 |
| B_scarcity | 168 | mae | 5.586 | [5.059, 6.184] | 0.0492 |
| B_scarcity | 168 | crps_quantile_approx | 1.820 | [1.455, 2.301] | 0.0492 |
| A_history | 168 | mae | 6.672 | [6.054, 7.390] | 0.0492 |
| A_history | 168 | crps_quantile_approx | 3.947 | [3.126, 4.995] | 0.0492 |
| pd_calibrator | 168 | mae | 0.587 | [0.373, 0.831] | 0.0492 |
| pd_calibrator | 168 | crps_quantile_approx | 0.629 | [0.102, 1.269] | 1 |
| pd_calibrator+floor | 168 | mae | 0.609 | [0.397, 0.851] | 0.0492 |
| pd_calibrator+floor | 168 | crps_quantile_approx | 0.712 | [0.188, 1.351] | 1 |
| pd_nofields | 168 | mae | 1.351 | [0.974, 1.808] | 0.0492 |
| pd_nofields | 168 | crps_quantile_approx | 0.761 | [0.245, 1.388] | 1 |
| pd_nofields+floor | 168 | mae | 1.451 | [1.093, 1.894] | 0.0492 |
| pd_nofields+floor | 168 | crps_quantile_approx | 0.738 | [0.222, 1.364] | 1 |
| event_joint_dgf | 168 | mae | 0.043 | [-0.029, 0.114] | 1 |
| event_joint_dgf | 168 | crps_quantile_approx | 0.199 | [0.096, 0.309] | 0.2169 |
| event_joint_raw | 168 | mae | 0.323 | [0.212, 0.439] | 0.0492 |
| event_joint_raw | 168 | crps_quantile_approx | 0.178 | [0.102, 0.259] | 0.0492 |
| event_joint_none | 168 | mae | 0.559 | [0.419, 0.715] | 0.0492 |
| event_joint_none | 168 | crps_quantile_approx | -0.004 | [-0.118, 0.099] | 1 |
| event_joint_raw_no_extremes | 168 | mae | 0.723 | [0.534, 0.958] | 0.0492 |
| event_joint_raw_no_extremes | 168 | crps_quantile_approx | 0.167 | [0.086, 0.252] | 0.1076 |
| xgboost | 168 | mae | -0.955 | [-1.317, -0.596] | 0.0492 |
| xgboost | 168 | crps_quantile_approx | -0.409 | [-0.803, 0.044] | 1 |
| xgboost+floor | 168 | mae | -0.953 | [-1.316, -0.595] | 0.0492 |
| xgboost+floor | 168 | crps_quantile_approx | -0.434 | [-0.830, 0.018] | 1 |
| linear | 168 | mae | 2.753 | [1.982, 3.691] | 0.0492 |
| linear | 168 | crps_quantile_approx | 0.919 | [0.290, 1.686] | 1 |
| dlinear | 168 | mae | 2.908 | [2.142, 3.847] | 0.0492 |
| dlinear | 168 | crps_quantile_approx | 1.081 | [0.424, 1.881] | 1 |
| patchtst | 168 | mae | 24.248 | [22.984, 25.379] | 0.0492 |
| patchtst | 168 | crps_quantile_approx | 11.130 | [10.213, 12.112] | 0.0492 |
| gatets | 168 | mae | 9.625 | [8.864, 10.523] | 0.0492 |
| gatets | 168 | crps_quantile_approx | 4.702 | [3.843, 5.802] | 0.0492 |
| gatets_adapter | 168 | mae | 2.783 | [2.194, 3.450] | 0.0492 |
| gatets_adapter | 168 | crps_quantile_approx | 0.374 | [-0.013, 0.855] | 1 |
| no_tail | 24 | mae | 0.014 | [-0.041, 0.066] | 1 |
| no_tail | 24 | crps_quantile_approx | 0.329 | [0.059, 0.648] | 1 |
| no_tail+floor | 24 | mae | 0.055 | [-0.000, 0.107] | 1 |
| no_tail+floor | 24 | crps_quantile_approx | 0.438 | [0.164, 0.759] | 0.3564 |
| concat | 24 | mae | 0.381 | [0.280, 0.486] | 0.012 |
| concat | 24 | crps_quantile_approx | 0.680 | [0.396, 1.029] | 0.0336 |
| concat+floor | 24 | mae | 0.489 | [0.387, 0.594] | 0.012 |
| concat+floor | 24 | crps_quantile_approx | 0.751 | [0.466, 1.101] | 0.012 |
| no_ic | 24 | mae | 0.396 | [0.229, 0.584] | 0.012 |
| no_ic | 24 | crps_quantile_approx | -0.137 | [-0.226, -0.042] | 0.2737 |
| xgboost | 24 | mae | -0.955 | [-1.251, -0.656] | 0.012 |
| xgboost | 24 | crps_quantile_approx | -0.409 | [-0.721, -0.039] | 1 |
| xgboost+floor | 24 | mae | -0.953 | [-1.249, -0.655] | 0.012 |
| xgboost+floor | 24 | crps_quantile_approx | -0.434 | [-0.746, -0.064] | 0.8205 |

Frozen-event forecasts are listed separately and excluded from paired price-forecast comparisons because their frozen trunks generate the same price forecasts.

### Equal-floor comparisons against full+floor

The same training candidate quantiles and validation selection rule are applied independently to every model, seed and region.

| Candidate | Metric | Difference | 95% CI | Holm p |
|---|---|---:|---|---:|
| no_tail+floor | mae | 0.010 | [-0.065, 0.082] | 1 |
| no_tail+floor | crps_quantile_approx | 0.373 | [0.051, 0.757] | 1 |
| concat+floor | mae | 0.445 | [0.285, 0.622] | 0.0492 |
| concat+floor | crps_quantile_approx | 0.686 | [0.357, 1.096] | 0.6608 |
| pd_calibrator+floor | mae | 0.564 | [0.349, 0.811] | 0.0492 |
| pd_calibrator+floor | crps_quantile_approx | 0.647 | [0.120, 1.289] | 1 |
| pd_nofields+floor | mae | 1.407 | [1.044, 1.854] | 0.0492 |
| pd_nofields+floor | crps_quantile_approx | 0.673 | [0.149, 1.307] | 1 |
| xgboost+floor | mae | -0.998 | [-1.355, -0.641] | 0.0492 |
| xgboost+floor | crps_quantile_approx | -0.499 | [-0.892, -0.042] | 1 |

Case figures: `static/figures/`; cases selected from forecasts by a fixed rule, not by observed outcomes.

## Computational cost

Sum of worker elapsed time: 32.59 hours.

Sum of per-worker elapsed times including data preparation and export; not wall time or exclusive GPU time.

| Job | GPU | Worker seconds |
|---|---|---:|
| static/full/2026/NSW1 | 6 | 345.2 |
| static/full/2026/QLD1 | 6 | 326.1 |
| static/full/2026/TAS1 | 6 | 337.2 |
| static/full/2027/NSW1 | 6 | 392.5 |
| static/full/2027/QLD1 | 7 | 404.6 |
| static/full/2027/TAS1 | 7 | 333.5 |
| static/full/2028/NSW1 | 7 | 358.3 |
| static/full/2028/QLD1 | 7 | 340.0 |
| static/full/2028/TAS1 | 6 | 357.0 |
| static/no_tail/2026/NSW1 | 7 | 327.5 |
| static/no_tail/2026/QLD1 | 6 | 329.8 |
| static/no_tail/2026/TAS1 | 7 | 332.6 |
| static/no_tail/2027/NSW1 | 6 | 350.2 |
| static/no_tail/2027/QLD1 | 7 | 334.8 |
| static/no_tail/2027/TAS1 | 6 | 385.0 |
| static/no_tail/2028/NSW1 | 7 | 377.9 |
| static/no_tail/2028/QLD1 | 7 | 334.3 |
| static/no_tail/2028/TAS1 | 6 | 392.6 |
| static/concat/2026/NSW1 | 7 | 323.2 |
| static/concat/2026/QLD1 | 6 | 326.1 |
| static/concat/2026/TAS1 | 7 | 333.8 |
| static/concat/2027/NSW1 | 6 | 320.8 |
| static/concat/2027/QLD1 | 6 | 391.8 |
| static/concat/2027/TAS1 | 7 | 337.1 |
| static/concat/2028/NSW1 | 7 | 328.4 |
| static/concat/2028/QLD1 | 7 | 332.3 |
| static/concat/2028/TAS1 | 6 | 321.3 |
| static/concat_no_tail/2026/NSW1 | 6 | 320.1 |
| static/concat_no_tail/2026/QLD1 | 7 | 319.6 |
| static/concat_no_tail/2026/TAS1 | 6 | 386.5 |
| static/concat_no_tail/2027/NSW1 | 7 | 335.4 |
| static/concat_no_tail/2027/QLD1 | 6 | 328.2 |
| static/concat_no_tail/2027/TAS1 | 7 | 318.4 |
| static/concat_no_tail/2028/NSW1 | 7 | 322.9 |
| static/concat_no_tail/2028/QLD1 | 6 | 318.7 |
| static/concat_no_tail/2028/TAS1 | 6 | 317.3 |
| static/no_ic/2026/NSW1 | 7 | 346.3 |
| static/no_ic/2026/QLD1 | 6 | 393.4 |
| static/no_ic/2026/TAS1 | 7 | 383.4 |
| static/no_ic/2027/NSW1 | 6 | 346.6 |
| static/no_ic/2027/QLD1 | 7 | 344.3 |
| static/no_ic/2027/TAS1 | 6 | 335.2 |
| static/no_ic/2028/NSW1 | 6 | 330.9 |
| static/no_ic/2028/QLD1 | 7 | 341.0 |
| static/no_ic/2028/TAS1 | 7 | 342.4 |
| static/no_fields/2026/NSW1 | 6 | 304.5 |
| static/no_fields/2026/QLD1 | 7 | 368.1 |
| static/no_fields/2026/TAS1 | 6 | 368.4 |
| static/no_fields/2027/NSW1 | 6 | 295.0 |
| static/no_fields/2027/QLD1 | 7 | 355.5 |
| static/no_fields/2027/TAS1 | 6 | 295.3 |
| static/no_fields/2028/NSW1 | 7 | 300.4 |
| static/no_fields/2028/QLD1 | 7 | 301.9 |
| static/no_fields/2028/TAS1 | 6 | 298.2 |
| static/uniform/2026/NSW1 | 7 | 314.0 |
| static/uniform/2026/QLD1 | 6 | 326.3 |
| static/uniform/2026/TAS1 | 6 | 310.8 |
| static/uniform/2027/NSW1 | 6 | 352.9 |
| static/uniform/2027/QLD1 | 7 | 339.9 |
| static/uniform/2027/TAS1 | 7 | 299.9 |
| static/uniform/2028/NSW1 | 7 | 330.4 |
| static/uniform/2028/QLD1 | 6 | 314.3 |
| static/uniform/2028/TAS1 | 7 | 309.8 |
| static/no_quantile_gate/2026/NSW1 | 6 | 379.3 |
| static/no_quantile_gate/2026/QLD1 | 6 | 327.0 |
| static/no_quantile_gate/2026/TAS1 | 7 | 345.0 |
| static/no_quantile_gate/2027/NSW1 | 6 | 383.5 |
| static/no_quantile_gate/2027/QLD1 | 7 | 312.8 |
| static/no_quantile_gate/2027/TAS1 | 7 | 320.9 |
| static/no_quantile_gate/2028/NSW1 | 6 | 319.1 |
| static/no_quantile_gate/2028/QLD1 | 7 | 320.9 |
| static/no_quantile_gate/2028/TAS1 | 6 | 369.7 |
| static/no_event/2026/NSW1 | 6 | 334.8 |
| static/no_event/2026/QLD1 | 7 | 315.0 |
| static/no_event/2026/TAS1 | 7 | 349.2 |
| static/no_event/2027/NSW1 | 7 | 327.3 |
| static/no_event/2027/QLD1 | 6 | 330.4 |
| static/no_event/2027/TAS1 | 6 | 321.8 |
| static/no_event/2028/NSW1 | 7 | 325.0 |
| static/no_event/2028/QLD1 | 6 | 318.8 |
| static/no_event/2028/TAS1 | 7 | 318.5 |
| static/raw_history/2026/NSW1 | 6 | 338.3 |
| static/raw_history/2026/QLD1 | 7 | 401.6 |
| static/raw_history/2026/TAS1 | 7 | 356.5 |
| static/raw_history/2027/NSW1 | 6 | 336.0 |
| static/raw_history/2027/QLD1 | 6 | 336.5 |
| static/raw_history/2027/TAS1 | 7 | 350.4 |
| static/raw_history/2028/NSW1 | 6 | 339.9 |
| static/raw_history/2028/QLD1 | 7 | 376.9 |
| static/raw_history/2028/TAS1 | 6 | 342.8 |
| static/no_pd/2026/NSW1 | 6 | 340.4 |
| static/no_pd/2026/QLD1 | 7 | 398.5 |
| static/no_pd/2026/TAS1 | 7 | 337.1 |
| static/no_pd/2027/NSW1 | 6 | 343.2 |
| static/no_pd/2027/QLD1 | 7 | 330.3 |
| static/no_pd/2027/TAS1 | 6 | 339.7 |
| static/no_pd/2028/NSW1 | 7 | 343.5 |
| static/no_pd/2028/QLD1 | 6 | 343.3 |
| static/no_pd/2028/TAS1 | 6 | 370.0 |
| static/no_scarcity/2026/NSW1 | 7 | 331.0 |
| static/no_scarcity/2026/QLD1 | 6 | 372.7 |
| static/no_scarcity/2026/TAS1 | 7 | 378.3 |
| static/no_scarcity/2027/NSW1 | 7 | 349.4 |
| static/no_scarcity/2027/QLD1 | 6 | 419.7 |
| static/no_scarcity/2027/TAS1 | 7 | 343.2 |
| static/no_scarcity/2028/NSW1 | 6 | 362.9 |
| static/no_scarcity/2028/QLD1 | 7 | 355.5 |
| static/no_scarcity/2028/TAS1 | 6 | 407.3 |
| static/gabor/2026/NSW1 | 6 | 337.3 |
| static/gabor/2026/QLD1 | 7 | 345.9 |
| static/gabor/2026/TAS1 | 7 | 337.3 |
| static/gabor/2027/NSW1 | 7 | 358.2 |
| static/gabor/2027/QLD1 | 6 | 373.2 |
| static/gabor/2027/TAS1 | 6 | 393.0 |
| static/gabor/2028/NSW1 | 7 | 392.3 |
| static/gabor/2028/QLD1 | 6 | 340.5 |
| static/gabor/2028/TAS1 | 6 | 400.2 |
| static/adaptive/2026/NSW1 | 7 | 482.3 |
| static/adaptive/2026/QLD1 | 7 | 481.9 |
| static/adaptive/2026/TAS1 | 7 | 493.7 |
| static/adaptive/2027/NSW1 | 6 | 522.0 |
| static/adaptive/2027/QLD1 | 6 | 525.1 |
| static/adaptive/2027/TAS1 | 6 | 513.4 |
| static/adaptive/2028/NSW1 | 7 | 482.3 |
| static/adaptive/2028/QLD1 | 6 | 555.6 |
| static/adaptive/2028/TAS1 | 7 | 450.1 |
| static/echo/2026/NSW1 | 7 | 439.7 |
| static/echo/2026/QLD1 | 7 | 399.6 |
| static/echo/2026/TAS1 | 6 | 404.9 |
| static/echo/2027/NSW1 | 6 | 408.1 |
| static/echo/2027/QLD1 | 7 | 402.3 |
| static/echo/2027/TAS1 | 6 | 449.2 |
| static/echo/2028/NSW1 | 7 | 458.9 |
| static/echo/2028/QLD1 | 6 | 395.1 |
| static/echo/2028/TAS1 | 7 | 409.4 |
| static/D_all_inputs/2026/NSW1 | 7 | 316.0 |
| static/D_all_inputs/2026/QLD1 | 6 | 317.6 |
| static/D_all_inputs/2026/TAS1 | 6 | 309.8 |
| static/D_all_inputs/2027/NSW1 | 7 | 317.8 |
| static/D_all_inputs/2027/QLD1 | 6 | 384.0 |
| static/D_all_inputs/2027/TAS1 | 6 | 370.7 |
| static/D_all_inputs/2028/NSW1 | 7 | 313.4 |
| static/D_all_inputs/2028/QLD1 | 7 | 307.3 |
| static/D_all_inputs/2028/TAS1 | 7 | 312.6 |
| static/C_predispatch/2026/NSW1 | 6 | 322.4 |
| static/C_predispatch/2026/QLD1 | 6 | 314.1 |
| static/C_predispatch/2026/TAS1 | 7 | 314.6 |
| static/C_predispatch/2027/NSW1 | 6 | 315.8 |
| static/C_predispatch/2027/QLD1 | 7 | 316.1 |
| static/C_predispatch/2027/TAS1 | 7 | 335.8 |
| static/C_predispatch/2028/NSW1 | 6 | 323.5 |
| static/C_predispatch/2028/QLD1 | 7 | 317.8 |
| static/C_predispatch/2028/TAS1 | 6 | 373.9 |
| static/B_scarcity/2026/NSW1 | 6 | 318.2 |
| static/B_scarcity/2026/QLD1 | 7 | 311.5 |
| static/B_scarcity/2026/TAS1 | 6 | 315.4 |
| static/B_scarcity/2027/NSW1 | 7 | 323.2 |
| static/B_scarcity/2027/QLD1 | 6 | 320.1 |
| static/B_scarcity/2027/TAS1 | 7 | 305.7 |
| static/B_scarcity/2028/NSW1 | 7 | 319.2 |
| static/B_scarcity/2028/QLD1 | 6 | 348.7 |
| static/B_scarcity/2028/TAS1 | 7 | 320.3 |
| static/A_history/2026/NSW1 | 6 | 367.0 |
| static/A_history/2026/QLD1 | 6 | 314.3 |
| static/A_history/2026/TAS1 | 7 | 309.6 |
| static/A_history/2027/NSW1 | 7 | 313.8 |
| static/A_history/2027/QLD1 | 6 | 326.8 |
| static/A_history/2027/TAS1 | 7 | 325.0 |
| static/A_history/2028/NSW1 | 6 | 309.0 |
| static/A_history/2028/QLD1 | 7 | 311.8 |
| static/A_history/2028/TAS1 | 6 | 368.6 |
| static/pd_calibrator/2026/NSW1 | 6 | 339.6 |
| static/pd_calibrator/2026/QLD1 | 7 | 337.8 |
| static/pd_calibrator/2026/TAS1 | 7 | 327.4 |
| static/pd_calibrator/2027/NSW1 | 6 | 317.2 |
| static/pd_calibrator/2027/QLD1 | 7 | 333.8 |
| static/pd_calibrator/2027/TAS1 | 6 | 324.2 |
| static/pd_calibrator/2028/NSW1 | 7 | 330.7 |
| static/pd_calibrator/2028/QLD1 | 6 | 379.8 |
| static/pd_calibrator/2028/TAS1 | 6 | 323.4 |
| static/pd_nofields/2026/NSW1 | 7 | 305.7 |
| static/pd_nofields/2026/QLD1 | 7 | 305.5 |
| static/pd_nofields/2026/TAS1 | 6 | 358.9 |
| static/pd_nofields/2027/NSW1 | 7 | 294.5 |
| static/pd_nofields/2027/QLD1 | 6 | 288.5 |
| static/pd_nofields/2027/TAS1 | 7 | 289.9 |
| static/pd_nofields/2028/NSW1 | 6 | 284.0 |
| static/pd_nofields/2028/QLD1 | 7 | 331.4 |
| static/pd_nofields/2028/TAS1 | 7 | 296.0 |
| static/event_joint_dgf/2026/NSW1 | 6 | 454.1 |
| static/event_joint_dgf/2026/QLD1 | 7 | 513.5 |
| static/event_joint_dgf/2026/TAS1 | 6 | 471.6 |
| static/event_joint_dgf/2027/NSW1 | 6 | 513.7 |
| static/event_joint_dgf/2027/QLD1 | 7 | 458.5 |
| static/event_joint_dgf/2027/TAS1 | 6 | 445.5 |
| static/event_joint_dgf/2028/NSW1 | 7 | 467.9 |
| static/event_joint_dgf/2028/QLD1 | 7 | 495.3 |
| static/event_joint_dgf/2028/TAS1 | 6 | 475.0 |
| static/event_frozen_dgf/2026/NSW1 | 7 | 448.2 |
| static/event_frozen_dgf/2026/QLD1 | 6 | 418.1 |
| static/event_frozen_dgf/2026/TAS1 | 7 | 401.1 |
| static/event_frozen_dgf/2027/NSW1 | 6 | 395.0 |
| static/event_frozen_dgf/2027/QLD1 | 6 | 396.3 |
| static/event_frozen_dgf/2027/TAS1 | 7 | 402.1 |
| static/event_frozen_dgf/2028/NSW1 | 7 | 410.0 |
| static/event_frozen_dgf/2028/QLD1 | 6 | 405.1 |
| static/event_frozen_dgf/2028/TAS1 | 6 | 396.8 |
| static/event_joint_raw/2026/NSW1 | 7 | 499.7 |
| static/event_joint_raw/2026/QLD1 | 7 | 514.6 |
| static/event_joint_raw/2026/TAS1 | 6 | 450.0 |
| static/event_joint_raw/2027/NSW1 | 6 | 479.7 |
| static/event_joint_raw/2027/QLD1 | 7 | 449.5 |
| static/event_joint_raw/2027/TAS1 | 7 | 516.8 |
| static/event_joint_raw/2028/NSW1 | 6 | 443.4 |
| static/event_joint_raw/2028/QLD1 | 6 | 453.3 |
| static/event_joint_raw/2028/TAS1 | 6 | 447.9 |
| static/event_frozen_raw/2026/NSW1 | 7 | 409.6 |
| static/event_frozen_raw/2026/QLD1 | 7 | 390.6 |
| static/event_frozen_raw/2026/TAS1 | 6 | 401.2 |
| static/event_frozen_raw/2027/NSW1 | 7 | 400.2 |
| static/event_frozen_raw/2027/QLD1 | 7 | 460.8 |
| static/event_frozen_raw/2027/TAS1 | 6 | 401.0 |
| static/event_frozen_raw/2028/NSW1 | 6 | 415.1 |
| static/event_frozen_raw/2028/QLD1 | 7 | 401.0 |
| static/event_frozen_raw/2028/TAS1 | 7 | 390.4 |
| static/event_joint_none/2026/NSW1 | 6 | 450.2 |
| static/event_joint_none/2026/QLD1 | 6 | 502.2 |
| static/event_joint_none/2026/TAS1 | 7 | 448.4 |
| static/event_joint_none/2027/NSW1 | 6 | 444.5 |
| static/event_joint_none/2027/QLD1 | 7 | 499.5 |
| static/event_joint_none/2027/TAS1 | 6 | 429.8 |
| static/event_joint_none/2028/NSW1 | 7 | 442.9 |
| static/event_joint_none/2028/QLD1 | 7 | 492.2 |
| static/event_joint_none/2028/TAS1 | 6 | 435.4 |
| static/event_frozen_none/2026/NSW1 | 7 | 388.4 |
| static/event_frozen_none/2026/QLD1 | 6 | 421.8 |
| static/event_frozen_none/2026/TAS1 | 6 | 386.6 |
| static/event_frozen_none/2027/NSW1 | 6 | 385.4 |
| static/event_frozen_none/2027/QLD1 | 7 | 459.4 |
| static/event_frozen_none/2027/TAS1 | 7 | 393.5 |
| static/event_frozen_none/2028/NSW1 | 7 | 395.7 |
| static/event_frozen_none/2028/QLD1 | 6 | 381.7 |
| static/event_frozen_none/2028/TAS1 | 7 | 392.6 |
| static/event_joint_raw_no_extremes/2026/NSW1 | 6 | 447.6 |
| static/event_joint_raw_no_extremes/2026/QLD1 | 6 | 450.7 |
| static/event_joint_raw_no_extremes/2026/TAS1 | 6 | 445.6 |
| static/event_joint_raw_no_extremes/2027/NSW1 | 7 | 462.1 |
| static/event_joint_raw_no_extremes/2027/QLD1 | 7 | 500.4 |
| static/event_joint_raw_no_extremes/2027/TAS1 | 7 | 450.5 |
| static/event_joint_raw_no_extremes/2028/NSW1 | 6 | 503.9 |
| static/event_joint_raw_no_extremes/2028/QLD1 | 7 | 444.7 |
| static/event_joint_raw_no_extremes/2028/TAS1 | 6 | 447.3 |
| static/xgboost/2026/NSW1 | 6 | 719.0 |
| static/xgboost/2026/QLD1 | 6 | 523.9 |
| static/xgboost/2026/TAS1 | 7 | 1405.8 |
| static/xgboost/2027/NSW1 | 7 | 733.4 |
| static/xgboost/2027/QLD1 | 7 | 697.3 |
| static/xgboost/2027/TAS1 | 6 | 1919.1 |
| static/xgboost/2028/NSW1 | 7 | 760.6 |
| static/xgboost/2028/QLD1 | 6 | 569.1 |
| static/xgboost/2028/TAS1 | 6 | 1908.7 |
| static/linear/2026/NSW1 | 6 | 72.6 |
| static/linear/2026/QLD1 | 6 | 83.8 |
| static/linear/2026/TAS1 | 6 | 88.3 |
| static/linear/2027/NSW1 | 7 | 71.1 |
| static/linear/2027/QLD1 | 7 | 77.0 |
| static/linear/2027/TAS1 | 6 | 106.8 |
| static/linear/2028/NSW1 | 7 | 71.6 |
| static/linear/2028/QLD1 | 6 | 86.7 |
| static/linear/2028/TAS1 | 7 | 72.9 |
| static/dlinear/2026/NSW1 | 7 | 80.2 |
| static/dlinear/2026/QLD1 | 6 | 108.7 |
| static/dlinear/2026/TAS1 | 7 | 91.8 |
| static/dlinear/2027/NSW1 | 7 | 76.1 |
| static/dlinear/2027/QLD1 | 6 | 70.4 |
| static/dlinear/2027/TAS1 | 7 | 81.7 |
| static/dlinear/2028/NSW1 | 6 | 78.0 |
| static/dlinear/2028/QLD1 | 7 | 75.7 |
| static/dlinear/2028/TAS1 | 7 | 82.1 |
| static/patchtst/2026/NSW1 | 6 | 163.4 |
| static/patchtst/2026/QLD1 | 7 | 189.0 |
| static/patchtst/2026/TAS1 | 7 | 165.5 |
| static/patchtst/2027/NSW1 | 6 | 165.4 |
| static/patchtst/2027/QLD1 | 7 | 163.2 |
| static/patchtst/2027/TAS1 | 6 | 170.8 |
| static/patchtst/2028/NSW1 | 7 | 204.6 |
| static/patchtst/2028/QLD1 | 7 | 170.1 |
| static/patchtst/2028/TAS1 | 6 | 171.3 |
| static/gatets/2026/NSW1 | 7 | 598.9 |
| static/gatets/2026/QLD1 | 6 | 570.1 |
| static/gatets/2026/TAS1 | 7 | 614.5 |
| static/gatets/2027/NSW1 | 7 | 612.0 |
| static/gatets/2027/QLD1 | 7 | 621.7 |
| static/gatets/2027/TAS1 | 6 | 546.8 |
| static/gatets/2028/NSW1 | 7 | 607.2 |
| static/gatets/2028/QLD1 | 6 | 567.8 |
| static/gatets/2028/TAS1 | 6 | 611.1 |
| static/gatets_adapter/2026/NSW1 | 7 | 620.9 |
| static/gatets_adapter/2026/QLD1 | 6 | 588.2 |
| static/gatets_adapter/2026/TAS1 | 7 | 612.7 |
| static/gatets_adapter/2027/NSW1 | 7 | 636.5 |
| static/gatets_adapter/2027/QLD1 | 6 | 601.3 |
| static/gatets_adapter/2027/TAS1 | 7 | 555.7 |
| static/gatets_adapter/2028/NSW1 | 6 | 590.3 |
| static/gatets_adapter/2028/QLD1 | 6 | 617.7 |
| static/gatets_adapter/2028/TAS1 | 6 | 611.9 |
| static/event_controls/2026/NSW1 | 7 | 64.1 |
| static/event_controls/2026/QLD1 | 7 | 46.4 |
| static/event_controls/2026/TAS1 | 7 | 46.8 |
| static/event_controls/2027/NSW1 | 7 | 48.0 |
| static/event_controls/2027/QLD1 | 7 | 93.3 |
| static/event_controls/2027/TAS1 | 7 | 38.8 |
| static/event_controls/2028/NSW1 | 7 | 46.2 |
| static/event_controls/2028/QLD1 | 7 | 43.1 |
| static/event_controls/2028/TAS1 | 7 | 41.4 |
