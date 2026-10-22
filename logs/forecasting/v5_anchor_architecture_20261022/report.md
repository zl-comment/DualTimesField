# Anchor architecture development comparison

Profile: screen; epochs: 6; seed: 2026.

Selection guards are not guarantees on evaluation periods. Screen results are preliminary.
Raw compares Full-QS checkpoints; guarded compares independently selected shared-alpha checkpoints.
Bias = prediction minus actual. Metrics aggregate forecast origins and all 24 horizons.

| Region/fold | Variant | Mode | Epoch | Alpha | Fallback | Anchor MAE | Full MAE | Full QS | Bias | C80 | C90 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| NSW1/2021 | B | baseline | — | 0 | — | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2021 | balanced | raw | 2 | 1.0 | False | 25.8740 | 25.7867 | 17.6567 | -13.4962 | 0.7069 | 0.8262 |
| NSW1/2021 | balanced | guarded | 5 | 0.25 | False | 25.8327 | 25.7856 | 17.6107 | -13.3219 | 0.7290 | 0.8375 |
| NSW1/2021 | identity_anchor | raw | 3 | 1.0 | False | 25.9673 | 25.7428 | 17.5200 | -12.8887 | 0.7311 | 0.8394 |
| NSW1/2021 | identity_anchor | guarded | 3 | 0.25 | False | 25.9673 | 25.8761 | 17.6213 | -12.8043 | 0.7307 | 0.8386 |
| NSW1/2021 | location_only | raw | 5 | 1.0 | False | 25.9148 | 25.8663 | 17.4932 | -14.3694 | 0.7232 | 0.8464 |
| NSW1/2021 | location_only | guarded | 5 | 0.5 | False | 25.7995 | 25.7429 | 17.5015 | -13.5728 | 0.7386 | 0.8475 |
| NSW1/2021 | separate_heads | raw | 1 | 1.0 | False | 25.9392 | 25.8829 | 17.7371 | -12.7468 | 0.7199 | 0.8294 |
| NSW1/2021 | separate_heads | guarded | 5 | 0.25 | False | 25.8686 | 25.8234 | 17.6712 | -13.3407 | 0.7333 | 0.8408 |
| NSW1/2021 | staged | raw | 5 | 1.0 | False | 25.7961 | 25.7515 | 17.6182 | -13.2926 | 0.7036 | 0.8219 |
| NSW1/2021 | staged | guarded | 4 | 0.0 | True | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2021 | wide_head | raw | 1 | 1.0 | False | 25.9292 | 25.8752 | 17.7636 | -13.0289 | 0.7110 | 0.8251 |
| NSW1/2021 | wide_head | guarded | 1 | 0.0 | True | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2022 | B | baseline | — | 0 | — | 65.9162 | 65.9162 | 37.2070 | -56.1996 | 0.6362 | 0.8069 |
| NSW1/2022 | balanced | raw | 6 | 1.0 | False | 63.5144 | 62.1101 | 35.0812 | -52.0745 | 0.6388 | 0.8173 |
| NSW1/2022 | balanced | guarded | 6 | 1.0 | False | 63.5144 | 62.1101 | 35.0812 | -52.0745 | 0.6388 | 0.8173 |
| NSW1/2022 | identity_anchor | raw | 6 | 1.0 | False | 65.9162 | 61.2149 | 35.2198 | -50.1032 | 0.6526 | 0.8150 |
| NSW1/2022 | identity_anchor | guarded | 6 | 1.0 | False | 65.9162 | 61.2149 | 35.2198 | -50.1032 | 0.6526 | 0.8150 |
| NSW1/2022 | location_only | raw | 6 | 1.0 | False | 62.0363 | 60.3128 | 34.7880 | -49.7557 | 0.6559 | 0.8177 |
| NSW1/2022 | location_only | guarded | 6 | 1.0 | False | 62.0363 | 60.3128 | 34.7880 | -49.7557 | 0.6559 | 0.8177 |
| NSW1/2022 | separate_heads | raw | 6 | 1.0 | False | 63.2654 | 61.9823 | 34.7970 | -51.6635 | 0.6346 | 0.8164 |
| NSW1/2022 | separate_heads | guarded | 6 | 1.0 | False | 63.2654 | 61.9823 | 34.7970 | -51.6635 | 0.6346 | 0.8164 |
| NSW1/2022 | staged | raw | 6 | 1.0 | False | 66.6613 | 65.3161 | 36.9791 | -56.2259 | 0.6321 | 0.8056 |
| NSW1/2022 | staged | guarded | 6 | 1.0 | False | 66.6613 | 65.3161 | 36.9791 | -56.2259 | 0.6321 | 0.8056 |
| NSW1/2022 | wide_head | raw | 5 | 1.0 | False | 63.0483 | 61.5772 | 34.6879 | -51.0530 | 0.6581 | 0.8314 |
| NSW1/2022 | wide_head | guarded | 5 | 1.0 | False | 63.0483 | 61.5772 | 34.6879 | -51.0530 | 0.6581 | 0.8314 |
| QLD1/2021 | B | baseline | — | 0 | — | 45.5272 | 45.5272 | 34.2666 | -29.4881 | 0.7321 | 0.8604 |
| QLD1/2021 | balanced | raw | 4 | 1.0 | False | 45.2951 | 45.0396 | 34.6757 | -30.4749 | 0.6544 | 0.8007 |
| QLD1/2021 | balanced | guarded | 4 | 1.0 | False | 45.2951 | 45.0396 | 34.6757 | -30.4749 | 0.6544 | 0.8007 |
| QLD1/2021 | identity_anchor | raw | 3 | 1.0 | False | 45.5272 | 45.1599 | 34.2131 | -30.4410 | 0.7228 | 0.8517 |
| QLD1/2021 | identity_anchor | guarded | 3 | 1.0 | False | 45.5272 | 45.1599 | 34.2131 | -30.4410 | 0.7228 | 0.8517 |
| QLD1/2021 | location_only | raw | 4 | 1.0 | False | 45.2542 | 44.9942 | 34.0368 | -30.2889 | 0.7186 | 0.8498 |
| QLD1/2021 | location_only | guarded | 5 | 0.5 | False | 45.2951 | 45.0381 | 34.0230 | -30.0922 | 0.7307 | 0.8603 |
| QLD1/2021 | separate_heads | raw | 4 | 1.0 | False | 45.3388 | 44.9955 | 34.7748 | -29.9569 | 0.6751 | 0.8151 |
| QLD1/2021 | separate_heads | guarded | 4 | 1.0 | False | 45.3388 | 44.9955 | 34.7748 | -29.9569 | 0.6751 | 0.8151 |
| QLD1/2021 | staged | raw | 6 | 1.0 | False | 45.3165 | 45.1301 | 34.6169 | -29.5600 | 0.6777 | 0.8160 |
| QLD1/2021 | staged | guarded | 6 | 1.0 | False | 45.3165 | 45.1301 | 34.6169 | -29.5600 | 0.6777 | 0.8160 |
| QLD1/2021 | wide_head | raw | 4 | 1.0 | False | 45.3563 | 45.0214 | 34.6702 | -29.9165 | 0.7016 | 0.8355 |
| QLD1/2021 | wide_head | guarded | 4 | 1.0 | False | 45.3563 | 45.0214 | 34.6702 | -29.9165 | 0.7016 | 0.8355 |
| QLD1/2022 | B | baseline | — | 0 | — | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | balanced | raw | 1 | 1.0 | False | 91.7768 | 88.1403 | 57.6322 | -72.7544 | 0.7673 | 0.8939 |
| QLD1/2022 | balanced | guarded | 1 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | identity_anchor | raw | 5 | 1.0 | False | 91.4881 | 85.9202 | 58.3487 | -68.3077 | 0.7732 | 0.8943 |
| QLD1/2022 | identity_anchor | guarded | 6 | 0.5 | False | 91.4881 | 87.4909 | 58.5867 | -72.6468 | 0.7781 | 0.8988 |
| QLD1/2022 | location_only | raw | 5 | 1.0 | False | 92.8958 | 88.5986 | 58.4819 | -74.7599 | 0.7847 | 0.9039 |
| QLD1/2022 | location_only | guarded | 6 | 0.5 | False | 91.3938 | 88.4712 | 58.4576 | -73.6811 | 0.7833 | 0.9032 |
| QLD1/2022 | separate_heads | raw | 1 | 1.0 | False | 91.7978 | 88.9602 | 57.2880 | -73.6926 | 0.7469 | 0.8828 |
| QLD1/2022 | separate_heads | guarded | 1 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | staged | raw | 4 | 1.0 | False | 92.0594 | 90.4859 | 56.8986 | -77.5048 | 0.7534 | 0.8888 |
| QLD1/2022 | staged | guarded | 4 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | wide_head | raw | 1 | 1.0 | False | 91.4909 | 88.2537 | 56.8182 | -72.5903 | 0.7514 | 0.8850 |
| QLD1/2022 | wide_head | guarded | 1 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| TAS1/2021 | B | baseline | — | 0 | — | 13.1294 | 13.1294 | 7.8151 | 0.4754 | 0.7935 | 0.8882 |
| TAS1/2021 | balanced | raw | 2 | 1.0 | False | 13.5428 | 13.4484 | 7.8200 | -4.1667 | 0.8088 | 0.9085 |
| TAS1/2021 | balanced | guarded | 3 | 0.5 | False | 13.2239 | 13.1338 | 7.7265 | -2.7453 | 0.8110 | 0.9080 |
| TAS1/2021 | identity_anchor | raw | 2 | 1.0 | False | 13.1294 | 13.0706 | 7.7341 | -1.3053 | 0.8040 | 0.8998 |
| TAS1/2021 | identity_anchor | guarded | 2 | 1.0 | False | 13.1294 | 13.0706 | 7.7341 | -1.3053 | 0.8040 | 0.8998 |
| TAS1/2021 | location_only | raw | 2 | 1.0 | False | 13.6009 | 13.4990 | 7.8167 | -4.4923 | 0.8024 | 0.9065 |
| TAS1/2021 | location_only | guarded | 3 | 0.5 | False | 13.2870 | 13.1683 | 7.7278 | -3.0705 | 0.8078 | 0.9058 |
| TAS1/2021 | separate_heads | raw | 2 | 1.0 | False | 13.7080 | 13.5700 | 7.9051 | -4.5373 | 0.8025 | 0.9062 |
| TAS1/2021 | separate_heads | guarded | 3 | 0.5 | False | 13.1938 | 13.1126 | 7.7921 | -2.5818 | 0.7982 | 0.8953 |
| TAS1/2021 | staged | raw | 6 | 1.0 | False | 14.4940 | 14.1760 | 8.0739 | -6.5663 | 0.7968 | 0.9089 |
| TAS1/2021 | staged | guarded | 5 | 0.5 | False | 13.2776 | 13.2235 | 7.7635 | -3.1695 | 0.8112 | 0.9090 |
| TAS1/2021 | wide_head | raw | 2 | 1.0 | False | 13.4289 | 13.3383 | 7.8513 | -3.5926 | 0.8033 | 0.9023 |
| TAS1/2021 | wide_head | guarded | 2 | 1.0 | False | 13.4289 | 13.3383 | 7.8513 | -3.5926 | 0.8033 | 0.9023 |
| TAS1/2022 | B | baseline | — | 0 | — | 57.5496 | 57.5496 | 33.1751 | -39.5122 | 0.6379 | 0.7812 |
| TAS1/2022 | balanced | raw | 4 | 1.0 | False | 55.0924 | 54.4469 | 31.7593 | -32.8882 | 0.6696 | 0.8002 |
| TAS1/2022 | balanced | guarded | 4 | 0.5 | False | 56.2007 | 55.8519 | 32.3309 | -36.2002 | 0.6573 | 0.7926 |
| TAS1/2022 | identity_anchor | raw | 3 | 1.0 | False | 57.5496 | 55.1493 | 32.3868 | -34.5295 | 0.6443 | 0.7822 |
| TAS1/2022 | identity_anchor | guarded | 3 | 1.0 | False | 57.5496 | 55.1493 | 32.3868 | -34.5295 | 0.6443 | 0.7822 |
| TAS1/2022 | location_only | raw | 4 | 1.0 | False | 55.2685 | 54.6155 | 32.1504 | -33.8226 | 0.6492 | 0.7827 |
| TAS1/2022 | location_only | guarded | 4 | 0.5 | False | 56.3132 | 55.9638 | 32.5866 | -36.6674 | 0.6447 | 0.7833 |
| TAS1/2022 | separate_heads | raw | 1 | 1.0 | False | 56.7311 | 55.3255 | 32.3179 | -34.3894 | 0.6498 | 0.7843 |
| TAS1/2022 | separate_heads | guarded | 1 | 0.5 | False | 57.1111 | 56.3518 | 32.6805 | -36.9508 | 0.6450 | 0.7839 |
| TAS1/2022 | staged | raw | 6 | 1.0 | False | 55.2752 | 54.7245 | 31.9546 | -33.5921 | 0.6662 | 0.7979 |
| TAS1/2022 | staged | guarded | 6 | 0.5 | False | 56.2993 | 56.0015 | 32.4565 | -36.5521 | 0.6540 | 0.7903 |
| TAS1/2022 | wide_head | raw | 3 | 1.0 | False | 55.5280 | 54.7230 | 32.2080 | -33.3425 | 0.6430 | 0.7790 |
| TAS1/2022 | wide_head | guarded | 3 | 0.5 | False | 56.4438 | 56.0084 | 32.6011 | -36.4274 | 0.6435 | 0.7811 |

JSON metrics include matched raw/guarded gaps, quarters, 2022H1/H2, Australian seasons, B forecast price bands and their crosses, with group counts.
All checkpoint and alpha decisions precede evaluation. Full history remains available causally.
