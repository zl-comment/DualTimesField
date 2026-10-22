# Anchor balance development comparison

Profile: screen; epochs: 3; seed: 2026.

Selection guards are not guarantees on evaluation periods. Screen results are preliminary.
Raw compares Full-QS checkpoints; guarded compares independently selected shared-alpha checkpoints.
Bias = prediction minus actual. Metrics aggregate forecast origins and all 24 horizons.

| Region/fold | Variant | Mode | Epoch | Alpha | Fallback | Anchor MAE | Full MAE | Full QS | Bias | C80 | C90 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| NSW1/2021 | B | baseline | — | 0 | — | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2021 | legacy | raw | 1 | 1.0 | False | 25.9788 | 25.9219 | 17.7296 | -12.3679 | 0.7154 | 0.8269 |
| NSW1/2021 | legacy | guarded | 3 | 0.25 | False | 25.8539 | 25.8327 | 17.6606 | -13.0154 | 0.7286 | 0.8376 |
| NSW1/2021 | balanced | raw | 2 | 1.0 | False | 25.8740 | 25.7867 | 17.6567 | -13.4962 | 0.7069 | 0.8262 |
| NSW1/2021 | balanced | guarded | 1 | 0.0 | True | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2021 | identity_anchor | raw | 1 | 1.0 | False | 25.9673 | 25.9052 | 17.6396 | -12.6256 | 0.7274 | 0.8364 |
| NSW1/2021 | identity_anchor | guarded | 1 | 0.0 | True | 25.9673 | 25.9673 | 17.6805 | -12.7762 | 0.7279 | 0.8375 |
| NSW1/2022 | B | baseline | — | 0 | — | 65.9162 | 65.9162 | 37.2070 | -56.1996 | 0.6362 | 0.8069 |
| NSW1/2022 | legacy | raw | 3 | 1.0 | False | 65.1890 | 64.1152 | 36.0128 | -54.8065 | 0.6277 | 0.8101 |
| NSW1/2022 | legacy | guarded | 3 | 0.5 | False | 65.5238 | 64.9793 | 36.5551 | -55.5031 | 0.6334 | 0.8094 |
| NSW1/2022 | balanced | raw | 3 | 1.0 | False | 66.5843 | 65.1500 | 36.8313 | -56.1132 | 0.6267 | 0.8022 |
| NSW1/2022 | balanced | guarded | 3 | 1.0 | False | 66.5843 | 65.1500 | 36.8313 | -56.1132 | 0.6267 | 0.8022 |
| NSW1/2022 | identity_anchor | raw | 3 | 1.0 | False | 65.9162 | 63.5915 | 36.1459 | -53.3793 | 0.6455 | 0.8125 |
| NSW1/2022 | identity_anchor | guarded | 3 | 1.0 | False | 65.9162 | 63.5915 | 36.1459 | -53.3793 | 0.6455 | 0.8125 |
| QLD1/2021 | B | baseline | — | 0 | — | 45.5272 | 45.5272 | 34.2666 | -29.4881 | 0.7321 | 0.8604 |
| QLD1/2021 | legacy | raw | 3 | 1.0 | False | 45.3812 | 45.2658 | 35.2109 | -30.9681 | 0.6632 | 0.8092 |
| QLD1/2021 | legacy | guarded | 3 | 1.0 | False | 45.3812 | 45.2658 | 35.2109 | -30.9681 | 0.6632 | 0.8092 |
| QLD1/2021 | balanced | raw | 3 | 1.0 | False | 45.2524 | 45.0647 | 34.6323 | -29.2371 | 0.6499 | 0.7939 |
| QLD1/2021 | balanced | guarded | 3 | 1.0 | False | 45.2524 | 45.0647 | 34.6323 | -29.2371 | 0.6499 | 0.7939 |
| QLD1/2021 | identity_anchor | raw | 3 | 1.0 | False | 45.5272 | 45.0369 | 34.2143 | -27.4956 | 0.7093 | 0.8370 |
| QLD1/2021 | identity_anchor | guarded | 3 | 1.0 | False | 45.5272 | 45.0369 | 34.2143 | -27.4956 | 0.7093 | 0.8370 |
| QLD1/2022 | B | baseline | — | 0 | — | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | legacy | raw | 1 | 1.0 | False | 91.3703 | 89.7988 | 57.4256 | -75.3566 | 0.7572 | 0.8899 |
| QLD1/2022 | legacy | guarded | 1 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | balanced | raw | 1 | 1.0 | False | 91.7768 | 88.1403 | 57.6322 | -72.7544 | 0.7673 | 0.8939 |
| QLD1/2022 | balanced | guarded | 1 | 0.0 | True | 91.4881 | 91.4881 | 59.5152 | -77.8638 | 0.7764 | 0.8997 |
| QLD1/2022 | identity_anchor | raw | 3 | 1.0 | False | 91.4881 | 88.4911 | 58.8434 | -72.9212 | 0.7747 | 0.8975 |
| QLD1/2022 | identity_anchor | guarded | 3 | 0.5 | False | 91.4881 | 89.7155 | 59.0986 | -75.3925 | 0.7764 | 0.8996 |
| TAS1/2021 | B | baseline | — | 0 | — | 13.1294 | 13.1294 | 7.8151 | 0.4754 | 0.7935 | 0.8882 |
| TAS1/2021 | legacy | raw | 2 | 1.0 | False | 13.5815 | 13.5251 | 7.8737 | -4.3174 | 0.8057 | 0.9080 |
| TAS1/2021 | legacy | guarded | 3 | 0.5 | False | 13.1920 | 13.1654 | 7.7572 | -3.0708 | 0.8067 | 0.9057 |
| TAS1/2021 | balanced | raw | 2 | 1.0 | False | 13.5428 | 13.4484 | 7.8200 | -4.1667 | 0.8088 | 0.9085 |
| TAS1/2021 | balanced | guarded | 3 | 0.5 | False | 13.2239 | 13.1338 | 7.7265 | -2.7453 | 0.8110 | 0.9080 |
| TAS1/2021 | identity_anchor | raw | 2 | 1.0 | False | 13.1294 | 13.0090 | 7.7224 | -1.0207 | 0.8045 | 0.8992 |
| TAS1/2021 | identity_anchor | guarded | 2 | 1.0 | False | 13.1294 | 13.0090 | 7.7224 | -1.0207 | 0.8045 | 0.8992 |
| TAS1/2022 | B | baseline | — | 0 | — | 57.5496 | 57.5496 | 33.1751 | -39.5122 | 0.6379 | 0.7812 |
| TAS1/2022 | legacy | raw | 1 | 1.0 | False | 57.0572 | 56.1260 | 32.5086 | -36.5180 | 0.6531 | 0.7895 |
| TAS1/2022 | legacy | guarded | 1 | 0.5 | False | 57.2887 | 56.8001 | 32.8100 | -38.0151 | 0.6458 | 0.7862 |
| TAS1/2022 | balanced | raw | 1 | 1.0 | False | 56.2773 | 55.3620 | 32.3222 | -34.5670 | 0.6517 | 0.7873 |
| TAS1/2022 | balanced | guarded | 3 | 0.5 | False | 56.2563 | 55.8307 | 32.3839 | -36.1478 | 0.6558 | 0.7910 |
| TAS1/2022 | identity_anchor | raw | 3 | 1.0 | False | 57.5496 | 55.8596 | 32.7323 | -35.1297 | 0.6431 | 0.7790 |
| TAS1/2022 | identity_anchor | guarded | 3 | 1.0 | False | 57.5496 | 55.8596 | 32.7323 | -35.1297 | 0.6431 | 0.7790 |

JSON metrics include matched raw/guarded gaps, quarters, 2022H1/H2, Australian seasons, B forecast price bands and their crosses, with group counts.
All checkpoint and alpha decisions precede evaluation. Full history remains available causally.
