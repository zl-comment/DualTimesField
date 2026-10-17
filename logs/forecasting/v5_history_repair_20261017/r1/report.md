# V5 history-only repair — retrospective development

Frozen C_roll; history weights fit October–November 2022; evaluated December 2022.
December has been previously inspected. This is not a blind test or the original V5 gate protocol.

| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |
|---|---:|---:|---:|---:|---:|
| B | 25.710190 | 26.291141 | 27.044704 | 26.348678 | 15.219256 |
| C_roll | 31.216519 | 29.512742 | 27.226839 | 29.318700 | 17.428757 |
| H_recent | 25.850057 | 25.771022 | 27.284182 | 26.301754 | 15.480187 |
| H_matched | 34.245663 | 33.199710 | 28.009585 | 31.818319 | 19.991982 |

## Acceptance

- mechanism_pass: **True**
- incremental_value_pass: **False**
- matching_value_pass: **False**
- per_region_non_regression: **False**

## NSW1

- Matched weights: [0.6034092743938654, 0.5516351518021525, 0.5321909750624891, 0.5004433510712911]
- Recent weights: [0.7141775139474055, 0.6704653075605116, 0.6592145912740146, 0.6176956716781092]
- Matched forecast change: {'fraction': 1.0, 'mean_abs': 14.4172646546762, 'max_abs': 90.51139115907978}
- Reconstructed boundary origins: 23
- Wall time: 5.2 seconds.

## QLD1

- Matched weights: [0.49697234942504254, 0.5204810767519374, 0.47186131743573834, 0.41981878388665317]
- Recent weights: [0.6882166275992672, 0.6950789188964647, 0.6653216625357292, 0.6177051776595368]
- Matched forecast change: {'fraction': 1.0, 'mean_abs': 11.991759033701367, 'max_abs': 78.07216151279061}
- Reconstructed boundary origins: 23
- Wall time: 5.1 seconds.

## TAS1

- Matched weights: [0.29271929229454935, 0.3298182127986417, 0.31587899293900523, 0.25937211406042365]
- Recent weights: [0.4519909226926198, 0.4603857832532341, 0.4532523215769663, 0.42339104293243524]
- Matched forecast change: {'fraction': 1.0, 'mean_abs': 7.185516357570435, 'max_abs': 33.55417379736002}
- Reconstructed boundary origins: 23
- Wall time: 5.2 seconds.
