# All river forecasting results

All ten model/input combinations are evaluated at identical held-out target timestamps for each horizon. Lower is better. Units are meters. Models were selected by validation RMSE, never by the test numbers below.

## Overall error (RMSE)

| Inputs | Ahead | Persistence | Tree | Tuned tree | Boosting | Tuned boosting |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 1 h | 0.0822 | 0.1002 | 0.0638 | 0.0639 | 0.0499 |
| 30 min | 3 h | 0.2387 | 0.1844 | 0.1908 | 0.1468 | 0.1313 |
| 30 min | 6 h | 0.4509 | 0.3094 | 0.3119 | 0.2387 | 0.2136 |
| 30 min | 9 h | 0.6351 | 0.4302 | 0.3959 | 0.3742 | 0.3536 |
| 60 min | 1 h | 0.0822 | 0.0769 | 0.0702 | 0.0644 | 0.0535 |
| 60 min | 3 h | 0.2387 | 0.1841 | 0.1738 | 0.1419 | 0.1272 |
| 60 min | 6 h | 0.4509 | 0.3031 | 0.3042 | 0.2379 | 0.2116 |
| 60 min | 9 h | 0.6351 | 0.4224 | 0.4358 | 0.3760 | 0.3305 |

## Error at the observed peak time (RMSE)

| Inputs | Ahead | Persistence | Tree | Tuned tree | Boosting | Tuned boosting |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 1 h | 0.0292 | 0.0581 | 0.0869 | 0.0772 | 0.0747 |
| 30 min | 3 h | 0.1715 | 0.1220 | 0.3821 | 0.2203 | 0.2344 |
| 30 min | 6 h | 0.7782 | 0.2254 | 0.2807 | 0.3224 | 0.1816 |
| 30 min | 9 h | 1.2852 | 0.5874 | 0.4807 | 0.4161 | 0.3802 |
| 60 min | 1 h | 0.0292 | 0.1221 | 0.1109 | 0.0780 | 0.0606 |
| 60 min | 3 h | 0.1715 | 0.3578 | 0.4107 | 0.2967 | 0.2701 |
| 60 min | 6 h | 0.7782 | 0.4236 | 0.2548 | 0.3524 | 0.3479 |
| 60 min | 9 h | 1.2852 | 0.3992 | 0.6447 | 0.3978 | 0.2570 |

## Event peak-height error (RMSE; timing ignored)

| Inputs | Ahead | Persistence | Tree | Tuned tree | Boosting | Tuned boosting |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 1 h | 0.0000 | 0.1176 | 0.1233 | 0.0988 | 0.0825 |
| 30 min | 3 h | 0.0000 | 0.3775 | 0.4559 | 0.2465 | 0.2503 |
| 30 min | 6 h | 0.0000 | 0.3714 | 0.3748 | 0.3973 | 0.3580 |
| 30 min | 9 h | 0.0000 | 0.6021 | 0.5885 | 0.6770 | 0.4086 |
| 60 min | 1 h | 0.0000 | 0.1507 | 0.0824 | 0.1110 | 0.0635 |
| 60 min | 3 h | 0.0000 | 0.3850 | 0.4477 | 0.2933 | 0.2744 |
| 60 min | 6 h | 0.0000 | 0.4330 | 0.5681 | 0.4883 | 0.3627 |
| 60 min | 9 h | 0.0000 | 0.5150 | 0.6940 | 0.6595 | 0.4237 |

## Event peak-height error (mean absolute error)

| Inputs | Ahead | Persistence | Tree | Tuned tree | Boosting | Tuned boosting |
|---|---:|---:|---:|---:|---:|---:|
| 30 min | 1 h | 0.0000 | 0.0871 | 0.0999 | 0.0817 | 0.0696 |
| 30 min | 3 h | 0.0000 | 0.3183 | 0.3647 | 0.1865 | 0.2031 |
| 30 min | 6 h | 0.0000 | 0.2703 | 0.2887 | 0.3533 | 0.2502 |
| 30 min | 9 h | 0.0000 | 0.5146 | 0.4891 | 0.5928 | 0.3461 |
| 60 min | 1 h | 0.0000 | 0.1157 | 0.0666 | 0.0944 | 0.0512 |
| 60 min | 3 h | 0.0000 | 0.2991 | 0.3543 | 0.2387 | 0.2125 |
| 60 min | 6 h | 0.0000 | 0.3310 | 0.4659 | 0.4225 | 0.3048 |
| 60 min | 9 h | 0.0000 | 0.4419 | 0.5841 | 0.5833 | 0.3856 |

## Published reference values (not a matched comparison)

Paper Tables 3–4: testing metrics averaged over 20 splits. SVR overall RMSE: 0.072, 0.131, 0.250 m at 3, 6, 9 hours. LSTM overall RMSE: 0.075, 0.138, 0.250 m. LSTM peak RMSE: 0.135, 0.150, 0.424 m. The paper restricts its peak analysis to 25 events above the first alert threshold; our peak analysis covers eligible targets in six merged test groups. Even the same metric name does not make these evaluations equivalent.

Source: https://doi.org/10.3390/w13121612

## Interpretation and limits

Persistence has zero eventual-peak-height error here because it repeats the peak later. This does not mean it predicts the flood peak in time. The observed-peak-time table compares each prediction with the observation at the first maximum among eligible timestamps; it penalizes that delay.

Peak-height error compares maxima over eligible targets in each merged event group, not necessarily the complete flood peak or the paper’s peak metric. All runs use the same six test groups. See the CSV for MAE, sample counts and event counts.

Each tuned family received 24 trials for each of eight configurations (192 trials per family). The baseline plus five random starts were followed by 18 Gaussian-process expected-improvement steps. Tuning minimizes validation RMSE; it does not directly minimize peak error. Selected models were trained on training groups only to preserve baseline comparability.

These test periods had already been inspected during the pilot. Results are exploratory, not an untouched final evaluation. Sampling resolution, smoothing, assumed event timezone, approximate event boundaries, normalization and split strategy differ from the paper. These results cannot establish superiority over its LSTM/SVR.

Checks passed: shared-target equality, 40 comparison rows, source-hash agreement, baseline validation-score reproduction, search bounds/selection tests and prediction sample-count checks. Not run: exact replication (source/preprocessing mismatch); a new untouched confirmatory evaluation (not part of this pilot).
