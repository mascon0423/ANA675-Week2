# Bayesian decision-tree results

24 trials per configuration; validation RMSE selects settings. Test RMSE below is in meters; lower is better. Comparisons are paired within each row, not necessarily across sampling resolutions.

| Inputs | Forecast | Original tree | Tuned tree | Untuned boosting |
|---|---:|---:|---:|---:|
| 30 min | 1 h | 0.0998 | 0.0667 | 0.0642 |
| 30 min | 3 h | 0.1856 | 0.1908 | 0.1467 |
| 30 min | 6 h | 0.3104 | 0.3090 | 0.2405 |
| 30 min | 9 h | 0.4274 | 0.3924 | 0.3746 |
| 60 min | 1 h | 0.0768 | 0.0701 | 0.0642 |
| 60 min | 3 h | 0.1836 | 0.1736 | 0.1417 |
| 60 min | 6 h | 0.3024 | 0.3034 | 0.2383 |
| 60 min | 9 h | 0.4225 | 0.4366 | 0.3764 |

Five of eight tuned trees improved test RMSE. Untuned gradient boosting still had lower test RMSE in all eight configurations. No test results were used by the search. These previously inspected pilot test periods are not an untouched confirmatory evaluation. Preprocessing and event splits differ from the paper, so no claim of outperforming the published models is supported.

Checks passed: search self-test, baseline preprocessing self-test, dependency check, input hashes, baseline validation-score reproduction, matching test counts, 192-trial audit, and best-validation-score selection. Not run: gradient-boosting tuning (this experiment targeted decision trees), exact replication (source/preprocessing mismatch).
