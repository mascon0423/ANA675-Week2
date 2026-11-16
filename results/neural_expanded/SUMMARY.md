# Expanded neural results

Averages across the same 20 published splits. All errors below are centimeters; lower is better. These are repeated exploratory evaluations on reconstructed data, not independent confirmation.

## Overall RMSE

| Ahead | Prior tuned MLP, one seed | Prior architecture, 10 seeds | Expanded MLP, 10 seeds | Expanded NEAT, 3 seeds | Paper reference |
|---|---:|---:|---:|---:|---:|
| 3 h | 9.40 | 8.31 | 9.72 | 19.95 | 7.20 |
| 6 h | 15.72 | 14.82 | 16.91 | 30.72 | 13.10 |
| 9 h | 27.18 | 26.46 | 29.16 | 42.91 | 25.00 |

## Peak-height RMSE

| Ahead | Prior tuned MLP, one seed | Prior architecture, 10 seeds | Expanded MLP, 10 seeds | Expanded NEAT, 3 seeds | Paper reference |
|---|---:|---:|---:|---:|---:|
| 3 h | 15.82 | 13.20 | 14.73 | 23.91 | 13.50 |
| 6 h | 19.51 | 16.82 | 19.81 | 34.57 | 15.00 |
| 9 h | 34.68 | 34.32 | 34.53 | 55.88 | 42.40 |

Selected expanded MLP settings: `{"layers": [50, 59, 16, 60, 28], "learning_rate_init": 0.009751384813126474, "alpha": 8.746801196484261e-06}`.

The search uses 64 Bayesian candidates, up to 400 epochs, independently sized layers and three-seed ensemble validation across splits 1–3 and horizons 3/6/9. The balanced objective is 50% overall RMSE plus 50% eligible-event peak-height RMSE, both in meters. Final MLP predictions average ten seeds. The prior-architecture ensemble retains ordinary validation-RMSE checkpoint selection; the new architecture uses the balanced objective. Both have a larger epoch budget than the pilot, so changes cannot be attributed solely to ensembling.

Expanded NEAT uses population 150, 200 generations and three seeds per split/horizon. Training MSE drives evolution; the balanced validation score chooses each seed’s generation champion. Three predictions are averaged. This is still a bounded search, not a best-possible performance claim; computational budgets differ across methods.

Paper references are its best reported overall RMSE and its LSTM peak RMSE (Tables 3–4). All reconstruction differences remain: incomplete windows omitted, coarser original sampling, adapted centered smoothing, assumed timestamps, and training-only standardization rather than physical scaling. Centered inputs use later archived readings. Peak-height error can hide timing errors; observed-peak-time error is also saved in the metrics. Reusing published test events across experiments limits confirmatory claims.

Checks: layer encoding, Bayesian selection, scalar/batch NEAT parity, training-only scaling, custom checkpoint scoring, objective arithmetic and complete split/model coverage. Not run: exact replication, fresh-event confirmation or equal-compute algorithm benchmarking.

Source: https://doi.org/10.3390/w13121612.
