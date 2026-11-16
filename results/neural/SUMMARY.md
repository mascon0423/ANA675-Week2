# Neural-model pilot results

Mean of 20 published splits, using the same reconstructed centered hourly inputs as the closest-available boosting experiment. Lower errors are better; all table values are centimeters.

## Overall RMSE

| Ahead | Initial MLP | Bayesian MLP | NEAT | Tuned boosting | Paper best |
|---|---:|---:|---:|---:|---:|
| 3 h | 14.86 | 9.40 | 30.67 | 11.69 | 7.20 |
| 6 h | 19.75 | 15.72 | 40.65 | 20.15 | 13.10 |
| 9 h | 29.91 | 27.18 | 52.60 | 32.86 | 25.00 |

## Peak-height RMSE

| Ahead | Initial MLP | Bayesian MLP | NEAT | Tuned boosting | Paper LSTM |
|---|---:|---:|---:|---:|---:|
| 3 h | 17.15 | 15.82 | 52.08 | 34.99 | 13.50 |
| 6 h | 23.46 | 19.51 | 59.22 | 38.10 | 15.00 |
| 9 h | 33.01 | 34.68 | 88.18 | 41.31 | 42.40 |

## Search and limits

Selected MLP settings: `{"layers": [60], "learning_rate_init": 0.0007868553027753759, "alpha": 6.468646876975688e-06}`.

MLP search: 24 candidates by validation RMSE across splits 1–3 and 3/6/9-hour horizons. ReLU and Adam; 1–5 layers, independent widths 8–64, learning rate and regularization. Each fit uses training-only standardization and published validation events for early stopping (maximum 200 epochs, patience 20). Initial MLP: one layer of 16 neurons.

NEAT: population 60 for 60 generations per split/horizon; topology and weights evolve on training MSE, validation chooses a generation champion. Linear output and ReLU hidden nodes. One seed per fit; no ten-model ensemble. Budgets are unequal across methods, so this is an exploratory configuration comparison.

All earlier reconstruction limits apply: coarser native measurements, adapted centered smoothing that uses later readings, rounded event endpoints, assumed UTC and omitted incomplete windows. Neural scaling uses training-only standardization instead of unavailable physical normalization bounds. Test events are reused across published splits and prior experiments; these are not untouched independent confirmation results. The original paper models are reference numbers, not reruns on our data.

Checks: independent layer-width encoding, eight-dimensional Bayesian search, vectorized/scalar NEAT parity, training-only scaling and both fitting paths. Not run: exact paper replication, multiple random-seed ensembles, or equal-compute optimizer benchmarking.

Paper source: https://doi.org/10.3390/w13121612 (Tables 3–4).
