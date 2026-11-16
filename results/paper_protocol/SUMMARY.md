# Closest available paper-protocol experiment

All 47 published event IDs, all 20 published splits, all nine hourly forecast horizons, and 12 past readings per station. Peak analysis uses events with published peaks above 27.7 m. One new boosting configuration was selected by mean validation RMSE across published splits 1–3 and horizons 3/6/9, then held fixed.

Each reported metric is the arithmetic mean of its per-split value, not a pooled error across splits. Units are meters.

| Ahead | Raw boosting RMSE | Centered boosting RMSE | Centered peak RMSE | Paper best overall RMSE |
|---|---:|---:|---:|---:|
| 1 h | 0.0585 | 0.0587 | 0.2399 | — |
| 2 h | 0.0887 | 0.0890 | 0.3025 | — |
| 3 h | 0.1171 | 0.1169 | 0.3499 | 0.0720 |
| 4 h | 0.1437 | 0.1449 | 0.3295 | — |
| 5 h | 0.1711 | 0.1728 | 0.3451 | — |
| 6 h | 0.2004 | 0.2015 | 0.3810 | 0.1310 |
| 7 h | 0.2337 | 0.2329 | 0.3882 | — |
| 8 h | 0.2731 | 0.2722 | 0.4139 | — |
| 9 h | 0.3284 | 0.3286 | 0.4131 | 0.2500 |

## Coverage and interpretation

One-hour paired samples: 7,523 (paper: 9,484). Nine-hour paired samples: 7,211 (paper: 9,108). See DISCUSSION_NOTES.md for the missing-observation/window distinction.

Raw and centered boosting use identical targets and the same selected settings. Parameters were selected using centered inputs, so this is not a separately optimized raw-versus-smoothed contest. Changes from the earlier chronological pilot also include event splits, eligible samples and the tuning objective; they cannot be attributed to smoothing alone.

Splits with shared train/test target timestamps: []. Published event memberships were retained. The audit counts shared target times, not every shared input observation.

## Remaining differences

- Native 15/30-minute data, not original ten-minute data; normalized discrete cubic kernel adaptation.
- Rounded durations and unconfirmed event timezone; missing/conflicting windows excluded.
- No gauge offsets or physical normalization bounds invented; tree inputs retained in exported meters.
- Published memberships retained; overlapping sample times are audited, not silently purged.
- Centered filter is historical/as-described and uses later readings; not a real-time causal forecast.
- Published 20 splits are reused across tuning/evaluation; this is not nested independent confirmation.

The neural and SVR models from the article were not retrained here. This is the existing boosting experiment moved closer to the article’s data protocol. Missing-value handling in the paper is unknown; do not claim the authors necessarily filled the gaps present in our current exports.

Source: https://doi.org/10.3390/w13121612 (Sections 3.2–3.3; Tables 3–4; supplementary Tables S1–S2). Cubic kernel convention: https://pysph.readthedocs.io/en/main/reference/kernels.html

Checks: kernel shape and constant preservation, missing neighborhoods, centered look-ahead, peak timing, all 20 partition counts, paired target equality, trial-selection audit and full split/horizon coverage. Not run: exact ten-minute replication (observations unavailable in these exports), physical normalization (numeric bounds unreported), or new untouched confirmation (published split reuse).
