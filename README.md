# ANA675 Week 2 — River forecasting experiments

Decision tree and histogram gradient boosting regression, with a persistence baseline (future Colorno level equals its current level). Python/scikit-learn; PyTorch is not required. This is a pilot extension using available ARPAE measurements, not an exact replication or evidence of beating the paper.

## Run in PyCharm or a terminal

Open this folder as a PyCharm project. Use Python 3.12 or newer (tested with 3.13). Create a Python virtual environment, install requirements, then select it as the project interpreter.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --self-test
.venv/bin/python main.py --step-minutes 60
.venv/bin/python main.py --step-minutes 30
.venv/bin/python compare.py
```

Local data is already in `data/combined_river_data.csv`. This large provenance-bearing file is excluded from Git. On another computer, copy the deduplicated CSV into that location. The event inventory is included. Originals remain in the National University Week 2 folder. No credentials or environment variables are required. Set OMP_NUM_THREADS=2 if limiting CPU usage is useful.

## Method

- Use actual observations exactly on the hour or half-hour; do not interpolate, average, or invent observations. Exclude missing/conflicting readings and windows requiring them.
- Hourly: 12 observations per station, times t-11 hours through t (36 features), following the paper's input layout. Half-hourly: 24 observations per station, t-11.5 hours through t (72 features). Both represent 12 hours of regularly sampled history; the oldest timestamp differs by half an hour.
- Forecast Colorno 1, 3, 6 and 9 hours ahead, with a separate model per horizon. The paper tested every integer horizon 1–9; this pilot uses four.
- Restrict all inputs and targets to event windows. The supplement's rounded durations define approximate, exclusive endpoints. Event start times are provisionally interpreted as UTC, not verified as such.
- Merge overlapping/touching windows to avoid cross-split leakage. The 47 listed events produce 35 independent groups: first 24 chronological groups train, next 5 validate, final 6 test. This differs from the paper's randomized splits. Never split individual rows randomly.
- Fixed initial settings: tree depth 8, minimum leaf size 10; gradient boosting 200 iterations, learning rate 0.05, 15 leaves, minimum leaf size 20. Seed 42. Boosting early stopping is disabled to avoid an internal random row split. These are the fixed baseline settings; Bayesian tree tuning is described below.
- Measure RMSE, MAE and average absolute difference of predicted versus observed event maxima, over eligible targets only. This peak metric is not necessarily the paper's metric; incomplete target coverage can omit actual peaks.
- `compare.py` evaluates both runs at identical test target times. Other metrics files evaluate each run's full available sample set and should not alone determine which sampling interval is better.

## Important differences from the paper

Exports contain 15/30-minute measurements plus a sparse secondary series, not the stated original 10-minute series. The paper smooths Ponte Verdi with a cubic-spline SPH kernel over ±40 minutes before hourly subsampling. We do not apply that smoothing. Gauge offsets to sea-level elevation and river-bed/levee normalization bounds remain unverified. Models use meters exactly as exported; tree models do not require feature scaling. Event timezone, exact endpoints, archive revisions, splits and sampling differ, so results cannot establish superiority over the published LSTM/SVR.

Paper: Dazzi, Vacondio and Mignosa (2021), *Flood Stage Forecasting Using Machine-Learning Methods: A Case Study on the Parma River (Italy)*, Water 13, 1612. https://doi.org/10.3390/w13121612
Data: https://simc.arpae.it/dext3r/
Models: https://scikit-learn.org/stable/modules/tree.html and https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

## Results and next experiment

Results are in `results/30min`, `results/60min`, and `results/matched_comparison.csv`. Each run saves coverage, split membership, metrics, local predictions, input hashes and library versions. Rerunning replaces the corresponding generated results.

Bayesian optimization of tree depth, leaf size, split size and feature fraction uses training and validation groups only. Do not choose settings on test scores. Since these pilot test scores have been inspected, a final confirmatory comparison should use a newly reserved untouched period or nested group validation. More frequent readings may add useful detail or noise; improvement is an empirical question.

## Checks

`main.py --self-test` checks lag order, target timing, boundaries, missing-value exclusion, both sample resolutions and the peak metric. Runtime checks reject duplicates/nonfinite data and ensure disjoint event groups. `compare.py` checks observed targets agree across resolutions. No pre-existing lint or CI setup in this new project. Exact-replication validation has not run because the required source resolution and preprocessing details remain unresolved.

## Bayesian decision-tree experiment

```sh
.venv/bin/python tune_tree.py --self-test
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python tune_tree.py --trials 24
```

Eight searches: both input resolutions and four forecast horizons. Each search evaluates 24 settings: the original baseline, five seeded random settings, then 18 Gaussian-process expected-improvement selections. The existing scikit-learn GaussianProcessRegressor and SciPy normal CDF avoid a new optimization package. A fixed Matern kernel and 512 random candidate settings per step make this a small, reproducible pilot, not an exhaustive optimum. Leaf/split sizes are sampled on log scales; depth and feature fraction on linear scales. Search bounds and seed are saved in search_config.json.

Selection minimizes sample-weighted validation RMSE, not peak error. Test values are excluded from the optimization objective. Selected trees are refitted on training only (not train plus validation), retaining a fair training-size comparison with the baseline. Baseline validation scores are reproduced by assertion. Input hashes, event manifests and test sample counts must match the baseline. This does not undo prior exposure to test results: treat the comparison as exploratory.

`results/bayesian_tree/metrics.csv` compares tuned and baseline tree RMSE plus unchanged boosting RMSE on the same samples within each resolution. `trials.csv` records every setting and validation score. Predictions remain local. Directly comparing 30-minute versus hourly numbers in this file can reflect differing target samples; the baseline matched comparison remains in its separate file.

Checks: optimizer bounds, acquisition edge cases, unique trials, minimum-objective selection, and execution of the Bayesian phase are covered by tune_tree.py --self-test.

## Bayesian gradient boosting and combined results

```sh
.venv/bin/python tune_tree.py --self-test
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python tune_tree.py --model boosting --trials 24
.venv/bin/python compare_all.py
```

The shared search runner now supports boosting as well as trees; the default remains tree. Boosting searches learning rate 0.01–0.2, 50–350 iterations, 5–31 leaves, and minimum leaf size 5–80. Learning rate and leaf size use log scales. Seed, search budget, validation objective and event splits match the tree experiment. Internal early stopping remains disabled, and test results do not enter the optimization objective.

`results/ALL_RESULTS.md` contains overall RMSE, peak RMSE and peak MAE tables for every model, at both sampling intervals. `results/all_models_matched.csv` also includes MAE and sample counts. All ten combinations use the intersection of target timestamps for each horizon, so sampling comparisons are aligned. Detailed boosting settings, every trial and full-sample metrics are in `results/bayesian_boosting/`. Per-target predictions remain local and untracked.
