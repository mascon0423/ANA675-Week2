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

## Closest available approximation to the paper

```sh
.venv/bin/python paper_protocol.py --self-test
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python paper_protocol.py --trials 24
```

This separate experiment preserves the previous pilot results. It uses all 47 event IDs from supplementary Table S1, all 20 explicit train/validation/test memberships in Table S2 (32/7/8), 12 hourly observations per station, and all nine lead times. Peak metrics include only events whose published peak exceeds 27.7 m above sea level (25 eligible events across the full inventory). Metrics are calculated within each split and averaged across 20 splits, matching the paper's reporting structure.

Ponte Verdi alone receives a centered cubic SPH-kernel filter with support ±40 minutes, interpreted as h=20 minutes for the usual support-2h cubic kernel. The finest complete native local grid is used: 15-minute values where available, otherwise 30-minute values. No synthetic ten-minute readings are created. On these coarse grids only neighbors up to ±30 minutes exist inside the support. Discrete weights are normalized to preserve constant levels. This is an explicit numerical adaptation, not the original ten-minute quadrature. Missing complete neighborhoods are excluded from both filtered and raw comparisons. Other station readings and all Colorno targets remain unfiltered. Kernel reference: https://pysph.readthedocs.io/en/main/reference/kernels.html

The symmetric filter uses later measurements, intentionally reproducing the paper's described historical processing. It is not a real-time causal forecast. Event timezone is provisionally UTC and event endpoints use published rounded day durations. The original gauge offsets and riverbed/levee bounds are not specified numerically in the paper; no offsets or bounds are invented. Tree models use the original meter values: fixed positive affine feature scaling is generally unnecessary for tree splits. This shortcut does not apply automatically to reproducing SVR/MLP/LSTM.

A new 24-trial Bayesian boosting search minimizes mean validation RMSE across splits 1–3 and horizons 3/6/9. The paper used three splits but does not identify which three for tuning; choosing the first three is documented. One parameter set is then used at every horizon, as in the paper. The same settings are used for raw and filtered inputs to isolate the processing change, although settings were selected on filtered inputs. This is not a reproduction of the authors' SVR/MLP/LSTM architectures.

Published event assignments are preserved rather than replaced with our earlier merged chronological groups. Because rounded windows can overlap, overlap_audit.csv counts shared target timestamps across partitions. It does not guarantee absence of shared input observations. Do not describe these estimates as independently leakage-free. Reuse of events across published splits and earlier experiments also prevents treating this as a new untouched test.

Results and detailed coverage are in `results/paper_protocol/`. `protocol.json` records sample counts versus the paper's 9,484 one-hour and 9,108 nine-hour samples, source hashes, smoothing availability and limitations. The script checks the kernel, missing-neighbor handling, centered behavior, peak timing, all 20 split counts, target equality and source event IDs. No new dependencies are required.

## MLP architecture search and NEAT pilot

```sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python neural_experiments.py --self-test
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python neural_experiments.py --trials 24 --epochs 200 --generations 60
```

This exploratory extension uses the same centered inputs and eligible samples as `paper_protocol.py`, at 3/6/9 hours across all 20 published splits. Bayesian search considers 1–5 hidden layers with independently selected widths of 8–64 neurons each, Adam learning rate 0.0001–0.01, and L2 regularization 0.000001–0.01. ReLU activation is fixed. Inactive width dimensions are canonicalized. The initial candidate has one 16-neuron layer. There are 24 candidates, selected on mean validation RMSE across splits 1–3 and the three horizons. The shared Gaussian-process search now infers its dimension from the encoder, retaining earlier tree behavior.

Inputs and target are standardized using only each split's training data. MLP weights are trained by Adam for up to 200 epochs, retaining the checkpoint with lowest published-validation RMSE and stopping after 20 epochs without improvement. No random internal validation holdout is used. This scaling differs from the paper's physical scaling; the baseline is paper-inspired, not a reproduced author model. Each fit uses one seed, not the authors' ten-initialization ensemble.

NEAT-Python 1.1.0 is the one new dependency, needed for genuine topology-and-weight evolution. It starts with direct input-output connections, evolves ReLU hidden nodes and connections, and retains a linear output. Population 60, 60 generations, seed 42: training MSE determines evolutionary fitness; validation RMSE chooses among generation training champions. No Adam or CMA-ES is used. Batch prediction is checked against NEAT's scalar evaluator. These budgets are bounded pilots, not equal computational budgets or exhaustive optimization. Results must not be presented as algorithm superiority or independent confirmation: published splits and previously viewed test events are reused.

Results are written to `results/neural/` as trials, selected settings, per-split metrics, NEAT generation histories and mean metrics. The MLP search runs first, followed by baseline MLP, tuned MLP and NEAT evaluation. Source documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html and https://neat-python.readthedocs.io/en/latest/.
