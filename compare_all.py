"""Compare all five forecasts and both sampling intervals on identical test targets."""
import csv
from pathlib import Path
import numpy as np
from main import metrics

ROOT=Path(__file__).resolve().parent
MODELS=['persistence','decision_tree','tuned_tree','gradient_boosting','tuned_boosting']


def run():
    runs={}
    for step in [30,60]:
        for model in MODELS:
            runs[step,model]={}
        with (ROOT/f'results/{step}min/predictions.csv').open() as f:
            for r in csv.DictReader(f):
                if r['split']=='test':
                    key=(int(r['horizon_h']),int(r['group']),r['target_utc'])
                    runs[step,r['model']][key]=(float(r['observed_m']),float(r['predicted_m']))
    for kind in ['tree','boosting']:
        with (ROOT/f'results/bayesian_{kind}/predictions.csv').open() as f:
            for r in csv.DictReader(f):
                key=(int(r['horizon_h']),int(r['group']),r['target_utc'])
                target=runs[int(r['sampling_minutes']),f'tuned_{kind}']
                assert key not in target
                target[key]=(float(r['observed_m']),float(r['predicted_m']))
    shared=set.intersection(*(set(r) for r in runs.values()))
    rows=[]
    for horizon in [1,3,6,9]:
        keys=sorted(k for k in shared if k[0]==horizon)
        assert keys,'No shared evaluation timestamps'
        truth=np.array([runs[30,'persistence'][k][0] for k in keys])
        groups=np.array([k[1] for k in keys])
        for step in [30,60]:
            for model in MODELS:
                vals=[runs[step,model][k] for k in keys]
                assert np.array_equal(truth,np.array([v[0] for v in vals]))
                pred=np.array([v[1] for v in vals])
                peak_errors=np.array([pred[groups==g].max()-truth[groups==g].max() for g in np.unique(groups)])
                peak_indices=[np.flatnonzero(groups==g)[np.argmax(truth[groups==g])] for g in np.unique(groups)]
                at_peak_errors=pred[peak_indices]-truth[peak_indices]
                rows.append(dict(sampling_minutes=step,horizon_h=horizon,model=model,
                    event_peak_rmse_m=float(np.sqrt(np.mean(peak_errors**2))),
                    observed_peak_time_rmse_m=float(np.sqrt(np.mean(at_peak_errors**2))),
                    **metrics(truth,pred,groups)))
    assert len(rows)==40
    with (ROOT/'results/all_models_matched.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text=['# All river forecasting results','',
          'All ten model/input combinations are evaluated at identical held-out target timestamps for each horizon. Lower is better. Units are meters. Models were selected by validation RMSE, never by the test numbers below.','']
    for title,metric in [('Overall error (RMSE)','rmse_m'),('Error at the observed peak time (RMSE)','observed_peak_time_rmse_m'),('Event peak-height error (RMSE; timing ignored)','event_peak_rmse_m'),('Event peak-height error (mean absolute error)','mean_event_peak_height_error_m')]:
        text += ['## '+title,'','| Inputs | Ahead | Persistence | Tree | Tuned tree | Boosting | Tuned boosting |','|---|---:|---:|---:|---:|---:|---:|']
        for step in [30,60]:
            for h in [1,3,6,9]:
                group={r['model']:r for r in rows if r['sampling_minutes']==step and r['horizon_h']==h}
                values=[f"{group[m][metric]:.4f}" for m in MODELS]
                text.append(f'| {step} min | {h} h | '+' | '.join(values)+' |')
        text.append('')
    text += ['## Published reference values (not a matched comparison)','',
        'Paper Tables 3–4: testing metrics averaged over 20 splits. SVR overall RMSE: 0.072, 0.131, 0.250 m at 3, 6, 9 hours. LSTM overall RMSE: 0.075, 0.138, 0.250 m. LSTM peak RMSE: 0.135, 0.150, 0.424 m. The paper restricts its peak analysis to 25 events above the first alert threshold; our peak analysis covers eligible targets in six merged test groups. Even the same metric name does not make these evaluations equivalent.',
        '', 'Source: https://doi.org/10.3390/w13121612', '', '## Interpretation and limits','',
        'Persistence has zero eventual-peak-height error here because it repeats the peak later. This does not mean it predicts the flood peak in time. The observed-peak-time table compares each prediction with the observation at the first maximum among eligible timestamps; it penalizes that delay.', '',
        'Peak-height error compares maxima over eligible targets in each merged event group, not necessarily the complete flood peak or the paper’s peak metric. All runs use the same six test groups. See the CSV for MAE, sample counts and event counts.',
        '', 'Each tuned family received 24 trials for each of eight configurations (192 trials per family). The baseline plus five random starts were followed by 18 Gaussian-process expected-improvement steps. Tuning minimizes validation RMSE; it does not directly minimize peak error. Selected models were trained on training groups only to preserve baseline comparability.',
        '', 'These test periods had already been inspected during the pilot. Results are exploratory, not an untouched final evaluation. Sampling resolution, smoothing, assumed event timezone, approximate event boundaries, normalization and split strategy differ from the paper. These results cannot establish superiority over its LSTM/SVR.',
        '', 'Checks passed: shared-target equality, 40 comparison rows, source-hash agreement, baseline validation-score reproduction, search bounds/selection tests and prediction sample-count checks. Not run: exact replication (source/preprocessing mismatch); a new untouched confirmatory evaluation (not part of this pilot).']
    (ROOT/'results/ALL_RESULTS.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:17]))

if __name__=='__main__':
    run()
