"""Compare both sampling runs on identical held-out targets. Run after main.py twice."""
import csv
from pathlib import Path
import numpy as np
from main import metrics

root=Path(__file__).resolve().parent/'results'
runs={}
for step in [30,60]:
    with (root/f'{step}min/predictions.csv').open() as f:
        runs[step]={(r['model'],r['horizon_h'],r['group'],r['target_utc']):r
                    for r in csv.DictReader(f) if r['split']=='test'}
common=sorted(runs[30].keys() & runs[60].keys())
rows=[]
for horizon in [1,3,6,9]:
    for model in ['persistence','decision_tree','gradient_boosting']:
        keys=[k for k in common if k[:2]==(model,str(horizon))]
        if not keys:
            raise ValueError('No common test targets')
        assert all(runs[30][k]['observed_m']==runs[60][k]['observed_m'] for k in keys)
        for step in [30,60]:
            values=[runs[step][k] for k in keys]
            rows.append(dict(horizon_h=horizon,model=model,sampling_minutes=step,
                **metrics(np.array([float(r['observed_m']) for r in values]),
                          np.array([float(r['predicted_m']) for r in values]),
                          np.array([int(r['group']) for r in values]))))
with (root/'matched_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:
    if r['model']=='gradient_boosting':
        print(f"{r['horizon_h']}h, {r['sampling_minutes']}min inputs: RMSE={r['rmse_m']:.4f}m, peak MAE={r['mean_event_peak_height_error_m']:.4f}m, n={r['samples']}")
