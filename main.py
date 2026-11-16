"""Event-held-out river forecasting pilot; see README for departures from the paper."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
STATIONS = ['Parma Ponte Verdi', 'Casalmaggiore', 'Colorno AIPO']
HOUR = timedelta(hours=1)


def event_windows(path):
    windows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            # ponytail: event timezone is unconfirmed; explicit UTC assumption for this pilot.
            start = datetime.fromisoformat(r['start']).replace(tzinfo=timezone.utc)
            windows.append([start, start + timedelta(days=float(r['duration_days'])), [r['event']]])
    merged = []
    for start, end, ids in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].extend(ids)
        else:
            merged.append([start, end, ids])
    return merged


def load_readings(path, step_minutes=60):
    data = {}
    seen = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            station = r['station']
            t = datetime.fromisoformat(r['timestamp_utc'])
            key = (station, t)
            if station not in STATIONS or t.utcoffset() != timedelta(0) or key in seen:
                raise ValueError('Invalid station, timestamp, or duplicate key')
            seen.add(key)
            if r['status'] == 'observed' and t.minute % step_minutes == 0 and t.second == t.microsecond == 0:
                v = float(r['water_level_m_as_exported'])
                if not np.isfinite(v):
                    raise ValueError('Nonfinite observation')
                data[key] = v
    return data


def samples(data, start, end, horizon, step_minutes=60):
    step = timedelta(minutes=step_minutes)
    history = 12 * 60 // step_minutes
    x, y, stamps = [], [], []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t < start:
        t += step
    t += (history - 1) * step
    while t + horizon * HOUR < end:
        keys = [(s, t - lag * step) for s in STATIONS for lag in range(history - 1, -1, -1)]
        target = ('Colorno AIPO', t + horizon * HOUR)
        if all(k in data for k in keys) and target in data:
            x.append([data[k] for k in keys]); y.append(data[target]); stamps.append(target[1])
        t += step
    return np.asarray(x, dtype=float).reshape(-1, len(STATIONS) * history), np.asarray(y), stamps


def metrics(y, pred, groups):
    errors = pred - y
    peak_errors = [abs(float(pred[groups == g].max() - y[groups == g].max())) for g in np.unique(groups)]
    return {'rmse_m': float(np.sqrt(np.mean(errors**2))), 'mae_m': float(np.mean(abs(errors))),
            'mean_event_peak_height_error_m': float(np.mean(peak_errors)),
            'evaluated_events': len(peak_errors), 'samples': len(y)}


def self_test():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    d = {(s, start+i*HOUR): float(i) for s in STATIONS for i in range(30)}
    x, y, stamps = samples(d, start, start+30*HOUR, 3)
    assert x.shape == (16, 36) and y[0] == 14 and list(x[0, :12]) == list(range(12))
    assert x[0, -1] == 11 and stamps[-1] == start+29*HOUR
    del d[STATIONS[0], start]
    assert len(samples(d, start, start+30*HOUR, 3)[1]) == 15
    m = metrics(np.array([1.,3.,2.]), np.array([1.,2.,2.]), np.array([0,0,1]))
    assert m['mean_event_peak_height_error_m'] == .5
    fine = {(s, start+timedelta(minutes=30*i)): float(i) for s in STATIONS for i in range(60)}
    fx, fy, _ = samples(fine, start, start+30*HOUR, 3, 30)
    assert fx.shape == (31, 72) and fy[0] == 29 and fx[0, -1] == 23
    print('Checks passed: lag order, future target, window bounds, missing-data exclusion, peak metric.')


def run(step_minutes):
    data_path = ROOT/'data/combined_river_data.csv'
    event_path = ROOT/'data/flood-events.csv'
    data = load_readings(data_path, step_minutes)
    windows = event_windows(event_path)
    n = len(windows)
    a, b = int(n*.70), int(n*.85)
    assert 0 < a < b < n
    splits = ['train' if i < a else 'validation' if i < b else 'test' for i in range(n)]
    assert all(windows[i][1] < windows[i+1][0] for i in range(n-1))
    out = ROOT/'results'/f'{step_minutes}min'; out.mkdir(parents=True, exist_ok=True)
    manifest = [{'group':i,'events':ids,'start_assumed_utc':start.isoformat(),'end_exclusive':end.isoformat(),'split':splits[i]}
                for i,(start,end,ids) in enumerate(windows)]
    (out/'event_splits.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results, predictions, coverage = [], [], []
    for horizon in [1,3,6,9]:
        parts = {s: [] for s in ['train','validation','test']}
        for i,(start,end,ids) in enumerate(windows):
            x,y,times = samples(data,start,end,horizon,step_minutes)
            coverage.append({'horizon_h':horizon,'group':i,'split':splits[i],'samples':len(y)})
            if len(y):
                parts[splits[i]].append((x,y,np.full(len(y),i),times))
        arrays = {}
        for split, chunks in parts.items():
            if not chunks:
                raise ValueError(f'No usable samples for {split}')
            arrays[split] = (np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks]),
                             np.concatenate([c[2] for c in chunks]), [t for c in chunks for t in c[3]])
        models = {'persistence':None,
                  'decision_tree':DecisionTreeRegressor(max_depth=8,min_samples_leaf=10,random_state=42),
                  'gradient_boosting':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,
                      learning_rate=.05,min_samples_leaf=20,early_stopping=False,random_state=42)}
        for name,model in models.items():
            begin = perf_counter()
            if model is not None:
                model.fit(arrays['train'][0],arrays['train'][1])
            elapsed = perf_counter()-begin
            for split in ['validation','test']:
                x,y,g,times = arrays[split]
                pred = x[:,-1] if model is None else model.predict(x)
                result = dict(horizon_h=horizon,model=name,split=split,train_samples=len(arrays['train'][1]),
                              fit_seconds=elapsed,**metrics(y,pred,g))
                results.append(result)
                predictions.extend({'horizon_h':horizon,'model':name,'split':split,'group':int(group),
                    'target_utc':t.isoformat(),'observed_m':float(obs),'predicted_m':float(p)}
                    for group,t,obs,p in zip(g,times,y,pred))
            print(f'{horizon}h {name}: test RMSE {results[-1]["rmse_m"]:.3f} m', flush=True)
    for name,rows in [('metrics.csv',results),('predictions.csv',predictions),('coverage.csv',coverage)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meta = {'sampling_minutes':step_minutes,'sklearn_version':sklearn.__version__,'numpy_version':np.__version__,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [data_path,event_path]},
            'event_groups':n,'split_groups':{s:splits.count(s) for s in set(splits)},
            'warning':'Approximate event UTC windows; no original smoothing or datum conversion. Not directly comparable to paper. Test results now exposed; future tuning uses validation only.'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--step-minutes',type=int,choices=[30,60],default=30); args=p.parse_args()
    self_test() if args.self_test else run(args.step_minutes)
