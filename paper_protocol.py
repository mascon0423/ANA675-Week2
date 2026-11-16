"""Closest documented paper protocol supported by the available exports; not exact replication."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from main import ROOT, STATIONS, samples
from tune_tree import search, decode_boost, encode_boost

OUT=ROOT/'results/paper_protocol'


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def cubic_shape(minutes):
    # Cubic SPH kernel has support 2h; h=20 min gives the paper's +/-40 min support.
    q=np.abs(np.asarray(minutes,dtype=float))/20
    return np.where(q<=1,1-1.5*q*q+.75*q**3,np.where(q<2,.25*(2-q)**3,0.))


def smooth(data):
    result={}
    counts=Counter()
    for (station,t),value in data.items():
        if t.minute or t.second:
            continue
        if station!='Parma Ponte Verdi':
            result[station,t]=value
            continue
        # Use the finest complete native local grid, never synthesize ten-minute observations.
        for spacing in [15,30]:
            offsets=np.arange(-30,31,spacing)
            keys=[(station,t+timedelta(minutes=int(m))) for m in offsets]
            if all(k in data for k in keys):
                weights=cubic_shape(offsets)
                # ponytail: normalized discrete kernel preserves constant levels on coarse grids;
                # replace with the original quadrature if ten-minute observations become available.
                result[station,t]=float(np.dot(weights,[data[k] for k in keys])/weights.sum())
                counts[f'{spacing}min_native_window']+=1
                break
        else:
            counts['missing_window']+=1
    return result,dict(counts)


def prepare():
    with (ROOT/'data/combined_river_data.csv').open() as f:
        rows=list(csv.DictReader(f))
    data={}
    for r in rows:
        t=datetime.fromisoformat(r['timestamp_utc']);key=(r['station'],t)
        if r['status']=='observed':
            value=float(r['water_level_m_as_exported'])
            assert key not in data and np.isfinite(value)
            data[key]=value
    filtered,counts=smooth(data)
    raw={k:v for k,v in data.items() if k[1].minute==k[1].second==0}
    write_csv(OUT/'hourly_measurements.csv',[dict(station=k[0],timestamp_utc=k[1].isoformat(),raw_level_m=v,
        processed_level_m=filtered.get(k,'')) for k,v in sorted(raw.items())])
    with (ROOT/'data/flood-events.csv').open() as f:
        events={r['event']:r for r in csv.DictReader(f)}
    with (ROOT/'data/paper_splits.csv').open() as f:
        assignments={r['event']:r for r in csv.DictReader(f)}
    assert set(events)==set(assignments) and len(events)==47
    for i in range(1,21):
        assert Counter(r[f'split_{i}'] for r in assignments.values())=={'t':32,'v':7,'x':8}
    prepared={};coverage=[]
    for event,r in events.items():
        start=datetime.fromisoformat(r['start']).replace(tzinfo=timezone.utc)
        end=start+timedelta(days=float(r['duration_days']))
        for horizon in range(1,10):
            raw_x,raw_y,raw_t=samples(raw,start,end,horizon)
            sx,sy,st=samples(filtered,start,end,horizon)
            raw_lookup={t:(x,y) for t,x,y in zip(raw_t,raw_x,raw_y)}
            selected=[i for i,t in enumerate(st) if t in raw_lookup]
            times=[st[i] for i in selected]
            assert len(times)>0, (event,horizon)
            xraw=np.asarray([raw_lookup[t][0] for t in times])
            y=np.asarray([raw_lookup[t][1] for t in times])
            xsmooth=sx[selected]
            assert np.array_equal(y,sy[selected])
            prepared[event,horizon]=(xraw,xsmooth,y,times)
            coverage.append(dict(event=event,horizon_h=horizon,raw_samples=len(raw_y),matched_samples=len(y),
                                 paper_peak_m=float(r['peak_m']),start_assumed_utc=start.isoformat(),end_exclusive=end.isoformat()))
    write_csv(OUT/'coverage.csv',coverage)
    return prepared,events,assignments,counts


def combine(prepared,assignments,split,horizon,partition):
    selected=[(event,prepared[event,horizon]) for event,a in assignments.items() if a[f'split_{split}']==partition]
    return (np.concatenate([p[0] for _,p in selected]),np.concatenate([p[1] for _,p in selected]),
            np.concatenate([p[2] for _,p in selected]),np.concatenate([np.full(len(p[2]),event) for event,p in selected]),
            [t for _,p in selected for t in p[3]])


def build(params):
    return HistGradientBoostingRegressor(random_state=42,early_stopping=False,**params)


def measure(y,pred,ids,events):
    error=pred-y;eligible=[e for e in np.unique(ids) if float(events[e]['peak_m'])>27.7]
    peaks=np.array([pred[ids==e].max()-y[ids==e].max() for e in eligible])
    locations=[np.flatnonzero(ids==e)[np.argmax(y[ids==e])] for e in eligible]
    assert len(eligible)>0
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))),mae_m=float(np.mean(abs(error))),
        peak_rmse_m=float(np.sqrt(np.mean(peaks**2))),
        observed_peak_time_rmse_m=float(np.sqrt(np.mean(error[locations]**2))),
        samples=len(y),peak_events=len(eligible))


def self_test():
    assert np.allclose(cubic_shape([0,20,40]),[1,.25,0])
    t=datetime(2020,1,1,12,tzinfo=timezone.utc)
    d={('Parma Ponte Verdi',t+timedelta(minutes=m)):3. for m in range(-30,31,15)}
    out,_=smooth(d);assert np.isclose(out['Parma Ponte Verdi',t],3.)
    d['Parma Ponte Verdi',t+timedelta(minutes=30)]=6.
    out,_=smooth(d);assert out['Parma Ponte Verdi',t]>3.,'Centered smoothing must use future neighbor'
    d={('Parma Ponte Verdi',t):3.}
    out,_=smooth(d);assert ('Parma Ponte Verdi',t) not in out
    m=measure(np.array([1.,3.,1.]),np.array([1.,1.,3.]),np.array(['a']*3),{'a':{'peak_m':28}})
    assert m['peak_rmse_m']==0 and m['observed_peak_time_rmse_m']==2
    print('Paper protocol tests passed: kernel, constant preservation, centered look-ahead, missing windows and peak timing.')


def run(trials):
    OUT.mkdir(exist_ok=True)
    prepared,events,assignments,counts=prepare()
    # Match the paper's three-split, 3/6/9-hour tuning structure. Which original three
    # splits were used is unreported, so we explicitly choose published splits 1, 2, 3.
    tuning=[(combine(prepared,assignments,s,h,'t'),combine(prepared,assignments,s,h,'v'))
            for s in [1,2,3] for h in [3,6,9]]
    def objective(p):
        scores=[]
        for train,val in tuning:
            pred=build(p).fit(train[1],train[2]).predict(val[1])
            scores.append(float(np.sqrt(np.mean((pred-val[2])**2))))
        print(f'Tuning candidate mean validation RMSE: {np.mean(scores):.4f} m',flush=True)
        return np.mean(scores)
    initial={'learning_rate':.05,'max_iter':200,'max_leaf_nodes':15,'min_samples_leaf':20}
    best,history=search(objective,trials,42,initial,decode_boost,encode_boost)
    write_csv(OUT/'tuning_trials.csv',history)
    (OUT/'selected_parameters.json').write_text(json.dumps(best,indent=2)+'\n')
    results=[];overlap=[]
    for split in range(1,21):
        for horizon in range(1,10):
            train=combine(prepared,assignments,split,horizon,'t')
            val=combine(prepared,assignments,split,horizon,'v')
            test=combine(prepared,assignments,split,horizon,'x')
            overlap.append(dict(split=split,horizon_h=horizon,
                train_validation_shared_targets=len(set(train[4])&set(val[4])),
                train_test_shared_targets=len(set(train[4])&set(test[4])),
                validation_test_shared_targets=len(set(val[4])&set(test[4]))))
            for model,column in [('persistence',None),('boost_raw',0),('boost_centered',1)]:
                pred=test[0][:,-1] if column is None else build(best).fit(train[column],train[2]).predict(test[column])
                results.append(dict(split=split,horizon_h=horizon,model=model,**measure(test[2],pred,test[3],events)))
        write_csv(OUT/'split_metrics.csv',results)
        write_csv(OUT/'overlap_audit.csv',overlap)
        print(f'Completed published split {split}/20',flush=True)
    summary=[]
    for horizon in range(1,10):
        for model in ['persistence','boost_raw','boost_centered']:
            subset=[r for r in results if r['horizon_h']==horizon and r['model']==model]
            assert len(subset)==20
            r=dict(horizon_h=horizon,model=model,splits=20)
            for metric in ['rmse_m','mae_m','peak_rmse_m','observed_peak_time_rmse_m']:
                values=[x[metric] for x in subset]
                r[f'mean_{metric}']=float(np.mean(values));r[f'std_{metric}']=float(np.std(values,ddof=1))
            summary.append(r)
    write_csv(OUT/'mean_metrics.csv',summary)
    meta=dict(native_smoothing_counts=counts,centered_lookahead_minutes=30,event_timezone_assumption='UTC',
        nominal_support_minutes=40,kernel_h_minutes=20,trials=trials,tuning_splits=[1,2,3],tuning_horizons=[3,6,9],
        samples_by_horizon={str(h):sum(len(prepared[e,h][2]) for e in events) for h in range(1,10)},
        paper_samples={'1':9484,'9':9108},splits_with_train_test_shared_targets=sorted({r['split'] for r in overlap if r['train_test_shared_targets']}),
        source_sha256={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                      for name in ['combined_river_data.csv','flood-events.csv','paper_splits.csv']},
        limitations=['Native 15/30-minute data, not original ten-minute data; normalized discrete cubic kernel adaptation.',
         'Rounded durations and unconfirmed event timezone; missing/conflicting windows excluded.',
         'No gauge offsets or physical normalization bounds invented; tree inputs retained in exported meters.',
         'Published memberships retained; overlapping sample times are audited, not silently purged.',
         'Centered filter is historical/as-described and uses later readings; not a real-time causal forecast.',
         'Published 20 splits are reused across tuning/evaluation; this is not nested independent confirmation.'])
    (OUT/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


def report():
    with (OUT/'mean_metrics.csv').open() as f:
        results=list(csv.DictReader(f))
    meta=json.loads((OUT/'protocol.json').read_text())
    lines=['# Closest available paper-protocol experiment','',
        'All 47 published event IDs, all 20 published splits, all nine hourly forecast horizons, and 12 past readings per station. Peak analysis uses events with published peaks above 27.7 m. One new boosting configuration was selected by mean validation RMSE across published splits 1–3 and horizons 3/6/9, then held fixed.','',
        'Each reported metric is the arithmetic mean of its per-split value, not a pooled error across splits. Units are meters.','',
        '| Ahead | Raw boosting RMSE | Centered boosting RMSE | Centered peak RMSE | Paper best overall RMSE |',
        '|---|---:|---:|---:|---:|']
    reference={3:.072,6:.131,9:.250}
    for h in range(1,10):
        rows={r['model']:r for r in results if int(r['horizon_h'])==h}
        raw=float(rows['boost_raw']['mean_rmse_m']);center=float(rows['boost_centered']['mean_rmse_m'])
        peak=float(rows['boost_centered']['mean_peak_rmse_m'])
        published=f'{reference[h]:.4f}' if h in reference else '—'
        lines.append(f'| {h} h | {raw:.4f} | {center:.4f} | {peak:.4f} | {published} |')
    lines += ['', '## Coverage and interpretation','',
        f"One-hour paired samples: {meta['samples_by_horizon']['1']:,} (paper: 9,484). Nine-hour paired samples: {meta['samples_by_horizon']['9']:,} (paper: 9,108). See DISCUSSION_NOTES.md for the missing-observation/window distinction.", '',
        'Raw and centered boosting use identical targets and the same selected settings. Parameters were selected using centered inputs, so this is not a separately optimized raw-versus-smoothed contest. Changes from the earlier chronological pilot also include event splits, eligible samples and the tuning objective; they cannot be attributed to smoothing alone.', '',
        f"Splits with shared train/test target timestamps: {meta['splits_with_train_test_shared_targets']}. Published event memberships were retained. The audit counts shared target times, not every shared input observation.", '',
        '## Remaining differences','']
    lines += ['- '+x for x in meta['limitations']]
    lines += ['', 'The neural and SVR models from the article were not retrained here. This is the existing boosting experiment moved closer to the article’s data protocol. Missing-value handling in the paper is unknown; do not claim the authors necessarily filled the gaps present in our current exports.', '',
        'Source: https://doi.org/10.3390/w13121612 (Sections 3.2–3.3; Tables 3–4; supplementary Tables S1–S2). Cubic kernel convention: https://pysph.readthedocs.io/en/main/reference/kernels.html', '',
        'Checks: kernel shape and constant preservation, missing neighborhoods, centered look-ahead, peak timing, all 20 partition counts, paired target equality, trial-selection audit and full split/horizon coverage. Not run: exact ten-minute replication (observations unavailable in these exports), physical normalization (numeric bounds unreported), or new untouched confirmation (published split reuse).']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--trials',type=int,default=24)
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.trials)
        report()
