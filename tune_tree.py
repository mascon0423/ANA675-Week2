"""Small Gaussian-process Bayesian search; never uses test data to select settings."""
import argparse
import csv
import json
from datetime import datetime
from time import perf_counter

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from main import ROOT, load_readings, samples, metrics


def decode(z):
    return {'max_depth':int(round(3+21*z[0])),
            'min_samples_leaf':int(round(2**(6*z[1]))),
            'min_samples_split':int(round(2**(1+6*z[2]))),
            'max_features':float(round(.5+.5*z[3],3))}


def encode(p):
    return [(p['max_depth']-3)/21,np.log2(p['min_samples_leaf'])/6,
            (np.log2(p['min_samples_split'])-1)/6,(p['max_features']-.5)/.5]


def decode_boost(z):
    return {'learning_rate':float(.01*20**z[0]), 'max_iter':int(round(50+300*z[1])),
            'max_leaf_nodes':int(round(5+26*z[2])), 'min_samples_leaf':int(round(5*16**z[3]))}


def encode_boost(p):
    return [np.log(p['learning_rate']/.01)/np.log(20), (p['max_iter']-50)/300,
            (p['max_leaf_nodes']-5)/26, np.log(p['min_samples_leaf']/5)/np.log(16)]


def expected_improvement(mean, std, best):
    sigma=np.maximum(std,1e-12)
    improvement=best-mean
    z=improvement/sigma
    ei=improvement*ndtr(z)+sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(std>1e-12,ei,np.maximum(improvement,0))


def search(objective, trials, seed, initial=None, decoder=decode, encoder=encode):
    if trials < 7:
        raise ValueError('At least seven trials required for Bayesian phase')
    rng=np.random.default_rng(seed)
    params=[initial if initial is not None else {'max_depth':8,'min_samples_leaf':10,'min_samples_split':2,'max_features':1.0}]
    scores=[]; history=[]; seen=set()
    for trial in range(trials):
        if trial:
            candidates=[]; candidate_keys=set()
            while len(candidates)<512:
                p=decoder(rng.random(4)); key=tuple(p.values())
                if key not in seen and key not in candidate_keys:
                    candidates.append(p); candidate_keys.add(key)
            if trial < 6:
                p=candidates[0]
            else:
                # ponytail: fixed Matern length scale and a 512-point candidate pool keep
                # this pilot cheap; larger studies can fit kernel scales and refine acquisition.
                gp=GaussianProcessRegressor(kernel=Matern(length_scale=.3,nu=2.5),
                    optimizer=None,alpha=1e-6,normalize_y=True)
                gp.fit(np.asarray([encoder(p) for p in params]),scores)
                mean,std=gp.predict(np.asarray([encoder(p) for p in candidates]),return_std=True)
                p=candidates[int(np.argmax(expected_improvement(mean,std,min(scores))))]
            params.append(p)
        p=params[-1]; seen.add(tuple(p.values()))
        score=float(objective(p))
        if not np.isfinite(score):
            raise ValueError('Objective must be finite')
        scores.append(score)
        history.append(dict(trial=trial+1,phase='baseline' if trial==0 else 'random' if trial<6 else 'bayesian',
                            validation_rmse_m=score,**p))
    return params[int(np.argmin(scores))],history


def self_test():
    assert decode(np.zeros(4))=={'max_depth':3,'min_samples_leaf':1,'min_samples_split':2,'max_features':.5}
    assert decode(np.ones(4))=={'max_depth':24,'min_samples_leaf':64,'min_samples_split':128,'max_features':1.0}
    assert np.allclose(expected_improvement(np.array([0.,2.]),np.zeros(2),1.),[1.,0.])
    objective=lambda p: (p['max_depth']-7)**2 + p['min_samples_leaf']/100
    p,h=search(objective,8,42)
    assert len(h)==8 and len({tuple(r[k] for k in p) for r in h})==8
    assert objective(p)==min(r['validation_rmse_m'] for r in h)
    assert h[-1]['phase']=='bayesian'
    assert np.allclose(encode_boost(decode_boost(np.zeros(4))), np.zeros(4))
    assert np.allclose(encode_boost(decode_boost(np.ones(4))), np.ones(4))
    initial=decode_boost(np.zeros(4))
    bp,bh=search(lambda p: p['learning_rate'],8,42,initial,decode_boost,encode_boost)
    assert bp['learning_rate']==.01 and bh[-1]['phase']=='bayesian'
    print('Bayesian search checks passed: bounds, EI, uniqueness, selection and Bayesian phase.')


def run(trials, kind='tree'):
    is_boost=kind=='boosting'
    model_name='gradient_boosting' if is_boost else 'decision_tree'
    initial={'learning_rate':.05,'max_iter':200,'max_leaf_nodes':15,'min_samples_leaf':20} if is_boost else None
    decoder,encoder=(decode_boost,encode_boost) if is_boost else (decode,encode)
    def build(p):
        return HistGradientBoostingRegressor(early_stopping=False,random_state=42,**p) if is_boost else DecisionTreeRegressor(random_state=42,**p)
    output=ROOT/f'results/bayesian_{kind}'; output.mkdir(exist_ok=True)
    summaries=[]; all_trials=[]; predicted=[]
    for step in [30,60]:
        data=load_readings(ROOT/'data/combined_river_data.csv',step)
        baseline_path=ROOT/f'results/{step}min'
        manifest=json.loads((baseline_path/'event_splits.json').read_text())
        baseline_meta=json.loads((baseline_path/'run_metadata.json').read_text())
        import hashlib
        for name,digest in baseline_meta['input_sha256'].items():
            assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest,'Input changed since baseline'
        assert all(datetime.fromisoformat(a['end_exclusive'])<datetime.fromisoformat(b['start_assumed_utc'])
                   for a,b in zip(manifest,manifest[1:]))
        with (baseline_path/'metrics.csv').open() as f:
            baseline={(r['model'],r['horizon_h'],r['split']):r for r in csv.DictReader(f)}
        for horizon in [1,3,6,9]:
            parts={s:[] for s in ['train','validation','test']}
            for event in manifest:
                x,y,t=samples(data,datetime.fromisoformat(event['start_assumed_utc']),
                    datetime.fromisoformat(event['end_exclusive']),horizon,step)
                if len(y): parts[event['split']].append((x,y,np.full(len(y),event['group']),t))
            arrays={s:(np.concatenate([c[0] for c in chunks]),np.concatenate([c[1] for c in chunks]),
                       np.concatenate([c[2] for c in chunks]),[t for c in chunks for t in c[3]])
                    for s,chunks in parts.items()}
            train_x,train_y=arrays['train'][:2]; val_x,val_y=arrays['validation'][:2]
            # This objective closes over training and validation only, never test arrays.
            def objective(p):
                model=build(p).fit(train_x,train_y)
                return np.sqrt(np.mean((model.predict(val_x)-val_y)**2))
            begin=perf_counter()
            best,history=search(objective,trials,42,initial,decoder,encoder)
            assert np.isclose(history[0]['validation_rmse_m'],
                float(baseline[model_name,str(horizon),'validation']['rmse_m']))
            model=build(best).fit(train_x,train_y)
            x,y,g,times=arrays['test']; pred=model.predict(x)
            row=dict(sampling_minutes=step,horizon_h=horizon,trials=trials,search_seconds=perf_counter()-begin,
                best_validation_rmse_m=min(r['validation_rmse_m'] for r in history),
                baseline_validation_rmse_m=history[0]['validation_rmse_m'],
                baseline_test_rmse_m=float(baseline[model_name,str(horizon),'test']['rmse_m']),
                boosting_test_rmse_m=float(baseline['gradient_boosting',str(horizon),'test']['rmse_m']),
                **metrics(y,pred,g),**best)
            assert len(y)==int(baseline[model_name,str(horizon),'test']['samples'])
            summaries.append(row)
            all_trials.extend(dict(sampling_minutes=step,horizon_h=horizon,**r) for r in history)
            predicted.extend(dict(sampling_minutes=step,horizon_h=horizon,group=int(group),target_utc=t.isoformat(),
                observed_m=float(obs),predicted_m=float(p)) for group,t,obs,p in zip(g,times,y,pred))
            for name,rows in [('metrics.csv',summaries),('trials.csv',all_trials),('predictions.csv',predicted)]:
                with (output/name).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            print(f"{step}min {horizon}h: baseline {kind} {row['baseline_test_rmse_m']:.4f}, tuned {row['rmse_m']:.4f}, boosting {row['boosting_test_rmse_m']:.4f} m RMSE",flush=True)
    (output/'search_config.json').write_text(json.dumps(dict(trials=trials,seed=42,initial_points=6,
        surrogate='Gaussian process, fixed Matern 2.5 kernel, length scale 0.3, normalized inputs/targets',
        acquisition='expected improvement; 512 random untried candidates per iteration',
        objective='validation sample-weighted RMSE; no test-guided selection; training-only refit',
        model=kind, bounds=({'learning_rate':[.01,.2],'max_iter':[50,350],'max_leaf_nodes':[5,31],'min_samples_leaf':[5,80]} if is_boost else {'max_depth':[3,24],'min_samples_leaf':[1,64],'min_samples_split':[2,128],'max_features':[.5,1.]}),
        limitation='Previously inspected pilot test periods; not an untouched confirmatory evaluation.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--trials',type=int,default=24)
    p.add_argument('--model',choices=['tree','boosting'],default='tree')
    args=p.parse_args()
    self_test() if args.self_test else run(args.trials,args.model)
