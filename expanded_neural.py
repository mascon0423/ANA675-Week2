"""Resumable expanded neural experiment; historical outputs are preserved."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import neural_experiments as n
from paper_protocol import prepare,combine,measure,write_csv,ROOT
from tune_tree import search

OUT=ROOT/'results/neural_expanded'
SEEDS=tuple(range(42,52))


def save_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def score_function(val,events):
    masks=[val[3]==e for e in np.unique(val[3]) if float(events[e]['peak_m'])>27.7]
    if not masks: raise ValueError('No eligible validation peak events')
    peaks=np.array([val[2][m].max() for m in masks])
    def score(pred):
        overall=np.sqrt(np.mean((pred-val[2])**2))
        peak=np.sqrt(np.mean((np.array([pred[m].max() for m in masks])-peaks)**2))
        return float(.5*overall+.5*peak)
    return score


def mlp_predictions(p,train,val,test,epochs,seeds,scorer):
    vp=[];tp=[];stops=[]
    for seed in seeds:
        model,xs,ys,_,epoch=n.fit_mlp(p,train,val,epochs,seed,scorer=scorer)
        vp.append(ys.inverse_transform(model.predict(xs.transform(val[1]))[:,None]).ravel())
        if test is not None:
            tp.append(ys.inverse_transform(model.predict(xs.transform(test[1]))[:,None]).ravel())
        stops.append(epoch)
    return np.mean(vp,axis=0),np.mean(tp,axis=0) if tp else None,stops


def self_test():
    y=np.array([1.,3.,1.]);v=(None,None,y,np.array(['a']*3))
    score=score_function(v,{'a':{'peak_m':28}})
    assert score(y)==0 and np.isclose(score(np.array([1.,1.,3.])),.5*np.sqrt(8/3))
    assert np.isclose(score(y+1),1)
    n.self_test()
    # Test custom scoring is actually applied to checkpoint selection.
    x=np.random.default_rng(7).normal(size=(12,36));y=x[:,0]
    train=(x[:8],x[:8],y[:8]);val=(x[8:],x[8:],y[8:])
    p=dict(layers=(8,4),learning_rate_init=.001,alpha=.0001)
    assert n.fit_mlp(p,train,val,2,42,scorer=lambda pred:7.)[3]==7.
    assert n.fit_neat(train,val,2,42,population=8,scorer=lambda pred:7.)[3]==7.
    print('Balanced objective, optional checkpoint scoring and small-population NEAT passed.',flush=True)


def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=dict(trials=args.trials,epochs=args.epochs,population=args.population,generations=args.generations,
        search_seeds=list(SEEDS[:3]),mlp_ensemble_seeds=list(SEEDS),neat_ensemble_seeds=list(SEEDS[:3]),
        tuning_splits=[1,2,3],horizons=list(n.HORIZONS),objective='0.5 overall RMSE + 0.5 eligible-event peak RMSE',
        data_hashes={f:hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest() for f in
            ['combined_river_data.csv','flood-events.csv','paper_splits.csv']})
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Existing expanded run uses different settings or inputs; preserve it in a separate folder.')
    save_json(path,manifest)
    prepared,events,assignments,_=prepare()
    groups={(s,h,k):combine(prepared,assignments,s,h,k)
            for s in range(1,21) for h in n.HORIZONS for k in ('t','v','x')}
    old=json.loads((n.OUT/'selected_mlp.json').read_text());old['layers']=tuple(old['layers'])
    cache_path=OUT/'search_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def objective(p):
        key=json.dumps(p,sort_keys=True)
        if key in cache: return cache[key]['score']
        scores=[];started=perf_counter()
        for split in (1,2,3):
            for h in n.HORIZONS:
                train,val=[groups[split,h,k] for k in ('t','v')]
                scorer=score_function(val,events)
                vp,_,_=mlp_predictions(p,train,val,None,args.epochs,SEEDS[:3],scorer)
                scores.append(scorer(vp))
        value=float(np.mean(scores))
        cache[key]=dict(score=value,seconds=perf_counter()-started)
        save_json(cache_path,cache)
        print(f'Expanded candidate {len(cache)}/{args.trials}: {p}; balanced validation score={value:.4f} m',flush=True)
        return value
    best,history=search(objective,args.trials,42,old,n.decode,n.encode)
    # Shared search calls its scalar column validation_rmse_m; here it is a balanced score.
    for r in history: r['validation_balanced_score_m']=r.pop('validation_rmse_m')
    write_csv(OUT/'mlp_trials.csv',history);save_json(OUT/'selected_mlp.json',best)
    results_path=OUT/'split_metrics.json'
    results=json.loads(results_path.read_text()) if results_path.exists() else []
    done={(r['split'],r['horizon_h'],r['model']) for r in results}
    for split in range(1,21):
        for h in n.HORIZONS:
            train,val,test=[groups[split,h,k] for k in ('t','v','x')]
            scorer=score_function(val,events)
            for name,params,objective_fn in [('prior_mlp_ensemble',old,None),('balanced_mlp_ensemble',best,scorer)]:
                if (split,h,name) in done: continue
                vp,pred,stops=mlp_predictions(params,train,val,test,args.epochs,SEEDS,objective_fn)
                results.append(dict(split=split,horizon_h=h,model=name,validation_balanced_score_m=scorer(vp),
                    seeds=len(SEEDS),**measure(test[2],pred,test[3],events)))
                save_json(OUT/f'mlp_stops_{split}_{h}_{name}.json',stops)
                save_json(results_path,results)
                write_csv(OUT/'split_metrics.csv',results)
                print(f'Completed {name}: split {split}/20, {h}h',flush=True)
            name='neat_expanded_ensemble'
            if (split,h,name) in done: continue
            vp=[];tp=[]
            for seed in SEEDS[:3]:
                net,xs,ys,_,hist=n.fit_neat(train,val,args.generations,seed,population=args.population,scorer=scorer)
                vp.append(ys.inverse_transform(n.batch_predict(net,xs.transform(val[1]))[:,None]).ravel())
                tp.append(ys.inverse_transform(n.batch_predict(net,xs.transform(test[1]))[:,None]).ravel())
                for r in hist: r['validation_balanced_score_m']=r.pop('validation_rmse_m')
                write_csv(OUT/f'neat_history_{split}_{h}_{seed}.csv',hist)
            results.append(dict(split=split,horizon_h=h,model=name,validation_balanced_score_m=scorer(np.mean(vp,axis=0)),
                seeds=3,**measure(test[2],np.mean(tp,axis=0),test[3],events)))
            save_json(results_path,results);write_csv(OUT/'split_metrics.csv',results)
            print(f'Completed expanded NEAT: split {split}/20, {h}h',flush=True)
    summary=[]
    for h in n.HORIZONS:
        for name in ('prior_mlp_ensemble','balanced_mlp_ensemble','neat_expanded_ensemble'):
            rows=[r for r in results if r['horizon_h']==h and r['model']==name]
            assert len(rows)==20
            summary.append(dict(horizon_h=h,model=name,**{f'mean_{m}':float(np.mean([r[m] for r in rows]))
                for m in ('rmse_m','mae_m','peak_rmse_m','observed_peak_time_rmse_m')}))
    write_csv(OUT/'mean_metrics.csv',summary)
    print('Expanded experiment complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true')
    p.add_argument('--trials',type=int,default=64);p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--population',type=int,default=150);p.add_argument('--generations',type=int,default=200)
    args=p.parse_args()
    if args.self_test: self_test()
    else:
        if args.trials<7 or min(args.epochs,args.population,args.generations)<1: p.error('Invalid budget')
        run(args)
