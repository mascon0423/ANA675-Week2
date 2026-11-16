"""Bounded MLP Bayesian architecture search and separate NEAT pilot on published splits."""
import argparse
import copy
import csv
import json
import random
from time import perf_counter
import neat
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from paper_protocol import ROOT, prepare, combine, measure, write_csv
from tune_tree import search

OUT=ROOT/'results/neural'
HORIZONS=(3,6,9)


def decode(z):
    depth=int(round(1+4*z[0]))
    return dict(layers=tuple(int(round(8+56*x)) for x in z[1:depth+1]),
                learning_rate_init=float(10**(-4+2*z[6])),alpha=float(10**(-6+4*z[7])))


def encode(p):
    widths=[(n-8)/56 for n in p['layers']]
    return [(len(widths)-1)/4]+widths+[0.]*(5-len(widths))+[(np.log10(p['learning_rate_init'])+4)/2,(np.log10(p['alpha'])+6)/4]


def scaled(train,val):
    xs=StandardScaler().fit(train[1]);ys=StandardScaler().fit(train[2][:,None])
    return xs,ys,xs.transform(train[1]),ys.transform(train[2][:,None]).ravel(),xs.transform(val[1])


def fit_mlp(p,train,val,epochs,seed,scorer=None):
    xs,ys,x,y,xv=scaled(train,val)
    model=MLPRegressor(hidden_layer_sizes=p['layers'],activation='relu',solver='adam',
        learning_rate_init=p['learning_rate_init'],alpha=p['alpha'],batch_size=min(256,len(x)),
        early_stopping=False,random_state=seed)
    best=None;score=float('inf');stale=0;best_epoch=0
    for epoch in range(1,epochs+1):
        model.partial_fit(x,y)
        pred=ys.inverse_transform(model.predict(xv)[:,None]).ravel()
        current=float(scorer(pred) if scorer else np.sqrt(np.mean((pred-val[2])**2)))
        if not np.isfinite(current): raise ValueError('Nonfinite MLP validation error')
        if current < score-1e-6:
            score=current;best=copy.deepcopy(model);stale=0;best_epoch=epoch
        else: stale+=1
        if stale>=20: break
    return best,xs,ys,score,best_epoch


def neat_config():
    return neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,
                       neat.DefaultStagnation,str(ROOT/'neat_config.ini'))


def batch_predict(network,x):
    # ponytail: only sum aggregation and relu/identity are supported; scalar parity
    # is tested below. Extend this dispatch before enabling other gene functions.
    values={k:x[:,i] for i,k in enumerate(network.input_nodes)}
    for k in network.output_nodes: values[k]=np.zeros(len(x))
    for node,act,agg,bias,response,links in network.node_evals:
        if agg.__name__!='sum_aggregation': raise ValueError('Unsupported aggregation')
        v=bias+response*sum((values[i]*w for i,w in links),np.zeros(len(x)))
        if act.__name__=='relu_activation': v=np.maximum(0,v)
        elif act.__name__!='identity_activation': raise ValueError('Unsupported activation')
        values[node]=v
    return values[network.output_nodes[0]]


def fit_neat(train,val,generations,seed,population=60,scorer=None):
    random.seed(seed);np.random.seed(seed)
    xs,ys,x,y,xv=scaled(train,val)
    config=neat_config();config.pop_size=population;pop=neat.Population(config)
    # Output must be linear for standardized regression. Structural mutations
    # create ReLU hidden nodes; output activation mutation is disabled in config.
    for genome in pop.population.values(): genome.nodes[0].activation='identity'
    best=None;best_score=float('inf');history=[]
    def evaluate(genomes,config):
        nonlocal best,best_score
        champion=None
        for _,genome in genomes:
            net=neat.nn.FeedForwardNetwork.create(genome,config)
            err=np.mean((batch_predict(net,x)-y)**2)
            genome.fitness=-float(err) if np.isfinite(err) else -1e30
            if champion is None or genome.fitness>champion.fitness: champion=genome
        net=neat.nn.FeedForwardNetwork.create(champion,config)
        pred=ys.inverse_transform(batch_predict(net,xv)[:,None]).ravel()
        score=float(scorer(pred) if scorer else np.sqrt(np.mean((pred-val[2])**2)))
        if score<best_score: best_score=score;best=copy.deepcopy(champion)
        history.append(dict(generation=len(history)+1,training_mse_scaled=-champion.fitness,
                            validation_rmse_m=score,nodes=len(champion.nodes),connections=len(champion.connections)))
    pop.run(evaluate,generations)
    if best is None: raise ValueError('No finite NEAT validation result')
    return neat.nn.FeedForwardNetwork.create(best,config),xs,ys,best_score,history


def self_test():
    p=dict(layers=(64,32,16),learning_rate_init=.001,alpha=.0001)
    q=decode(encode(p));assert q['layers']==p['layers']
    assert len(encode(p))==8 and decode(np.ones(8))['layers']==(64,)*5
    random.seed(42);c=neat_config();pop=neat.Population(c)
    g=next(iter(pop.population.values()));g.nodes[0].activation='identity'
    g.mutate_add_node(c.genome_config)
    net=neat.nn.FeedForwardNetwork.create(g,c)
    x=np.random.default_rng(42).normal(size=(8,36))
    assert np.allclose(batch_predict(net,x),[net.activate(row)[0] for row in x])
    p,h=search(lambda p:sum(p['layers']),7,42,p,decode,encode)
    assert len(h)==7 and h[-1]['phase']=='bayesian'
    y=x[:,0]*.2+2
    train=(x[:6],x[:6],y[:6]);val=(x[6:]+10,x[6:]+10,y[6:])
    _,xs,_,score,_=fit_mlp(dict(layers=(8,4),learning_rate_init=.001,alpha=.0001),train,val,2,42)
    assert np.allclose(xs.mean_,x[:6].mean(axis=0)) and np.isfinite(score)
    _,xs,_,score,hist=fit_neat(train,val,2,42)
    assert len(hist)==2 and np.isfinite(score) and np.allclose(xs.mean_,x[:6].mean(axis=0))
    print('Independent widths, Bayesian search, NEAT parity, train-only scaling and both fit paths passed.',flush=True)


def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    prepared,events,assignments,_=prepare()
    groups={(s,h,k):combine(prepared,assignments,s,h,k)
            for s in range(1,21) for h in HORIZONS for k in ('t','v','x')}
    initial=dict(layers=(16,),learning_rate_init=.001,alpha=.0001)
    def objective(p):
        scores=[fit_mlp(p,groups[s,h,'t'],groups[s,h,'v'],args.epochs,42)[3]
                for s in (1,2,3) for h in HORIZONS]
        score=float(np.mean(scores))
        print(f'MLP candidate {p}: mean validation RMSE {score:.4f} m',flush=True)
        return score
    started=perf_counter()
    best,history=search(objective,args.trials,42,initial,decode,encode)
    write_csv(OUT/'mlp_trials.csv',history)
    (OUT/'selected_mlp.json').write_text(json.dumps(best,indent=2)+'\n')
    results=[];evolution=[]
    for split in range(1,21):
        for horizon in HORIZONS:
            train,val,test=[groups[split,horizon,k] for k in ('t','v','x')]
            for name,p in [('mlp_baseline',initial),('mlp_bayesian',best)]:
                model,xs,ys,score,epochs=fit_mlp(p,train,val,args.epochs,42)
                pred=ys.inverse_transform(model.predict(xs.transform(test[1]))[:,None]).ravel()
                results.append(dict(split=split,horizon_h=horizon,model=name,validation_rmse_m=score,
                    budget_used=epochs,**measure(test[2],pred,test[3],events)))
            model,xs,ys,score,hist=fit_neat(train,val,args.generations,42)
            pred=ys.inverse_transform(batch_predict(model,xs.transform(test[1]))[:,None]).ravel()
            results.append(dict(split=split,horizon_h=horizon,model='neat',validation_rmse_m=score,
                budget_used=args.generations,**measure(test[2],pred,test[3],events)))
            evolution.extend(dict(split=split,horizon_h=horizon,**r) for r in hist)
            write_csv(OUT/'split_metrics.csv',results);write_csv(OUT/'neat_generations.csv',evolution)
            print(f'Completed neural evaluation split {split}/20, horizon {horizon}h',flush=True)
    summary=[]
    for h in HORIZONS:
        for name in ('mlp_baseline','mlp_bayesian','neat'):
            rows=[r for r in results if r['horizon_h']==h and r['model']==name]
            assert len(rows)==20
            summary.append(dict(horizon_h=h,model=name,**{f'mean_{m}':float(np.mean([r[m] for r in rows]))
                for m in ('rmse_m','mae_m','peak_rmse_m','observed_peak_time_rmse_m')}))
    write_csv(OUT/'mean_metrics.csv',summary)
    (OUT/'run.json').write_text(json.dumps(dict(trials=args.trials,epochs=args.epochs,generations=args.generations,
        population=60,seed=42,horizons=HORIZONS,elapsed_seconds=perf_counter()-started,
        note='Exploratory, one seed per fit; train-only standardization, published validation early stopping; unequal MLP/NEAT compute budgets; inherited paper-protocol limitations.'),indent=2)+'\n')
    print('All neural experiments complete.',flush=True)


def report():
    with (OUT/'mean_metrics.csv').open() as f: neural=list(csv.DictReader(f))
    with (ROOT/'results/paper_protocol/mean_metrics.csv').open() as f: trees=list(csv.DictReader(f))
    params=json.loads((OUT/'selected_mlp.json').read_text())
    paper_overall={3:.072,6:.131,9:.250};paper_peak={3:.135,6:.150,9:.424}
    lines=['# Neural-model pilot results','',
        'Mean of 20 published splits, using the same reconstructed centered hourly inputs as the closest-available boosting experiment. Lower errors are better; all table values are centimeters.','',
        '## Overall RMSE','',
        '| Ahead | Initial MLP | Bayesian MLP | NEAT | Tuned boosting | Paper best |',
        '|---|---:|---:|---:|---:|---:|']
    for metric,title,reference in [('rmse_m','Overall RMSE',paper_overall),('peak_rmse_m','Peak-height RMSE',paper_peak)]:
        if metric!='rmse_m':
            lines+=['','## '+title,'','| Ahead | Initial MLP | Bayesian MLP | NEAT | Tuned boosting | Paper LSTM |','|---|---:|---:|---:|---:|---:|']
        for h in HORIZONS:
            rows={r['model']:r for r in neural if int(r['horizon_h'])==h}
            boost=next(r for r in trees if int(r['horizon_h'])==h and r['model']=='boost_centered')
            vals=[float(rows[n]['mean_'+metric])*100 for n in ('mlp_baseline','mlp_bayesian','neat')]
            vals += [float(boost['mean_'+metric])*100,reference[h]*100]
            lines.append('| '+str(h)+' h | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['','## Search and limits','',f'Selected MLP settings: `{json.dumps(params)}`.','',
        'MLP search: 24 candidates by validation RMSE across splits 1–3 and 3/6/9-hour horizons. ReLU and Adam; 1–5 layers, independent widths 8–64, learning rate and regularization. Each fit uses training-only standardization and published validation events for early stopping (maximum 200 epochs, patience 20). Initial MLP: one layer of 16 neurons.','',
        'NEAT: population 60 for 60 generations per split/horizon; topology and weights evolve on training MSE, validation chooses a generation champion. Linear output and ReLU hidden nodes. One seed per fit; no ten-model ensemble. Budgets are unequal across methods, so this is an exploratory configuration comparison.','',
        'All earlier reconstruction limits apply: coarser native measurements, adapted centered smoothing that uses later readings, rounded event endpoints, assumed UTC and omitted incomplete windows. Neural scaling uses training-only standardization instead of unavailable physical normalization bounds. Test events are reused across published splits and prior experiments; these are not untouched independent confirmation results. The original paper models are reference numbers, not reruns on our data.','',
        'Checks: independent layer-width encoding, eight-dimensional Bayesian search, vectorized/scalar NEAT parity, training-only scaling and both fitting paths. Not run: exact paper replication, multiple random-seed ensembles, or equal-compute optimizer benchmarking.','',
        'Paper source: https://doi.org/10.3390/w13121612 (Tables 3–4).']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--trials',type=int,default=24);parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--generations',type=int,default=60);args=parser.parse_args()
    if args.self_test: self_test()
    else:
        if args.trials<7 or args.epochs<1 or args.generations<1: parser.error('Invalid search budget')
        run(args)
        report()
