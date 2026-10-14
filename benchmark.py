"""Predict entirely withheld combinations of already measured single genes."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from composer import ResidualComposer,symmetric_features


def split_pairs(pairs,seed=311):
    order = np.random.default_rng(seed).permutation(len(pairs))
    splits = np.empty(len(pairs),dtype='U10')
    for rank,index in enumerate(order):
        splits[index] = 'train' if rank < .6*len(pairs) else 'validation' if rank < .8*len(pairs) else 'test'
    return splits


def metrics(target,prediction,additive):
    result = []
    for truth,pred,base in zip(target,prediction,additive):
        top = np.argsort(np.abs(truth))[-20:]
        result.append({'all_gene_rmse':float(np.sqrt(np.mean((truth-pred)**2))),
            'oracle_top20_rmse':float(np.sqrt(np.mean((truth[top]-pred[top])**2))),
            'delta_pearson':float(np.corrcoef(truth,pred)[0,1]) if np.std(pred)>0 and np.std(truth)>0 else None,
            'interaction_rmse':float(np.sqrt(np.mean(((truth-base)-(pred-base))**2)))})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=Path('data/pseudobulk.npz'))
    args = parser.parse_args(); torch.set_num_threads(4)
    data = np.load(args.data); lookup = {str(c):i for i,c in enumerate(data['conditions'])}
    control = data['means'][lookup['ctrl']]
    single_names = sorted(c for c in lookup if c != 'ctrl' and '+' not in c)
    singles = np.array([data['means'][lookup[c]]-control for c in single_names])
    single_lookup = {c:i for i,c in enumerate(single_names)}
    pairs, excluded = [], []
    for name in sorted(lookup):
        genes = name.split('+')
        if len(genes) != 2: continue
        if any(g not in single_lookup for g in genes):
            excluded.append({'condition':name,'reason':'missing_single_gene_reference'}); continue
        pairs.append(name)
    splits = split_pairs(pairs); train,validation,test = [splits == s for s in ['train','validation','test']]
    target = np.array([data['means'][lookup[c]]-control for c in pairs])
    first = np.array([singles[single_lookup[c.split('+')[0]]] for c in pairs])
    second = np.array([singles[single_lookup[c.split('+')[1]]] for c in pairs]); additive = first+second
    # The basis is fitted on known singles and training pairs only.
    latent = min(32,len(singles)-1,int(train.sum())-1)
    basis = PCA(n_components=latent,svd_solver='full').fit(np.r_[singles,target[train]])
    z1,z2 = [basis.transform(x).astype('float32') for x in [first,second]]
    residual = (target-additive)@basis.components_.T
    location,scale = residual[train].mean(0),np.maximum(residual[train].std(0),.05)
    y = ((residual-location)/scale).astype('float32')
    x = symmetric_features(torch.from_numpy(z1),torch.from_numpy(z2)).numpy()
    xloc,xscale = x[train].mean(0),np.maximum(x[train].std(0),.05)
    # Scaling singles identically preserves first/second symmetry.
    zloc = np.r_[z1[train],z2[train]].mean(0); zscale = np.maximum(np.r_[z1[train],z2[train]].std(0),.05)
    a,b = [torch.tensor((z-zloc)/zscale) for z in [z1,z2]]
    ridge = Ridge(alpha=10.).fit((x[train]-xloc)/xscale,y[train])
    ridge_prediction = additive+((ridge.predict((x-xloc)/xscale)*scale+location)@basis.components_)
    predictions,traces,checkpoints = [],[],[]; out = Path('results');out.mkdir(exist_ok=True)
    for seed in [311,312,313]:
        torch.manual_seed(seed); model = ResidualComposer(latent)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        best,stale,state,trace = float('inf'),0,None,[]
        for epoch in range(500):
            model.train(); optimizer.zero_grad()
            loss = ((model(a[train],b[train])-torch.tensor(y[train]))**2).mean()
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): val = float(((model(a[validation],b[validation])-torch.tensor(y[validation]))**2).mean())
            trace.append({'epoch':epoch+1,'train_mse':float(loss.detach()),'validation_mse':val})
            if val < best:
                best,stale,state = val,0,copy.deepcopy(model.state_dict())
            else: stale += 1
            if stale == 40: break
        model.load_state_dict(state);model.eval()
        with torch.no_grad(): predictions.append(model(a,b).numpy())
        torch.save({'state':state,'latent':latent},out/f'composer-{seed}.pt')
        traces.append(trace);checkpoints.append(int(np.argmin([t['validation_mse'] for t in trace]))+1)
    neural = additive+((np.mean(predictions,0)*scale+location)@basis.components_)
    methods = {'no_perturbation':np.zeros_like(target),'single_gene_additivity':additive,
               'ridge_residual':ridge_prediction,'symmetric_neural_residual':neural}
    report = {'source_cells':int(data['counts'].sum()),'genes':len(control),'single_genes':len(singles),
              'eligible_combinations':len(pairs),'excluded_combinations':excluded,'latent_dimensions':latent,
              'split_combinations':{s:int((splits == s).sum()) for s in np.unique(splits)},
              'selected_epochs':checkpoints,'models':{}}
    records = []; bootstrap = np.random.default_rng(314).integers(0,int(test.sum()),size=(2000,int(test.sum())))
    base_metrics = np.array([r['all_gene_rmse'] for r in metrics(target[test],additive[test],additive[test])])
    for name,prediction in methods.items():
        values = metrics(target[test],prediction[test],additive[test])
        errors = np.array([r['all_gene_rmse'] for r in values]); delta = (errors-base_metrics)[bootstrap].mean(1)
        report['models'][name] = {'mean_condition_rmse':float(errors.mean()),
            'paired_rmse_difference_vs_additive':float((errors-base_metrics).mean()),
            'condition_bootstrap_95_interval':np.quantile(delta,[.025,.975]).tolist(),
            'mean_oracle_top20_rmse':float(np.mean([r['oracle_top20_rmse'] for r in values]))}
        for condition,values_row in zip(np.array(pairs)[test],values):
            records.append({'condition':condition,'model':name,**values_row})
    pd.DataFrame({'condition':pairs,'split':splits,'cell_count':[int(data['counts'][lookup[c]]) for c in pairs]}).to_csv(out/'split-ledger.csv',index=False)
    pd.DataFrame(records).to_csv(out/'per-condition.csv',index=False)
    np.savez_compressed(out/'heldout-profiles.npz',conditions=np.array(pairs)[test],genes=data['genes'],
                        truth=target[test],**{k:v[test] for k,v in methods.items()})
    np.savez_compressed(out/'preprocessing.npz',basis=basis.components_,basis_mean=basis.mean_,
        zloc=zloc,zscale=zscale,residual_location=location,residual_scale=scale,
        singles=singles,single_names=np.array(single_names),genes=data['genes'],
        ridge_coef=ridge.coef_,ridge_intercept=ridge.intercept_,xloc=xloc,xscale=xscale)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'training-history.json').write_text(json.dumps(traces,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
