"""Predict a combination from measured single-gene responses and released heads."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from composer import ResidualComposer


def predict(first,second):
    if first==second:raise ValueError('Two distinct perturbation genes are required')
    data=np.load('results/preprocessing.npz');lookup={g:i for i,g in enumerate(data['single_names'])}
    if first not in lookup or second not in lookup:raise ValueError('No measured single-gene reference')
    a,b=[data['singles'][lookup[g]] for g in [first,second]]
    z1,z2=[((x-data['basis_mean'])@data['basis'].T-data['zloc'])/data['zscale'] for x in [a,b]]
    residuals=[]
    for seed in [311,312,313]:
        checkpoint=torch.load(Path('results')/f'composer-{seed}.pt',weights_only=True,map_location='cpu')
        model=ResidualComposer(checkpoint['latent']).eval();model.load_state_dict(checkpoint['state'])
        with torch.no_grad():residuals.append(model(torch.tensor(z1)[None],torch.tensor(z2)[None]).numpy()[0])
    residual=np.mean(residuals,0)*data['residual_scale']+data['residual_location']
    return pd.DataFrame({'gene':data['genes'],'additive_delta':a+b,'neural_delta':a+b+residual@data['basis']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second');parser.add_argument('--output',default='prediction.csv')
    args=parser.parse_args();predict(args.first,args.second).to_csv(args.output,index=False)
    print('Saved processed-expression deltas; not a clinical or new-cell-type prediction.')
