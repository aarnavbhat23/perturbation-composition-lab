"""Aggregate a backed AnnData matrix without loading its cells into RAM."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://dataverse.harvard.edu/api/access/datafile/6154020'


def canonical_condition(condition):
    genes = sorted(g for g in condition.split('+') if g != 'ctrl')
    if len(genes) != len(set(genes)):
        raise ValueError('Repeated gene perturbation')
    return '+'.join(genes) or 'ctrl'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('data/pseudobulk.npz'))
    args = parser.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    frame = ad.read_h5ad(args.data,backed='r')
    conditions = np.array([canonical_condition(c) for c in frame.obs.condition.astype(str)])
    names, indexes = np.unique(conditions,return_inverse=True)
    sums = np.zeros((len(names),frame.n_vars),dtype='float64')
    square = np.zeros_like(sums); counts = np.bincount(indexes)
    for offset in range(0,frame.n_obs,512):
        block = frame.X[offset:offset+512]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(block).all(): raise ValueError('Non-finite expression')
        np.add.at(sums,indexes[offset:offset+512],block)
        np.add.at(square,indexes[offset:offset+512],block.astype('float64')**2)
        if offset % 10240 == 0: print('aggregated',offset,flush=True)
    means = sums/counts[:,None]
    variance = np.maximum(square/counts[:,None]-means**2,0)
    np.savez_compressed(args.output,conditions=names,means=means.astype('float32'),
                        standard_errors=np.sqrt(variance/counts[:,None]).astype('float32'),
                        counts=counts,genes=np.asarray(frame.var.gene_name.astype(str),dtype='U'))
    digest = hashlib.sha256()
    with args.data.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    manifest = {'source':SOURCE,'processed_h5ad_sha256':digest.hexdigest(),
        'cells':int(frame.n_obs),'genes':int(frame.n_vars),'canonical_conditions':len(names),
        'upstream_cell_type_labels':frame.obs.cell_type.astype(str).unique().tolist(),
        'identity_note':'GEARS export labels cells A549; Norman et al. original experiment is K562. This project does not use the cell_type field.',
        'expression_note':'Means of the upstream processed expression scale; not raw counts or fold changes.',
        'uncertainty_note':'Cell-level standard errors are descriptive, not independent biological-replicate uncertainty.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
