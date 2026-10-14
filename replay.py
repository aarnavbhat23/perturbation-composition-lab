import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import metrics
from predict import predict


def main():
    profiles=np.load('results/heldout-profiles.npz');report=json.loads(Path('results/metrics.json').read_text())
    ledger=pd.read_csv('results/split-ledger.csv');assert ledger.condition.is_unique
    for name,values in report['models'].items():
        actual=metrics(profiles['truth'],profiles[name],profiles['single_gene_additivity'])
        assert np.isclose(np.mean([v['all_gene_rmse'] for v in actual]),values['mean_condition_rmse'])
        assert np.isclose(np.mean([v['oracle_top20_rmse'] for v in actual]),values['mean_oracle_top20_rmse'])
    first,second=str(profiles['conditions'][0]).split('+')
    generated=predict(first,second).neural_delta.to_numpy()
    np.testing.assert_allclose(generated,profiles['symmetric_neural_residual'][0],atol=1e-6,rtol=1e-5)
    print('Replayed all-gene and top-20 errors, held-out combinations and released model inference.')


if __name__=='__main__':main()
