import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

frame=pd.read_csv('results/per-condition.csv');report=json.loads(Path('results/metrics.json').read_text())
names=list(report['models']);labels=['No perturbation','Additive singles','Ridge residual','Neural residual']
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
axes[0].bar(range(4),[report['models'][m]['mean_condition_rmse'] for m in names],color=['#888','#ad661b','#75995f','#216e91'])
axes[0].set_xticks(range(4),labels,rotation=25,ha='right');axes[0].set_ylabel('Mean held-out combination RMSE')
pivot=frame.pivot(index='condition',columns='model',values='all_gene_rmse')
axes[1].scatter(pivot.single_gene_additivity,pivot.symmetric_neural_residual,color='#216e91')
maximum=float(pivot.single_gene_additivity.max());axes[1].plot([0,maximum],[0,maximum],'--',color='#444')
axes[1].set_xlabel('Additive RMSE');axes[1].set_ylabel('Neural residual RMSE')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig('results/combinations.png',dpi=180);fig.savefig('results/combinations.svg')
