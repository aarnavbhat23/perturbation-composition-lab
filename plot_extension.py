from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "font.size": 11,
        "figure.facecolor": "#fafbfd",
        "axes.facecolor": "#fafbfd",
    }
)
colors = {
    "additive": "#95a5ad",
    "matched_ridge": "#2d7790",
    "neural_ensemble": "#ba5b39",
    "endpoint_gaussian": "#95a5ad",
    "observation_likelihood": "#ba5b39",
    "constant_observation": "#2d7790",
}
p = Path("results/replication")
df = pd.read_csv(p / "partition-results.csv")
table = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for r, dataset in enumerate(["norman", "wessels"]):
    for c, setting in enumerate(["random", "heldout-combination-gene"]):
        subset = df[(df.dataset == dataset) & df.partition.str.startswith(setting)]
        ax = axes[r, c]
        labels = sorted(subset.partition.unique())
        for method in ["additive", "matched_ridge", "neural_ensemble"]:
            rows = subset[subset.method == method].set_index("partition").loc[labels]
            ax.plot(
                range(len(labels)),
                rows.mean_condition_rmse,
                marker="o",
                label=method.replace("_", " "),
                color=colors[method],
            )
            table.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "method": method,
                    "partitions": len(rows),
                    "mean_rmse": float(rows.mean_condition_rmse.mean()),
                    "min_rmse": float(rows.mean_condition_rmse.min()),
                    "max_rmse": float(rows.mean_condition_rmse.max()),
                }
            )
        ax.set(
            title=f"{dataset.title()} · "
            + ("random combinations" if c == 0 else "gene combination holdout"),
            ylabel="Mean combination RMSE",
            xticks=range(len(labels)),
            xticklabels=[x.split("-")[-1] for x in labels],
        )
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.15)
axes[0, 0].legend(frameon=False)
fig.suptitle(
    "Two experiments · matched inputs · every evaluated partition", fontsize=18
)
fig.savefig(p / "comparison.png", dpi=150)
fig.savefig(p / "comparison.svg")
plt.close(fig)
(p / "summary.json").write_text(
    json.dumps(
        {
            "comparison": table,
            "note": "Means/min/max across dependent partitions; not biological replicate intervals.",
        },
        indent=2,
    )
    + "\n"
)
