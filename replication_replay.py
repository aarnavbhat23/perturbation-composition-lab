"""Recompute every partition metric and recover both fitted heads without training."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark import metrics
from replication import gene_holdout
from replication_inference import predict


def main():
    root = Path("results/replication")
    summary = pd.read_csv(root / "partition-results.csv")
    folders = 0
    for (dataset, label), rows in summary.groupby(["dataset", "partition"]):
        folder = root / dataset / label
        source_path = Path(
            "data/wessels-pseudobulk.npz"
            if dataset == "wessels"
            else "data/pseudobulk.npz"
        )
        config = json.loads((folder / "config.json").read_text())
        assert (
            config["dataset_sha256"]
            == hashlib.sha256(source_path.read_bytes()).hexdigest()
        )
        profiles = np.load(folder / "heldout-profiles.npz")
        ledger = pd.read_csv(folder / "partition.csv")
        folders += 1
        assert ledger.condition.is_unique
        assert ledger.partition.eq("validation").any()
        assert set(profiles["conditions"]) == set(
            ledger.loc[ledger.partition == "test", "condition"]
        )
        if label.startswith("heldout-combination-gene-"):
            gene = label.split("heldout-combination-gene-")[1]
            assert np.array_equal(
                gene_holdout(ledger.condition.tolist(), gene), ledger.partition
            )
        for row in rows.itertuples():
            actual = metrics(
                profiles["truth"], profiles[row.method], profiles["additive"]
            )
            np.testing.assert_allclose(
                np.mean([v["all_gene_rmse"] for v in actual]),
                row.mean_condition_rmse,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                np.mean([v["oracle_top20_rmse"] for v in actual]),
                row.oracle_top20_rmse,
                rtol=1e-6,
            )
        first, second = str(profiles["conditions"][0]).split("+")
        values = predict(first, second, dataset, label)
        for method in ["neural_ensemble", "matched_ridge", "additive"]:
            np.testing.assert_allclose(
                values[method], profiles[method][0], atol=2e-6, rtol=1e-4
            )
    manifest = json.loads((root / "manifest.json").read_text())
    for dataset, path in [
        ("norman", "data/pseudobulk.npz"),
        ("wessels", "data/wessels-pseudobulk.npz"),
    ]:
        assert (
            np.load(path)["counts"].sum()
            == manifest["datasets"][dataset]["source_cells"]
        )
    print(
        f"Replayed {folders} dependent partitions, all metrics and neural/ridge inference; two independent experiments."
    )


if __name__ == "__main__":
    main()
