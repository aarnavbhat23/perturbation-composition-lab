"""Use one released partition model with measured single-gene references."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from composer import ResidualComposer, symmetric_features
from replication import build
from sklearn.linear_model import Ridge


def add_ridge_artifacts(root=Path("results/replication")):
    for dataset, path in [
        ("norman", "data/pseudobulk.npz"),
        ("wessels", "data/wessels-pseudobulk.npz"),
    ]:
        pairs, a, b, target, _, _ = build(np.load(path))
        for folder in sorted((root / dataset).iterdir()):
            if not folder.is_dir():
                continue
            prep = dict(np.load(folder / "preprocessing.npz"))
            ledger = pd.read_csv(folder / "partition.csv")
            assert ledger.condition.tolist() == pairs
            train = ledger.partition.to_numpy() == "train"
            z = [
                ((x - prep["basis_mean"]) @ prep["basis"].T - prep["input_location"])
                / prep["input_scale"]
                for x in [a, b]
            ]
            features = symmetric_features(
                *[torch.tensor(x, dtype=torch.float32) for x in z]
            ).numpy()
            loc, scale = (
                features[train].mean(0),
                np.maximum(features[train].std(0), 0.05),
            )
            residual = (target - a - b) @ prep["basis"].T
            y = (
                (residual - prep["residual_location"]) / prep["residual_scale"]
            ).astype("float32")
            model = Ridge(alpha=10.0).fit((features[train] - loc) / scale, y[train])
            prep.update(
                ridge_coef=model.coef_,
                ridge_intercept=model.intercept_,
                ridge_location=loc,
                ridge_scale=scale,
            )
            np.savez_compressed(folder / "preprocessing.npz", **prep)


def predict(
    first,
    second,
    dataset="wessels",
    partition="random-311",
    root=Path("results/replication"),
):
    if first == second:
        raise ValueError("Distinct genes required")
    data = np.load(
        "data/wessels-pseudobulk.npz" if dataset == "wessels" else "data/pseudobulk.npz"
    )
    lookup = {str(c): i for i, c in enumerate(data["conditions"])}
    if any(g not in lookup or g == "ctrl" or "+" in g for g in [first, second]):
        raise ValueError("Both measured single references required")
    folder = root / dataset / partition
    prep = np.load(folder / "preprocessing.npz")
    checkpoint = torch.load(
        folder / "neural-checkpoints.pt", map_location="cpu", weights_only=True
    )
    a, b = [
        data["means"][lookup[g]] - data["means"][lookup["ctrl"]]
        for g in [first, second]
    ]
    z = [
        torch.tensor(
            (
                ((x - prep["basis_mean"]) @ prep["basis"].T - prep["input_location"])
                / prep["input_scale"]
            )[None, :],
            dtype=torch.float32,
        )
        for x in [a, b]
    ]
    features = symmetric_features(*z).numpy()
    residuals = []
    for state in checkpoint["states"].values():
        model = ResidualComposer(checkpoint["latent"])
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            residuals.append(model(*z).numpy()[0])
    neural = (
        a
        + b
        + (np.mean(residuals, 0) * prep["residual_scale"] + prep["residual_location"])
        @ prep["basis"]
    )
    ridge_z = ((features - prep["ridge_location"]) / prep["ridge_scale"]) @ prep[
        "ridge_coef"
    ].T + prep["ridge_intercept"]
    ridge = (
        a
        + b
        + (ridge_z[0] * prep["residual_scale"] + prep["residual_location"])
        @ prep["basis"]
    )
    return pd.DataFrame(
        {
            "gene": data["genes"],
            "additive": a + b,
            "matched_ridge": ridge,
            "neural_ensemble": neural,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first", nargs="?")
    parser.add_argument("second", nargs="?")
    parser.add_argument("--dataset", choices=["norman", "wessels"], default="wessels")
    parser.add_argument("--partition", default="random-311")
    parser.add_argument("--output", type=Path, default=Path("prediction.csv"))
    parser.add_argument("--add-ridge-artifacts", action="store_true")
    args = parser.parse_args()
    if args.add_ridge_artifacts:
        add_ridge_artifacts()
    else:
        if not args.first or not args.second:
            parser.error("Two single-gene references required")
        predict(args.first, args.second, args.dataset, args.partition).to_csv(
            args.output, index=False
        )
