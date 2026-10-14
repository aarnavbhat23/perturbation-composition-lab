"""Matched feature comparisons over partitions and a genuinely independent experiment."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from benchmark import metrics, split_pairs
from composer import ResidualComposer, symmetric_features
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def build(data):
    lookup = {str(c): i for i, c in enumerate(data["conditions"])}
    control = data["means"][lookup["ctrl"]]
    singles = {
        g: data["means"][i] - control
        for g, i in lookup.items()
        if g != "ctrl" and "+" not in g
    }
    pairs = []
    excluded = []
    for pair in sorted(lookup):
        genes = pair.split("+")
        if len(genes) != 2:
            continue
        if any(g not in singles for g in genes):
            excluded.append(pair)
            continue
        pairs.append(pair)
    first = np.array([singles[p.split("+")[0]] for p in pairs])
    second = np.array([singles[p.split("+")[1]] for p in pairs])
    target = np.array([data["means"][lookup[p]] - control for p in pairs])
    return pairs, first, second, target, np.array(list(singles.values())), excluded


def gene_holdout(pairs, gene, seed=311):
    test = np.array([gene in p.split("+") for p in pairs])
    remaining = np.flatnonzero(~test)
    order = np.random.default_rng(seed).permutation(remaining)
    validation = order[: max(5, int(0.2 * len(order)))]
    partition = np.where(test, "test", "train").astype(object)
    partition[validation] = "validation"
    assert all(
        gene not in p.split("+") for p, s in zip(pairs, partition) if s == "train"
    )
    return partition


def train_neural(features, y, train, val, seed):
    if not train.any() or not val.any():
        raise ValueError("Training and validation must both be nonempty")
    torch.manual_seed(seed)
    model = ResidualComposer(y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    a, b = features
    best, stale, state, trace = float("inf"), 0, None, []
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = ((model(a[train], b[train]) - torch.tensor(y[train])) ** 2).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(((model(a[val], b[val]) - torch.tensor(y[val])) ** 2).mean())
        trace.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(loss.detach()),
                "validation_mse": score,
            }
        )
        if score < best:
            best, stale, state = score, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale == 40:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(a, b).numpy()
    return prediction, state, trace


def experiment(first, second, target, singles, partition, negative=False):
    train, val = [partition == s for s in ["train", "validation"]]
    additive = first + second
    latent = min(32, len(singles) - 1, int(train.sum()) - 1)
    basis = PCA(n_components=latent, svd_solver="full").fit(
        np.r_[singles, target[train]]
    )
    z1, z2 = [basis.transform(x).astype("float32") for x in [first, second]]
    loc = np.r_[z1[train], z2[train]].mean(0)
    scale = np.maximum(np.r_[z1[train], z2[train]].std(0), 0.05)
    a, b = [torch.tensor((z - loc) / scale) for z in [z1, z2]]
    features = symmetric_features(
        a, b
    ).numpy()  # Identical features for ridge and neural heads.
    xloc, xscale = features[train].mean(0), np.maximum(features[train].std(0), 0.05)
    residual = (target - additive) @ basis.components_.T
    rloc, rscale = residual[train].mean(0), np.maximum(residual[train].std(0), 0.05)
    y = ((residual - rloc) / rscale).astype("float32")
    ridge = Ridge(alpha=10.0).fit((features[train] - xloc) / xscale, y[train])
    ridge_pred = (
        additive
        + (ridge.predict((features - xloc) / xscale) * rscale + rloc)
        @ basis.components_
    )
    predictions = []
    checkpoints = {}
    histories = {}
    for seed in [311, 312, 313]:
        prediction, state, trace = train_neural((a, b), y, train, val, seed)
        predictions.append(prediction)
        checkpoints[str(seed)] = state
        histories[str(seed)] = trace
    neural = additive + (np.mean(predictions, 0) * rscale + rloc) @ basis.components_
    outputs = {
        "no_perturbation": np.zeros_like(target),
        "additive": additive,
        "matched_ridge": ridge_pred,
        "neural_ensemble": neural,
    }
    if negative:
        # Fit only shuffled residual training labels; validation labels stay real.
        shuffled = y.copy()
        shuffled[train] = y[
            np.random.default_rng(321).permutation(np.flatnonzero(train))
        ]
        prediction, _, trace = train_neural((a, b), shuffled, train, val, 311)
        outputs["shuffled_residual_control"] = (
            additive + (prediction * rscale + rloc) @ basis.components_
        )
        histories["shuffled"] = trace
    preprocessing = {
        "basis": basis.components_,
        "basis_mean": basis.mean_,
        "input_location": loc,
        "input_scale": scale,
        "residual_location": rloc,
        "residual_scale": rscale,
        "ridge_coef": ridge.coef_,
        "ridge_intercept": ridge.intercept_,
        "ridge_location": xloc,
        "ridge_scale": xscale,
    }
    return outputs, checkpoints, histories, preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/replication"))
    parser.add_argument("--norman", type=Path, default=Path("data/pseudobulk.npz"))
    parser.add_argument(
        "--wessels", type=Path, default=Path("data/wessels-pseudobulk.npz")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    source_info = {}
    for name, path in [("norman", args.norman), ("wessels", args.wessels)]:
        config = {
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "algorithm": "matched-residual-v1",
            "ridge_alpha": 10.0,
            "neural_seeds": [311, 312, 313],
            "max_epochs": 500,
            "early_stopping_patience": 40,
            "feature_basis": "training targets and known single references",
            "gene_holdout": "highest five combination-degree genes; known singles allowed",
        }
        data = np.load(path)
        pairs, a, b, y, singles, excluded = build(data)
        degrees = {
            g: sum(g in p.split("+") for p in pairs)
            for g in sorted({g for p in pairs for g in p.split("+")})
        }
        chosen = sorted(
            (g for g in degrees if degrees[g] >= 3), key=lambda g: (-degrees[g], g)
        )[:5]
        partitions = [
            (f"random-{s}", split_pairs(pairs, s)) for s in [311, 312, 313, 314, 315]
        ] + [(f"heldout-combination-gene-{g}", gene_holdout(pairs, g)) for g in chosen]
        source_info[name] = {
            "source_cells": int(data["counts"].sum()),
            "genes": y.shape[1],
            "single_genes": len(singles),
            "eligible_pairs": len(pairs),
            "excluded_missing_single": excluded,
            "selected_gene_holdouts": chosen,
            "minimum_condition_cell_count": int(data["counts"].min()),
        }
        for label, partition in partitions:
            train, test = partition == "train", partition == "test"
            if train.sum() < 20 or test.sum() < 3:
                raise ValueError("Insufficient combinations for fixed protocol")
            folder = args.output / name / label
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "heldout-profiles.npz").exists():
                if (
                    not (folder / "config.json").exists()
                    or json.loads((folder / "config.json").read_text()) != config
                ):
                    raise ValueError(
                        "Cached source/config differs; use a new output directory for a new experiment"
                    )
                ledger = pd.read_csv(folder / "partition.csv")
                assert ledger.condition.tolist() == pairs and np.array_equal(
                    ledger.partition, partition
                )
                saved = np.load(folder / "heldout-profiles.npz")
                outputs = {
                    k: saved[k]
                    for k in saved.files
                    if k not in ["truth", "genes", "conditions"]
                }
                np.testing.assert_allclose(saved["truth"], y[test])
            else:
                output, checkpoints, histories, preprocessing = experiment(
                    a, b, y, singles, partition, negative=label == "random-311"
                )
                torch.save(
                    {"states": checkpoints, "latent": preprocessing["basis"].shape[0]},
                    folder / "neural-checkpoints.pt",
                )
                np.savez_compressed(folder / "preprocessing.npz", **preprocessing)
                (folder / "histories.json").write_text(
                    json.dumps(histories, indent=2) + "\n"
                )
                pd.DataFrame({"condition": pairs, "partition": partition}).to_csv(
                    folder / "partition.csv", index=False
                )
                outputs = {k: v[test] for k, v in output.items()}
                (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
                np.savez_compressed(
                    folder / "heldout-profiles.npz",
                    conditions=np.array(pairs)[test],
                    genes=data["genes"],
                    truth=y[test],
                    **outputs,
                )
            for method, pred in outputs.items():
                values = metrics(y[test], pred, outputs["additive"])
                mean_rmse = float(np.mean([v["all_gene_rmse"] for v in values]))
                summary.append(
                    {
                        "dataset": name,
                        "partition": label,
                        "method": method,
                        "train_pairs": int(train.sum()),
                        "test_pairs": int(test.sum()),
                        "mean_condition_rmse": mean_rmse,
                        "oracle_top20_rmse": float(
                            np.mean([v["oracle_top20_rmse"] for v in values])
                        ),
                    }
                )
                for pair, row in zip(np.array(pairs)[test], values):
                    all_rows.append(
                        {
                            "dataset": name,
                            "partition": label,
                            "method": method,
                            "condition": pair,
                            **row,
                        }
                    )
            print("completed", name, label, flush=True)
    pd.DataFrame(all_rows).to_csv(args.output / "per-condition.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "partition-results.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": source_info,
                "note": "Twenty partitions are dependent repeated evaluations, not twenty independent datasets. Held-out combination genes have measured single references.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
