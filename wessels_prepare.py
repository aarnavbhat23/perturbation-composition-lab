"""Process primary GEO raw counts, preserving guide/lane provenance."""

import argparse
import csv
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213957/suppl/"
ARCHIVE = "GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz"
METADATA = "GSE213957_THP1-CaRPool-seq.metadata.tsv.gz"


def condition(pair):
    genes = sorted(g for g in pair.split("_") if g != "NT")
    if len(genes) != len(set(genes)):
        return None
    return "+".join(genes) or "ctrl"


def read_metadata(path):
    with gzip.open(path, "rt") as handle:
        reader = csv.reader(handle, delimiter="\t")
        columns = next(reader)
        rows = [row for row in reader if row]
    if any(len(row) != len(columns) + 1 for row in rows):
        raise ValueError("Unexpected metadata schema")
    frame = pd.DataFrame(
        [r[1:] for r in rows], columns=columns, index=[r[0] for r in rows]
    )
    if not frame.index.is_unique:
        raise ValueError("Duplicate barcodes")
    frame["canonical_condition"] = [condition(p) for p in frame.GenePair]
    return frame


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/wessels-pseudobulk.npz")
    )
    args = parser.parse_args()
    frame = read_metadata(args.metadata)
    eligible = frame.canonical_condition.notna()
    labels = sorted(frame.loc[eligible, "canonical_condition"].unique())
    lookup = {name: i for i, name in enumerate(labels)}
    counts = np.zeros(len(labels), dtype=int)
    sums = squares = None
    genes = None
    joined = []
    lane_counts = []
    with tarfile.open(args.archive) as archive:
        names = [m.name for m in archive.getmembers()]
        for lane in range(1, 5):
            prefix = next(
                n[: -len("features.tsv.gz")]
                for n in names
                if f"GEXGDO{lane}.features.tsv.gz" in n
            )

            def lines(suffix, prefix=prefix):
                with gzip.GzipFile(
                    fileobj=archive.extractfile(prefix + suffix)
                ) as handle:
                    return handle.read().decode().splitlines()

            features = [line.split("\t") for line in lines("features.tsv.gz")]
            barcodes = lines("barcodes.tsv.gz")
            gene_mask = np.array([r[2] == "Gene Expression" for r in features])
            symbols = np.array([r[1] for r in features])[gene_mask]
            if genes is None:
                genes = symbols
                sums = np.zeros((len(labels), len(genes)), dtype=np.float64)
                squares = np.zeros_like(sums)
            elif not np.array_equal(genes, symbols):
                raise ValueError("Lane gene axes differ")
            full_names = [f"L{lane}_{b}" for b in barcodes]
            selected = [
                i
                for i, name in enumerate(full_names)
                if name in frame.index and eligible.loc[name]
            ]
            identifiers = [full_names[i] for i in selected]
            joined.extend(identifiers)
            print(
                "reading lane", lane, "selected THP1 cells", len(selected), flush=True
            )
            with gzip.GzipFile(
                fileobj=archive.extractfile(prefix + "matrix.mtx.gz")
            ) as handle:
                matrix = (
                    sio.mmread(handle)
                    .tocsr()[gene_mask, :][:, selected]
                    .tocsc()
                    .astype(np.float64)
                )
            totals = np.asarray(matrix.sum(axis=0)).ravel()
            if (totals <= 0).any():
                raise ValueError("Zero expression library")
            matrix = matrix @ sparse.diags(1e4 / totals)
            matrix.data = np.log1p(matrix.data)
            assignment = np.array(
                [lookup[frame.loc[name, "canonical_condition"]] for name in identifiers]
            )
            for group in np.unique(assignment):
                block = matrix[:, assignment == group]
                sums[group] += np.asarray(block.sum(axis=1)).ravel()
                squares[group] += np.asarray(block.power(2).sum(axis=1)).ravel()
                counts[group] += block.shape[1]
            lane_counts.append(
                {
                    "lane": lane,
                    "cells": len(selected),
                    "control_cells": int((assignment == lookup["ctrl"]).sum()),
                }
            )
            del matrix
    missing = sorted(set(frame.index[eligible]) - set(joined))
    if missing:
        raise ValueError(f"{len(missing)} metadata cells missing from count matrices")
    means = sums / counts[:, None]
    variance = np.maximum(squares / counts[:, None] - means**2, 0)
    control = lookup["ctrl"]
    selected_genes = np.argsort(-variance[control], kind="stable")[:5000]
    selected_genes = np.sort(selected_genes)
    args.output.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        args.output,
        conditions=np.array(labels, dtype="U"),
        means=means[:, selected_genes].astype("float32"),
        standard_errors=np.sqrt(variance[:, selected_genes] / counts[:, None]).astype(
            "float32"
        ),
        counts=counts,
        genes=genes[selected_genes].astype("U"),
    )
    manifest = {
        "source": BASE + ARCHIVE,
        "metadata_source": BASE + METADATA,
        "archive_sha256": sha256(args.archive),
        "metadata_sha256": sha256(args.metadata),
        "metadata_cells": len(frame),
        "retained_cells": int(counts.sum()),
        "excluded_same_gene_cells": int((~eligible).sum()),
        "conditions": len(labels),
        "source_expression_genes": len(genes),
        "selected_genes": len(selected_genes),
        "selection": "Top 5000 variance genes in non-targeting control cells only; stable tie order.",
        "normalization": "Primary raw Gene Expression counts; per-cell library normalized to 10000, then natural log1p.",
        "lane_counts": lane_counts,
        "intervention": "Cas13 RNA knockdown; THP1; not Norman CRISPRa/K562.",
        "join": "Every eligible metadata barcode found exactly in lane-prefix count barcode join.",
        "pseudobulk_sha256": sha256(args.output),
        "uncertainty": "Cell standard errors are descriptive, not biological replicate intervals.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    frame[
        ["GenePair", "Guide.Class", "TenX.Lane", "CRISPR.Array", "canonical_condition"]
    ].to_csv(
        args.output.with_suffix(".cell-ledger.csv.gz"),
        index_label="barcode",
        compression="gzip",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
