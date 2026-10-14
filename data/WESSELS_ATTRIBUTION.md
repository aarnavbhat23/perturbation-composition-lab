# Primary Cas13 source and derived records

Source: [Wessels et al., 2023, Nature Methods](https://doi.org/10.1038/s41592-022-01705-x), CaRPool-seq, [GEO GSE213957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957). The manifest records the exact GEO archive/metadata URLs and SHA-256 checksums. No new ownership or experimental collection is claimed.

`wessels-pseudobulk.npz` and its barcode ledger are derived from the deposited primary THP1 counts and metadata. The source experimental data retain their depositor attribution and applicable source terms; the repository's MIT code license does not relicense experimental data. The raw archive is downloaded separately, not included in Git.

Processing changes: join all 30,707 THP1 metadata barcodes to their four count lanes; keep Gene Expression features; normalize each cell to 10,000 RNA counts; log1p; canonicalize non-targeting and gene order; aggregate by condition. Select 5,000 genes by control-cell variance only. Store condition counts and descriptive cell standard errors. This is not a biological replicate error estimate. Some single conditions have only 15 cells; results depend on noisy reference means, control selection and preprocessing.

The top-five gene holdouts are selected by combination degree (at least three pairs), with alphabetical tie-breaking, before model scores are used. A test gene's measured single response is available. Its combination targets are all held out. Those assumptions are deliberately different from prediction of an entirely unmeasured gene.
