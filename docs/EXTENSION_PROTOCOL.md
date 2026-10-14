# Extension protocol: dependence and known-gene combination prediction

Fixed before external-dataset modeling on 2026-10-14. The initial Norman split results have already been seen. This is a versioned plan, not formal independent preregistration.

Question: is a residual model's advantage over additivity robust to combination partition, experimental dataset, and access to overlapping perturbation genes?

1. Use the primary Wessels et al. 2023 THP1 CaRPool-seq/Cas13 RNA perturbation experiment, GEO GSE213957. Replogle single-gene and Tahoe chemical perturbation records cannot directly replicate a two-gene combination task.
2. Keep source guide assignments, gene-pair labels, lanes and explicit controls. Verify metadata/count-barcode join; exclude ambiguous, non-targeting-only mistakes and missing singles using recorded rules. Preserve raw counts and process with library-size normalization to 10k plus log1p. Report effective combinations and genes, not cells as supervised sample count.
3. Predict wholly held-out combinations with both measured single-gene references available. Compare zero response, additive singles, a regularized residual baseline, and a symmetric neural residual with identical input features. Latent basis and scaling fit training targets/known singles only.
4. Run matched random split seeds 311/312/313/314/315 on Norman and Wessels. Also run leave-perturbed-gene-combinations-out: training combination targets cannot contain test gene, though its measured single is explicitly allowed. Report this setting separately from unseen-gene prediction.
5. Primary unit: complete combination, primary metric mean combination all-gene delta RMSE. Secondary oracle top-20 RMSE, Pearson delta, and residual interaction error. Do not inflate N using cells, genes or overlapping repeated test partitions. Report each partition and method, known-single assumptions, and pair dependencies.
6. Preserve initial experiment as historical artifact. Pin raw source hashes, processing recipe and excluded conditions. Include a shuffled-target negative control; a gain that survives only cell-row splitting is not generalization.

Norman CRISPRa and Wessels Cas13 RNA knockdown differ in cell system and intervention. Replication of the qualitative comparison would not establish a universal claim about virtual-cell models.
