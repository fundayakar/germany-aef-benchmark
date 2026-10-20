# Diagnostic and sensitivity analyses

This directory contains analyses that complement the primary benchmark in `src/`. Folder names describe the scientific purpose of each analysis rather than the revision round in which it was added.

| Folder | Purpose | Main manuscript location |
|---|---|---|
| `benchmark_diagnostics/` | Vegetation greenness-anomaly PR-AUC, paired ROC-AUC/PR-AUC, sample-size accounting | Tables 1-2; Methods 2.3.2 |
| `spatial_block_sensitivity/` | k = 5/10/15 block sensitivity and exported block assignments | Results 3.6; Supplementary Table S1; Figure 1 |
| `dimensionality_sensitivity/` | Fold-internal PCA matching of AEF dimensionality to engineered stacks | Results 3.3-3.4; Supplementary Table S2 |
| `same_year_embedding_diagnostic/` | Same-year AEF upper-bound diagnostic | Limitations; Supplementary Table S3 |
| `temporal_transfer/` | Held-out-year-excluded LOYO and region-year analyses | Results 3.7; Table 6 |
| `linear_learner_sensitivity/` | Ridge/logistic learner-family sensitivity | Results 3.2; Limitations; Supplementary Table S4 |
| `year_specific_spatial_cv/` | Year-specific ROC-AUC/PR-AUC from the primary vegetation spatial-CV OOF predictions; no model refit | Results 3.1; Supplementary Table S5 |
| `archive/` | Superseded variants retained for provenance only | Not used for reported results |

See the repository-level `REPRODUCIBILITY.md` for commands, expected inputs, and interpretation notes.
