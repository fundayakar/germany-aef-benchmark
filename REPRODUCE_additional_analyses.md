# Additional analyses and reproducibility notes

This document supplements the main `REPRODUCE.md` file and describes additional diagnostic and sensitivity analyses used in the revised manuscript. It lists the scripts, output files, and manuscript locations associated with each analysis.

All scripts assume `veg_stress_pointlevel.csv` in the working directory, as in the original pipeline. See the main `DATA.md` and `REPRODUCE.md` files for the core input files, directory structure, and locked modelling configuration.

| Analysis / purpose | Script or workflow | Output | Reported in |
|---|---|---|---|
| Vegetation PR-AUC reporting for the main spatial and random cross-validation benchmark | `veg_prauc.py` | `veg_prauc_table1_folds.csv` | Table 1 |
| Vegetation paired ROC-AUC/PR-AUC fold-level diagnostics | `veg_paired_prauc.py` | `veg_prauc_rocauc_fold_level.csv`, `veg_paired_rocauc_prauc.csv` | Table 2 |
| Vegetation sample-size accounting for the final 13,832 point-year panel | `veg_sample_size_diagnostic.py` | `veg_sample_size_diagnostic_output.txt` | Methods 2.3.2 |
| Same-year AlphaEarth diagnostic for the vegetation-stress task | `veg_sameyear_aef.py` | `veg_sameyear_aef_summary.csv`, `veg_sameyear_aef_paired.csv` | Supplementary Table S3; Limitations |
| Held-out-year-excluded stress-label baseline for LOYO validation | `veg_loyo_trainonly_label_7yr.py` | `veg_loyo_trainonly_v2_*.csv` | Results 3.7; Table 6 |
| Superseded six-year LOYO baseline variant retained for reference | `veg_loyo_trainonly_label_6yr_NOT_USED_IN_MANUSCRIPT.py` | `veg_loyo_trainonly_6yr_NOT_USED_*.csv` | Not reported in the manuscript |
| Held-out-year-excluded stress-label baseline for region-year blocked validation | `region_year_blocked_trainonly.py` | `region_year_blocked_trainonly_folds.csv`, `region_year_blocked_trainonly_summary.csv`, `region_year_blocked_trainonly_CI.csv` | Results 3.7; Table 6 |
| SOC dimensionality sensitivity, reducing AlphaEarth from 64 dimensions to 17 fold-internal PCA components | `soc_pca_dim_sensitivity.py` | `soc_pca_dim_fold_level.csv`, `soc_pca_dim_summary.csv`, `soc_pca_dim_paired.csv` | Results 3.3; Supplementary Table S2 |
| Vegetation dimensionality sensitivity, reducing AlphaEarth from 64 dimensions to 5 fold-internal PCA components | `veg_pca_dim_sensitivity.py` | `veg_pca_dim_fold_level.csv`, `veg_pca_dim_summary.csv`, `veg_pca_dim_paired.csv` | Results 3.4; Supplementary Table S2 |
| Additional paired contrasts for the vegetation PCA dimensionality sensitivity | `veg_pca_dim_paired_extra.py` | `veg_pca_dim_paired_extra.csv` | Results 3.4; Supplementary Table S2 |
| Spatial block-number sensitivity for k = 5, 10, and 15 coordinate-based k-means blocks | `soc_block_k_sensitivity.py`, `veg_block_k_sensitivity.py` | `soc_block_k_sensitivity_folds.csv`, `soc_block_k_sensitivity_summary.csv`, `soc_block_k_sensitivity_paired.csv`, `veg_block_k_sensitivity_folds.csv`, `veg_block_k_sensitivity_summary.csv`, `veg_block_k_sensitivity_paired.csv` | Results 3.6; Supplementary Table S1 |
| Spatial block map for the study-area figure | Manual QGIS layout using `soc_points_blocks.csv` and `vegetation_points_blocks.csv` | `Figure_spatial_blocks.png` and/or `Figure_spatial_blocks.tif` | Figure 1 |

## Notes

- `veg_prauc.py` and `veg_paired_prauc.py` reuse the exact locked configuration as the main vegetation pipeline: seed 42, the same coordinate-based k-means spatial blocks, and the same XGBoost hyperparameters. Re-running them reproduces the main ROC-AUC values before adding PR-AUC reporting.

- `veg_sameyear_aef.py` changes exactly one element relative to the main vegetation pipeline: AlphaEarth predictors are taken from the same-year AlphaEarth bands (`A00` to `A63`) rather than the prior-year shifted bands (`A00_p` to `A63_p`). The retained row mask is kept identical to the main analysis by applying the same prior-year availability filter, so the same 13,832 point-years are used. This isolates the embedding source year as the intended difference. This analysis is treated as a circularity-prone upper-bound diagnostic, not as an alternative main design.

- `veg_sample_size_diagnostic.py` is a read-only diagnostic. It does not change any modelling result. It explains why 24 of the 2,000 sampled locations are absent from the final panel: ERA5-Land climate predictors are missing for all eight years at those locations. The remaining 1,976 locations have complete climate and NDVI records across the analysis panel; after the prior-year embedding design removes the first available year, the final modelling panel contains 1,976 locations × 7 years = 13,832 point-years.

- **Held-out-year-excluded stress labels.** The original spatial and random benchmarks use the full 2017-2024 per-point NDVI record to define the vegetation-stress label. For temporal validation, this full-period label would allow the held-out test year to contribute to the baseline used to label that same year. The LOYO and region-year temporal diagnostics therefore use a held-out-year-excluded baseline. For each possible test year `y`, the per-point NDVI mean and standard deviation are recomputed using 2017-2024 excluding `y`. The year 2017 is included in the baseline calculation even though it is not itself a modelled point-year, because it has valid NDVI for the retained locations and provides a larger baseline pool.

- `veg_loyo_trainonly_label_6yr_NOT_USED_IN_MANUSCRIPT.py` is retained only for provenance. It uses a six-year baseline based on 2018-2024 excluding the held-out year, without 2017. It was superseded by the seven-year baseline version because including 2017 gives a larger and more stable baseline pool. The seven-year version is the version reported in the manuscript.

- For the region-year blocked design, all spatial folds sharing the same test year use that year's held-out-year-excluded label. The label depends only on which year is excluded from the baseline, not on which spatial block is held out. The block-exclusion logic is otherwise unchanged: training excludes both the test year in all blocks and the test block in all years.

- **SOC PCA dimensionality sensitivity.** `soc_pca_dim_sensitivity.py` tests whether the difference between AlphaEarth and the engineered SOC stack is driven simply by feature count. For each of the 10 spatial-CV folds, a `StandardScaler` and `PCA(n_components=17)` are fitted on the training fold only and then used to transform both the training and held-out folds. No PCA fitting uses test-fold data. `Stack-17`, `AEF-64`, and `AEF-64+Stack-17` reproduce the main Table 1 setup, while `AEF-PCA17` and `AEF-PCA17+Stack-17` provide the dimensionality-matched diagnostic. The same locked random-forest configuration and spatial blocks are used throughout.

- **Vegetation PCA dimensionality sensitivity.** `veg_pca_dim_sensitivity.py` applies the same fold-internal PCA discipline to the vegetation task, reducing AlphaEarth to 5 components to match the five-variable antecedent climate stack. The diagnostic uses the same locked XGBoost configuration, spatial blocks, and main-analysis stress label as the vegetation spatial-CV benchmark.

- `veg_pca_dim_paired_extra.py` reads the fold-level output from the vegetation PCA diagnostic and computes additional paired contrasts: `AEF-PCA5+Stack-5` versus `Stack-5`, and `AEF-64` versus `AEF-PCA5`. It does not refit models.

- **Block-number sensitivity.** `soc_block_k_sensitivity.py` and `veg_block_k_sensitivity.py` repeat the main spatial-CV benchmark at k = 5, 10, and 15 coordinate-based k-means blocks. The locked model configurations, predictors, and labels are kept unchanged; only the block count and therefore the fold structure change. The k = 10 setting is the one used in the main manuscript.

- **Spatial block map.** The point-level block assignment files (`soc_points_blocks.csv` and `vegetation_points_blocks.csv`) contain the deterministic coordinate-based k-means block labels used in the spatial cross-validation. These files were used to create Figure 1 manually in QGIS. No separate Python script was used to generate the final map layout.
