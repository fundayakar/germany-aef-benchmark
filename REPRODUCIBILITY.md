# Reproducibility guide

This document maps the analysis code to the reported manuscript outputs and distinguishes the primary benchmark from diagnostic and sensitivity analyses.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Google Earth Engine extraction additionally requires authentication:

```bash
earthengine authenticate
```

Primary and diagnostic settings are documented in `config/benchmark.yaml` and `config/diagnostics.yaml`.

## 2. Input data

Raw source data are not redistributed. See `DATA.md` for products, versions, and access information.

Place locally prepared modelling tables at:

```text
data/SOC_master_aligned.csv
data/veg_stress_pointlevel.csv
```

The vegetation filename is retained for backward compatibility with the existing scripts; the modeled outcome is described in the manuscript as a negative vegetation greenness anomaly.

The final vegetation modelling panel contains 1,976 locations × 7 modelled years (2018-2024) = 13,832 point-years.

## 3. Earth Engine extraction

The Earth Engine scripts are stored in `gee/` and extract the AlphaEarth, remote-sensing, terrain, climate, land-cover, and MODIS inputs described in the manuscript and `DATA.md`.

## 4. Primary benchmark

```bash
python src/lock_and_run.py soc_spatial
python src/lock_and_run.py soc_random
python src/lock_and_run.py veg_spatial
python src/lock_and_run.py veg_random
```

The script uses the fixed settings documented in `config/benchmark.yaml` and writes `results/final_benchmark_locked.csv`. Existing rows for the same task and validation scheme are replaced rather than duplicated.

The primary script retains the legacy internal label name `stress`; it corresponds to the negative vegetation greenness-anomaly class defined from location-specific growing-season NDVI.

### Table 1 vegetation PR-AUC

PR-AUC reporting for the same vegetation folds and learner configuration is reproduced by:

```bash
python analyses/benchmark_diagnostics/veg_prauc.py
```

Values repeated in Supplementary Tables S1-S2 should be reported from the same unrounded fold-level output before final three-decimal rounding.

## 5. Paired fold-level comparisons

Core paired statistics are produced by `src/paired_stats.py`. Vegetation ROC-AUC and PR-AUC paired diagnostics use:

```bash
python analyses/benchmark_diagnostics/veg_paired_prauc.py
```

The exact sign-flip tests use the ten matched spatial folds as paired units; bootstrap confidence intervals resample paired fold-level differences.

## 6. SOC target sensitivities

The reported fixed-block SOC exclusion analysis is:

```bash
python src/soc_sensitivity_fixedblocks.py
```

Reported outputs are `results/soc_sensitivity_fixedblocks_summary.csv` and `results/soc_sensitivity_fixedblocks_paired.csv`.

The earlier variant that recomputed blocks after exclusions is retained in `analyses/archive/` and is not used for manuscript results.

## 7. Vegetation greenness-anomaly threshold sensitivity

```bash
python src/veg_threshold_sensitivity.py
```

Only the NDVI anomaly threshold changes. The -1.0 SD row corresponds to the primary benchmark and should reproduce the canonical Table 1 values.

## 8. Spatial block-number sensitivity — Supplementary Table S1

```bash
python analyses/spatial_block_sensitivity/soc_block_k_sensitivity.py
python analyses/spatial_block_sensitivity/veg_block_k_sensitivity.py
```

The benchmark is repeated for k = 5, 10, and 15 coordinate-based blocks. Point-level block assignments used for Figure 1 are stored in the same folder; the final map layout was prepared in QGIS.

## 9. Dimensionality sensitivity — Supplementary Table S2

```bash
python analyses/dimensionality_sensitivity/soc_pca_dim_sensitivity.py
python analyses/dimensionality_sensitivity/veg_pca_dim_sensitivity.py
python analyses/dimensionality_sensitivity/veg_pca_dim_paired_extra.py
```

Scaling and PCA are fitted within each training fold and applied to the corresponding held-out fold. SOC uses 17 components; the vegetation greenness-anomaly benchmark uses 5 components to match the five winter-spring hydroclimate predictors.

## 10. Same-year AlphaEarth diagnostic — Supplementary Table S3

```bash
python analyses/same_year_embedding_diagnostic/veg_sameyear_aef.py
```

This changes the embedding source year from prior-year to same-year while retaining the main-analysis row mask. It is treated as a circularity-prone upper-bound diagnostic because a same-year annual embedding may contain surface information from the growing season used to derive the NDVI anomaly target.

## 11. Temporal and joint spatio-temporal transfer

```bash
python analyses/temporal_transfer/veg_loyo_trainonly_label_7yr.py
python analyses/temporal_transfer/region_year_blocked_trainonly.py
```

For each evaluation year, the per-location NDVI mean and standard deviation are recomputed from 2017-2024 excluding that year. The year 2017 contributes to the baseline but is not itself a modelled point-year.

The superseded six-year baseline variant is retained under `analyses/archive/` for provenance only.

## 12. Regularized linear-learner sensitivity — Supplementary Table S4

```bash
python analyses/linear_learner_sensitivity/soc_linear_learner_sensitivity.py
python analyses/linear_learner_sensitivity/veg_linear_learner_sensitivity.py
```

Both analyses reuse the same samples, feature sets, and ten spatial folds as the primary comparison. Predictor standardization is fitted on each training fold. Ridge alpha and logistic-regression C are evaluated over 0.01, 0.1, 1, 10, and 100; no value is selected according to held-out-fold performance.

The folder contains fold-level results, summaries, paired contrasts, and ranking diagnostics for both tasks: `soc_linear_fold_level.csv`, `soc_linear_summary.csv`, `soc_linear_paired.csv`, `soc_linear_ranking.csv`, and the corresponding `veg_linear_*` files.

## 13. Year-specific spatial-CV evaluation — Supplementary Table S5

The current revision adds a year-specific evaluation of the **existing** primary vegetation spatial-CV out-of-fold predictions. No model is refitted.

```bash
python analyses/year_specific_spatial_cv/veg_year_specific_spatial_cv.py
```

Input:

```text
results/oof_predictions_veg_spatial.csv
```

Output:

```text
results/veg_year_specific_spatial_cv.csv
```

ROC-AUC and PR-AUC are calculated separately within each year from 2018 through 2024, and unweighted macro-averages are reported across the seven years. This diagnostic separates within-year spatial discrimination from performance that can arise from ranking observations across years with strongly different positive-class prevalence.

## 14. Supplementary material

The journal supplementary file is stored at `supplementary/Supplementary_Material.docx` and contains Supplementary Tables S1-S5.

## 15. Archived analyses

`analyses/archive/` contains superseded variants retained only to preserve analytical provenance. This includes the earlier six-year LOYO label baseline, the SOC sensitivity variant that recomputed spatial blocks, and the original full-period temporal diagnostic scripts. These files are not used to generate the corrected temporal results reported in Table 6.
