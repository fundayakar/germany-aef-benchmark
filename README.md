# germany-aef-benchmark

[![DOI](https://zenodo.org/badge/1268689707.svg)](https://doi.org/10.5281/zenodo.20683594)

Code, diagnostics, and reproducibility materials for:

**Substitute or complement? Benchmarking AlphaEarth embeddings against engineered features for soil organic carbon prediction and vegetation greenness-anomaly classification in Germany under spatial cross-validation**

Funda Yakar, Ministry of Agriculture and Forestry, Türkiye.

## Overview

This repository benchmarks 64-dimensional AlphaEarth annual embeddings against task-specific engineered predictors for two environmental prediction tasks over Germany:

1. soil organic carbon (SOC) regression at LUCAS 2018 topsoil sites; and
2. classification of negative growing-season vegetation greenness anomalies over 2018-2024.

The SOC benchmark compares AlphaEarth embeddings with a 17-variable engineered stack comprising Sentinel-1/2, SRTM terrain, and ERA5-Land predictors. The vegetation benchmark compares prior-year AlphaEarth embeddings with five winter-spring ERA5-Land hydroclimate variables. The vegetation outcome is an NDVI-based greenness-anomaly class and is treated as a remote-sensing proxy for vegetation stress, not as a direct physiological stress measurement.

The primary evaluation uses spatially blocked cross-validation, with random, temporal, joint spatio-temporal, dimensionality, learner, label-threshold, and year-specific diagnostics used to test sensitivity to the comparator and transfer setting.

## Main findings represented in this repository

The two task-comparator configurations produced different patterns under spatial validation.

For SOC, full AlphaEarth embeddings ranked above the engineered stack in the primary benchmark, but the standalone AEF-over-Stack contrast was not statistically clear after dimensionality matching. The more stable result was the absence of detectable incremental predictive gain when the evaluated engineered SOC stack was added to AEF. In the manuscript this is described as **operational substitution relative to the specific comparator evaluated**, not as informational equivalence or replacement of all possible digital soil-mapping covariates.

For vegetation greenness-anomaly classification, pooled spatial-CV ROC-AUC was 0.779 for AEF, 0.779 for the winter-spring hydroclimate stack, and 0.821 for their union. Year-specific evaluation of the same spatial-CV out-of-fold predictions materially qualified that pooled result: macro-average ROC-AUC across 2018-2024 was 0.543, 0.554, and 0.602, respectively. The combined representation retained the highest macro-average performance, but the within-year incremental-gain pattern was weaker and heterogeneous. Temporal and joint spatio-temporal transfer were weak and did not support reliable extrapolation to unseen years.

These findings are conditional on the comparator sets and validation targets used here. The vegetation comparator is intentionally described as winter-spring hydroclimate rather than strictly antecedent hydroclimate because the March-May summaries overlap the April-September NDVI target window during April and May.

SHAP attribution and Pearson correlations are used only as descriptive feature-association diagnostics; they are not treated as direct evidence of information redundancy, equivalence, or the source of complementarity.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── DATA.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   ├── benchmark.yaml
│   └── diagnostics.yaml
├── gee/
│   ├── 01_soc_aef_embeddings.js
│   ├── 02_soc_master_export.js
│   └── 03_vegetation_pointlevel.js
├── src/
│   └── main benchmark and core analysis scripts
├── analyses/
│   ├── benchmark_diagnostics/
│   ├── spatial_block_sensitivity/
│   ├── dimensionality_sensitivity/
│   ├── same_year_embedding_diagnostic/
│   ├── temporal_transfer/
│   ├── linear_learner_sensitivity/
│   ├── year_specific_spatial_cv/
│   └── archive/
├── results/
│   └── main benchmark and diagnostic result tables
├── figures/
│   └── manuscript figure files
└── supplementary/
    └── Supplementary_Material.docx
```

`analyses/` contains named diagnostic and sensitivity analyses rather than revision-specific folders. `analyses/archive/` contains superseded variants retained only for provenance and not used for the reported manuscript results.

## Data

Raw and intermediate input tables are not redistributed in the GitHub repository. Public source products and access information are listed in [`DATA.md`](DATA.md). Locally exported modelling tables should be placed in a `data/` directory, which is excluded from version control.

Expected local inputs for the benchmark are:

```text
data/SOC_master_aligned.csv
data/veg_stress_pointlevel.csv
```

The second filename is retained as a legacy input name because the benchmark scripts expect it; the manuscript target is now described as a vegetation greenness anomaly.

The analysis uses AlphaEarth annual embeddings, Sentinel-1/2, SRTM, ERA5-Land, MODIS MOD13A3 NDVI, ESA WorldCover, and LUCAS topsoil data.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Google Earth Engine extraction additionally requires an authenticated Earth Engine account.

```bash
earthengine authenticate
```

## Reproducing the analysis

Detailed commands, expected inputs, and manuscript-output mappings are provided in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The primary benchmark uses fixed learner configurations across feature sets within each task. The SOC task uses a random forest regressor; the vegetation greenness-anomaly task uses XGBoost. Ten coordinate-based k-means blocks are used for the primary spatial cross-validation, with block assignments reused across feature sets to keep comparisons matched.

The year-specific diagnostic added for the current revision re-scores the archived primary spatial-CV out-of-fold predictions and does not refit any model:

```bash
python analyses/year_specific_spatial_cv/veg_year_specific_spatial_cv.py
```

Supplementary diagnostic analyses are organized by scientific purpose under [`analyses/`](analyses/README.md), including block-number sensitivity, fold-internal PCA dimensionality checks, the same-year embedding diagnostic, held-out-year-excluded temporal transfer, regularized linear-learner sensitivity, and the year-specific spatial-CV evaluation.

## Manuscript-output map

| Manuscript output | Main repository location |
|---|---|
| Table 1 | `src/lock_and_run.py`; `analyses/benchmark_diagnostics/` |
| Table 2 | `src/paired_stats.py`; `analyses/benchmark_diagnostics/` |
| Figure 1 | `analyses/spatial_block_sensitivity/` point-block exports; final layout prepared in QGIS |
| Figures 2-4 | `figures/` and corresponding `results/` files |
| Table 3 | `src/soc_sensitivity_fixedblocks.py`; `results/soc_sensitivity_fixedblocks_*` |
| Table 4 | `src/veg_threshold_sensitivity.py`; `results/veg_threshold_sensitivity_*` |
| Table 5 / yearly greenness-anomaly prevalence | `results/veg_yearly_prevalence.csv`; archived full-period temporal diagnostic in `analyses/archive/full_period_temporal/` |
| Table 6 / corrected temporal diagnostics | `analyses/temporal_transfer/` |
| Supplementary Table S1 | `analyses/spatial_block_sensitivity/` |
| Supplementary Table S2 | `analyses/dimensionality_sensitivity/` |
| Supplementary Table S3 | `analyses/same_year_embedding_diagnostic/` |
| Supplementary Table S4 | `analyses/linear_learner_sensitivity/` |
| Supplementary Table S5 | `analyses/year_specific_spatial_cv/`; `results/veg_year_specific_spatial_cv.csv` |

## Citation

If you use this code or analysis package, please cite the associated study and the archived repository. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

Zenodo concept DOI: **10.5281/zenodo.20683594**

The concept DOI resolves to the latest archived repository version.

## License

Code is released under the MIT License. Third-party data products retain their own licenses and terms of use; see [`DATA.md`](DATA.md).

## ORCID

0000-0002-7082-3956
