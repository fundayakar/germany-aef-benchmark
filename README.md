# germany-aef-benchmark

[![DOI](https://zenodo.org/badge/1268689707.svg)](https://doi.org/10.5281/zenodo.20683594)

Code, diagnostics, and reproducibility materials for:

**Substitute or complement? Benchmarking AlphaEarth embeddings against engineered features for soil organic carbon and vegetation-stress prediction in Germany under spatial cross-validation**

Funda Yakar, Ministry of Agriculture and Forestry, Türkiye.

## Overview

This repository benchmarks 64-dimensional AlphaEarth annual embeddings against task-specific engineered predictors for two environmental prediction tasks over Germany:

1. soil organic carbon (SOC) regression at LUCAS 2018 topsoil sites; and
2. vegetation-stress classification from antecedent hydroclimate over 2018-2024.

The SOC benchmark compares AlphaEarth embeddings with a 17-variable engineered stack comprising Sentinel-1/2, SRTM terrain, and ERA5-Land predictors. The vegetation benchmark compares prior-year AlphaEarth embeddings with five antecedent ERA5-Land hydroclimate variables. The primary evaluation uses spatially blocked cross-validation, with random, temporal, and joint spatio-temporal diagnostics used to test sensitivity to the transfer setting.

## Main findings represented in this repository

The two task-comparator configurations produced different patterns under spatial validation.

For SOC, full AlphaEarth embeddings ranked above the engineered stack in the primary benchmark, but that standalone contrast weakened under dimensionality matching. The more stable result was that adding the engineered SOC stack to AlphaEarth produced essentially no additional predictive gain, including in the dimensionality-matched and regularized linear-learner sensitivities.

For vegetation stress, prior-year AlphaEarth embeddings and antecedent hydroclimate were complementary under spatial transfer: the combined feature set performed best. The size of this gain was smaller under logistic regression and weakened sharply under temporal and joint spatio-temporal validation.

These findings are conditional on the comparator sets used here. Because the SOC and vegetation engineered baselines differ substantially in composition, the contrast should not be interpreted as an intrinsic rule that embeddings substitute for stable targets and complement dynamic targets.

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
│   └── archive/
├── results/
│   └── main benchmark and core-result tables
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

The primary benchmark uses fixed learner configurations across feature sets within each task. The SOC task uses a random forest regressor; the vegetation-stress task uses XGBoost. Ten coordinate-based k-means blocks are used for the primary spatial cross-validation, with block assignments reused across feature sets to keep comparisons matched.

Supplementary diagnostic analyses are organized by scientific purpose under [`analyses/`](analyses/README.md), including block-number sensitivity, fold-internal PCA dimensionality checks, the same-year embedding diagnostic, held-out-year-excluded temporal transfer, and regularized linear-learner sensitivity.

## Manuscript-output map

| Manuscript output | Main repository location |
|---|---|
| Table 1 | `src/lock_and_run.py`; `analyses/benchmark_diagnostics/` |
| Table 2 | `src/paired_stats.py`; `analyses/benchmark_diagnostics/` |
| Figure 1 | `analyses/spatial_block_sensitivity/` point-block exports; final layout prepared in QGIS |
| Figures 2-4 | `figures/` and corresponding `results/` files |
| Table 3 | `src/soc_sensitivity_fixedblocks.py`; `results/soc_sensitivity_fixedblocks_*` |
| Table 4 | `src/veg_threshold_sensitivity.py`; `results/veg_threshold_sensitivity_*` |
| Table 6 / temporal diagnostics | `analyses/temporal_transfer/` |
| Supplementary Table S1 | `analyses/spatial_block_sensitivity/` |
| Supplementary Table S2 | `analyses/dimensionality_sensitivity/` |
| Supplementary Table S3 | `analyses/same_year_embedding_diagnostic/` |
| Supplementary Table S4 | `analyses/linear_learner_sensitivity/` |

## Citation

If you use this code or analysis package, please cite the associated study and the archived repository. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

Zenodo concept DOI: **10.5281/zenodo.20683594**

The concept DOI resolves to the latest archived repository version.

## License

Code is released under the MIT License. Third-party data products retain their own licenses and terms of use; see [`DATA.md`](DATA.md).

## ORCID

0000-0002-7082-3956
