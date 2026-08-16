# germany-aef-benchmark

Code and reproducible analysis package for the study:

[![DOI](https://zenodo.org/badge/1268689707.svg)](https://doi.org/10.5281/zenodo.20683594)

**Substitute or complement? Benchmarking AlphaEarth embeddings against engineered features for soil organic carbon and vegetation-stress prediction in Germany under spatial cross-validation**

Funda Yakar, Ministry of Agriculture and Forestry, Türkiye.

## Overview

This repository benchmarks 64-dimensional AlphaEarth annual embeddings against task-specific engineered predictors for two contrasting environmental prediction tasks over Germany:

1. soil organic carbon regression at LUCAS 2018 topsoil sites; and
2. binary vegetation-stress classification from antecedent hydroclimate over 2017-2024.

The soil organic carbon task compares AlphaEarth embeddings with an engineered stack based on Sentinel-1/2, SRTM terrain and ERA5-Land predictors. The vegetation-stress task compares prior-year AlphaEarth embeddings with five antecedent ERA5-Land hydroclimate predictors. Primary comparisons use spatially blocked cross-validation. Random, temporal and joint spatio-temporal validation are included as diagnostic checks of sensitivity to the validation target.

## Key findings

The value of AlphaEarth embeddings is task-dependent and validation-target-dependent.

For soil organic carbon, AlphaEarth substitutes for the engineered SOC stack used here: it performs better than the engineered stack under spatial cross-validation, and adding the stack to AlphaEarth provides little additional gain.

For vegetation stress, prior-year AlphaEarth embeddings complement antecedent hydroclimate under spatial transfer: the two predictor sets are comparable in isolation but perform best when combined. This complementarity weakens under temporal and joint spatio-temporal validation, where performance remains weak and metric-dependent.

A SHAP-based analysis relates these patterns to task-specific representational redundancy. Predictive embedding dimensions show moderate association with engineered SOC predictors, but weak association with the antecedent hydroclimate predictors used in the vegetation task. Random cross-validation inflates apparent skill and, for vegetation stress, changes the feature-set ranking, so spatial and temporal validation test different forms of transfer.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── DATA.md
├── REPRODUCE.md
├── REPRODUCE_additional_analyses.md
├── config/
│   └── config.yaml
├── gee/
│   └── Google Earth Engine extraction scripts
├── src/
│   ├── lock_and_run.py
│   └── modelling, diagnostics, sensitivity and figure scripts
├── results/
│   └── output tables and fold-level diagnostics
└── figures/
    └── manuscript and supplementary figures
```

## Data

Raw data are not redistributed in this repository. All source datasets are public, except where external access constraints apply to platform-hosted assets. `DATA.md` lists the products, access points and collection or version information used in the analysis.

The analysis uses AlphaEarth annual embeddings, Sentinel-1/2, SRTM, ERA5-Land, MODIS MOD13A3 NDVI, ESA WorldCover and LUCAS topsoil data. Exported intermediate tables should be placed under a local `data/` directory, which is excluded from version control.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Google Earth Engine extraction additionally requires an authenticated Earth Engine account:

```bash
earthengine authenticate
```

## Reproducing the main analysis

1. Extract predictors in Google Earth Engine using the scripts in `gee/`.
   - Soil task: AlphaEarth embeddings, Sentinel-1/2, SRTM terrain and ERA5-Land predictors.
   - Vegetation task: AlphaEarth embeddings, ERA5-Land hydroclimate predictors, WorldCover strata and MODIS NDVI.
2. Save exported tables under a local `data/` directory.
3. Build the modelling tables with the dataset scripts in `src/`.
4. Run the locked benchmark:

   ```bash
   python src/lock_and_run.py
   ```

This reproduces the main locked spatial and random cross-validation benchmark and the core SHAP outputs from the fixed configuration in `config/config.yaml`.

Additional diagnostics and sensitivity analyses are documented in `REPRODUCE_additional_analyses.md`. These include PR-AUC reporting, paired PR-AUC diagnostics, held-out-year-excluded temporal labels, same-year AlphaEarth diagnostics, PCA dimensionality checks, and block-number sensitivity analyses.

## Locked configuration

All main benchmark results use fixed learner settings rather than feature-set-specific hyperparameter tuning.

The soil organic carbon task uses a random forest regressor with 300 trees. The vegetation-stress task uses XGBoost with 300 trees, maximum depth 5, learning rate 0.05, subsample 0.8, colsample_bytree 0.8 and histogram tree construction. A common random seed of 42 is used. The primary spatial cross-validation uses ten coordinate-based k-means spatial blocks, with block assignments computed once and reused across feature sets so that comparisons are matched.

## Additional analyses

The revised analysis package includes additional diagnostic outputs beyond the initial locked benchmark:

- vegetation PR-AUC reporting for Table 1;
- paired ROC-AUC and PR-AUC diagnostics for Table 2;
- sample-size accounting for the 13,832 point-year vegetation panel;
- same-year AlphaEarth diagnostic for the vegetation-stress task;
- held-out-year-excluded LOYO and region-year temporal diagnostics;
- PCA dimensionality sensitivity analyses for SOC and vegetation stress;
- spatial block-number sensitivity for k = 5, 10 and 15;
- point-level spatial block files used to create the study-area map in QGIS.

See `REPRODUCE_additional_analyses.md` for the corresponding scripts, output files and manuscript locations.

## Citation

If you use this code or analysis package, please cite the associated paper and the archived repository DOI:

https://doi.org/10.5281/zenodo.20683594

The DOI above is the Zenodo concept DOI and resolves to the latest archived version of the repository.

## License

Code is released under the MIT License. Third-party data products retain their own licenses; see `DATA.md`.

## ORCID

0000-0002-7082-3956
