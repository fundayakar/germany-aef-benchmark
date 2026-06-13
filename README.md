# germany-aef-benchmark

Code and reproducible pipeline for the study:

https://doi.org/10.5281/zenodo.20683594

**Substitute or complement? Benchmarking Earth observation foundation-model embeddings against engineered features for soil and vegetation prediction under spatial cross-validation**

Funda Yakar, Ministry of Agriculture and Forestry, Türkiye.

This repository benchmarks 64-dimensional AlphaEarth annual embeddings against multi-source engineered feature stacks (Sentinel-1/2, terrain, ERA5-Land) on two contrasting tasks over Germany: soil organic carbon regression at LUCAS sites, and binary vegetation-stress classification from antecedent hydroclimate (2017 to 2024). All model comparisons use spatially blocked cross-validation, with random cross-validation reported for contrast.

## Key finding

The value of the embeddings is task-dependent. For soil organic carbon they substitute for the engineered stack (matching or exceeding it, with no gain from adding it). For vegetation stress they complement antecedent climate (comparable alone, clearly better together). A SHAP-based analysis ties the contrast to representational redundancy. Random cross-validation not only inflates scores but, for vegetation, reverses the ranking and hides the complementary value, so spatial validation is a precondition rather than a refinement.

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── DATA.md                 # data sources and how to obtain them
├── config/
│   └── config.yaml         # locked hyperparameters, seeds, block count
├── gee/                    # Google Earth Engine extraction scripts
├── src/                    # Python pipeline (dataset build, benchmark, SHAP, figures)
│   └── lock_and_run.py     # single locked-config reproducible run
├── results/                # output tables (e.g. final_benchmark_locked.csv)
└── figures/                # output figures
```

## Data

Raw data are not redistributed here. All sources are public; `DATA.md` lists each product, its access point, and the exact collection or version used. The LUCAS extraction point set is held as a Google Earth Engine asset (see `DATA.md`).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Earth Engine extraction additionally requires an authenticated `earthengine-api` account (`earthengine authenticate`).

## Reproducing the analysis

1. Extract predictors in Google Earth Engine using the scripts in `gee/` (AlphaEarth embeddings, Sentinel-1/2, SRTM, ERA5-Land for the soil task; AlphaEarth, ERA5-Land, WorldCover strata and MODIS NDVI for the vegetation task). Exported tables go under a local `data/` directory (git-ignored).
2. Build the modelling tables with the dataset scripts in `src/`.
3. Run the locked benchmark:
   ```bash
   python src/lock_and_run.py
   ```
   This reproduces all numbers in Table 1 and the SHAP outputs from the fixed configuration in `config/config.yaml`.
4. Generate figures with the figure script in `src/`.

## Locked configuration

All reported numbers come from one fixed configuration: random forest (300 trees) for the soil task; XGBoost (300 trees, max depth 5, learning rate 0.05, subsample 0.8, colsample 0.8, histogram method) for the vegetation task; a common random seed (42); and ten spatial blocks formed by k-means on coordinates. Block assignments are computed once and reused across all feature sets so comparisons are matched.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and this repository via its archived DOI (added after the first Zenodo release).

## License

Code is released under the MIT License (see `LICENSE`). Third-party data products retain their own licenses; see `DATA.md`.

## ORCID

0000-0002-7082-3956
