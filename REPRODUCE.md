# Reproducing the robustness and temporal-transfer analyses

This document maps each additions including (statistical robustness,
SOC and vegetation sensitivity checks, temporal and spatio-temporal transfer) to
the script and result files that produced it. It complements, and does not
replace, `DATA.md` (raw data sources) and `config.yaml` (the locked config for the
original Table 1 benchmark) — see also `config_addendum.yaml` for the settings
specific to these additions.

No raw or intermediate data files are redistributed here, consistent with
`DATA.md`. All scripts expect the same inputs as the original pipeline:
`SOC_master_aligned.csv` and `veg_stress_pointlevel.csv`, produced by the GEE
export scripts already in this repository (`gee/01_soc_aef_embeddings.js`,
`gee/02_soc_master_export.js`, `gee/03_vegetation_pointlevel.js`).

## Table 2 — Statistical robustness of the substitution and complementarity patterns

| Step | Script | Output |
|---|---|---|
| Reproduce Table 1's spatial-CV benchmark while saving fold-level and point-level out-of-fold predictions | `src/spatial_cv_robustness/rerun_with_folds.py` | `results/spatial_cv_robustness/fold_level_soc_spatial.csv`, `fold_level_veg_spatial.csv`, `oof_predictions_soc_spatial.csv`, `oof_predictions_veg_spatial.csv` |
| Paired bootstrap CI + exact permutation test on fold-level differences | `src/spatial_cv_robustness/paired_stats.py` | `results/spatial_cv_robustness/paired_bootstrap_results.csv` (= Table 2) |

## Table 5 — Sensitivity to organic-rich soils and statistical outliers (SOC)

| Step | Script | Output |
|---|---|---|
| Locked SOC benchmark rerun on two trimmed samples, **original spatial blocks retained** (used in manuscript) | `src/soc_sensitivity/soc_sensitivity_fixedblocks.py` | `results/soc_sensitivity/soc_sensitivity_fixedblocks_summary.csv` (= Table 5), `soc_sensitivity_fixedblocks_paired.csv` |
| Earlier version, blocks recomputed per trimmed sample — **not used in manuscript**, kept for reference only | `src/soc_sensitivity/soc_sensitivity_recomputed_blocks_NOT_USED_IN_MANUSCRIPT.py` | `results/soc_sensitivity/soc_sensitivity_summary.csv`, `soc_sensitivity_paired.csv` |

## Table 6 — Sensitivity to the vegetation-stress label threshold

| Step | Script | Output |
|---|---|---|
| Locked vegetation benchmark rerun at \u22120.5 SD and \u22121.5 SD thresholds (\u22121.0 SD is Table 1) | `src/veg_threshold_sensitivity/veg_threshold_sensitivity.py` | `results/veg_threshold_sensitivity/veg_threshold_sensitivity_summary.csv` (= Table 6), `..._paired.csv`, `..._prevalence.csv` |

## Table 3 & Table 4 — Temporal and spatio-temporal transfer

| Step | Script | Output |
|---|---|---|
| Leave-one-year-out (LOYO): 7 folds, one held-out year each | `src/temporal_transfer/loyo_vegetation.py` | `results/temporal_transfer/fold_level_veg_loyo.csv`, `paired_bootstrap_veg_loyo.csv` |
| Yearly prevalence table | (same script; also standalone) | `results/temporal_transfer/veg_yearly_prevalence.csv` (= Table 3) |
| Full LOYO diagnostics: pooled AUC, PR-AUC, Brier, mean-predicted-probability-vs-prevalence, and baseline feature sets (LC-only, Climate-anomaly-only) used only to interpret the AEF pooled/year-wise divergence, not reported in Table 4 | `src/temporal_transfer/loyo_diagnostics.py` | `results/temporal_transfer/loyo_full_diagnostics_per_year.csv`, `loyo_pooled_metrics.csv` (feeds Table 4's "Pooled ROC-AUC" column), `loyo_meanprob_vs_prevalence.csv`, `loyo_yearlevel_correlation.csv` |
| Region-year blocked CV: 70 folds (7 years \u00d7 10 spatial blocks), both test year and test block excluded from training | `src/temporal_transfer/region_year_blocked.py` | `results/temporal_transfer/region_year_blocked_folds.csv`, `region_year_blocked_summary.csv` (feeds Table 4's "Region-year blocked ROC-AUC" column) |
| Year-clustered bootstrap CI for region-year blocked AUC/PR-AUC vs. chance/no-skill baselines | `src/temporal_transfer/region_year_ci.py` | `results/temporal_transfer/region_year_blocked_CI.csv` |

## Assembling Table 4 from the pieces above

Table 4's four columns come from three separate result files:

- **Pooled ROC-AUC (LOYO)** — `loyo_pooled_metrics.csv`, column `pooled_ROC_AUC`, rows AEF/Stack/AEF+Stack.
- **Mean year-wise ROC-AUC (LOYO)** — `loyo_pooled_metrics.csv`, column `mean_of_yearwise_AUC` (mean) and `fold_level_veg_loyo.csv` (per-year values, for the SD).
- **Region-year blocked ROC-AUC** — `region_year_blocked_summary.csv`, mean and SD columns.

## Notes for reviewers reproducing this locally

- All model configuration (seeds, hyperparameters, spatial block count) matches
  `config.yaml`; see `config_addendum.yaml` for the settings specific to these
  additions (exclusion thresholds, bootstrap/permutation parameters, LOYO and
  region-year-blocked fold design).
- Everything here ran on a single CPU core; `soc_sensitivity_fixedblocks.py` and
  `soc_sensitivity_recomputed_blocks_NOT_USED_IN_MANUSCRIPT.py` are the slowest
  steps (~9 minutes each, random forest with 300 trees x 10 folds x 3 feature sets
  x multiple samples). Everything else runs in well under a minute.
