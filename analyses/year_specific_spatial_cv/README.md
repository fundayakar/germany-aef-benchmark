# Year-specific spatial-CV diagnostic

This analysis addresses the strong year-to-year variation in positive-class prevalence in the vegetation greenness-anomaly benchmark.

It **does not refit the models**. Instead, it re-scores the archived out-of-fold predictions from the primary spatial cross-validation separately within each year (2018-2024). ROC-AUC and PR-AUC are therefore calculated only among locations observed under the same annual conditions. The script also reports unweighted macro-averages across the seven years.

## Input

`results/oof_predictions_veg_spatial.csv`

Required columns:

- `year`
- `y_true`
- `pred_Stack`
- `pred_AEF`
- `pred_AEF+Stack`

## Run

```bash
python analyses/year_specific_spatial_cv/veg_year_specific_spatial_cv.py
```

Optional paths can be supplied with `--input` and `--output`.

## Output

`results/veg_year_specific_spatial_cv.csv`

This output underlies Supplementary Table S5 and the year-specific results reported in Section 3.1 of the manuscript. Because annual positive-class prevalence varies strongly, year-specific PR-AUC should be interpreted relative to the corresponding annual prevalence.
