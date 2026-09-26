"""
Formalize the "approached chance" claim for region-year blocked CV:
year-clustered bootstrap 95% CI on mean AUC/PR-AUC per feature set, testing
whether 0.5 (AUC) / prevalence-baseline (PR-AUC) falls inside the interval.
Clustering by year (not treating all 70 folds as independent) because year
is the dominant source of correlated variation across the 10 blocks that
share a year.
"""
import numpy as np, pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 10000

df = pd.read_csv("region_year_blocked_folds.csv")
years = sorted(df['year'].unique())
overall_prevalence = 0.1983  # from veg_yearly_prevalence.csv (pooled)

rows = []
for name in df['feature_set'].unique():
    sub = df[df.feature_set == name]
    # index rows by year for cluster resampling
    by_year = {yr: sub[sub.year == yr] for yr in years}

    boot_auc_means = []
    boot_prauc_means = []
    for _ in range(N_BOOT):
        drawn_years = RNG.choice(years, size=len(years), replace=True)
        resampled = pd.concat([by_year[yr] for yr in drawn_years], ignore_index=True)
        boot_auc_means.append(resampled['ROC_AUC'].mean())
        boot_prauc_means.append(resampled['PR_AUC'].mean())

    auc_mean = sub['ROC_AUC'].mean()
    auc_lo, auc_hi = np.percentile(boot_auc_means, [2.5, 97.5])
    prauc_mean = sub['PR_AUC'].mean()
    prauc_lo, prauc_hi = np.percentile(boot_prauc_means, [2.5, 97.5])

    rows.append({
        'feature_set': name,
        'mean_AUC': round(auc_mean, 4), 'AUC_CI_lo': round(auc_lo, 4), 'AUC_CI_hi': round(auc_hi, 4),
        'CI_excludes_0.5': bool(auc_lo > 0.5 or auc_hi < 0.5),
        'mean_PRAUC': round(prauc_mean, 4), 'PRAUC_CI_lo': round(prauc_lo, 4), 'PRAUC_CI_hi': round(prauc_hi, 4),
        'CI_excludes_noskill_baseline': bool(prauc_lo > overall_prevalence or prauc_hi < overall_prevalence),
    })

out = pd.DataFrame(rows)
out.to_csv("region_year_blocked_CI.csv", index=False)
print(out.to_string(index=False))
