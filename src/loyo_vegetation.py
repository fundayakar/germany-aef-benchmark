"""
Leave-one-year-out (LOYO) temporal validation for the vegetation stress task.

Uses the exact same label construction, antecedent-embedding shift, and
locked model config as lock_and_run.py / config.yaml, but groups folds by
YEAR instead of spatial block: train on all years except Y, test on Y,
for each Y in {2018..2024} (7 folds, since 2017 is dropped by the
antecedent-embedding shift). This tests temporal transfer -- whether the
predictor-target relationship learned in other years holds in an unseen
year -- as a complement to the spatial-block transfer already reported.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
from itertools import product
import xgboost as xgb

SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]

def load_veg():
    df = pd.read_csv(VEG_PATH)
    clim = ['precip_winter','precip_spring','temp_spring','sm_winter','sm_spring']
    df = df.dropna(subset=['gs_ndvi']+clim).copy()
    g = df.groupby('id')['gs_ndvi']
    df['z'] = (df['gs_ndvi']-g.transform('mean'))/g.transform('std')
    df['stress'] = (df['z'] < -1).astype(int)
    df = df.sort_values(['id','year'])
    prev = df.groupby('id')[BANDS].shift(1); prev.columns = [b+'_p' for b in BANDS]
    df = pd.concat([df, prev], axis=1)
    pb = [b+'_p' for b in BANDS]
    df = df.dropna(subset=pb).reset_index(drop=True)
    y = df['stress'].values
    sets = {'Stack':clim, 'AEF':pb, 'AEF+Stack':pb+clim}
    return df, y, sets

df, y, sets = load_veg()
years = sorted(df['year'].unique().astype(int).tolist())
print("Years used as LOYO folds:", years, f"(n={len(years)})")
print("Total point-years:", len(df))
print("Stress prevalence overall:", round(df['stress'].mean(), 3))
print("Stress prevalence by year:")
print(df.groupby('year')['stress'].agg(['mean','count']))

fold_rows = []
oof_preds = {name: np.full(len(df), np.nan) for name in sets}

for name, cols in sets.items():
    X = df[cols].values
    for yr in years:
        te = df['year'].values == yr
        tr = ~te
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof_preds[name][te] = p
        auc = roc_auc_score(y[te], p)
        fold_rows.append({'feature_set': name, 'test_year': yr,
                           'ROC_AUC': auc, 'n_test': int(te.sum()),
                           'n_train': int(tr.sum())})
    print(f"[{name}] done.")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("fold_level_veg_loyo.csv", index=False)

summary = fold_df.groupby('feature_set')['ROC_AUC'].agg(['mean','std']).round(3)
print("\n=== LOYO temporal-CV summary (ROC-AUC mean/std across 7 year-folds) ===")
print(summary)

# ---- paired stats across the 7 year-folds (same structure as spatial Table 2) ----
RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b):
    d = vals_a - vals_b
    n = len(d)
    mean_d = d.mean()
    boot_means = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=n)))
    perm_means = (signs * d).mean(axis=1)
    p_perm = np.mean(np.abs(perm_means) >= np.abs(mean_d) - 1e-12)
    t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, p_w = stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    return {
        'comparison': f'{label_a} - {label_b}', 'metric': 'ROC_AUC', 'n_folds': n,
        'mean_diff': round(mean_d, 4), 'ci95_lo': round(ci_lo, 4), 'ci95_hi': round(ci_hi, 4),
        'ci_excludes_zero': bool(ci_lo > 0 or ci_hi < 0),
        'perm_p': round(p_perm, 4), 'paired_t_p': round(p_t, 4),
        'wilcoxon_p': round(p_w, 4) if not np.isnan(p_w) else np.nan,
    }

piv = fold_df.pivot(index='test_year', columns='feature_set', values='ROC_AUC').sort_index()
pairs = [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]
rows = [paired_stats(piv[a].values, piv[b].values, a, b) for a, b in pairs]
paired_df = pd.DataFrame(rows)
paired_df.insert(0, 'task', 'veg_loyo')
paired_df.to_csv("paired_bootstrap_veg_loyo.csv", index=False)
print("\n=== Paired LOYO comparison (n=7 year-folds) ===")
print(paired_df.to_string(index=False))
