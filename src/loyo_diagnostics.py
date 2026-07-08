"""
Diagnostics requested for the LOYO temporal-validation finding.

1. Pooled out-of-year AUC (all LOYO OOF predictions pooled, one ROC-AUC)
2. PR-AUC (average precision) and Brier score, per year and pooled
3. Region-year blocked CV (test = year Y AND spatial block B; train excludes
   ALL of year Y and ALL of block B, so the model never sees that location
   in any year and never sees that year in any other location)
4. Yearly prevalence table (positives/negatives per year)
5. Simple baselines: LC-only (land-cover one-hot), climate-anomaly-only
   (point-relative z-scored climate, same normalization convention as the
   label itself)
6. Mean predicted probability per year vs actual prevalence per year, to
   check whether the model captures the coarse "bad year" signal even
   when within-year ranking (AUC) is poor.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats
import xgboost as xgb

SEED = 42
N_BLOCKS = 10
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
CLIM = ['precip_winter','precip_spring','temp_spring','sm_winter','sm_spring']

def load_veg_full():
    df = pd.read_csv(VEG_PATH)
    df = df.dropna(subset=['gs_ndvi']+CLIM).copy()
    g = df.groupby('id')['gs_ndvi']
    df['ndvi_mean'] = g.transform('mean'); df['ndvi_std'] = g.transform('std')
    df['z'] = (df['gs_ndvi']-df['ndvi_mean'])/df['ndvi_std']
    df['stress'] = (df['z'] < -1).astype(int)

    # point-relative climate anomalies (same normalization convention as the label)
    for c in CLIM:
        gc = df.groupby('id')[c]
        df[c+'_anom'] = (df[c] - gc.transform('mean')) / gc.transform('std')

    df = df.sort_values(['id','year'])
    prev = df.groupby('id')[BANDS].shift(1); prev.columns = [b+'_p' for b in BANDS]
    df = pd.concat([df, prev], axis=1)
    pb = [b+'_p' for b in BANDS]
    df = df.dropna(subset=pb).reset_index(drop=True)

    # land-cover one-hot (lc: 1 forest, 2 cropland, 3 grassland, 4 other)
    lc_dummies = pd.get_dummies(df['lc'].astype(int), prefix='lc').astype(float)
    df = pd.concat([df, lc_dummies], axis=1)
    lc_cols = list(lc_dummies.columns)

    # spatial blocks (same convention as the spatial-CV script: assigned at point level)
    pts = df.drop_duplicates('id')[['id','lon','lat']].copy()
    pts['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(pts[['lon','lat']].values)
    df = df.merge(pts[['id','blk']], on='id')

    anom_cols = [c+'_anom' for c in CLIM]
    feature_sets = {
        'Stack': CLIM,
        'AEF': pb,
        'AEF+Stack': pb + CLIM,
        'LC-only': lc_cols,
        'Climate-anomaly-only': anom_cols,
    }
    return df, feature_sets, pb, lc_cols, anom_cols

df, feature_sets, pb, lc_cols, anom_cols = load_veg_full()
years = sorted(df['year'].unique().astype(int).tolist())
y_all = df['stress'].values

# ---- 4. Yearly prevalence table ----
prev_tab = df.groupby('year')['stress'].agg(n='count', positives='sum')
prev_tab['negatives'] = prev_tab['n'] - prev_tab['positives']
prev_tab['prevalence'] = (prev_tab['positives'] / prev_tab['n']).round(3)
prev_tab = prev_tab[['n','positives','negatives','prevalence']]
print("=== Yearly prevalence ===")
print(prev_tab)
prev_tab.to_csv("veg_yearly_prevalence.csv")

# ============================================================
# LOYO for all feature sets (original 3 + 2 baselines), saving OOF probs
# ============================================================
oof = {name: np.full(len(df), np.nan) for name in feature_sets}
fold_metrics = []

for name, cols in feature_sets.items():
    X = df[cols].values
    for yr in years:
        te = df['year'].values == yr
        tr = ~te
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y_all[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[name][te] = p
        yt = y_all[te]
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        brier = brier_score_loss(yt, p)
        fold_metrics.append({'feature_set': name, 'test_year': yr, 'ROC_AUC': auc,
                              'PR_AUC': ap, 'Brier': brier, 'n_test': int(te.sum()),
                              'prevalence': round(yt.mean(), 3),
                              'mean_pred_prob': round(p.mean(), 4)})
    print(f"[{name}] LOYO done.")

fold_df = pd.DataFrame(fold_metrics)
fold_df.to_csv("loyo_full_diagnostics_per_year.csv", index=False)
print("\n=== Per-year metrics (all feature sets) ===")
print(fold_df.pivot(index='test_year', columns='feature_set', values='ROC_AUC').round(3))

# ---- 1 & 2. Pooled metrics across all LOYO OOF predictions ----
pooled_rows = []
for name in feature_sets:
    p = oof[name]
    auc = roc_auc_score(y_all, p)
    ap = average_precision_score(y_all, p)
    brier = brier_score_loss(y_all, p)
    mean_auc_of_years = fold_df.loc[fold_df.feature_set == name, 'ROC_AUC'].mean()
    pooled_rows.append({'feature_set': name, 'pooled_ROC_AUC': round(auc, 4),
                         'pooled_PR_AUC': round(ap, 4), 'pooled_Brier': round(brier, 4),
                         'mean_of_yearwise_AUC': round(mean_auc_of_years, 4)})
pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv("loyo_pooled_metrics.csv", index=False)
print("\n=== Pooled (all years combined) vs mean-of-yearwise AUC ===")
print(pooled_df.to_string(index=False))

# ---- 6. Mean predicted probability per year vs actual prevalence ----
print("\n=== Mean predicted probability by year vs actual prevalence ===")
diag6 = fold_df.pivot(index='test_year', columns='feature_set', values='mean_pred_prob').round(3)
diag6['actual_prevalence'] = prev_tab['prevalence'].values
print(diag6)
diag6.to_csv("loyo_meanprob_vs_prevalence.csv")

print("\n=== Correlation (across 7 years) between mean predicted prob and actual prevalence ===")
corr_rows = []
for name in feature_sets:
    sub = fold_df[fold_df.feature_set == name].sort_values('test_year')
    r, p_r = stats.pearsonr(sub['mean_pred_prob'], sub['prevalence'])
    corr_rows.append({'feature_set': name, 'pearson_r_meanprob_vs_prevalence': round(r, 3), 'p_value': round(p_r, 4)})
corr_df = pd.DataFrame(corr_rows)
print(corr_df.to_string(index=False))
corr_df.to_csv("loyo_yearlevel_correlation.csv", index=False)
