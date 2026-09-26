"""
Training-year-only stress-label sensitivity for LOYO (addresses Referee 2's
leakage concern: the original label standardized each point's NDVI using its
full 2017-2024 mean/SD, so the held-out test year's own NDVI value
contributed to the baseline used to label that same test year).

For each LOYO fold with test year y in {2018..2024}: recompute each point's
NDVI mean/SD using ONLY the six training years (2018-2024 minus y, i.e. NOT
including 2017), then apply this fold-specific, training-only baseline to
define stress = (z <= -1 SD) for BOTH the six training years and the held-out
year y. Predictors, feature sets, model config, and the underlying row mask
are identical to the main LOYO analysis; only the label's baseline changes,
and it now varies by fold.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
MODELING_YEARS = list(range(2018, 2025))  # 2018..2024, matches main analysis

# ---- base panel: same row mask as the main LOYO analysis (antecedent shift) ----
raw = pd.read_csv(VEG_PATH)
raw = raw.dropna(subset=['gs_ndvi'] + CLIM).copy()
raw = raw.sort_values(['id', 'year'])
prev = raw.groupby('id')[BANDS].shift(1)
prev.columns = [b + '_p' for b in BANDS]
raw = pd.concat([raw, prev], axis=1)
pb = [b + '_p' for b in BANDS]
raw = raw.dropna(subset=pb).reset_index(drop=True)  # -> 13,832 rows, 2018-2024 only
assert sorted(raw['year'].unique().astype(int).tolist()) == MODELING_YEARS
print(f"Base panel: {len(raw)} point-years, years {MODELING_YEARS}")

sets = {'AEF': pb, 'Stack': CLIM, 'AEF+Stack': pb + CLIM}

fold_rows = []
prevalence_rows = []
oof_by_set = {name: np.full(len(raw), np.nan) for name in sets}

for y_test in MODELING_YEARS:
    train_years = [yy for yy in MODELING_YEARS if yy != y_test]

    # fold-specific, training-year-only baseline per point
    train_mask = raw['year'].isin(train_years)
    base_stats = (raw.loc[train_mask]
                     .groupby('id')['gs_ndvi']
                     .agg(mu='mean', sd='std'))

    fold_df = raw.merge(base_stats, on='id', how='left')
    fold_df['z'] = (fold_df['gs_ndvi'] - fold_df['mu']) / fold_df['sd']
    fold_df['stress'] = (fold_df['z'] <= -1).astype(int)

    te_mask = (fold_df['year'] == y_test).values
    tr_mask = fold_df['year'].isin(train_years).values
    y_all = fold_df['stress'].values

    prevalence_rows.append({
        'test_year': y_test,
        'n_test': int(te_mask.sum()),
        'positives_test': int(y_all[te_mask].sum()),
        'negatives_test': int(te_mask.sum() - y_all[te_mask].sum()),
        'prevalence_test': round(y_all[te_mask].mean(), 4),
    })

    for name, cols in sets.items():
        X = fold_df[cols].values
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr_mask], y_all[tr_mask])
        p = m.predict_proba(X[te_mask])[:, 1]
        oof_by_set[name][te_mask] = p  # global index aligns 1:1 with `raw`/`fold_df` row order
        yt = y_all[te_mask]
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        brier = brier_score_loss(yt, p)
        fold_rows.append({'feature_set': name, 'test_year': y_test,
                           'ROC_AUC': auc, 'PR_AUC': ap, 'Brier': brier,
                           'n_test': int(te_mask.sum()), 'positives_test': int(yt.sum()),
                           'prevalence_test': round(yt.mean(), 4),
                           'mean_pred_prob': round(p.mean(), 4)})
    print(f"[y_test={y_test}] done. prevalence={y_all[te_mask].mean():.4f} "
          f"(n_pos={int(y_all[te_mask].sum())}/{int(te_mask.sum())})")

fold_df_out = pd.DataFrame(fold_rows)
prev_df = pd.DataFrame(prevalence_rows)
fold_df_out.to_csv("veg_loyo_trainonly_per_year.csv", index=False)
prev_df.to_csv("veg_loyo_trainonly_prevalence.csv", index=False)

# ---- pooled metrics (all 7 folds' out-of-fold predictions combined) ----
# NOTE: because the label is fold-specific, "pooled" here pools each row's
# prediction against the STRESS LABEL DEFINED IN THE FOLD WHERE THAT ROW WAS
# THE TEST YEAR (i.e., the only label each row ever received as a test point).
pooled_rows = []
for name in sets:
    p = oof_by_set[name]
    valid = ~np.isnan(p)
    # reconstruct the label each row had when it was in the test fold: recompute
    # using that row's own year as y_test (matches the loop above exactly)
    y_label_as_test = np.full(len(raw), np.nan)
    for y_test in MODELING_YEARS:
        train_years = [yy for yy in MODELING_YEARS if yy != y_test]
        train_mask = raw['year'].isin(train_years)
        base_stats = raw.loc[train_mask].groupby('id')['gs_ndvi'].agg(mu='mean', sd='std')
        tmp = raw.merge(base_stats, on='id', how='left')
        tmp['z'] = (tmp['gs_ndvi'] - tmp['mu']) / tmp['sd']
        tmp_stress = (tmp['z'] <= -1).astype(int).values
        te_mask = (raw['year'] == y_test).values
        y_label_as_test[te_mask] = tmp_stress[te_mask]
    yt = y_label_as_test[valid]
    pv = p[valid]
    auc = roc_auc_score(yt, pv)
    ap = average_precision_score(yt, pv)
    brier = brier_score_loss(yt, pv)
    mean_yw_auc = fold_df_out.loc[fold_df_out.feature_set == name, 'ROC_AUC'].mean()
    pooled_rows.append({'feature_set': name, 'pooled_ROC_AUC': round(auc, 4),
                         'pooled_PR_AUC': round(ap, 4), 'pooled_Brier': round(brier, 4),
                         'mean_of_yearwise_ROC_AUC': round(mean_yw_auc, 4)})

pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv("veg_loyo_trainonly_pooled.csv", index=False)

print("\n=== Yearly prevalence (training-year-only baseline) ===")
print(prev_df.to_string(index=False))
print("\n=== Year-wise ROC-AUC by feature set ===")
print(fold_df_out.pivot(index='test_year', columns='feature_set', values='ROC_AUC').round(3))
print("\n=== Pooled metrics ===")
print(pooled_df.to_string(index=False))
