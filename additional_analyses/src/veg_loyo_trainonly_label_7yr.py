"""
Training-year-only stress-label sensitivity for LOYO -- CORRECTED baseline.

Per reviewer guidance: the per-point NDVI baseline (mean/SD) for each LOYO fold
uses all years 2017-2024 EXCLUDING the held-out test year (7 years), not just
the six modelling years 2018-2024 excluding the test year. 2017 is available
for defining the baseline (it has valid gs_ndvi for all 1,976 retained
locations) even though it is never itself a modelled point-year (it lacks a
prior-year AlphaEarth embedding for the antecedent design).

Everything else (predictors, feature sets, model config, the 13,832-row
modelling panel) is identical to the main LOYO analysis and to the previous
(6-year-baseline) version of this sensitivity check.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy import stats
from itertools import product
import xgboost as xgb

SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
MODELING_YEARS = list(range(2018, 2025))   # 2018..2024 -- possible test years / modelled rows
BASELINE_YEARS = list(range(2017, 2025))   # 2017..2024 -- pool available for the NDVI baseline

# ---- full raw file: used only as the source of 2017-2024 gs_ndvi for the baseline ----
raw_full = pd.read_csv(VEG_PATH)

# ---- modelling panel: identical construction to the main LOYO analysis ----
panel = raw_full.dropna(subset=['gs_ndvi'] + CLIM).copy()
panel = panel.sort_values(['id', 'year'])
prev = panel.groupby('id')[BANDS].shift(1)
prev.columns = [b + '_p' for b in BANDS]
panel = pd.concat([panel, prev], axis=1)
pb = [b + '_p' for b in BANDS]
panel = panel.dropna(subset=pb).reset_index(drop=True)  # -> 13,832 rows, years 2018-2024
assert sorted(panel['year'].unique().astype(int).tolist()) == MODELING_YEARS
retained_ids = panel['id'].unique()
print(f"Modelling panel: {len(panel)} point-years, {len(retained_ids)} locations, years {MODELING_YEARS}")

# ---- baseline source: same 1,976 locations, all available years 2017-2024 ----
baseline_source = raw_full[raw_full['id'].isin(retained_ids) & raw_full['year'].isin(BASELINE_YEARS)][['id', 'year', 'gs_ndvi']].copy()
assert baseline_source['gs_ndvi'].notna().all(), "unexpected missing gs_ndvi in baseline source for a retained location"
print(f"Baseline source: {len(baseline_source)} point-years across {baseline_source['id'].nunique()} "
      f"locations, years {BASELINE_YEARS} (should be {len(retained_ids)*8} = 8 years/location)")

sets = {'AEF': pb, 'Stack': CLIM, 'AEF+Stack': pb + CLIM}

fold_rows = []
prevalence_rows = []
oof_by_set = {name: np.full(len(panel), np.nan) for name in sets}
label_as_test_by_year = {}  # cache each row's label-when-it-was-the-test-year, for pooling

for y_test in MODELING_YEARS:
    baseline_years_this_fold = [yy for yy in BASELINE_YEARS if yy != y_test]  # 7 years
    base_stats = (baseline_source[baseline_source['year'].isin(baseline_years_this_fold)]
                  .groupby('id')['gs_ndvi'].agg(mu='mean', sd='std'))

    fold_df = panel.merge(base_stats, on='id', how='left')
    fold_df['z'] = (fold_df['gs_ndvi'] - fold_df['mu']) / fold_df['sd']
    fold_df['stress'] = (fold_df['z'] <= -1).astype(int)
    label_as_test_by_year[y_test] = fold_df['stress'].values  # full-panel labels under this fold's baseline

    te_mask = (fold_df['year'] == y_test).values
    tr_mask = fold_df['year'].isin([yy for yy in MODELING_YEARS if yy != y_test]).values
    y_all = fold_df['stress'].values

    prevalence_rows.append({
        'test_year': y_test, 'n_test': int(te_mask.sum()),
        'positives_test': int(y_all[te_mask].sum()),
        'negatives_test': int(te_mask.sum() - y_all[te_mask].sum()),
        'prevalence_test': round(y_all[te_mask].mean(), 4),
        'baseline_years_used': f"{min(baseline_years_this_fold)}-{max(baseline_years_this_fold)} excl. {y_test} (n={len(baseline_years_this_fold)})",
    })

    for name, cols in sets.items():
        X = fold_df[cols].values
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr_mask], y_all[tr_mask])
        p = m.predict_proba(X[te_mask])[:, 1]
        oof_by_set[name][te_mask] = p
        yt = y_all[te_mask]
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        brier = brier_score_loss(yt, p)
        fold_rows.append({'feature_set': name, 'test_year': y_test, 'ROC_AUC': auc, 'PR_AUC': ap,
                           'Brier': brier, 'n_test': int(te_mask.sum()), 'positives_test': int(yt.sum()),
                           'prevalence_test': round(yt.mean(), 4), 'mean_pred_prob': round(p.mean(), 4)})
    print(f"[y_test={y_test}] baseline={baseline_years_this_fold} prevalence={y_all[te_mask].mean():.4f} "
          f"(n_pos={int(y_all[te_mask].sum())}/{int(te_mask.sum())})")

fold_df_out = pd.DataFrame(fold_rows)
prev_df = pd.DataFrame(prevalence_rows)
fold_df_out.to_csv("veg_loyo_trainonly_v2_per_year.csv", index=False)
prev_df.to_csv("veg_loyo_trainonly_v2_prevalence.csv", index=False)

# ---- pooled metrics: each row scored against the label it received when IT was the test year ----
pooled_rows = []
for name in sets:
    p = oof_by_set[name]
    y_label_as_test = np.full(len(panel), np.nan)
    for y_test in MODELING_YEARS:
        te_mask = (panel['year'] == y_test).values
        y_label_as_test[te_mask] = label_as_test_by_year[y_test][te_mask]
    auc = roc_auc_score(y_label_as_test, p)
    ap = average_precision_score(y_label_as_test, p)
    brier = brier_score_loss(y_label_as_test, p)
    mean_yw_auc = fold_df_out.loc[fold_df_out.feature_set == name, 'ROC_AUC'].mean()
    pooled_rows.append({'feature_set': name, 'pooled_ROC_AUC': round(auc, 4), 'pooled_PR_AUC': round(ap, 4),
                         'pooled_Brier': round(brier, 4), 'mean_of_yearwise_ROC_AUC': round(mean_yw_auc, 4)})
pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv("veg_loyo_trainonly_v2_pooled.csv", index=False)

# ---- one-sample test vs chance (0.5) for mean-of-yearwise AUC ----
RNG = np.random.default_rng(42)
N_BOOT = 10000
chance_rows = []
for name in sets:
    vals = fold_df_out[fold_df_out.feature_set == name].sort_values('test_year')['ROC_AUC'].values
    d = vals - 0.5
    boot = np.array([RNG.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=len(d))))
    pm = (signs * d).mean(axis=1)
    p_perm = np.mean(np.abs(pm) >= np.abs(d.mean()) - 1e-12)
    chance_rows.append({'feature_set': name, 'mean_AUC': round(vals.mean(), 4), 'sd_AUC': round(vals.std(), 4),
                         'ci95_lo': round(lo + 0.5, 4), 'ci95_hi': round(hi + 0.5, 4),
                         'excludes_0.5': bool(lo > 0 or hi < 0), 'perm_p': round(p_perm, 4)})
chance_df = pd.DataFrame(chance_rows)
chance_df.to_csv("veg_loyo_trainonly_v2_vs_chance.csv", index=False)

print("\n=== Yearly prevalence (2017-2024-based, test-year-excluded baseline) ===")
print(prev_df.to_string(index=False))
print("\n=== Year-wise ROC-AUC by feature set ===")
print(fold_df_out.pivot(index='test_year', columns='feature_set', values='ROC_AUC').round(3))
print("\n=== Pooled metrics ===")
print(pooled_df.to_string(index=False))
print("\n=== vs. chance (0.5) ===")
print(chance_df.to_string(index=False))
