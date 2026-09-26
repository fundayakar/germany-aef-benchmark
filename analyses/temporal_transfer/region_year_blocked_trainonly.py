"""
Region-year blocked validation, re-run with the 7-year test-year-excluded
stress-label baseline (same label-correction logic as
veg_loyo_trainonly_label_v2.py): for each test year y, the per-point NDVI
baseline uses all of 2017-2024 except y (2017 included, even though it is
not itself a modelled point-year). All 10 block-folds that share the same
test year y use that year's label (the label depends only on which year is
held out, not on which spatial block is held out).

Fold structure is unchanged from the original region-year-blocked design:
train excludes BOTH the test year y (any block) AND the test block b (any
year); test = (year == y) AND (block == b). 70 folds (7 years x 10 blocks).
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

SEED = 42
N_BLOCKS = 10
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
MODELING_YEARS = list(range(2018, 2025))
BASELINE_YEARS = list(range(2017, 2025))

# ---- modelling panel (identical construction to main analysis) ----
raw_full = pd.read_csv(VEG_PATH)
panel = raw_full.dropna(subset=['gs_ndvi'] + CLIM).copy()
panel = panel.sort_values(['id', 'year'])
prev = panel.groupby('id')[BANDS].shift(1)
prev.columns = [b + '_p' for b in BANDS]
panel = pd.concat([panel, prev], axis=1)
pb = [b + '_p' for b in BANDS]
panel = panel.dropna(subset=pb).reset_index(drop=True)
retained_ids = panel['id'].unique()
print(f"Modelling panel: {len(panel)} point-years, {len(retained_ids)} locations")

# ---- spatial blocks: same convention as the main spatial-CV analysis ----
pts = panel.drop_duplicates('id')[['id', 'lon', 'lat']].copy()
pts['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(pts[['lon', 'lat']].values)
panel = panel.merge(pts[['id', 'blk']], on='id')

# ---- baseline source: same 1,976 locations, all available years 2017-2024 ----
baseline_source = raw_full[raw_full['id'].isin(retained_ids) & raw_full['year'].isin(BASELINE_YEARS)][['id', 'year', 'gs_ndvi']].copy()
assert baseline_source['gs_ndvi'].notna().all()

# ---- precompute the 7-year test-year-excluded label, once per possible test year ----
label_by_test_year = {}
prevalence_rows = []
for y_test in MODELING_YEARS:
    baseline_years_this = [yy for yy in BASELINE_YEARS if yy != y_test]  # 7 years, incl. 2017
    base_stats = (baseline_source[baseline_source['year'].isin(baseline_years_this)]
                  .groupby('id')['gs_ndvi'].agg(mu='mean', sd='std'))
    tmp = panel[['id', 'year', 'gs_ndvi']].merge(base_stats, on='id', how='left')
    z = (tmp['gs_ndvi'] - tmp['mu']) / tmp['sd']
    label_by_test_year[y_test] = (z <= -1).astype(int).values
    te_y = (panel['year'] == y_test).values
    yl = label_by_test_year[y_test]
    prevalence_rows.append({'test_year': y_test, 'prevalence': round(yl[te_y].mean(), 4),
                             'n_pos': int(yl[te_y].sum()), 'n': int(te_y.sum())})
print(pd.DataFrame(prevalence_rows).to_string(index=False))

years = MODELING_YEARS
blocks_list = sorted(panel['blk'].unique().tolist())
sets = {'Stack': CLIM, 'AEF': pb, 'AEF+Stack': pb + CLIM}
yr_arr = panel['year'].values
blk_arr = panel['blk'].values

rows = []
for name, cols in sets.items():
    X_full = panel[cols].values
    for yr in years:
        y_full = label_by_test_year[yr]  # this year's fold-specific label, applied panel-wide
        for b in blocks_list:
            te = (yr_arr == yr) & (blk_arr == b)
            tr = (yr_arr != yr) & (blk_arr != b)
            if te.sum() < 5 or len(np.unique(y_full[te])) < 2:
                continue
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X_full[tr], y_full[tr])
            p = m.predict_proba(X_full[te])[:, 1]
            yt = y_full[te]
            rows.append({'feature_set': name, 'year': yr, 'block': b,
                         'ROC_AUC': roc_auc_score(yt, p), 'PR_AUC': average_precision_score(yt, p),
                         'n_test': int(te.sum()), 'prevalence': round(yt.mean(), 3)})
    print(f"[{name}] region-year blocked (7-yr baseline) done.")

reg_df = pd.DataFrame(rows)
reg_df.to_csv("region_year_blocked_trainonly_folds.csv", index=False)

summary = reg_df.groupby('feature_set')[['ROC_AUC', 'PR_AUC']].agg(['mean', 'std']).round(4)
print("\n=== Region-year blocked (7-yr test-year-excluded baseline): mean/std over folds ===")
print(summary)
summary.to_csv("region_year_blocked_trainonly_summary.csv")

# ---- year-clustered bootstrap CI vs chance (0.5) / no-skill PR-AUC baseline ----
RNG = np.random.default_rng(42)
N_BOOT = 10000
overall_prevalence = float(np.mean([label_by_test_year[yr][(yr_arr == yr)].mean() for yr in years]))
print(f"\nApprox. overall no-skill PR-AUC baseline (mean of yearly prevalences): {overall_prevalence:.4f}")

ci_rows = []
for name in sets:
    sub = reg_df[reg_df.feature_set == name]
    by_year = {yr: sub[sub.year == yr] for yr in years}
    boot_auc, boot_prauc = [], []
    for _ in range(N_BOOT):
        drawn = RNG.choice(years, size=len(years), replace=True)
        resampled = pd.concat([by_year[yr] for yr in drawn], ignore_index=True)
        boot_auc.append(resampled['ROC_AUC'].mean())
        boot_prauc.append(resampled['PR_AUC'].mean())
    auc_mean = sub['ROC_AUC'].mean(); auc_lo, auc_hi = np.percentile(boot_auc, [2.5, 97.5])
    pr_mean = sub['PR_AUC'].mean(); pr_lo, pr_hi = np.percentile(boot_prauc, [2.5, 97.5])
    ci_rows.append({'feature_set': name, 'mean_AUC': round(auc_mean, 4),
                     'AUC_CI_lo': round(auc_lo, 4), 'AUC_CI_hi': round(auc_hi, 4),
                     'CI_excludes_0.5': bool(auc_lo > 0.5 or auc_hi < 0.5),
                     'mean_PRAUC': round(pr_mean, 4), 'PRAUC_CI_lo': round(pr_lo, 4),
                     'PRAUC_CI_hi': round(pr_hi, 4),
                     'CI_excludes_noskill': bool(pr_lo > overall_prevalence or pr_hi < overall_prevalence)})
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv("region_year_blocked_trainonly_CI.csv", index=False)
print("\n=== Year-clustered bootstrap CI vs chance/no-skill ===")
print(ci_df.to_string(index=False))
