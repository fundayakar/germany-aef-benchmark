import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

SEED = 42
N_BLOCKS = 10
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
CLIM = ['precip_winter','precip_spring','temp_spring','sm_winter','sm_spring']

df = pd.read_csv(VEG_PATH)
df = df.dropna(subset=['gs_ndvi']+CLIM).copy()
g = df.groupby('id')['gs_ndvi']
df['z'] = (df['gs_ndvi']-g.transform('mean'))/g.transform('std')
df['stress'] = (df['z'] < -1).astype(int)
df = df.sort_values(['id','year'])
prev = df.groupby('id')[BANDS].shift(1); prev.columns = [b+'_p' for b in BANDS]
df = pd.concat([df, prev], axis=1)
pb = [b+'_p' for b in BANDS]
df = df.dropna(subset=pb).reset_index(drop=True)

pts = df.drop_duplicates('id')[['id','lon','lat']].copy()
pts['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(pts[['lon','lat']].values)
df = df.merge(pts[['id','blk']], on='id')

years = sorted(df['year'].unique().astype(int).tolist())
blocks = sorted(df['blk'].unique().tolist())
feature_sets = {'Stack': CLIM, 'AEF': pb, 'AEF+Stack': pb + CLIM}
y_all = df['stress'].values
yr_arr = df['year'].values
blk_arr = df['blk'].values

rows = []
n_folds = len(years) * len(blocks)
i = 0
for name, cols in feature_sets.items():
    X = df[cols].values
    for yr in years:
        for b in blocks:
            i += 1
            te = (yr_arr == yr) & (blk_arr == b)
            tr = (yr_arr != yr) & (blk_arr != b)  # exclude target year AND target block entirely
            if te.sum() < 5 or len(np.unique(y_all[te])) < 2:
                continue  # skip degenerate folds (too few points or single-class test set)
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y_all[tr])
            p = m.predict_proba(X[te])[:, 1]
            yt = y_all[te]
            auc = roc_auc_score(yt, p)
            ap = average_precision_score(yt, p)
            brier = brier_score_loss(yt, p)
            rows.append({'feature_set': name, 'year': yr, 'block': b, 'ROC_AUC': auc,
                         'PR_AUC': ap, 'Brier': brier, 'n_test': int(te.sum()),
                         'n_train': int(tr.sum()), 'prevalence': round(yt.mean(), 3)})
    print(f"[{name}] region-year blocked CV done ({i} folds attempted so far).")

reg_df = pd.DataFrame(rows)
reg_df.to_csv("region_year_blocked_folds.csv", index=False)

print(f"\nUsable folds per feature set (out of up to {len(years)*len(blocks)}):")
print(reg_df.groupby('feature_set').size())

print("\n=== Region-year blocked: mean/std ROC-AUC and PR-AUC across usable folds ===")
summary = reg_df.groupby('feature_set')[['ROC_AUC','PR_AUC','Brier']].agg(['mean','std']).round(3)
print(summary)
summary.to_csv("region_year_blocked_summary.csv")
