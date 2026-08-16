import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
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
K_VALUES = [5, 10, 15]

df = pd.read_csv(VEG_PATH)
df = df.dropna(subset=['gs_ndvi'] + CLIM).copy()
g = df.groupby('id')['gs_ndvi']
df['z'] = (df['gs_ndvi'] - g.transform('mean')) / g.transform('std')
df['stress'] = (df['z'] < -1).astype(int)
df = df.sort_values(['id', 'year'])
prev = df.groupby('id')[BANDS].shift(1); prev.columns = [b + '_p' for b in BANDS]
df = pd.concat([df, prev], axis=1)
pb = [b + '_p' for b in BANDS]
df = df.dropna(subset=pb).reset_index(drop=True)
y = df['stress'].values
sets = {'AEF': pb, 'Stack': CLIM, 'AEF+Stack': pb + CLIM}
pts = df.drop_duplicates('id')[['id', 'lon', 'lat']].copy()

RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b):
    d = vals_a - vals_b
    n = len(d)
    mean_d = d.mean()
    boot = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=n)))
    pm = (signs * d).mean(axis=1)
    p = np.mean(np.abs(pm) >= np.abs(mean_d) - 1e-12)
    return {'mean_diff': round(mean_d, 4), 'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
            'ci_excludes_zero': bool(lo > 0 or hi < 0), 'perm_p': round(p, 4), 'n_folds': n}

all_fold_rows = []
all_summary = []
all_paired = []

for k in K_VALUES:
    blk = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(pts[['lon', 'lat']].values)
    pts_k = pts.copy(); pts_k['blk'] = blk
    dfk = df.merge(pts_k[['id', 'blk']], on='id')
    groups = dfk['blk'].values
    gkf = GroupKFold(n_splits=k)

    fold_rows = []
    for name, cols in sets.items():
        X = dfk[cols].values
        for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            yt = y[te]
            fold_rows.append({'k': k, 'feature_set': name, 'fold': fold_i,
                               'ROC_AUC': roc_auc_score(yt, p), 'PR_AUC': average_precision_score(yt, p),
                               'n_test': len(te)})
    fold_df = pd.DataFrame(fold_rows)
    all_fold_rows.append(fold_df)

    summ = fold_df.groupby('feature_set')[['ROC_AUC', 'PR_AUC']].agg(['mean', 'std']).round(4)
    print(f"\n=== Vegetation k={k} ===")
    print(summ)
    summ_flat = fold_df.groupby('feature_set')[['ROC_AUC', 'PR_AUC']].agg(['mean', 'std']).round(4)
    summ_flat.columns = ['_'.join(c) for c in summ_flat.columns]
    summ_flat['k'] = k
    all_summary.append(summ_flat.reset_index())

    for metric in ['ROC_AUC', 'PR_AUC']:
        piv = fold_df.pivot(index='fold', columns='feature_set', values=metric).sort_index()
        for a, b in [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]:
            row = paired_stats(piv[a].values, piv[b].values, a, b)
            row.update({'k': k, 'metric': metric, 'comparison': f'{a} - {b}'})
            all_paired.append(row)
    print(f"[k={k}] done.")

fold_all = pd.concat(all_fold_rows, ignore_index=True)
summary_all = pd.concat(all_summary, ignore_index=True)
paired_all = pd.DataFrame(all_paired)

fold_all.to_csv("veg_block_k_sensitivity_folds.csv", index=False)
summary_all.to_csv("veg_block_k_sensitivity_summary.csv", index=False)
paired_all.to_csv("veg_block_k_sensitivity_paired.csv", index=False)

print("\n\n=== Vegetation summary (all k) ===")
print(summary_all.to_string(index=False))
print("\n=== Vegetation paired (all k) ===")
print(paired_all[['k','metric','comparison','mean_diff','ci95_lo','ci95_hi','ci_excludes_zero','perm_p']].to_string(index=False))
print("\nDONE")
