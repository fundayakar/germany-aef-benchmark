import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from itertools import product
import xgboost as xgb

SEED = 42
N_BLOCKS = 10
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]

def blocks(coords):
    return KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(coords)

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
    pts = df.drop_duplicates('id')[['id','lon','lat']].copy()
    pts['blk'] = blocks(pts[['lon','lat']].values)
    df = df.merge(pts[['id','blk']], on='id')
    y = df['stress'].values
    sets = {'AEF': pb, 'Stack': clim, 'AEF+Stack': pb + clim}
    return df, y, sets

df, y, sets = load_veg()
groups = df['blk'].values
gkf = GroupKFold(n_splits=N_BLOCKS)

fold_rows = []
for name, cols in sets.items():
    X = df[cols].values
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        fold_rows.append({
            'feature_set': name, 'fold': k,
            'ROC_AUC': roc_auc_score(y[te], p),
            'PR_AUC': average_precision_score(y[te], p),
        })
    print(f"[{name}] done.")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("veg_prauc_rocauc_fold_level.csv", index=False)

RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b, metric_name):
    d = vals_a - vals_b
    n = len(d)
    mean_d = d.mean()
    boot_means = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=n)))
    perm_means = (signs * d).mean(axis=1)
    p_perm = np.mean(np.abs(perm_means) >= np.abs(mean_d) - 1e-12)
    return {
        'metric': metric_name, 'comparison': f'{label_a} - {label_b}', 'n_folds': n,
        'mean_diff': round(mean_d, 4), 'ci95_lo': round(ci_lo, 4), 'ci95_hi': round(ci_hi, 4),
        'ci_excludes_zero': bool(ci_lo > 0 or ci_hi < 0), 'perm_p': round(p_perm, 4),
    }

pairs = [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]
rows = []
for metric in ['ROC_AUC', 'PR_AUC']:
    piv = fold_df.pivot(index='fold', columns='feature_set', values=metric).sort_index()
    for a, b in pairs:
        rows.append(paired_stats(piv[a].values, piv[b].values, a, b, metric))

out = pd.DataFrame(rows)
out.to_csv("veg_paired_rocauc_prauc.csv", index=False)
print("\n=== Paired fold-level comparison: vegetation ROC-AUC and PR-AUC (spatial CV, n=10 folds) ===")
print(out.to_string(index=False))
