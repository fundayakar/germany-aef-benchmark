"""
Fold-internal PCA dimensionality sensitivity, vegetation-stress spatial-CV
benchmark. Analogous to soc_pca_dim_sensitivity.py: tests whether AEF's
64-vs-5 dimensionality advantage over the climate Stack drives the
complementarity result. For each of the 10 spatial-CV folds, the 64
prior-year AlphaEarth dimensions are standardized and reduced to 5 components
via PCA fit on the TRAINING FOLD ONLY; the fitted scaler/PCA then transform
the test fold. Same label (-1 SD, full-period baseline, as in Table 1), same
spatial blocks, same XGBoost config as the main spatial-CV benchmark.

Feature sets:
  Stack-5          : the 5 antecedent hydroclimate variables (raw, Table 1)
  AEF-64           : the 64 raw prior-year AlphaEarth bands (Table 1)
  AEF-PCA5         : AlphaEarth reduced to 5 components, fold-internal PCA
  AEF-PCA5+Stack-5 : PCA-reduced AlphaEarth (5) + climate stack (5)
  AEF-64+Stack-5   : raw AlphaEarth (64) + climate stack (5), as in Table 1
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
PCA_N = 5

def blocks(coords):
    return KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(coords)

# ---- exact same panel construction as the main analysis (Table 1) ----
df = pd.read_csv(VEG_PATH)
df = df.dropna(subset=['gs_ndvi'] + CLIM).copy()
g = df.groupby('id')['gs_ndvi']
df['z'] = (df['gs_ndvi'] - g.transform('mean')) / g.transform('std')
df['stress'] = (df['z'] < -1).astype(int)   # main-analysis label: strict '<', full-period baseline
df = df.sort_values(['id', 'year'])
prev = df.groupby('id')[BANDS].shift(1)
prev.columns = [b + '_p' for b in BANDS]
df = pd.concat([df, prev], axis=1)
pb = [b + '_p' for b in BANDS]
df = df.dropna(subset=pb).reset_index(drop=True)
pts = df.drop_duplicates('id')[['id', 'lon', 'lat']].copy()
pts['blk'] = blocks(pts[['lon', 'lat']].values)
df = df.merge(pts[['id', 'blk']], on='id')
print(f"n = {len(df)} (should match main analysis: 13832)")

y = df['stress'].values
groups = df['blk'].values
gkf = GroupKFold(n_splits=N_BLOCKS)

X_aef = df[pb].values
X_stack = df[CLIM].values

fold_rows = []
folds = list(gkf.split(X_aef, y, groups))

for k, (tr, te) in enumerate(folds):
    scaler = StandardScaler().fit(X_aef[tr])
    aef_tr_scaled = scaler.transform(X_aef[tr])
    aef_te_scaled = scaler.transform(X_aef[te])
    pca = PCA(n_components=PCA_N, random_state=SEED).fit(aef_tr_scaled)
    aef_tr_pca = pca.transform(aef_tr_scaled)
    aef_te_pca = pca.transform(aef_te_scaled)
    evr = pca.explained_variance_ratio_.sum()

    feature_sets = {
        'Stack-5':          (X_stack[tr], X_stack[te]),
        'AEF-64':            (X_aef[tr], X_aef[te]),
        'AEF-PCA5':          (aef_tr_pca, aef_te_pca),
        'AEF-PCA5+Stack-5':  (np.hstack([aef_tr_pca, X_stack[tr]]), np.hstack([aef_te_pca, X_stack[te]])),
        'AEF-64+Stack-5':    (np.hstack([X_aef[tr], X_stack[tr]]), np.hstack([X_aef[te], X_stack[te]])),
    }

    for name, (Xtr, Xte) in feature_sets.items():
        m = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        yt = y[te]
        fold_rows.append({'feature_set': name, 'fold': k,
                           'ROC_AUC': roc_auc_score(yt, p),
                           'PR_AUC': average_precision_score(yt, p),
                           'n_test': len(te),
                           'pca_explained_var': round(evr, 4) if 'PCA' in name else np.nan})
    print(f"fold {k}: PCA(5) explained variance ratio = {evr:.4f}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("veg_pca_dim_fold_level.csv", index=False)

order = ['Stack-5', 'AEF-64', 'AEF-PCA5', 'AEF-PCA5+Stack-5', 'AEF-64+Stack-5']
summary = fold_df.groupby('feature_set')[['ROC_AUC', 'PR_AUC']].agg(['mean', 'std']).round(4)
summary = summary.reindex(order)
summary.to_csv("veg_pca_dim_summary.csv")
print("\n=== Spatial-CV ROC-AUC / PR-AUC by feature set (mean +/- SD over 10 folds) ===")
print(summary)

mean_evr = fold_df.loc[fold_df.pca_explained_var.notna(), 'pca_explained_var'].mean()
print(f"\nMean PCA(5) explained variance ratio across folds: {mean_evr:.4f}")

# ---- paired stats ----
RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b, metric):
    d = vals_a - vals_b
    n = len(d)
    mean_d = d.mean()
    boot = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=n)))
    pm = (signs * d).mean(axis=1)
    p = np.mean(np.abs(pm) >= np.abs(mean_d) - 1e-12)
    return {'metric': metric, 'comparison': f'{label_a} - {label_b}', 'mean_diff': round(mean_d, 4),
            'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
            'ci_excludes_zero': bool(lo > 0 or hi < 0), 'perm_p': round(p, 4)}

pairs = [('AEF-PCA5', 'Stack-5'), ('AEF-PCA5+Stack-5', 'AEF-PCA5')]
paired_rows = []
for metric in ['ROC_AUC', 'PR_AUC']:
    piv = fold_df.pivot(index='fold', columns='feature_set', values=metric).sort_index()
    for a, b in pairs:
        paired_rows.append(paired_stats(piv[a].values, piv[b].values, a, b, metric))
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv("veg_pca_dim_paired.csv", index=False)
print("\n=== Paired comparisons ===")
print(paired_df.to_string(index=False))
