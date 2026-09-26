"""
Fold-internal PCA dimensionality sensitivity, SOC spatial-CV benchmark.

Addresses Referee 2: AEF has 64 dims vs. the engineered Stack's 17. For each
of the 10 spatial-CV folds, AlphaEarth is standardized (StandardScaler) and
reduced to 17 components (PCA) using ONLY the training fold; the fitted
scaler and PCA are then applied to transform the test fold. No fitting ever
sees test-fold data (neither the scaler nor the PCA). Same locked RF config,
same spatial blocks, same target (log1p(Lucas_OC)) as the main analysis.

Feature sets:
  Stack-17          : the 17 engineered predictors (raw, as in Table 1)
  AEF-64            : the 64 raw AlphaEarth bands (as in Table 1)
  AEF-PCA17         : AlphaEarth reduced to 17 components, fold-internal PCA
  AEF-PCA17+Stack-17: PCA-reduced AlphaEarth (17) + engineered stack (17)
  AEF-64+Stack-17   : raw AlphaEarth (64) + engineered stack (17), as in Table 1
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats
from itertools import product

SEED = 42
N_BLOCKS = 10
RF_PARAMS = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
STACK = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_div_VH','VV_minus_VH',
         'aspect','elev','slope','sm_annual','t2m_summer','tp_winter']
BANDS = [f"A{i:02d}" for i in range(64)]
PCA_N = 17

df = pd.read_csv("SOC_master_aligned.csv")
df['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(df[['lon','lat']].values)
y = np.log1p(df['Lucas_OC'].values)
groups = df['blk'].values
gkf = GroupKFold(n_splits=N_BLOCKS)

X_aef = df[BANDS].values
X_stack = df[STACK].values

fold_rows = []
folds = list(gkf.split(X_aef, y, groups))  # fixed fold assignment, reused for every feature set

for k, (tr, te) in enumerate(folds):
    # ---- fold-internal scaler + PCA, fit on TRAIN ONLY ----
    scaler = StandardScaler().fit(X_aef[tr])
    aef_tr_scaled = scaler.transform(X_aef[tr])
    aef_te_scaled = scaler.transform(X_aef[te])
    pca = PCA(n_components=PCA_N, random_state=SEED).fit(aef_tr_scaled)
    aef_tr_pca = pca.transform(aef_tr_scaled)
    aef_te_pca = pca.transform(aef_te_scaled)
    evr = pca.explained_variance_ratio_.sum()

    feature_sets = {
        'Stack-17':           (X_stack[tr], X_stack[te]),
        'AEF-64':             (X_aef[tr], X_aef[te]),
        'AEF-PCA17':          (aef_tr_pca, aef_te_pca),
        'AEF-PCA17+Stack-17': (np.hstack([aef_tr_pca, X_stack[tr]]), np.hstack([aef_te_pca, X_stack[te]])),
        'AEF-64+Stack-17':    (np.hstack([X_aef[tr], X_stack[tr]]), np.hstack([X_aef[te], X_stack[te]])),
    }

    for name, (Xtr, Xte) in feature_sets.items():
        m = RandomForestRegressor(**RF_PARAMS).fit(Xtr, y[tr])
        p = m.predict(Xte)
        fold_rows.append({'feature_set': name, 'fold': k, 'R2': r2_score(y[te], p),
                           'n_test': len(te), 'pca_explained_var': round(evr, 4) if 'PCA' in name else np.nan})
    print(f"fold {k}: PCA(17) explained variance ratio = {evr:.4f}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("soc_pca_dim_fold_level.csv", index=False)

summary = fold_df.groupby('feature_set')['R2'].agg(['mean', 'std']).round(4)
order = ['Stack-17', 'AEF-64', 'AEF-PCA17', 'AEF-PCA17+Stack-17', 'AEF-64+Stack-17']
summary = summary.reindex(order)
summary.to_csv("soc_pca_dim_summary.csv")
print("\n=== Spatial-CV R2 by feature set (mean +/- SD over 10 folds) ===")
print(summary)

mean_evr = fold_df.loc[fold_df.pca_explained_var.notna(), 'pca_explained_var'].mean()
print(f"\nMean PCA(17) explained variance ratio across folds: {mean_evr:.4f}")

# ---- paired stats (requested comparisons) ----
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
    return {'comparison': f'{label_a} - {label_b}', 'mean_diff': round(mean_d, 4),
            'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
            'ci_excludes_zero': bool(lo > 0 or hi < 0), 'perm_p': round(p, 4)}

piv = fold_df.pivot(index='fold', columns='feature_set', values='R2').sort_index()
pairs = [('AEF-PCA17', 'Stack-17'), ('AEF-64', 'AEF-PCA17'), ('AEF-PCA17+Stack-17', 'AEF-PCA17')]
paired_rows = [paired_stats(piv[a].values, piv[b].values, a, b) for a, b in pairs]
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv("soc_pca_dim_paired.csv", index=False)
print("\n=== Paired comparisons ===")
print(paired_df.to_string(index=False))
