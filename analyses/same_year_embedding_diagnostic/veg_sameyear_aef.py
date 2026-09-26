"""
Same-year AlphaEarth diagnostic (circularity-prone upper bound), vegetation-stress task.

Identical to the locked Table 1 pipeline (same seed, same spatial blocks / random
folds, same XGBoost config, same stress-label definition and same ERA5-Land
antecedent Stack), except AEF now uses the SAME-YEAR AlphaEarth embedding
(A00..A63 as exported) instead of the prior-year shifted embedding (A00_p..A63_p)
used in the main analysis. Because annual AlphaEarth embeddings may encode
growing-season surface state that is itself used to derive the MODIS NDVI stress
label, this is explicitly an upper-bound / circularity-prone diagnostic, not an
alternative main design.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
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
CLIM = ['precip_winter','precip_spring','temp_spring','sm_winter','sm_spring']

def blocks(coords):
    return KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(coords)

def load_veg_sameyear():
    df = pd.read_csv(VEG_PATH)
    df = df.dropna(subset=['gs_ndvi']+CLIM).copy()
    g = df.groupby('id')['gs_ndvi']
    df['z'] = (df['gs_ndvi']-g.transform('mean'))/g.transform('std')
    df['stress'] = (df['z'] < -1).astype(int)
    # SAME-YEAR AEF: no shift. But to keep the sample identical to the main
    # analysis (same 1,976 locations x 7 years = 13,832), we still apply the
    # antecedent-embedding dropna mask from the prior-year version, then just
    # swap in the same-year bands as predictors. This isolates "which year's
    # embedding" as the only change -- same rows, same label, same Stack.
    df = df.sort_values(['id','year'])
    prev = df.groupby('id')[BANDS].shift(1); prev.columns = [b+'_p' for b in BANDS]
    df = pd.concat([df, prev], axis=1)
    pb = [b+'_p' for b in BANDS]
    df = df.dropna(subset=pb).reset_index(drop=True)  # same row mask as main analysis
    pts = df.drop_duplicates('id')[['id','lon','lat']].copy()
    pts['blk'] = blocks(pts[['lon','lat']].values)
    df = df.merge(pts[['id','blk']], on='id')
    y = df['stress'].values
    # same-year AEF bands (A00..A63, already present, unshifted) replace the
    # prior-year (A00_p..A63_p) bands as the embedding predictor set
    sets = {'AEF_sameyear': BANDS, 'Stack': CLIM, 'AEF_sameyear+Stack': BANDS + CLIM}
    return df, y, sets

df, y, sets = load_veg_sameyear()
print(f"n = {len(df)} (should match main analysis: 13832)")
groups = df['blk'].values

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
    return {'metric': metric_name, 'comparison': f'{label_a} - {label_b}', 'n_folds': n,
            'mean_diff': round(mean_d, 4), 'ci95_lo': round(ci_lo, 4), 'ci95_hi': round(ci_hi, 4),
            'ci_excludes_zero': bool(ci_lo > 0 or ci_hi < 0), 'perm_p': round(p_perm, 4)}

summary_rows = []
fold_store = {}  # scheme -> feature_set -> {'ROC_AUC':[...], 'PR_AUC':[...]}

for scheme in ['spatial', 'random']:
    splitter = (GroupKFold(n_splits=N_BLOCKS) if scheme == 'spatial'
                else KFold(n_splits=10, shuffle=True, random_state=SEED))
    fold_store[scheme] = {}
    for name, cols in sets.items():
        X = df[cols].values
        fold_rows = []
        it = (splitter.split(X, y, groups) if scheme == 'spatial' else splitter.split(X))
        for k, (tr, te) in enumerate(it):
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            fold_rows.append({'fold': k, 'ROC_AUC': roc_auc_score(y[te], p),
                               'PR_AUC': average_precision_score(y[te], p)})
        fd = pd.DataFrame(fold_rows)
        fold_store[scheme][name] = fd
        summary_rows.append({
            'scheme': scheme, 'feature_set': name,
            'ROC_AUC_mean': round(fd['ROC_AUC'].mean(),3), 'ROC_AUC_std': round(fd['ROC_AUC'].std(),3),
            'PR_AUC_mean': round(fd['PR_AUC'].mean(),3), 'PR_AUC_std': round(fd['PR_AUC'].std(),3),
        })
        print(f"[{scheme}] {name:20s} ROC_AUC={fd['ROC_AUC'].mean():.3f}+/-{fd['ROC_AUC'].std():.3f}  "
              f"PR_AUC={fd['PR_AUC'].mean():.3f}+/-{fd['PR_AUC'].std():.3f}")

summary_df = pd.DataFrame(summary_rows)
order = {'AEF_sameyear': 0, 'Stack': 1, 'AEF_sameyear+Stack': 2}
summary_df['_o'] = summary_df['feature_set'].map(order)
summary_df = summary_df.sort_values(['scheme','_o']).drop(columns='_o').reset_index(drop=True)
summary_df.to_csv("veg_sameyear_aef_summary.csv", index=False)

# paired stats, spatial scheme only (as requested, "at least for spatial")
pairs = [('AEF_sameyear+Stack', 'AEF_sameyear'), ('AEF_sameyear+Stack', 'Stack')]
paired_rows = []
for metric in ['ROC_AUC', 'PR_AUC']:
    fds = fold_store['spatial']
    piv = pd.DataFrame({name: fds[name].sort_values('fold')[metric].values for name in fds})
    for a, b in pairs:
        paired_rows.append(paired_stats(piv[a].values, piv[b].values, a, b, metric))
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv("veg_sameyear_aef_paired.csv", index=False)

print("\n=== Summary table ===")
print(summary_df.to_string(index=False))
print("\n=== Paired comparisons (spatial CV) ===")
print(paired_df.to_string(index=False))
