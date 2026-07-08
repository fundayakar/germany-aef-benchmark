import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
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

raw = pd.read_csv(VEG_PATH)
raw = raw.dropna(subset=['gs_ndvi']+CLIM).copy()
g = raw.groupby('id')['gs_ndvi']
raw['ndvi_mean'] = g.transform('mean'); raw['ndvi_std'] = g.transform('std')
raw = raw.sort_values(['id','year'])
prev = raw.groupby('id')[BANDS].shift(1); prev.columns = [b+'_p' for b in BANDS]
raw = pd.concat([raw, prev], axis=1)
pb = [b+'_p' for b in BANDS]
raw = raw.dropna(subset=pb).reset_index(drop=True)

pts = raw.drop_duplicates('id')[['id','lon','lat']].copy()
pts['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(pts[['lon','lat']].values)
raw = raw.merge(pts[['id','blk']], on='id')

RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b):
    d = vals_a - vals_b
    n = len(d)
    mean_d = d.mean()
    boot_means = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    signs = np.array(list(product([1, -1], repeat=n)))
    perm_means = (signs * d).mean(axis=1)
    p_perm = np.mean(np.abs(perm_means) >= np.abs(mean_d) - 1e-12)
    return {'comparison': f'{label_a} - {label_b}', 'mean_diff': round(mean_d, 4),
            'ci95_lo': round(ci_lo, 4), 'ci95_hi': round(ci_hi, 4),
            'ci_excludes_zero': bool(ci_lo > 0 or ci_hi < 0), 'perm_p': round(p_perm, 4)}

thresholds = [-0.5, -1.5]  # -1.0 already reported as the main result (Table 1)
sets = {'Stack': CLIM, 'AEF': pb, 'AEF+Stack': pb + CLIM}

all_summary, all_paired, prevalence_rows = [], [], []

for thr in thresholds:
    df = raw.copy()
    df['z'] = (df['gs_ndvi'] - df['ndvi_mean']) / df['ndvi_std']
    df['stress'] = (df['z'] < thr).astype(int)
    prevalence = df['stress'].mean()
    prevalence_rows.append({'threshold': thr, 'n': len(df), 'prevalence': round(prevalence, 4),
                             'n_positive': int(df['stress'].sum())})
    print(f"\nThreshold {thr} SD: prevalence = {prevalence:.4f} ({int(df['stress'].sum())} / {len(df)})")

    y = df['stress'].values
    groups = df['blk'].values
    gkf = GroupKFold(n_splits=N_BLOCKS)
    fold_rows = []
    for name, cols in sets.items():
        X = df[cols].values
        for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan
            fold_rows.append({'feature_set': name, 'fold': k, 'ROC_AUC': auc, 'n_test': len(te)})
        print(f"  [{name}] done.")

    fold_df = pd.DataFrame(fold_rows)
    summ = fold_df.groupby('feature_set')['ROC_AUC'].agg(['mean', 'std']).round(3)
    print(f"  Summary (threshold={thr}):")
    print("  " + summ.to_string().replace("\n", "\n  "))
    summ2 = summ.reset_index(); summ2['threshold'] = thr
    all_summary.append(summ2)

    piv = fold_df.pivot(index='fold', columns='feature_set', values='ROC_AUC').sort_index()
    for a, b in [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]:
        row = paired_stats(piv[a].values, piv[b].values, a, b)
        row['threshold'] = thr
        all_paired.append(row)

summary_df = pd.concat(all_summary, ignore_index=True)
paired_df = pd.DataFrame(all_paired)
prev_df = pd.DataFrame(prevalence_rows)
summary_df.to_csv("veg_threshold_sensitivity_summary.csv", index=False)
paired_df.to_csv("veg_threshold_sensitivity_paired.csv", index=False)
prev_df.to_csv("veg_threshold_sensitivity_prevalence.csv", index=False)

print("\n\n=== Threshold sensitivity: paired comparisons ===")
print(paired_df[['threshold','comparison','mean_diff','ci95_lo','ci95_hi','ci_excludes_zero','perm_p']].to_string(index=False))
print("\nDONE")
