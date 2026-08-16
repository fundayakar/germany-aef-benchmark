import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats
from itertools import product

SEED = 42
RF_PARAMS = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
STACK = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_div_VH','VV_minus_VH',
         'aspect','elev','slope','sm_annual','t2m_summer','tp_winter']
BANDS = [f"A{i:02d}" for i in range(64)]
K_VALUES = [5, 10, 15]

df = pd.read_csv("SOC_master_aligned.csv")
y = np.log1p(df['Lucas_OC'].values)
sets = {'AEF': BANDS, 'Stack': STACK, 'AEF+Stack': BANDS + STACK}

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
    blk = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(df[['lon', 'lat']].values)
    groups = blk
    gkf = GroupKFold(n_splits=k)
    fold_rows = []
    for name, cols in sets.items():
        X = df[cols].values
        for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
            m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
            p = m.predict(X[te])
            fold_rows.append({'k': k, 'feature_set': name, 'fold': fold_i,
                               'R2': r2_score(y[te], p), 'n_test': len(te)})
    fold_df = pd.DataFrame(fold_rows)
    all_fold_rows.append(fold_df)

    summ = fold_df.groupby('feature_set')['R2'].agg(['mean', 'std']).round(4)
    summ['k'] = k
    all_summary.append(summ.reset_index())
    print(f"\n=== SOC k={k} ===")
    print(summ)

    piv = fold_df.pivot(index='fold', columns='feature_set', values='R2').sort_index()
    for a, b in [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]:
        row = paired_stats(piv[a].values, piv[b].values, a, b)
        row.update({'k': k, 'comparison': f'{a} - {b}'})
        all_paired.append(row)

fold_all = pd.concat(all_fold_rows, ignore_index=True)
summary_all = pd.concat(all_summary, ignore_index=True)
paired_all = pd.DataFrame(all_paired)

fold_all.to_csv("soc_block_k_sensitivity_folds.csv", index=False)
summary_all.to_csv("soc_block_k_sensitivity_summary.csv", index=False)
paired_all.to_csv("soc_block_k_sensitivity_paired.csv", index=False)

print("\n\n=== SOC summary (all k) ===")
print(summary_all.to_string(index=False))
print("\n=== SOC paired (all k) ===")
print(paired_all[['k','comparison','mean_diff','ci95_lo','ci95_hi','ci_excludes_zero','perm_p']].to_string(index=False))
print("\nDONE")
