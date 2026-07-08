"""
Cleaner design per review: use the SAME spatial block assignment computed
once on the full 772-point sample, then simply drop high-OC points from
those fixed folds (rather than recomputing k-means blocks on each trimmed
sample, which would confound "removing points" with "changing the fold
structure").
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from itertools import product

SEED = 42
N_BLOCKS = 10
RF_PARAMS = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
STACK = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_div_VH','VV_minus_VH',
         'aspect','elev','slope','sm_annual','t2m_summer','tp_winter']
BANDS = [f"A{i:02d}" for i in range(64)]

full = pd.read_csv("SOC_master_aligned.csv")
# blocks computed ONCE on the full sample -- identical seed/procedure to the main analysis
full['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(full[['lon','lat']].values)
print("Full sample n:", len(full))
print("Points per block (full sample):")
print(full['blk'].value_counts().sort_index())

scenarios = {
    'Excl. high-OC (OC>120 g/kg), fixed blocks': full[full['Lucas_OC'] <= 120].reset_index(drop=True),
    'Excl. statistical outliers (Tukey, OC>169.1 g/kg), fixed blocks': full[full['Lucas_OC'] <= 169.1].reset_index(drop=True),
}

def run_scenario(df):
    print("  Points per block (this scenario):")
    print("  " + df['blk'].value_counts().sort_index().to_string().replace("\n", "\n  "))
    y = np.log1p(df['Lucas_OC'].values)
    sets = {'AEF': BANDS, 'Stack': STACK, 'AEF+Stack': BANDS + STACK}
    gkf = GroupKFold(n_splits=N_BLOCKS)
    groups = df['blk'].values
    fold_rows = []
    for name, cols in sets.items():
        X = df[cols].values
        for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
            m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
            p = m.predict(X[te])
            fold_rows.append({'feature_set': name, 'fold': k,
                               'R2': r2_score(y[te], p),
                               'n_test': len(te)})
    return pd.DataFrame(fold_rows), len(df)

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

all_summary = []
all_paired = []
for scen_name, scen_df in scenarios.items():
    fold_df, n = run_scenario(scen_df)
    summ = fold_df.groupby('feature_set')['R2'].agg(['mean', 'std']).round(3)
    print(f"\n=== {scen_name} (n={n}) ===")
    print(summ.to_string())
    summ2 = summ.reset_index(); summ2['scenario'] = scen_name; summ2['n'] = n
    all_summary.append(summ2)

    piv = fold_df.pivot(index='fold', columns='feature_set', values='R2').sort_index()
    for a, b in [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]:
        row = paired_stats(piv[a].values, piv[b].values, a, b)
        row['scenario'] = scen_name
        all_paired.append(row)

summary_df = pd.concat(all_summary, ignore_index=True)
paired_df = pd.DataFrame(all_paired)
summary_df.to_csv("soc_sensitivity_fixedblocks_summary.csv", index=False)
paired_df.to_csv("soc_sensitivity_fixedblocks_paired.csv", index=False)

print("\n\n=== Paired comparisons (fixed-block version) ===")
print(paired_df[['scenario','comparison','mean_diff','ci95_lo','ci95_hi','ci_excludes_zero','perm_p']].to_string(index=False))
print("\nDONE")
