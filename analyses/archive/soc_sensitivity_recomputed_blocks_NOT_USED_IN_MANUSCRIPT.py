"""
SOC sensitivity analysis: does the AEF > Stack, AEF+Stack ~ AEF pattern
hold after excluding organic-rich soils / high-OC statistical outliers?

Two exclusion scenarios, both applied on top of the exact locked config
(same RF params, same 10-fold spatial CV design; k-means blocks are
recomputed on each trimmed point set, following the same "compute once,
reuse across feature sets" convention used for the full sample):

  A. Organic-rich exclusion: Lucas_OC > 120 g/kg (a conventional proxy
     threshold separating organic/histic from mineral soils), n=30 removed.
  B. Statistical-outlier exclusion: Tukey's rule (Q3 + 1.5*IQR) applied on
     the modelling scale, log1p(Lucas_OC) > 5.136 i.e. Lucas_OC > 169.1
     g/kg, n=20 removed.
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
print("Full sample n:", len(full))

scenarios = {
    'Full sample': full,
    'Excl. organic-rich (OC>120 g/kg)': full[full['Lucas_OC'] <= 120].reset_index(drop=True),
    'Excl. statistical outliers (Tukey, OC>169.1 g/kg)': full[full['Lucas_OC'] <= 169.1].reset_index(drop=True),
}

def run_scenario(df):
    df = df.copy()
    df['blk'] = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(df[['lon','lat']].values)
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
                               'RMSE': mean_squared_error(y[te], p)**0.5,
                               'MAE': mean_absolute_error(y[te], p),
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
    summ['scenario'] = scen_name
    summ['n'] = n
    all_summary.append(summ.reset_index())

    piv = fold_df.pivot(index='fold', columns='feature_set', values='R2').sort_index()
    for a, b in [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]:
        row = paired_stats(piv[a].values, piv[b].values, a, b)
        row['scenario'] = scen_name
        all_paired.append(row)

    print(f"\n=== {scen_name} (n={n}) ===")
    print(summ[['mean', 'std']].to_string())

summary_df = pd.concat(all_summary, ignore_index=True)
paired_df = pd.DataFrame(all_paired)
summary_df.to_csv("soc_sensitivity_summary.csv", index=False)
paired_df.to_csv("soc_sensitivity_paired.csv", index=False)

print("\n\n=== Paired comparisons across scenarios ===")
print(paired_df[['scenario','comparison','mean_diff','ci95_lo','ci95_hi','ci_excludes_zero','perm_p']].to_string(index=False))
