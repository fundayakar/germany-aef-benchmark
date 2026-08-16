"""
Additional paired fold-level comparisons computed from the fold-level output
of veg_pca_dim_sensitivity.py (veg_pca_dim_fold_level.csv), without re-fitting
any models:

  AEF-PCA5+Stack-5 vs Stack-5 : does the dimension-matched AEF representation
                                 add anything beyond the climate stack alone?
  AEF-64 vs AEF-PCA5          : quantifies the compression loss from reducing
                                 AEF to 5 fold-internal PCA components.

Reads veg_pca_dim_fold_level.csv (must be run first); same paired-bootstrap /
exact sign-flip permutation methodology used throughout (seed 42, 10,000
resamples, 2^10 = 1024 sign assignments for the exact test).
"""
import pandas as pd, numpy as np
from itertools import product

fold_df = pd.read_csv("veg_pca_dim_fold_level.csv")

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
    return {'metric': metric, 'comparison': f'{label_a} - {label_b}', 'n_folds': n,
            'mean_diff': round(mean_d, 4), 'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
            'ci_excludes_zero': bool(lo > 0 or hi < 0), 'perm_p': round(p, 4)}

pairs = [('AEF-PCA5+Stack-5', 'Stack-5'), ('AEF-64', 'AEF-PCA5')]
rows = []
for metric in ['ROC_AUC', 'PR_AUC']:
    piv = fold_df.pivot(index='fold', columns='feature_set', values=metric).sort_index()
    for a, b in pairs:
        rows.append(paired_stats(piv[a].values, piv[b].values, a, b, metric))

out = pd.DataFrame(rows)
out.to_csv("veg_pca_dim_paired_extra.csv", index=False)
print(out.to_string(index=False))
