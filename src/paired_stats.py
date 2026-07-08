import numpy as np, pandas as pd
from scipy import stats
from itertools import product

RNG = np.random.default_rng(42)
N_BOOT = 10000

def paired_stats(vals_a, vals_b, label_a, label_b, metric_name):
    """vals_a, vals_b: arrays of length n_folds, same fold order (paired)."""
    d = vals_a - vals_b  # positive = a beats b
    n = len(d)
    mean_d = d.mean()

    # 1. Nonparametric bootstrap CI (resample folds with replacement)
    boot_means = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(N_BOOT)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # 2. Exact sign-flip permutation test (2^n sign combinations, n=10 -> 1024)
    signs = np.array(list(product([1, -1], repeat=n)))
    perm_means = (signs * d).mean(axis=1)
    p_perm = np.mean(np.abs(perm_means) >= np.abs(mean_d) - 1e-12)

    # 3. Paired t-test (parametric reference)
    t_stat, p_t = stats.ttest_rel(vals_a, vals_b)

    # 4. Wilcoxon signed-rank (nonparametric reference; small n so treat cautiously)
    try:
        w_stat, p_w = stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    return {
        'comparison': f'{label_a} - {label_b}',
        'metric': metric_name,
        'n_folds': n,
        'mean_diff': round(mean_d, 4),
        'ci95_lo': round(ci_lo, 4),
        'ci95_hi': round(ci_hi, 4),
        'ci_excludes_zero': bool(ci_lo > 0 or ci_hi < 0),
        'perm_p': round(p_perm, 4),
        'paired_t_p': round(p_t, 4),
        'wilcoxon_p': round(p_w, 4) if not np.isnan(p_w) else np.nan,
    }

def analyze(task, metric_col):
    fold = pd.read_csv(f"fold_level_{task}_spatial.csv")
    piv = fold.pivot(index='fold', columns='feature_set', values=metric_col).sort_index()
    pairs = [('AEF', 'Stack'), ('AEF+Stack', 'AEF'), ('AEF+Stack', 'Stack')]
    rows = []
    for a, b in pairs:
        rows.append(paired_stats(piv[a].values, piv[b].values, a, b, metric_col))
    out = pd.DataFrame(rows)
    out.insert(0, 'task', task)
    return out

soc_stats = analyze('soc', 'R2')
veg_stats = analyze('veg', 'ROC_AUC')
all_stats = pd.concat([soc_stats, veg_stats], ignore_index=True)
all_stats.to_csv("paired_bootstrap_results.csv", index=False)

pd.set_option('display.width', 140)
print(all_stats.to_string(index=False))
