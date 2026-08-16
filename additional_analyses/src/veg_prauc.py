"""
PR-AUC (average precision) for the vegetation-stress task, using the EXACT
same locked pipeline (seed, blocks, model config) as Table 1 / lock_and_run.py,
for both the spatial (GroupKFold on k-means blocks) and random (shuffled KFold)
10-fold schemes. Reports mean +/- SD per feature set, in AEF/Stack/AEF+Stack
order, plus the no-skill PR-AUC baseline (= overall stress prevalence).
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
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
    sets = {'AEF': pb, 'Stack': clim, 'AEF+Stack': pb + clim}  # reporting order
    return df, y, sets

df, y, sets = load_veg()
overall_prevalence = y.mean()
print(f"n = {len(df)}, overall stress prevalence (no-skill PR-AUC baseline) = {overall_prevalence:.4f}")

groups = df['blk'].values
rows = []

for scheme in ['spatial', 'random']:
    splitter = (GroupKFold(n_splits=N_BLOCKS) if scheme == 'spatial'
                else KFold(n_splits=10, shuffle=True, random_state=SEED))
    for name, cols in sets.items():
        X = df[cols].values
        fold_rows = []
        it = (splitter.split(X, y, groups) if scheme == 'spatial' else splitter.split(X))
        for k, (tr, te) in enumerate(it):
            m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            fold_rows.append({
                'fold': k,
                'ROC_AUC': roc_auc_score(y[te], p),
                'PR_AUC': average_precision_score(y[te], p),
                'n_test': len(te),
                'prevalence_test': y[te].mean(),
            })
        fd = pd.DataFrame(fold_rows)
        rec = {
            'scheme': scheme, 'feature_set': name, 'n': len(df),
            'ROC_AUC_mean': round(fd['ROC_AUC'].mean(), 3), 'ROC_AUC_std': round(fd['ROC_AUC'].std(), 3),
            'PR_AUC_mean': round(fd['PR_AUC'].mean(), 3), 'PR_AUC_std': round(fd['PR_AUC'].std(), 3),
        }
        rows.append(rec)
        print(f"[{scheme}] {name:10s} ROC_AUC={rec['ROC_AUC_mean']:.3f}+/-{rec['ROC_AUC_std']:.3f}  "
              f"PR_AUC={rec['PR_AUC_mean']:.3f}+/-{rec['PR_AUC_std']:.3f}")

out = pd.DataFrame(rows)
# enforce AEF, Stack, AEF+Stack row order within each scheme
order = {'AEF': 0, 'Stack': 1, 'AEF+Stack': 2}
out['_ord'] = out['feature_set'].map(order)
out = out.sort_values(['scheme', '_ord']).drop(columns='_ord').reset_index(drop=True)
out['no_skill_PR_AUC_baseline'] = round(overall_prevalence, 4)

out.to_csv("veg_prauc_table1_folds.csv", index=False)
print("\n=== Final table (AEF, Stack, AEF+Stack order) ===")
print(out.to_string(index=False))
print(f"\nNo-skill PR-AUC baseline (overall prevalence) = {overall_prevalence:.4f}")
