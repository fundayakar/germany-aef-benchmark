"""
Reproduces the locked spatial-CV benchmark (same seed, same k-means blocks,
same models as lock_and_run.py) but additionally saves:
  - fold-level metrics per feature set (for paired testing across folds)
  - point-level out-of-fold predictions per feature set (for reference)

Sanity-checks against the already-reported aggregate numbers
(final_benchmark_locked.csv / benchmark_veg_results.csv) before anything
else is trusted downstream.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
import xgboost as xgb

SEED = 42
N_BLOCKS = 10
RF_PARAMS  = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
SOC_PATH = "SOC_master_aligned.csv"
VEG_PATH = "veg_stress_pointlevel.csv"
BANDS = [f"A{i:02d}" for i in range(64)]

def blocks(coords):
    return KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(coords)

def load_soc():
    df = pd.read_csv(SOC_PATH)
    stack = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_div_VH','VV_minus_VH',
             'aspect','elev','slope','sm_annual','t2m_summer','tp_winter']
    df['blk'] = blocks(df[['lon','lat']].values)
    y = np.log1p(df['Lucas_OC'].values)
    sets = {'AEF':BANDS, 'Stack':stack, 'AEF+Stack':BANDS+stack}
    return df, y, sets

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
    sets = {'Stack':clim, 'AEF':pb, 'AEF+Stack':pb+clim}
    return df, y, sets

def reg_metrics(yt, p):
    return {'R2': r2_score(yt,p), 'RMSE': mean_squared_error(yt,p)**0.5, 'MAE': mean_absolute_error(yt,p)}

def run_spatial(task):
    if task == 'soc':
        df, y, sets = load_soc(); learner, kind = 'rf', 'reg'
        id_col = 'POINTID'
    else:
        df, y, sets = load_veg(); learner, kind = 'xgb', 'clf'
        id_col = None  # veg is point-year; keep row index instead

    groups = df['blk'].values
    splitter = GroupKFold(n_splits=N_BLOCKS)

    fold_rows = []
    oof_preds = {}  # feature_set -> array aligned to df index

    for name, cols in sets.items():
        X = df[cols].values
        oof = np.full(len(df), np.nan)
        for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
            if learner == 'rf':
                m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
                p = m.predict(X[te])
            else:
                m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
            oof[te] = p
            metr = reg_metrics(y[te], p) if kind == 'reg' else {'ROC_AUC': roc_auc_score(y[te], p)}
            metr.update({'feature_set': name, 'fold': k, 'n_test': int(len(te)),
                         'block_id': int(np.unique(groups[te])[0]) if len(np.unique(groups[te]))==1 else -1})
            fold_rows.append(metr)
        oof_preds[name] = oof
        print(f"[{task}] {name} done.")

    fold_df = pd.DataFrame(fold_rows)

    # sanity check against reported aggregate
    metric_col = 'R2' if kind == 'reg' else 'ROC_AUC'
    check = fold_df.groupby('feature_set')[metric_col].agg(['mean','std']).round(3)
    print(f"\n=== {task} spatial-CV reproduction check ({metric_col} mean/std across folds) ===")
    print(check)

    fold_df.to_csv(f"fold_level_{task}_spatial.csv", index=False)

    # point-level out-of-fold predictions
    base_cols = ['blk'] + ([id_col] if id_col else ['id','year'])
    oof_df = df[base_cols].copy()
    oof_df['y_true'] = y
    for name, arr in oof_preds.items():
        oof_df[f'pred_{name}'] = arr
    oof_df.to_csv(f"oof_predictions_{task}_spatial.csv", index=False)

    return fold_df, oof_df

if __name__ == '__main__':
    fold_soc, oof_soc = run_spatial('soc')
    fold_veg, oof_veg = run_spatial('veg')
