"""
Locked benchmark: one fixed config per learner; regenerates all reported numbers.
Usage: python lock_and_run.py {soc_spatial|soc_random|veg_spatial|veg_random}
Appends results to /mnt/user-data/outputs/final_benchmark_locked.csv
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
import xgboost as xgb

# ---- LOCKED CONFIG ----
SEED = 42
N_BLOCKS = 10
RF_PARAMS  = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                  n_jobs=-1, eval_metric="logloss")
SOC_PATH = "/mnt/user-data/outputs/SOC_master_aligned.csv"
VEG_PATH = "/mnt/user-data/uploads/veg_stress_pointlevel.csv"
OUT = "/mnt/user-data/outputs/final_benchmark_locked.csv"
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

def run(task, scheme):
    if task == 'soc':
        df, y, sets = load_soc(); learner, kind = 'rf', 'reg'
    else:
        df, y, sets = load_veg(); learner, kind = 'xgb', 'clf'
    groups = df['blk'].values
    splitter = (GroupKFold(n_splits=N_BLOCKS) if scheme=='spatial'
                else KFold(n_splits=10, shuffle=True, random_state=SEED))
    rows = []
    for name, cols in sets.items():
        X = df[cols].values; fold = []
        it = (splitter.split(X,y,groups) if scheme=='spatial' else splitter.split(X))
        for tr, te in it:
            if learner=='rf':
                m = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr]); p = m.predict(X[te])
            else:
                m = xgb.XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:,1]
            fold.append(reg_metrics(y[te],p) if kind=='reg' else {'ROC_AUC': roc_auc_score(y[te],p)})
        fd = pd.DataFrame(fold)
        rec = {'task':task,'scheme':scheme,'feature_set':name,'n':len(df)}
        for c in fd.columns:
            rec[c+'_mean']=round(fd[c].mean(),3); rec[c+'_std']=round(fd[c].std(),3)
        rows.append(rec)
        print(f"[{task}/{scheme}] {name:10s} " + " ".join(f"{c}={fd[c].mean():.3f}±{fd[c].std():.3f}" for c in fd.columns))
    out = pd.DataFrame(rows)
    if os.path.exists(OUT):
        out = pd.concat([pd.read_csv(OUT), out], ignore_index=True)
    out.to_csv(OUT, index=False)

if __name__ == '__main__':
    task, scheme = sys.argv[1].split('_')
    run(task, scheme)
