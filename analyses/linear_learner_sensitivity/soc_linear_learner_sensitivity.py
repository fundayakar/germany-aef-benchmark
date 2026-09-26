"""Regularized linear-learner sensitivity for the SOC spatial-CV benchmark."""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"/"SOC_master_aligned.csv"
OUT=Path(__file__).resolve().parent
SEED=42; N_BLOCKS=10; N_BOOT=10000
ALPHAS=[0.01,0.1,1.0,10.0,100.0]
BANDS=[f"A{i:02d}" for i in range(64)]
STACK=["B2","B3","B4","B8","B11","B12","NDVI","VV","VH","VV_div_VH","VV_minus_VH","aspect","elev","slope","sm_annual","t2m_summer","tp_winter"]

def paired(a,b,la,lb,rng):
    d=np.asarray(a)-np.asarray(b); n=len(d); m=d.mean()
    boots=np.array([rng.choice(d,size=n,replace=True).mean() for _ in range(N_BOOT)])
    lo,hi=np.percentile(boots,[2.5,97.5])
    signs=np.array(list(product([1,-1],repeat=n)))
    pm=(signs*d).mean(axis=1)
    p=np.mean(np.abs(pm)>=np.abs(m)-1e-12)
    return dict(comparison=f"{la} - {lb}",n_folds=n,mean_diff=m,ci95_lo=lo,ci95_hi=hi,ci_excludes_zero=bool(lo>0 or hi<0),perm_p=p)

df=pd.read_csv(DATA)
df["blk"]=KMeans(n_clusters=N_BLOCKS,random_state=SEED,n_init=10).fit_predict(df[["lon","lat"]])
y=np.log1p(df["Lucas_OC"].to_numpy(float)); groups=df["blk"].to_numpy()
sets={"AEF":BANDS,"Stack":STACK,"AEF+Stack":BANDS+STACK}
folds=list(GroupKFold(n_splits=N_BLOCKS).split(np.zeros(len(df)),y,groups))
rows=[]
for alpha in ALPHAS:
    for name,cols in sets.items():
        X=df[cols].to_numpy(float)
        for fold,(tr,te) in enumerate(folds):
            sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
            pred=Ridge(alpha=alpha).fit(Xtr,y[tr]).predict(Xte)
            rows.append(dict(alpha=alpha,feature_set=name,fold=fold,R2=r2_score(y[te],pred),RMSE=mean_squared_error(y[te],pred)**0.5,MAE=mean_absolute_error(y[te],pred),n_test=len(te)))
fold=pd.DataFrame(rows); fold.to_csv(OUT/"soc_linear_fold_level.csv",index=False)
summ=fold.groupby(["alpha","feature_set"])[["R2","RMSE","MAE"]].agg(["mean","std"]).reset_index()
summ.columns=["_".join(c).strip("_") if isinstance(c,tuple) else c for c in summ.columns]
summ.to_csv(OUT/"soc_linear_summary.csv",index=False)
rank=[]
for alpha in ALPHAS:
    s=fold[fold.alpha==alpha].groupby("feature_set").R2.mean().sort_values(ascending=False)
    rank += [dict(alpha=alpha,rank_by_mean_R2=i,feature_set=n,mean_R2=v) for i,(n,v) in enumerate(s.items(),1)]
pd.DataFrame(rank).to_csv(OUT/"soc_linear_ranking.csv",index=False)
rng=np.random.default_rng(SEED); out=[]
for alpha in ALPHAS:
    p=fold[fold.alpha==alpha].pivot(index="fold",columns="feature_set",values="R2")
    for a,b in [("AEF","Stack"),("AEF+Stack","AEF"),("AEF+Stack","Stack")]:
        r=paired(p[a],p[b],a,b,rng); r.update(alpha=alpha,metric="R2"); out.append(r)
pd.DataFrame(out).to_csv(OUT/"soc_linear_paired.csv",index=False)
