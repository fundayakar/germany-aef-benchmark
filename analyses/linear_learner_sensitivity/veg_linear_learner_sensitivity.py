"""Regularized linear-learner sensitivity for the vegetation-stress spatial-CV benchmark."""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"/"veg_stress_pointlevel.csv"
OUT=Path(__file__).resolve().parent
SEED=42; N_BLOCKS=10; N_BOOT=10000
C_VALUES=[0.01,0.1,1.0,10.0,100.0]
BANDS=[f"A{i:02d}" for i in range(64)]
CLIM=["precip_winter","precip_spring","temp_spring","sm_winter","sm_spring"]

def paired(a,b,la,lb,metric,rng):
    d=np.asarray(a)-np.asarray(b); n=len(d); m=d.mean()
    boots=np.array([rng.choice(d,size=n,replace=True).mean() for _ in range(N_BOOT)])
    lo,hi=np.percentile(boots,[2.5,97.5])
    signs=np.array(list(product([1,-1],repeat=n))); pm=(signs*d).mean(axis=1)
    p=np.mean(np.abs(pm)>=np.abs(m)-1e-12)
    return dict(metric=metric,comparison=f"{la} - {lb}",n_folds=n,mean_diff=m,ci95_lo=lo,ci95_hi=hi,ci_excludes_zero=bool(lo>0 or hi<0),perm_p=p)

df=pd.read_csv(DATA).dropna(subset=["gs_ndvi"]+CLIM).copy()
g=df.groupby("id")["gs_ndvi"]; df["z"]=(df.gs_ndvi-g.transform("mean"))/g.transform("std"); df["stress"]=(df.z<-1).astype(int)
df=df.sort_values(["id","year"]); prev=df.groupby("id")[BANDS].shift(1); prev.columns=[b+"_p" for b in BANDS]; df=pd.concat([df,prev],axis=1)
pb=[b+"_p" for b in BANDS]; df=df.dropna(subset=pb).reset_index(drop=True)
pts=df.drop_duplicates("id")[["id","lon","lat"]].copy(); pts["blk"]=KMeans(n_clusters=N_BLOCKS,random_state=SEED,n_init=10).fit_predict(pts[["lon","lat"]]); df=df.merge(pts[["id","blk"]],on="id")
y=df.stress.to_numpy(int); groups=df.blk.to_numpy(); sets={"AEF":pb,"Stack":CLIM,"AEF+Stack":pb+CLIM}
folds=list(GroupKFold(n_splits=N_BLOCKS).split(np.zeros(len(df)),y,groups)); rows=[]
for C in C_VALUES:
    for name,cols in sets.items():
        X=df[cols].to_numpy(float)
        for fold,(tr,te) in enumerate(folds):
            sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
            prob=LogisticRegression(penalty="l2",C=C,solver="newton-cholesky",max_iter=500).fit(Xtr,y[tr]).predict_proba(Xte)[:,1]
            rows.append(dict(C=C,feature_set=name,fold=fold,ROC_AUC=roc_auc_score(y[te],prob),PR_AUC=average_precision_score(y[te],prob),n_test=len(te),prevalence_test=y[te].mean()))
fold=pd.DataFrame(rows); fold.to_csv(OUT/"veg_linear_fold_level.csv",index=False)
summ=fold.groupby(["C","feature_set"])[["ROC_AUC","PR_AUC"]].agg(["mean","std"]).reset_index(); summ.columns=["_".join(c).strip("_") if isinstance(c,tuple) else c for c in summ.columns]; summ.to_csv(OUT/"veg_linear_summary.csv",index=False)
rank=[]
for C in C_VALUES:
    sub=fold[fold.C==C]
    for metric in ["ROC_AUC","PR_AUC"]:
        s=sub.groupby("feature_set")[metric].mean().sort_values(ascending=False)
        rank += [dict(C=C,metric=metric,rank=i,feature_set=n,mean_score=v) for i,(n,v) in enumerate(s.items(),1)]
pd.DataFrame(rank).to_csv(OUT/"veg_linear_ranking.csv",index=False)
rng=np.random.default_rng(SEED); out=[]
for C in C_VALUES:
    sub=fold[fold.C==C]
    for metric in ["ROC_AUC","PR_AUC"]:
        p=sub.pivot(index="fold",columns="feature_set",values=metric)
        for a,b in [("AEF","Stack"),("AEF+Stack","AEF"),("AEF+Stack","Stack")]:
            r=paired(p[a],p[b],a,b,metric,rng); r["C"]=C; out.append(r)
pd.DataFrame(out).to_csv(OUT/"veg_linear_paired.csv",index=False)
