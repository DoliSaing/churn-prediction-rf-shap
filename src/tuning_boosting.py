"""Tuning setara (12 kombinasi, 3-fold, F1) untuk XGBoost, LightGBM, CatBoost + evaluasi."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parent.parent
from data_loader import ensure_dataset
ensure_dataset(ROOT)
df=pd.read_csv(ROOT/'data'/'Churn_Modelling.csv').drop(columns=['RowNumber','CustomerId','Surname'])
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)
X=df.drop(columns='Exited');y=df.Exited
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
spw=(ytr==0).sum()/(ytr==1).sum()
bp=json.load(open(ROOT/'results'/'model_comparison.json'))['best_rf_params']
cv3=StratifiedKFold(3,shuffle=True,random_state=42)
cfg={
'XGBoost_Tuned':(XGBClassifier(scale_pos_weight=spw,random_state=42,eval_metric='logloss',n_jobs=1),
  {'n_estimators':[200,300,400],'max_depth':[3,4,6,8],'learning_rate':[0.01,0.05,0.1],'subsample':[0.7,0.85,1.0],'colsample_bytree':[0.7,0.85,1.0]}),
'LightGBM_Tuned':(LGBMClassifier(class_weight='balanced',random_state=42,verbose=-1,n_jobs=1),
  {'n_estimators':[200,300,400],'max_depth':[3,4,6,8],'learning_rate':[0.01,0.05,0.1],'num_leaves':[15,31,63],'subsample':[0.7,0.85,1.0],'subsample_freq':[1]}),
'CatBoost_Tuned':(CatBoostClassifier(auto_class_weights='Balanced',random_state=42,verbose=0,thread_count=1),
  {'iterations':[200,300,400],'depth':[4,6,8],'learning_rate':[0.03,0.05,0.1],'l2_leaf_reg':[1,3,5,9]})}
out={'best_params':{},'test':{},'ci':{},'cv':{},'paired':{}}
proba={}
rf=RandomForestClassifier(**bp,class_weight='balanced',random_state=42,n_jobs=1).fit(Xtr,ytr); proba['RF_Tuned']=rf.predict_proba(Xte)[:,1]
def met(yt,p):
    pr=(p>=.5).astype(int)
    return dict(accuracy=accuracy_score(yt,pr),precision=precision_score(yt,pr),recall=recall_score(yt,pr),f1=f1_score(yt,pr),roc_auc=roc_auc_score(yt,p),pr_auc=average_precision_score(yt,p))
best={}
for n,(est,grid) in cfg.items():
    s=RandomizedSearchCV(est,grid,n_iter=12,scoring='f1',cv=cv3,random_state=42,n_jobs=1).fit(Xtr,ytr)
    best[n]=s.best_estimator_; out['best_params'][n]={k:(v.item() if hasattr(v,'item') else v) for k,v in s.best_params_.items()}
    proba[n]=s.best_estimator_.predict_proba(Xte)[:,1]; out['test'][n]=met(yte.values,proba[n]); print(n,out['best_params'][n],{k:round(v,4) for k,v in out['test'][n].items()},flush=True)
rng=np.random.default_rng(42); yt=yte.values; B={n:{'f1':[],'roc_auc':[]} for n in proba}
for _ in range(1000):
    ii=rng.integers(0,len(yt),len(yt))
    for n,p in proba.items():
        B[n]['f1'].append(f1_score(yt[ii],p[ii]>=.5)); B[n]['roc_auc'].append(roc_auc_score(yt[ii],p[ii]))
for n in proba: out['ci'][n]={k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in B[n].items()}
for n in cfg:
    for k in ['f1','roc_auc']:
        d=np.array(B[n][k])-np.array(B['RF_Tuned'][k]); out['paired'][f'{n}-RF_Tuned|{k}']=[float(d.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
for n,m in best.items():
    r=cross_validate(m,Xtr,ytr,cv=rcv,scoring={'f1':'f1','roc_auc':'roc_auc'})
    out['cv'][n]={k:[float(r['test_'+k].mean()),float(r['test_'+k].std())] for k in ['f1','roc_auc']}; print(n,out['cv'][n],flush=True)
json.dump(out,open(ROOT/'results'/'tuning_boosting.json','w'),indent=1,default=float); print('DONE')
