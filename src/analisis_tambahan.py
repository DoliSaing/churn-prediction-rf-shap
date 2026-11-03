"""Analisis tambahan revisi (CI bootstrap, CV berulang, baseline, ambang, subkelompok, kurva PR).
Jalankan: python src/analisis_tambahan.py  (setelah churn_prediction_pipeline.py)"""
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IPipe
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
from data_loader import ensure_dataset
ensure_dataset(ROOT)
(ROOT/'results').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
NJ=1
df=pd.read_csv(ROOT/'data'/'Churn_Modelling.csv').drop(columns=['RowNumber','CustomerId','Surname'])
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)
X=df.drop(columns='Exited');y=df.Exited
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
bp=json.load(open(ROOT/'results'/'model_comparison.json'))['best_rf_params']
spw=(ytr==0).sum()/(ytr==1).sum()
def mk():
    return {
 'RF_Baseline':lambda:RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=NJ),
 'RF_Tuned':lambda:RandomForestClassifier(**bp,class_weight='balanced',random_state=42,n_jobs=NJ),
 'RF_SMOTE_Tuned':lambda:RandomForestClassifier(**bp,random_state=42,n_jobs=NJ),
 'XGBoost':lambda:XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,scale_pos_weight=spw,random_state=42,eval_metric='logloss',n_jobs=NJ),
 'LightGBM':lambda:LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,class_weight='balanced',random_state=42,verbose=-1,n_jobs=NJ),
 'CatBoost':lambda:CatBoostClassifier(iterations=300,depth=6,learning_rate=0.05,auto_class_weights='Balanced',random_state=42,verbose=0,thread_count=NJ),
 'LogReg':lambda:make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42)),
 'DecisionTree':lambda:DecisionTreeClassifier(max_depth=5,min_samples_leaf=20,class_weight='balanced',random_state=42)}
out={}
M=mk(); fitted={}; proba={}
for n,f in M.items():
    m=f()
    if n=='RF_SMOTE_Tuned':
        Xs,ys=SMOTE(random_state=42).fit_resample(Xtr,ytr); m.fit(Xs,ys)
    else: m.fit(Xtr,ytr)
    fitted[n]=m; proba[n]=m.predict_proba(Xte)[:,1]
def met(yt,p,t=0.5):
    pr=(p>=t).astype(int)
    return dict(accuracy=accuracy_score(yt,pr),precision=precision_score(yt,pr,zero_division=0),recall=recall_score(yt,pr),f1=f1_score(yt,pr),roc_auc=roc_auc_score(yt,p),pr_auc=average_precision_score(yt,p))
out['test']={n:met(yte,p) for n,p in proba.items()}
print('TEST',pd.DataFrame(out['test']).T.round(4),flush=True)
# bootstrap
rng=np.random.default_rng(42); yt=yte.values; B=1000
idx=[rng.integers(0,len(yt),len(yt)) for _ in range(B)]
boot={n:{k:[] for k in ['accuracy','precision','recall','f1','roc_auc','pr_auc']} for n in proba}
for ii in idx:
    for n,p in proba.items():
        if yt[ii].sum()==0: continue
        mm=met(yt[ii],p[ii])
        for k,v in mm.items(): boot[n][k].append(v)
out['boot_ci']={n:{k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in d.items()} for n,d in boot.items()}
# paired differences
def pd_diff(a,b,k):
    d=np.array(boot[a][k])-np.array(boot[b][k]); return [float(d.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
out['paired']={f'{a}-{b}|{k}':pd_diff(a,b,k) for a,b in [('RF_Tuned','RF_Baseline'),('CatBoost','RF_Tuned'),('RF_Tuned','RF_SMOTE_Tuned'),('RF_Tuned','LogReg')] for k in ['f1','roc_auc']}
print('CI',json.dumps(out['boot_ci']['RF_Tuned']),flush=True); print(json.dumps(out['paired'],indent=1),flush=True)
# threshold from OOF on train (5-fold)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
out['thr']={}
for n,f in M.items():
    if n=='RF_SMOTE_Tuned': continue
    oof=cross_val_predict(f(),Xtr,ytr,cv=skf,method='predict_proba')[:,1]
    ts=np.linspace(0.1,0.9,81); f1s=[f1_score(ytr,oof>=t) for t in ts]; t=float(ts[int(np.argmax(f1s))])
    out['thr'][n]=dict(threshold=t,**met(yte,proba[n],t))
print('THR',pd.DataFrame(out['thr']).T.round(4),flush=True)
# repeated CV on training data (SMOTE inside)
rcv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
out['cv']={}
sc={'f1':'f1','roc_auc':'roc_auc','recall':'recall','precision':'precision','pr_auc':'average_precision'}
for n,f in M.items():
    est=IPipe([('smote',SMOTE(random_state=42)),('rf',f())]) if n=='RF_SMOTE_Tuned' else f()
    r=cross_validate(est,Xtr,ytr,cv=rcv,scoring=sc)
    out['cv'][n]={k:[float(r['test_'+k].mean()),float(r['test_'+k].std())] for k in sc}
    print(n,{k:np.round(v,3) for k,v in out['cv'][n].items()},flush=True)
# SHAP gender / product
sv=shap.TreeExplainer(fitted['RF_Tuned']).shap_values(Xte)
sv=sv[:,:,1] if sv.ndim==3 else sv
ma=np.abs(sv).mean(0); out['shap_rank']=dict(sorted({c:float(v) for c,v in zip(Xte.columns,ma)}.items(),key=lambda x:-x[1]))
c=list(Xte.columns).index('NumOfProducts')
g=pd.DataFrame({'p':Xte.NumOfProducts.values,'s':sv[:,c]}).groupby('p').s.agg(['count','mean','min','max']); out['shap_prod']=g.round(4).to_dict()
print(out['shap_rank']);print(g.round(4),flush=True)
# gender-wise
gm=Xte['Gender_Male'].values.astype(bool); pr=(proba['RF_Tuned']>=.5).astype(int)
out['fair']={}
for lab,mask in [('Pria',gm),('Wanita',~gm)]+[(f'Geo_{n}',m) for n,m in [('Germany',Xte.Geography_Germany.values.astype(bool)),('Spain',Xte.Geography_Spain.values.astype(bool)),('France',~(Xte.Geography_Germany.values.astype(bool)|Xte.Geography_Spain.values.astype(bool)))]]:
    out['fair'][lab]=dict(n=int(mask.sum()),churn_rate=float(yt[mask].mean()),pred_pos_rate=float(pr[mask].mean()),recall=float(recall_score(yt[mask],pr[mask])),precision=float(precision_score(yt[mask],pr[mask])),fpr=float(((pr[mask]==1)&(yt[mask]==0)).sum()/(yt[mask]==0).sum()))
print(pd.DataFrame(out['fair']).T.round(3),flush=True)
json.dump(out,open(ROOT/'results'/'analisis_tambahan.json','w'),indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.2,5))
lab={'RF_Baseline':'RF Baseline','RF_Tuned':'RF Tuned','RF_SMOTE_Tuned':'RF SMOTE Tuned','XGBoost':'XGBoost','LightGBM':'LightGBM','CatBoost':'CatBoost','LogReg':'Regresi Logistik','DecisionTree':'Pohon Keputusan'}
for n,p in proba.items():
    pr,rc,_=precision_recall_curve(yte,p); ax.plot(rc,pr,lw=1.6,label=f"{lab[n]} (AP={average_precision_score(yte,p):.3f})")
ax.axhline(yte.mean(),ls='--',c='gray',label=f'Acuan acak ({yte.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Kurva Precision-Recall pada Data Uji'); ax.legend(fontsize=8,loc='upper right')
plt.tight_layout(); plt.savefig(ROOT/'figures'/'gambar8_pr_curve.png',dpi=200); print('DONE')
