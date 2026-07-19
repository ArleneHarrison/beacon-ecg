#!/usr/bin/env python3
"""Analyses required by the peer-review self-audit:
 F1  same-test-set baseline vs waveform + DeLong test (paired)
 M4  incremental prognostic: dAUROC bootstrap CI + IDI (not NRI alone)
 F3  discordance: multivariable Cox adjusted for age/sex/echo -> adjusted HR + CI
 M6  severe-AS clinical utility: calibration, decision curve, PPV/NPV at fixed sensitivity
"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats

OUT="/root/autodl-tmp/aiecg/outputs"
COH="/root/autodl-tmp/aiecg/cohort_labels.parquet"
R={}

# ---------------- DeLong (paired AUROC comparison) ----------------
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N,float); T2[J]=T; return T2
def delong(y,p1,p2):
    """paired DeLong test for two correlated ROC curves; returns auc1,auc2,z,p"""
    pos=y==1; neg=y==0; m=pos.sum(); n=neg.sum()
    preds=np.vstack([p1,p2])
    tx=np.array([_midrank(preds[k][pos]) for k in range(2)])
    ty=np.array([_midrank(preds[k][neg]) for k in range(2)])
    tz=np.array([_midrank(np.concatenate([preds[k][pos],preds[k][neg]])) for k in range(2)])
    aucs=(tz[:,:m].sum(1)/m-(m+1)/2)/n
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    sx=np.cov(v01); sy=np.cov(v10)
    S=sx/m+sy/n
    d=np.array([1,-1])
    var=d@S@d
    z=(aucs[0]-aucs[1])/np.sqrt(var) if var>0 else 0.0
    return float(aucs[0]),float(aucs[1]),float(z),float(2*(1-stats.norm.cdf(abs(z))))

def boot_delta(y,p1,p2,n=2000,seed=0):
    rng=np.random.default_rng(seed); d=[]
    for _ in range(n):
        b=rng.choice(len(y),len(y),True)
        if len(np.unique(y[b]))<2: continue
        d.append(roc_auc_score(y[b],p1[b])-roc_auc_score(y[b],p2[b]))
    return [float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]

coh=pd.read_parquet(COH); idx=coh[coh.is_index_ecg==1].copy()
pred=pd.read_parquet(f"{OUT}/final/test_predictions.parquet")
MM=["rr_interval","pr_interval","qrs_duration","qt_interval","qtc","heart_rate","p_axis","qrs_axis","t_axis","age_at_ecg","sex_male"]

# ---------------- F1: same-test-set comparison + DeLong ----------------
R["F1_same_testset_delong"]={}
tr=idx[idx.split=="train"]
for tgt,wcol in [("hfref_le40","pred_hfref_le40"),("as_severe","pred_as_severe"),("dead_365d","pred_dead_365d")]:
    te=pred.dropna(subset=[tgt,wcol]).copy()           # EXACT waveform test patients
    trn=tr.dropna(subset=[tgt])
    m=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,l2_regularization=1.0,
                                     early_stopping=True,validation_fraction=0.15,random_state=0)
    m.fit(trn[MM].astype(float),trn[tgt].astype(int))
    pb=m.predict_proba(te[MM].astype(float))[:,1]      # baseline on the SAME patients
    pw=te[wcol].values; y=te[tgt].astype(int).values
    a_w,a_b,z,p=delong(y,pw,pb)
    R["F1_same_testset_delong"][tgt]={
        "n_same_test":int(len(te)),"auroc_waveform":a_w,"auroc_baseline_same_n":a_b,
        "delta":a_w-a_b,"delta_95CI":boot_delta(y,pw,pb),"delong_z":z,"delong_p":p}

# ---------------- M4: incremental prognostic with CI + IDI ----------------
echo=[c for c in ["lvef_value","av_pk_vel","av_mean_grad","av_area_continuity","septal_thickness",
     "lvedd","lvesd","la_vol","tr_mmhg"] if c in pred.columns]
sig=[c for c in ["pred_hfref_le40","pred_as_severe","pred_lvh"] if c in pred.columns]
d=pred.dropna(subset=["dead_365d"]).copy(); y=d["dead_365d"].astype(int).values
def cvp(cols,seed=1):
    X=d[cols].astype(float).values; oof=np.zeros(len(y))
    for tr_i,te_i in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        mm=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                         LogisticRegression(max_iter=3000,class_weight="balanced"))
        mm.fit(X[tr_i],y[tr_i]); oof[te_i]=mm.predict_proba(X[te_i])[:,1]
    return oof
p_e=cvp(echo); p_b=cvp(echo+sig)
def idi(y,po,pn):
    return float((pn[y==1].mean()-po[y==1].mean())-(pn[y==0].mean()-po[y==0].mean()))
R["M4_incremental_prognostic"]={
    "n":int(len(y)),"auroc_echo_only":float(roc_auc_score(y,p_e)),
    "auroc_echo_plus_aiecg":float(roc_auc_score(y,p_b)),
    "delta_auroc":float(roc_auc_score(y,p_b)-roc_auc_score(y,p_e)),
    "delta_auroc_95CI":boot_delta(y,p_b,p_e),
    "delong_p":delong(y,p_b,p_e)[3],
    "IDI":idi(y,p_e,p_b),
    "note":"IDI reported alongside NRI; continuous NRI is known to be unstable/inflated"}

# ---------------- F3: discordance with multivariable Cox ----------------
try:
    from lifelines import CoxPHFitter
    en=pred[(pred.lvef_value>50)].dropna(subset=["pred_hfref_le40"]).copy()
    dtd=(en["days_to_death"]).astype(float)
    en["event"]=((dtd>=0)&(dtd<=365)).astype(int)
    en["T"]=np.where(en["event"]==1,dtd,365.0).clip(1,365)
    thr=en["pred_hfref_le40"].quantile(0.8)
    en["aiecg_high"]=(en["pred_hfref_le40"]>=thr).astype(int)
    cov=["aiecg_high","age_at_ecg","sex_male"]+[c for c in ["lvef_value","septal_thickness","av_pk_vel"] if c in en.columns]
    cx=en[cov+["T","event"]].copy()
    for c in cov: cx[c]=pd.to_numeric(cx[c],errors="coerce")
    cx=cx.fillna(cx.median(numeric_only=True))
    cph=CoxPHFitter().fit(cx,duration_col="T",event_col="event")
    s=cph.summary.loc["aiecg_high"]
    R["F3_discordance_adjusted_cox"]={
        "n":int(len(cx)),"events":int(cx.event.sum()),"covariates":cov,
        "adjusted_HR":float(s["exp(coef)"]),
        "HR_95CI":[float(s["exp(coef) lower 95%"]),float(s["exp(coef) upper 95%"])],
        "p":float(s["p"]),
        "note":"adjusted for age, sex and available echo measures; comorbidity codes not available in this table"}
except Exception as e:
    R["F3_discordance_adjusted_cox"]={"error":str(e)}

# ---------------- M6: severe-AS clinical utility ----------------
a=pred.dropna(subset=["as_severe","pred_as_severe"]).copy()
ya=a["as_severe"].astype(int).values; pa=a["pred_as_severe"].values
def ece(y,p,bins=10):
    e=0
    for b in range(bins):
        m=(p>=b/bins)&(p<(b+1)/bins)
        if m.sum(): e+=abs(p[m].mean()-y[m].mean())*m.sum()/len(y)
    return float(e)
def nb(y,p,pt):
    tp=((p>=pt)&(y==1)).sum(); fp=((p>=pt)&(y==0)).sum()
    return float(tp/len(y)-fp/len(y)*(pt/(1-pt)))
# operating point at 90% sensitivity
order=np.argsort(-pa); thr90=None
cum=np.cumsum(ya[order]); need=0.9*ya.sum()
k=int(np.searchsorted(cum,need)); thr90=float(pa[order][min(k,len(pa)-1)])
pred_pos=pa>=thr90
sens=float(((pred_pos)&(ya==1)).sum()/max(ya.sum(),1))
spec=float(((~pred_pos)&(ya==0)).sum()/max((ya==0).sum(),1))
ppv=float(((pred_pos)&(ya==1)).sum()/max(pred_pos.sum(),1))
npv=float(((~pred_pos)&(ya==0)).sum()/max((~pred_pos).sum(),1))
R["M6_severe_AS_utility"]={
    "n":int(len(ya)),"prevalence":float(ya.mean()),
    "brier":float(brier_score_loss(ya,pa)),"ece":ece(ya,pa),
    "net_benefit":{f"{t:.2f}":nb(ya,pa,t) for t in [0.02,0.05,0.10,0.20]},
    "at_90pct_sensitivity":{"threshold":thr90,"sensitivity":sens,"specificity":spec,
                            "PPV":ppv,"NPV":npv,
                            "echos_needed_per_case_found":float(1/ppv) if ppv>0 else None}}

json.dump(R,open(f"{OUT}/review_fixes.json","w"),indent=2)
print(json.dumps(R,indent=2))
