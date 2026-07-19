#!/usr/bin/env python3
"""Downstream evaluation from test_predictions.parquet:
 (A) discrimination + calibration (Brier/ECE) + decision-curve net benefit
 (B) INCREMENTAL PROGNOSTIC VALUE over echo: 1-yr mortality from echo-numbers alone
     vs echo + AI-ECG structural signature (5-fold CV on test), dAUROC + continuous NRI
 (C) DISCORDANCE / occult disease: among echo-normal (LVEF>50), does high AI-ECG HFrEF
     probability flag higher 1-yr mortality?  (the 'ECG sees what the echo misses' result)
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import argparse

ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
df=pd.read_parquet(a.pred)
R={}

def ece(y,p,bins=10):
    e=0; n=len(y)
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins; m=(p>=lo)&(p<hi)
        if m.sum(): e+=abs(p[m].mean()-y[m].mean())*m.sum()/n
    return float(e)

def net_benefit(y,p,pt):
    tp=((p>=pt)&(y==1)).sum(); fp=((p>=pt)&(y==0)).sum(); n=len(y)
    return tp/n - fp/n*(pt/(1-pt))

# ---------- (A) discrimination + calibration + DCA ----------
R["discrimination_calibration"]={}
for t in ["hfref_le40","as_severe","lvh","dead_365d"]:
    pc=f"pred_{t}"
    if pc not in df.columns: continue
    d=df.dropna(subset=[t,pc]); y=d[t].astype(int).values; p=d[pc].values
    if len(np.unique(y))<2: continue
    dca={f"{pt:.2f}":float(net_benefit(y,p,pt)) for pt in [0.05,0.1,0.15,0.2,0.3]}
    R["discrimination_calibration"][t]={"n":int(len(y)),"pos_rate":float(y.mean()),
        "auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),
        "brier":float(brier_score_loss(y,p)),"ece":ece(y,p),"net_benefit":dca}

# ---------- (B) incremental prognostic value over echo (1-yr mortality) ----------
echo_feats=[c for c in ["lvef_value","av_pk_vel","av_mean_grad","av_area_continuity",
            "septal_thickness","lvedd","lvesd","la_vol","tr_mmhg"] if c in df.columns]
sig_feats=[c for c in ["pred_hfref_le40","pred_as_severe","pred_lvh"] if c in df.columns]
dm=df.dropna(subset=["dead_365d"]).copy(); y=dm["dead_365d"].astype(int).values
def cv_auc(X):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); oof=np.zeros(len(y))
    for tr,te in skf.split(X,y):
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],y[tr]); oof[te]=clf.predict_proba(X[te])[:,1]
    return oof
p_echo=cv_auc(dm[echo_feats].astype(float).values)
p_both=cv_auc(dm[echo_feats+sig_feats].astype(float).values)
def cont_nri(y,p_old,p_new):
    up=(y==1)&(p_new>p_old); down=(y==1)&(p_new<p_old)
    nup=(y==0)&(p_new<p_old); ndown=(y==0)&(p_new>p_old)
    ev=(up.sum()-down.sum())/max((y==1).sum(),1); nonev=(nup.sum()-ndown.sum())/max((y==0).sum(),1)
    return float(ev+nonev)
R["incremental_prognostic_1yr_mortality"]={
    "n":int(len(y)),"echo_feats":echo_feats,"signature_feats":sig_feats,
    "auroc_echo_only":float(roc_auc_score(y,p_echo)),
    "auroc_echo_plus_aiecg":float(roc_auc_score(y,p_both)),
    "delta_auroc":float(roc_auc_score(y,p_both)-roc_auc_score(y,p_echo)),
    "continuous_NRI":cont_nri(y,p_echo,p_both),
    "note":"5-fold CV on test set; AI-ECG signature = model structural probabilities from ECG"}

# ---------- (C) discordance: occult disease among echo-normal ----------
if "pred_hfref_le40" in df.columns:
    en=df[(df["lvef_value"]>50)].dropna(subset=["pred_hfref_le40","dead_365d"]).copy()
    thr=en["pred_hfref_le40"].quantile(0.8)   # top-quintile AI-ECG risk despite normal echo
    hi=en[en["pred_hfref_le40"]>=thr]; lo=en[en["pred_hfref_le40"]<thr]
    R["discordance_echo_normal_occult"]={
        "n_echo_normal":int(len(en)),
        "mortality_1yr_high_aiecg_risk":float(hi["dead_365d"].mean()),
        "mortality_1yr_low_aiecg_risk":float(lo["dead_365d"].mean()),
        "risk_ratio":float(hi["dead_365d"].mean()/max(lo["dead_365d"].mean(),1e-6)),
        "n_high":int(len(hi)),"n_low":int(len(lo)),
        "note":"Among patients the echo called normal (LVEF>50), do those the AI-ECG flags high-risk die more? tests occult-disease detection"}

json.dump(R,open(f"{a.out}/eval_full.json","w"),indent=2)
print(json.dumps(R,indent=2))
