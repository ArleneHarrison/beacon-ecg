#!/usr/bin/env python3
"""1-year Kaplan-Meier + log-rank for the occult-disease (discordance) finding, from
test_predictions.parquet. No lifelines dependency (manual KM + log-rank).
Among echo-NORMAL patients (LVEF>50), compare survival of high vs low AI-ECG HFrEF risk."""
import json, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
df=pd.read_parquet(a.pred)

# 1-year time-to-event: event=died<=365d; time=min(days_to_death,365); alive censored at 365
def surv_cols(d):
    dtd=d["days_to_death"].values.astype(float)
    ev=((dtd>=0)&(dtd<=365)).astype(int)
    t=np.where(ev==1, dtd, 365.0)
    t=np.clip(t,0,365)
    return t, ev

def km(t,ev,grid):
    # Kaplan-Meier survival at grid times
    order=np.argsort(t); t=t[order]; ev=ev[order]
    S=[]; s=1.0; n=len(t)
    uniq=np.unique(t[ev==1])
    surv=1.0
    out=np.ones_like(grid,dtype=float)
    for gi,g in enumerate(grid):
        s=1.0
        for tt in uniq:
            if tt>g: break
            at_risk=(t>=tt).sum(); d=((t==tt)&(ev==1)).sum()
            if at_risk>0: s*=(1-d/at_risk)
        out[gi]=s
    return out

def logrank(t,ev,grp):
    # two-group log-rank
    times=np.unique(t[ev==1]); O1=E1=V=0.0
    for tt in times:
        at=(t>=tt); n=at.sum(); n1=(at&(grp==1)).sum()
        d=((t==tt)&(ev==1)).sum(); d1=((t==tt)&(ev==1)&(grp==1)).sum()
        if n>1:
            E1+=d*n1/n; O1+=d1
            V+=d*(n1/n)*(1-n1/n)*(n-d)/(n-1)
    chi2=(O1-E1)**2/V if V>0 else 0
    from scipy.stats import chi2 as chi2d
    p=1-chi2d.cdf(chi2,1)
    return float(chi2), float(p), float(O1), float(E1)

R={}
grid=np.linspace(0,365,74)
plt.rcParams.update({"figure.dpi":140,"font.size":11})

for name, sub in [("echo_normal_LVEF>50", df[df["lvef_value"]>50]),
                  ("whole_test", df)]:
    d=sub.dropna(subset=["pred_hfref_le40"]).copy()   # keep alive (censored); NaN days_to_death -> censored at 365
    if len(d)<50: continue
    thr=d["pred_hfref_le40"].quantile(0.8)
    grp=(d["pred_hfref_le40"]>=thr).astype(int).values
    t,ev=surv_cols(d)
    chi2,p,O1,E1=logrank(t,ev,grp)
    # hazard ratio approx (O/E)
    O0=ev[grp==0].sum();
    hr=(O1/E1)/((ev.sum()-O1)/max(ev.sum()-E1,1e-6)) if E1>0 else np.nan
    R[name]={"n":int(len(d)),"n_high":int((grp==1).sum()),"n_low":int((grp==0).sum()),
             "events_high":int(ev[grp==1].sum()),"events_low":int(ev[grp==0].sum()),
             "logrank_chi2":chi2,"logrank_p":p,"approx_HR_high_vs_low":float(hr),
             "mort365_high":float(ev[grp==1].mean()),"mort365_low":float(ev[grp==0].mean())}
    if name.startswith("echo_normal"):
        fig,ax=plt.subplots(figsize=(5,3.6))
        for g,lab,c in [(1,f"AI-ECG high risk (n={(grp==1).sum()})","#c0603a"),
                        (0,f"AI-ECG low risk (n={(grp==0).sum()})","#3b8a6e")]:
            S=km(t[grp==g],ev[grp==g],grid)
            ax.step(grid,S,where="post",color=c,label=lab,lw=2)
        ax.set_xlabel("days"); ax.set_ylabel("survival probability"); ax.set_ylim(0.7,1.0)
        ax.set_title(f"Occult disease in ECHO-NORMAL patients\nlog-rank p={p:.2e}, HR≈{hr:.2f}")
        ax.legend(frameon=False,loc="lower left"); plt.tight_layout()
        plt.savefig(f"{a.out}/fig7_km_discordance.png"); plt.close()

json.dump(R,open(f"{a.out}/survival.json","w"),indent=2)
print(json.dumps(R,indent=2))
