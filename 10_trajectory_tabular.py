#!/usr/bin/env python3
"""Structured (tabular) serial-ECG TRAJECTORY analysis — no waveform download needed.
Uses machine-measurements (intervals/axes) + timestamps for ALL of each cohort patient's ECGs.

Design (landmark, leakage-free):
  * include patients with >=2 ECGs;
  * landmark = time of the patient's LAST ECG;
  * SNAPSHOT features = that last ECG's measurements (+age/sex);
  * TRAJECTORY features = how the ECG changed up to the landmark (first->last delta, slope/yr, variability);
  * OUTCOME = death within 365 d AFTER the landmark (features are all <= landmark -> no leakage).
Question: does the ECG's trajectory add prognostic value over the latest single ECG?
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE="/data/s01011/cardio_m3t/data/raw/physionet"
ECG=f"{BASE}/mimic-iv-ecg/1.0"
OUT="/data/s01011/cardio_m3t/aiecg/outputs"

MEAS=["heart_rate","pr_interval","qrs_duration","qt_interval","qtc","p_axis","qrs_axis","t_axis"]

print(">> cohort labels (patient-level dod/split/sex/age)")
coh=pd.read_parquet(f"{OUT}/cohort_labels.parquet")
pat=(coh.sort_values("ecg_time").groupby("subject_id")
       .agg(dod=("dod","first"),split=("split","first"),
            sex_male=("sex_male","first"),age=("age_at_ecg","first")).reset_index())
pat["dod"]=pd.to_datetime(pat["dod"],errors="coerce")
cohort_ids=set(pat.subject_id)

print(">> record_list + machine_measurements for cohort patients")
rl=pd.read_csv(f"{ECG}/record_list.csv",usecols=["subject_id","study_id","ecg_time"])
rl=rl[rl.subject_id.isin(cohort_ids)].copy()
rl["ecg_time"]=pd.to_datetime(rl["ecg_time"],errors="coerce")
mm=pd.read_csv(f"{ECG}/machine_measurements.csv",
               usecols=["subject_id","study_id","rr_interval","p_onset","qrs_onset","qrs_end","t_end"])
mm["heart_rate"]=np.where(mm.rr_interval>0,60000.0/mm.rr_interval,np.nan)
mm["pr_interval"]=mm.qrs_onset-mm.p_onset
mm["qrs_duration"]=mm.qrs_end-mm.qrs_onset
mm["qt_interval"]=mm.t_end-mm.qrs_onset
mm["qtc"]=np.where(mm.rr_interval>0,mm.qt_interval/np.sqrt(mm.rr_interval/1000.0),np.nan)
mmx=pd.read_csv(f"{ECG}/machine_measurements.csv",usecols=["subject_id","study_id","p_axis","qrs_axis","t_axis"])
mm=mm.merge(mmx,on=["subject_id","study_id"],how="left")
df=rl.merge(mm[["subject_id","study_id"]+MEAS],on=["subject_id","study_id"],how="left").dropna(subset=["ecg_time"])
df=df.sort_values(["subject_id","ecg_time"])
print(f"   cohort ECGs with measurements: {len(df):,}")

print(">> building per-patient trajectory features (>=2 ECGs)")
rows=[]
for sid,g in df.groupby("subject_id"):
    if len(g)<2: continue
    g=g.sort_values("ecg_time")
    t=g["ecg_time"].values; span_yr=max((t[-1]-t[0])/np.timedelta64(365,"D"),1e-3)
    rec={"subject_id":sid,"n_ecgs":len(g),"span_days":(t[-1]-t[0])/np.timedelta64(1,"D"),
         "t_last":g["ecg_time"].iloc[-1]}
    for m in MEAS:
        v=g[m].values.astype(float); first=v[0]; last=v[-1]
        rec[f"{m}_last"]=last
        rec[f"{m}_delta"]=last-first
        rec[f"{m}_slope"]=(last-first)/span_yr
        rec[f"{m}_std"]=np.nanstd(v)
    rows.append(rec)
T=pd.DataFrame(rows)
print(f"   patients with >=2 ECGs: {len(T):,}")

T=T.merge(pat,on="subject_id",how="left")
dtl=(T["dod"]-T["t_last"]).dt.total_seconds()/86400.0
T["event"]=((dtl>=0)&(dtl<=365)).astype(int)
print(f"   event rate (death within 1yr after last ECG): {T['event'].mean():.3f} ({int(T['event'].sum())}/{len(T)})")

snap=[f"{m}_last" for m in MEAS]+["age","sex_male"]
traj=snap+[f"{m}_{s}" for m in MEAS for s in ("delta","slope","std")]+["n_ecgs","span_days"]

def evaluate(cols):
    tr=T[T.split=="train"]; te=T[T.split=="test"]
    Xtr,ytr=tr[cols].astype(float).values,tr["event"].values
    Xte,yte=te[cols].astype(float).values,te["event"].values
    m=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                    LogisticRegression(max_iter=3000,class_weight="balanced"))
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,p),p,yte

auc_s,ps,y=evaluate(snap)
auc_t,pt,_=evaluate(traj)
def cnri(y,po,pn):
    up=((y==1)&(pn>po)).sum()-((y==1)&(pn<po)).sum()
    dn=((y==0)&(pn<po)).sum()-((y==0)&(pn>po)).sum()
    return float(up/max((y==1).sum(),1)+dn/max((y==0).sum(),1))
def boot(y,p,n=1000,seed=0):
    rng=np.random.default_rng(seed);a=[]
    for _ in range(n):
        b=rng.choice(len(y),len(y),True)
        if len(np.unique(y[b]))>1: a.append(roc_auc_score(y[b],p[b]))
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

R={"n_patients":int(len(T)),"n_test":int((T.split=='test').sum()),
   "event_rate":float(T['event'].mean()),
   "median_n_ecgs":float(T['n_ecgs'].median()),"median_span_days":float(T['span_days'].median()),
   "auroc_snapshot_only":auc_s,"auroc_snapshot_ci":boot(y,ps),
   "auroc_snapshot_plus_trajectory":auc_t,"auroc_traj_ci":boot(y,pt),
   "delta_auroc":auc_t-auc_s,"continuous_NRI":cnri(y,ps,pt)}
json.dump(R,open(f"{OUT}/trajectory_tabular.json","w"),indent=2)
print("\n================ TABULAR TRAJECTORY ================")
print(json.dumps(R,indent=2))
