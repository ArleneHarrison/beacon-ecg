#!/usr/bin/env python3
"""Pack locally-present ECG waveforms (subset already on disk) into waves.npy + index.parquet
matching train_waveform.py's expected format. No download; uses whatever .hea/.dat exist."""
import os, argparse, numpy as np, pandas as pd, wfdb
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def read_wave(fp):
    rec=wfdb.rdrecord(fp); sig=rec.p_signal.astype(np.float32)
    n2i={n.upper():i for i,n in enumerate(rec.sig_name)}
    out=np.zeros((12,5000),np.float32); T=min(sig.shape[0],5000)
    for j,ln in enumerate(LEADS):
        i=n2i.get(ln.upper());
        if i is None: continue
        out[j,:T]=np.nan_to_num(sig[:T,i],nan=0.0)
    mu=out.mean(1,keepdims=True); sd=out.std(1,keepdims=True)+1e-6
    return ((out-mu)/sd).astype(np.float16)

ap=argparse.ArgumentParser()
ap.add_argument("--cohort",required=True); ap.add_argument("--files",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
df=pd.read_parquet(a.cohort)
def relpath(p):
    return p[6:] if p.startswith("files/") else p
present=[];
for _,r in df.iterrows():
    fp=os.path.join(a.files,relpath(r["path"]))
    if os.path.exists(fp+".hea") and os.path.exists(fp+".dat"): present.append(r)
sub=pd.DataFrame(present).reset_index(drop=True)
print(f"locally present records: {len(sub)}  (of {len(df)})")
W=np.zeros((len(sub),12,5000),np.float16); ok=np.zeros(len(sub),bool)
for i,r in sub.iterrows():
    try: W[i]=read_wave(os.path.join(a.files,relpath(r["path"]))); ok[i]=True
    except Exception as e:
        if i<10: print("skip",r["path"],e)
sub["ok"]=ok
np.save(f"{a.out}/waves.npy",W)
sub.to_parquet(f"{a.out}/index.parquet")
print(f"packed ok={ok.sum()}/{len(sub)}")
print("split counts:", sub[sub.ok].groupby("split").size().to_dict())
for t in ["hfref_le40","lvef_le50","dead_365d"]:
    s=sub.loc[sub.ok,t].dropna(); print(f"  {t}: pos {int(s.sum())}/{len(s)}")
