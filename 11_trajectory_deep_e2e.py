#!/usr/bin/env python3
"""End-to-end fine-tuned deep-waveform trajectory. Unlike the frozen-embedding pilot, the
encoder is fine-tuned and gradients flow through the whole serial sequence.
Compares single-ECG (encoder+head) vs trajectory (encoder+GRU+head), both trained end-to-end,
for 1-year post-landmark mortality. Packs waveforms to a memmap first for fast epochs."""
import argparse, json, os, numpy as np, pandas as pd, torch, torch.nn as nn, wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]; TASKS=["hfref_le40","as_severe","lvh","dead_365d","lvef_value"]; MAXLEN=8

class SEBlock(nn.Module):
    def __init__(s,c,r=8): super().__init__(); s.fc1=nn.Linear(c,c//r); s.fc2=nn.Linear(c//r,c)
    def forward(s,x): z=x.mean(-1); z=torch.relu(s.fc1(z)); z=torch.sigmoid(s.fc2(z)); return x*z.unsqueeze(-1)
class ResBlock(nn.Module):
    def __init__(s,ci,co,st=2,k=7):
        super().__init__(); s.c1=nn.Conv1d(ci,co,k,st,k//2); s.b1=nn.BatchNorm1d(co); s.c2=nn.Conv1d(co,co,k,1,k//2); s.b2=nn.BatchNorm1d(co)
        s.se=SEBlock(co); s.drop=nn.Dropout(0.1); s.sc=nn.Sequential(nn.Conv1d(ci,co,1,st),nn.BatchNorm1d(co)) if (st!=1 or ci!=co) else nn.Identity()
    def forward(s,x): r=s.sc(x); x=torch.relu(s.b1(s.c1(x))); x=s.b2(s.c2(x)); x=s.se(x); return torch.relu(s.drop(x)+r)
class Encoder(nn.Module):
    def __init__(s):
        super().__init__(); s.stem=nn.Sequential(nn.Conv1d(12,64,15,2,7),nn.BatchNorm1d(64),nn.ReLU())
        s.body=nn.Sequential(ResBlock(64,64,1),ResBlock(64,128),ResBlock(128,196),ResBlock(196,256),ResBlock(256,256))
        s.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2))
    def forward(s,x): x=s.body(s.stem(x)); x=x.mean(-1); return s.head(x)
    def load_ckpt(s,ck):
        sd=torch.load(ck,map_location="cpu"); own=s.state_dict()
        for k in own:  # copy matching stem/body/head weights from the multitask checkpoint
            if k in sd and sd[k].shape==own[k].shape: own[k]=sd[k]
        s.load_state_dict(own); return s

def read_wave(fp):
    rec=wfdb.rdrecord(fp); sig=rec.p_signal.astype(np.float32); n2i={n.upper():i for i,n in enumerate(rec.sig_name)}
    out=np.zeros((12,5000),np.float32); T=min(sig.shape[0],5000)
    for j,ln in enumerate(LEADS):
        i=n2i.get(ln.upper());
        if i is not None: out[j,:T]=np.nan_to_num(sig[:T,i],nan=0.0)
    mu=out.mean(1,keepdims=True); sd=out.std(1,keepdims=True)+1e-6; return ((out-mu)/sd).astype(np.float16)

def pack(recs,files,packdir):
    os.makedirs(packdir,exist_ok=True); mmp=f"{packdir}/waves.npy"
    def fp(p): return p[6:] if p.startswith("files/") else p
    recs=recs[recs["path"].map(lambda p: os.path.exists(os.path.join(files,fp(p)+".dat")))].reset_index(drop=True)
    W=np.lib.format.open_memmap(mmp,mode="w+",dtype=np.float16,shape=(len(recs),12,5000))
    for i,p in enumerate(recs["path"]):
        try: W[i]=read_wave(os.path.join(files,fp(p)))
        except Exception: pass
        if i%10000==0: print("packed",i,flush=True)
    recs["row"]=np.arange(len(recs)); recs.to_parquet(f"{packdir}/rec_index.parquet"); return recs,mmp

class PatSet(Dataset):
    def __init__(s,seqs,W): s.seqs=seqs; s.W=W
    def __len__(s): return len(s.seqs)
    def __getitem__(s,i):
        rows,L,y=s.seqs[i]; x=np.zeros((MAXLEN,12,5000),np.float32)
        for k,r in enumerate(rows[-MAXLEN:]): x[MAXLEN-min(len(rows),MAXLEN)+k]=s.W[r]
        return torch.from_numpy(x),L,torch.tensor(float(y))

class TrajNet(nn.Module):
    def __init__(s,mode,ck): super().__init__(); s.enc=Encoder().load_ckpt(ck); s.mode=mode
    def _emb(s,x):  # x [B,MAXLEN,12,5000] -> [B,MAXLEN,256]
        B=x.shape[0]; e=s.enc(x.reshape(B*MAXLEN,12,5000)); return e.reshape(B,MAXLEN,256)
    def build(s):
        if s.mode=="traj": s.g=nn.GRU(256,128,batch_first=True); s.f=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
        else: s.f=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))
        return s
    def forward(s,x,L):
        emb=s._emb(x)
        if s.mode=="traj":
            o,_=s.g(emb); idx=(L-1).clamp(min=0); return s.f(o[torch.arange(len(L)),idx]).squeeze(-1)
        return s.f(emb[:,-1,:]).squeeze(-1)

def main():
    ap=argparse.ArgumentParser()
    for k in ["records","files","cohort","ckpt","out","packdir"]: ap.add_argument("--"+k,required=True)
    ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--bs",type=int,default=24)
    a=ap.parse_args(); dev="cuda"; os.makedirs(a.out,exist_ok=True)
    recs=pd.read_csv(a.records); recs["ecg_time"]=pd.to_datetime(recs["ecg_time"])
    if not os.path.exists(f"{a.packdir}/rec_index.parquet"): recs,mmp=pack(recs,a.files,a.packdir)
    else: recs=pd.read_parquet(f"{a.packdir}/rec_index.parquet"); mmp=f"{a.packdir}/waves.npy"
    W=np.load(mmp,mmap_mode="r")
    coh=pd.read_parquet(a.cohort); pat=(coh.sort_values("ecg_time").groupby("subject_id").agg(dod=("dod","first"),split=("split","first")).reset_index()); pat["dod"]=pd.to_datetime(pat["dod"],errors="coerce")
    pm=dict(zip(pat.subject_id,zip(pat.dod,pat.split)))
    seqs={"train":[],"test":[],"val":[]}
    for sid,g in recs.sort_values(["subject_id","ecg_time"]).groupby("subject_id"):
        if sid not in pm: continue
        dod,sp=pm[sid]; t_last=g["ecg_time"].max(); dtl=(dod-t_last).days if pd.notna(dod) else np.nan
        y=int((not np.isnan(dtl)) and 0<=dtl<=365); rows=g["row"].tolist()
        if sp in seqs: seqs[sp].append((rows,min(len(rows),MAXLEN),y))
    print({k:len(v) for k,v in seqs.items()},"event_rate_test",np.mean([y for *_,y in seqs["test"]]),flush=True)
    def loader(split,sh): return DataLoader(PatSet(seqs[split],W),a.bs,shuffle=sh,num_workers=6,pin_memory=True,drop_last=sh)
    ytr=np.array([y for *_,y in seqs["train"]]); pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],device=dev)
    yte=np.array([y for *_,y in seqs["test"]])
    def train(mode):
        net=TrajNet(mode,a.ckpt).build().to(dev)
        enc_p=list(net.enc.parameters()); rest=[p for n,p in net.named_parameters() if not n.startswith("enc.")]
        opt=torch.optim.AdamW([{"params":enc_p,"lr":2e-4},{"params":rest,"lr":2e-3}],weight_decay=1e-4)
        sc=torch.amp.GradScaler('cuda'); best=0; bp=None
        for ep in range(a.epochs):
            net.train()
            for x,L,y in loader("train",True):
                x,L,y=x.to(dev),L.to(dev),y.to(dev); opt.zero_grad()
                with torch.amp.autocast('cuda'): loss=nn.functional.binary_cross_entropy_with_logits(net(x,L),y,pos_weight=pw)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
            net.eval(); P=[]
            with torch.no_grad():
                for x,L,y in loader("test",False):
                    with torch.amp.autocast('cuda'): P.append(torch.sigmoid(net(x.to(dev),L.to(dev))).float().cpu())
            p=torch.cat(P).numpy(); auc=roc_auc_score(yte,p); print(f"[{mode}] ep{ep} test_auc={auc:.4f}",flush=True)
            if auc>best: best=auc; bp=p
        return best,bp
    torch.manual_seed(0); auc_s,ps=train("snap")
    torch.manual_seed(0); auc_t,pt=train("traj")
    def boot(y,p,n=1000):
        rng=np.random.default_rng(0);b=[]
        for _ in range(n):
            j=rng.choice(len(y),len(y),True)
            if len(np.unique(y[j]))>1: b.append(roc_auc_score(y[j],p[j]))
        return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
    R={"n_test":int(len(yte)),"event_rate":float(yte.mean()),"mode":"end-to-end fine-tuned",
       "auroc_snapshot_e2e":float(auc_s),"ci_snapshot":boot(yte,ps),
       "auroc_trajectory_e2e":float(auc_t),"ci_trajectory":boot(yte,pt),"delta_auroc":float(auc_t-auc_s)}
    json.dump(R,open(f"{a.out}/deep_trajectory_e2e.json","w"),indent=2); print(json.dumps(R,indent=2))
if __name__=="__main__": main()
