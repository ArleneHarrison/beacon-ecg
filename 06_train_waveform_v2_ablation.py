#!/usr/bin/env python3
"""Improved waveform multitask model (v2) vs v1 baseline (HFrEF 0.886 / AS 0.825 / LVH 0.776).
Upgrades: (1) ECG-specific augmentation, (2) attention pooling instead of mean pooling,
(3) cosine LR + warmup + longer training + label smoothing. Same tasks/splits for a fair comparison."""
import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

CLS=["hfref_le40","as_severe","lvh","dead_365d"]; REG=["lvef_value"]; TASKS=CLS+REG

class ECGSet(Dataset):
    def __init__(s,waves,df,rows,train=False): s.w=waves; s.df=df; s.rows=rows; s.train=train
    def __len__(s): return len(s.rows)
    def __getitem__(s,k):
        i=s.rows[k]; x=np.asarray(s.w[i],np.float32).copy()
        if s.train:                                  # --- ECG augmentation ---
            if np.random.rand()<0.5: x=np.roll(x,np.random.randint(-250,250),axis=1)   # time shift
            if np.random.rand()<0.5: x*=np.random.uniform(0.8,1.2)                     # amplitude scale
            if np.random.rand()<0.3:                                                   # lead dropout
                x[np.random.randint(0,12)]=0.0
            if np.random.rand()<0.3: x+=np.random.normal(0,0.02,x.shape).astype(np.float32)  # noise
            if np.random.rand()<0.2:                                                   # baseline wander
                t=np.linspace(0,1,x.shape[1],dtype=np.float32)
                x+=0.05*np.sin(2*np.pi*np.random.uniform(0.1,0.5)*t)[None,:]
        row=s.df.iloc[i]
        y=torch.tensor([float(row[t]) if pd.notna(row[t]) else float("nan") for t in TASKS])
        return torch.from_numpy(x), y

class SEBlock(nn.Module):
    def __init__(s,c,r=8): super().__init__(); s.fc1=nn.Linear(c,c//r); s.fc2=nn.Linear(c//r,c)
    def forward(s,x): z=x.mean(-1); z=torch.relu(s.fc1(z)); z=torch.sigmoid(s.fc2(z)); return x*z.unsqueeze(-1)
class ResBlock(nn.Module):
    def __init__(s,ci,co,st=2,k=7):
        super().__init__(); s.c1=nn.Conv1d(ci,co,k,st,k//2); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,1,k//2); s.b2=nn.BatchNorm1d(co); s.se=SEBlock(co); s.drop=nn.Dropout(0.1)
        s.sc=nn.Sequential(nn.Conv1d(ci,co,1,st),nn.BatchNorm1d(co)) if (st!=1 or ci!=co) else nn.Identity()
    def forward(s,x): r=s.sc(x); x=torch.relu(s.b1(s.c1(x))); x=s.b2(s.c2(x)); x=s.se(x); return torch.relu(s.drop(x)+r)
class AttnPool(nn.Module):                      # learned attention over time (vs mean pooling)
    def __init__(s,c): super().__init__(); s.a=nn.Sequential(nn.Conv1d(c,c//4,1),nn.Tanh(),nn.Conv1d(c//4,1,1))
    def forward(s,x): w=torch.softmax(s.a(x),dim=-1); return (x*w).sum(-1)
class Net(nn.Module):
    def __init__(s,nt=len(TASKS)):
        super().__init__(); s.stem=nn.Sequential(nn.Conv1d(12,64,15,2,7),nn.BatchNorm1d(64),nn.ReLU())
        s.body=nn.Sequential(ResBlock(64,64,1),ResBlock(64,128),ResBlock(128,196),ResBlock(196,256),ResBlock(256,256))
        s.pool=AttnPool(256); s.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2))
        s.out=nn.Linear(256,nt); s.log_var=nn.Parameter(torch.zeros(nt))
    def forward(s,x): x=s.body(s.stem(x)); x=s.pool(x); return s.out(s.head(x))

def uw_loss(pred,y,log_var,smooth=0.02):
    tot=0.0
    for j,t in enumerate(TASKS):
        m=~torch.isnan(y[:,j])
        if m.sum()==0: continue
        p=pred[m,j]; tgt=y[m,j]
        if t in CLS:
            tgt=tgt*(1-smooth)+0.5*smooth                     # label smoothing
            l=nn.functional.binary_cross_entropy_with_logits(p,tgt)
        else: l=nn.functional.smooth_l1_loss(p,tgt/50.0)
        prec=torch.exp(-log_var[j]); sc=(0.5 if t in REG else 1.0)
        tot=tot+sc*prec*l+0.5*log_var[j]
    return tot

def boot(y,p,n=1000,seed=0):
    rng=np.random.default_rng(seed); a=[]
    for _ in range(n):
        b=rng.choice(len(y),len(y),True)
        if len(np.unique(y[b]))>1: a.append(roc_auc_score(y[b],p[b]))
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=45); ap.add_argument("--bs",type=int,default=256); ap.add_argument("--lr",type=float,default=3e-3)
    a=ap.parse_args(); dev="cuda"
    full=pd.read_parquet(f"{a.data}/index.parquet"); waves=np.load(f"{a.data}/waves.npy",mmap_mode="r")
    rows=lambda sp: full.index[(full["ok"])&(full["split"]==sp)].values
    tr,va,te=rows("train"),rows("val"),rows("test")
    print(f"train {len(tr)} val {len(va)} test {len(te)}",flush=True)
    mk=lambda r,t,sh: DataLoader(ECGSet(waves,full,r,t),a.bs,shuffle=sh,num_workers=8,pin_memory=True,drop_last=sh)
    net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),a.lr,weight_decay=1e-4)
    steps=max(len(mk(tr,True,True)),1)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,epochs=a.epochs,steps_per_epoch=steps,pct_start=0.15)
    scaler=torch.amp.GradScaler('cuda')
    def ev(r):
        net.eval(); P=[];Y=[]
        with torch.no_grad():
            for x,y in mk(r,False,False):
                with torch.amp.autocast('cuda'): P.append(torch.sigmoid(net(x.to(dev))).float().cpu())
                Y.append(y)
        P=torch.cat(P).numpy(); Y=torch.cat(Y).numpy(); res={}
        for j,t in enumerate(CLS):
            m=~np.isnan(Y[:,j])
            if m.sum() and len(np.unique(Y[m,j]))>1:
                res[t]={"auroc":float(roc_auc_score(Y[m,j],P[m,j])),"auprc":float(average_precision_score(Y[m,j],P[m,j])),"n":int(m.sum())}
        return res,P,Y
    best=0;bad=0
    for ep in range(a.epochs):
        net.train()
        for x,y in mk(tr,True,True):
            x,y=x.to(dev),y.to(dev); opt.zero_grad()
            with torch.amp.autocast('cuda'): loss=uw_loss(net(x),y,net.log_var)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        vr,_,_=ev(va); sc=np.mean([vr[t]["auroc"] for t in vr])
        print(f"ep{ep} val_mAUROC={sc:.4f} "+" ".join(f"{t}={vr[t]['auroc']:.3f}" for t in vr),flush=True)
        if sc>best: best=sc;bad=0; torch.save(net.state_dict(),f"{a.out}/best_v2.pt")
        else:
            bad+=1
            if bad>=10: print("early stop"); break
    net.load_state_dict(torch.load(f"{a.out}/best_v2.pt"))
    tr_res,P,Y=ev(te)
    for j,t in enumerate(CLS):
        m=~np.isnan(Y[:,j])
        if t in tr_res: tr_res[t]["auroc_ci"]=boot(Y[m,j].astype(int),P[m,j])
    json.dump(tr_res,open(f"{a.out}/waveform_metrics_v2.json","w"),indent=2)
    v1={"hfref_le40":0.886,"as_severe":0.825,"lvh":0.776,"dead_365d":0.722}
    print("\n==== V2 TEST (vs v1) ====")
    for t in CLS:
        if t in tr_res: print(f"{t:12} v2 {tr_res[t]['auroc']:.3f} CI{[round(c,3) for c in tr_res[t]['auroc_ci']]}  v1 {v1[t]:.3f}  delta {tr_res[t]['auroc']-v1[t]:+.3f}")
if __name__=="__main__": main()
