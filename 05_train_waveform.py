#!/usr/bin/env python3
"""
Waveform multitask model on seeta (5090). Input: one 12-lead ECG (12x5000).
Heads: HFrEF(LVEF<=40), severe AS, LVH  [BCE, NaN-masked] ; LVEF value [Huber] ;
       1-year mortality [BCE].  Homoscedastic uncertainty weighting (Kendall 2018).
Patient-level splits from index.parquet. Reports test AUROC + bootstrap CI vs baseline.
"""
import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

CLS = ["hfref_le40","as_severe","lvh","dead_365d"]
REG = ["lvef_value"]
TASKS = CLS + REG

class ECGSet(Dataset):
    def __init__(self, waves, df, idx):
        self.w = waves; self.df = df.reset_index(drop=True); self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(np.asarray(self.w[i], np.float32))
        row = self.df.iloc[i]
        y = torch.tensor([float(row[t]) if pd.notna(row[t]) else float("nan") for t in TASKS])
        return x, y

class SEBlock(nn.Module):
    def __init__(s, c, r=8):
        super().__init__(); s.fc1=nn.Linear(c,c//r); s.fc2=nn.Linear(c//r,c)
    def forward(s,x):
        z=x.mean(-1); z=torch.relu(s.fc1(z)); z=torch.sigmoid(s.fc2(z))
        return x*z.unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(s, cin, cout, stride=2, k=7):
        super().__init__()
        s.c1=nn.Conv1d(cin,cout,k,stride,k//2); s.b1=nn.BatchNorm1d(cout)
        s.c2=nn.Conv1d(cout,cout,k,1,k//2);     s.b2=nn.BatchNorm1d(cout)
        s.se=SEBlock(cout); s.drop=nn.Dropout(0.1)
        s.sc=nn.Sequential(nn.Conv1d(cin,cout,1,stride),nn.BatchNorm1d(cout)) if (stride!=1 or cin!=cout) else nn.Identity()
    def forward(s,x):
        r=s.sc(x); x=torch.relu(s.b1(s.c1(x))); x=s.b2(s.c2(x)); x=s.se(x)
        return torch.relu(s.drop(x)+r)

class Net(nn.Module):
    def __init__(s, nt=len(TASKS)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(12,64,15,2,7),nn.BatchNorm1d(64),nn.ReLU())
        s.body=nn.Sequential(ResBlock(64,64,1),ResBlock(64,128),ResBlock(128,196),
                             ResBlock(196,256),ResBlock(256,256))
        s.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2))
        s.out=nn.Linear(256,nt)
        s.log_var=nn.Parameter(torch.zeros(nt))    # uncertainty weights
    def forward(s,x):
        x=s.body(s.stem(x)); x=x.mean(-1); x=s.head(x); return s.out(x)

def uw_loss(pred, y, log_var, only=None, use_unc=True):
    tot=0.0
    for j,t in enumerate(TASKS):
        if only and t!=only: continue          # single-task ablation
        m=~torch.isnan(y[:,j])
        if m.sum()==0: continue
        p=pred[m,j]; tgt=y[m,j]
        if t in CLS:
            l=nn.functional.binary_cross_entropy_with_logits(p,tgt)
        else:
            l=nn.functional.smooth_l1_loss(p, tgt/50.0)   # scale LVEF
        if use_unc:                              # homoscedastic uncertainty weighting
            prec=torch.exp(-log_var[j]); scale=(0.5 if t in REG else 1.0)
            tot=tot+scale*prec*l+0.5*log_var[j]
        else:                                    # equal-weight ablation
            tot=tot+l
    return tot

def boot(y,p,n=1000,seed=0):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); a=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if len(np.unique(y[b]))<2: continue
        a.append(roc_auc_score(y[b],p[b]))
    return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=25); ap.add_argument("--bs",type=int,default=256)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--no_uncertainty",action="store_true"); ap.add_argument("--single_task",default="")
    ap.add_argument("--pretrained",default="",help="path to self-supervised encoder weights (PTB-XL)")
    a=ap.parse_args()
    dev="cuda"
    df=pd.read_parquet(f"{a.data}/index.parquet"); df=df[df["ok"]].reset_index(drop=True)
    waves=np.load(f"{a.data}/waves.npy",mmap_mode="r")
    # rebuild alignment: index.parquet was saved with 'ok'; waves rows align to df BEFORE filtering.
    # We saved df with ok and same order, so recover original positions:
    full=pd.read_parquet(f"{a.data}/index.parquet"); pos=np.where(full["ok"].values)[0]
    def make(split):
        rows=full.index[(full["ok"])&(full["split"]==split)].values
        return ECGSet(waves, full, rows)
    tr,va,te=make("train"),make("val"),make("test")
    print(f"train {len(tr)} val {len(va)} test {len(te)}")
    dl=lambda ds,sh: DataLoader(ds,a.bs,shuffle=sh,num_workers=8,pin_memory=True,drop_last=sh)
    net=Net().to(dev)
    if a.pretrained:                      # load PTB-XL self-supervised encoder (stem/body/head)
        sd=torch.load(a.pretrained,map_location=dev); own=net.state_dict(); n=0
        for k in own:
            if k in sd and sd[k].shape==own[k].shape: own[k]=sd[k]; n+=1
        net.load_state_dict(own); print(f"loaded {n} pretrained tensors from {a.pretrained}",flush=True)
    opt=torch.optim.AdamW(net.parameters(),a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,epochs=a.epochs,steps_per_epoch=len(dl(tr,True)))
    scaler=torch.amp.GradScaler('cuda')

    def evaluate(ds):
        net.eval(); P,Y=[],[]
        with torch.no_grad():
            for x,y in dl(ds,False):
                with torch.amp.autocast('cuda'):
                    p=torch.sigmoid(net(x.to(dev))).float().cpu()
                P.append(p); Y.append(y)
        P=torch.cat(P).numpy(); Y=torch.cat(Y).numpy(); res={}
        for j,t in enumerate(CLS):
            m=~np.isnan(Y[:,j])
            if m.sum() and len(np.unique(Y[m,j]))>1:
                res[t]={"auroc":float(roc_auc_score(Y[m,j],P[m,j])),
                        "auprc":float(average_precision_score(Y[m,j],P[m,j])),"n":int(m.sum())}
        return res,P,Y

    best=0; bad=0
    for ep in range(a.epochs):
        net.train()
        for x,y in dl(tr,True):
            x,y=x.to(dev),y.to(dev); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss=uw_loss(net(x),y,net.log_var,
                             only=(a.single_task or None), use_unc=not a.no_uncertainty)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        vres,_,_=evaluate(va); score=np.mean([vres[t]["auroc"] for t in vres])
        print(f"ep{ep} val_mAUROC={score:.4f}  " + " ".join(f"{t}={vres[t]['auroc']:.3f}" for t in vres))
        if score>best: best=score; bad=0; torch.save(net.state_dict(),f"{a.out}/best.pt")
        else:
            bad+=1
            if bad>=6: print("early stop"); break

    net.load_state_dict(torch.load(f"{a.out}/best.pt"))
    tres,P,Y=evaluate(te)
    for j,t in enumerate(CLS):
        m=~np.isnan(Y[:,j])
        if t in tres: tres[t]["auroc_ci"]=boot(Y[m,j].astype(int),P[m,j])
    json.dump(tres,open(f"{a.out}/waveform_metrics.json","w"),indent=2)
    # save aligned test predictions for downstream analyses (calibration/DCA/NRI/discordance/trajectory)
    test_rows=full.index[(full["ok"])&(full["split"]=="test")].values
    pred=full.loc[test_rows].copy()
    for j,t in enumerate(CLS): pred[f"pred_{t}"]=P[:,j]
    pred.to_parquet(f"{a.out}/test_predictions.parquet")
    print("saved test_predictions.parquet rows", len(pred))
    print("\n==== WAVEFORM TEST ====");
    for t in CLS:
        if t in tres: print(f"{t:12} AUROC {tres[t]['auroc']:.3f} CI{tres[t].get('auroc_ci')} AUPRC {tres[t]['auprc']:.3f} n={tres[t]['n']}")

if __name__=="__main__": main()
