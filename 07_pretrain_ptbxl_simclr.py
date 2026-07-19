#!/usr/bin/env python3
"""Self-supervised (SimCLR/NT-Xent) pre-training of the ECG encoder on PTB-XL (21,799 x 500Hz).
Encoder = same stem/body/head as the downstream multitask model, so weights transfer directly.
Outputs encoder weights loadable by train_waveform.py's Net."""
import argparse, glob, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, wfdb
from torch.utils.data import Dataset, DataLoader
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# ---------- encoder (identical to downstream Net stem/body/head) ----------
class SEBlock(nn.Module):
    def __init__(s,c,r=8): super().__init__(); s.fc1=nn.Linear(c,c//r); s.fc2=nn.Linear(c//r,c)
    def forward(s,x): z=x.mean(-1); z=torch.relu(s.fc1(z)); z=torch.sigmoid(s.fc2(z)); return x*z.unsqueeze(-1)
class ResBlock(nn.Module):
    def __init__(s,ci,co,st=2,k=7):
        super().__init__(); s.c1=nn.Conv1d(ci,co,k,st,k//2); s.b1=nn.BatchNorm1d(co)
        s.c2=nn.Conv1d(co,co,k,1,k//2); s.b2=nn.BatchNorm1d(co); s.se=SEBlock(co); s.drop=nn.Dropout(0.1)
        s.sc=nn.Sequential(nn.Conv1d(ci,co,1,st),nn.BatchNorm1d(co)) if (st!=1 or ci!=co) else nn.Identity()
    def forward(s,x): r=s.sc(x); x=torch.relu(s.b1(s.c1(x))); x=s.b2(s.c2(x)); x=s.se(x); return torch.relu(s.drop(x)+r)
class Encoder(nn.Module):
    def __init__(s):
        super().__init__(); s.stem=nn.Sequential(nn.Conv1d(12,64,15,2,7),nn.BatchNorm1d(64),nn.ReLU())
        s.body=nn.Sequential(ResBlock(64,64,1),ResBlock(64,128),ResBlock(128,196),ResBlock(196,256),ResBlock(256,256))
        s.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2))
    def forward(s,x): x=s.body(s.stem(x)); x=x.mean(-1); return s.head(x)
class SimCLR(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=Encoder(); s.proj=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,128))
    def forward(s,x): return F.normalize(s.proj(s.enc(x)),dim=1)

# ---------- data ----------
def read_wave(fp):
    rec=wfdb.rdrecord(fp); sig=rec.p_signal.astype(np.float32)
    n2i={n.upper():i for i,n in enumerate(rec.sig_name)}; out=np.zeros((12,5000),np.float32); T=min(sig.shape[0],5000)
    for j,ln in enumerate(LEADS):
        i=n2i.get(ln.upper())
        if i is not None: out[j,:T]=np.nan_to_num(sig[:T,i],nan=0.0)
    mu=out.mean(1,keepdims=True); sd=out.std(1,keepdims=True)+1e-6
    return ((out-mu)/sd).astype(np.float16)

def pack(root,packfile):
    heas=sorted(glob.glob(os.path.join(root,"**","records500","**","*.hea"),recursive=True))
    print(f"found {len(heas)} PTB-XL records",flush=True)
    W=np.lib.format.open_memmap(packfile,mode="w+",dtype=np.float16,shape=(len(heas),12,5000))
    ok=0
    for i,h in enumerate(heas):
        try: W[i]=read_wave(h[:-4]); ok+=1
        except Exception: pass
        if i%5000==0: print(f"  packed {i}/{len(heas)}",flush=True)
    print(f"packed ok={ok}",flush=True); return packfile

def augment(x):
    """strong ECG augmentation for contrastive views; x: (12,5000) float32"""
    x=x.copy()
    if np.random.rand()<0.8: x=np.roll(x,np.random.randint(-500,500),axis=1)          # time shift
    if np.random.rand()<0.8: x=x*np.random.uniform(0.7,1.3)                            # amplitude
    if np.random.rand()<0.5:                                                           # lead dropout
        for _ in range(np.random.randint(1,4)): x[np.random.randint(0,12)]=0.0
    if np.random.rand()<0.5:                                                           # temporal masking
        for _ in range(np.random.randint(1,4)):
            st=np.random.randint(0,4500); ln=np.random.randint(100,500); x[:,st:st+ln]=0.0
    if np.random.rand()<0.5: x=x+np.random.normal(0,0.05,x.shape).astype(np.float32)   # noise
    if np.random.rand()<0.4:                                                           # baseline wander
        t=np.linspace(0,1,x.shape[1],dtype=np.float32)
        x=x+0.1*np.sin(2*np.pi*np.random.uniform(0.05,0.5)*t+np.random.rand()*6.28)[None,:]
    return x

class Views(Dataset):
    def __init__(s,W): s.W=W
    def __len__(s): return len(s.W)
    def __getitem__(s,i):
        x=np.asarray(s.W[i],np.float32)
        return torch.from_numpy(augment(x)), torch.from_numpy(augment(x))

def nt_xent(z1,z2,t=0.2):
    # autocast would force the matmul to fp16 (mask value overflows), so disable it here
    with torch.amp.autocast('cuda', enabled=False):
        B=z1.shape[0]; z=torch.cat([z1.float(),z2.float()],0)
        sim=(z@z.T)/t
        sim.fill_diagonal_(-1e9)
        tgt=torch.cat([torch.arange(B,2*B),torch.arange(0,B)]).to(z.device)
        return F.cross_entropy(sim,tgt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--pack",default=""); ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--bs",type=int,default=256); ap.add_argument("--lr",type=float,default=1e-3)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); dev="cuda"
    packfile=a.pack or os.path.join(a.out,"ptbxl_waves.npy")
    if not os.path.exists(packfile): pack(a.root,packfile)
    W=np.load(packfile,mmap_mode="r"); print("pretrain samples:",len(W),flush=True)
    dl=DataLoader(Views(W),a.bs,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    net=SimCLR().to(dev)
    opt=torch.optim.AdamW(net.parameters(),a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,epochs=a.epochs,steps_per_epoch=len(dl),pct_start=0.1)
    scaler=torch.amp.GradScaler('cuda'); best=1e9; hist=[]
    for ep in range(a.epochs):
        net.train(); tot=0; n=0
        for v1,v2 in dl:
            v1,v2=v1.to(dev,non_blocking=True),v2.to(dev,non_blocking=True); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss=nt_xent(net(v1),net(v2))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot+=loss.item(); n+=1
        m=tot/max(n,1); hist.append(m)
        print(f"ep{ep} contrastive_loss={m:.4f}",flush=True)
        if m<best:
            best=m
            torch.save(net.enc.state_dict(), os.path.join(a.out,"ptbxl_encoder.pt"))
    json.dump({"best_loss":best,"epochs":a.epochs,"n_samples":int(len(W)),"loss_history":hist},
              open(os.path.join(a.out,"pretrain_summary.json"),"w"),indent=2)
    print("PRETRAIN DONE best_loss=%.4f -> %s/ptbxl_encoder.pt"%(best,a.out),flush=True)
if __name__=="__main__": main()
