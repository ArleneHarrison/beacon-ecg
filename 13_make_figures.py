#!/usr/bin/env python3
"""Publication figures from real results (baseline + Table1 + mortality feature-sets)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/data/s01011/cardio_m3t/aiecg/outputs"; FIG=f"{OUT}/figures"; import os; os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"figure.dpi":140,"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
C=["#2f6f9f","#3b8a6e","#c0603a","#7a5aa0"]

base=json.load(open(f"{OUT}/baseline_metrics.json"))
an=json.load(open(f"{OUT}/analyze_now.json"))
df=pd.read_parquet(f"{OUT}/cohort_labels.parquet"); idx=df[df.is_index_ecg==1]

# Fig 1: baseline AUROC with 95% CI
fig,ax=plt.subplots(figsize=(5,3.2))
tg=[("hfref_le40","HFrEF\n(LVEF≤40)"),("as_severe","Severe AS"),("dead_365d","1-yr mortality")]
xs=range(len(tg))
for i,(k,_) in enumerate(tg):
    m=base[k]["histgb"]; lo,hi=m["auroc_ci"]
    ax.errorbar(i,m["auroc"],yerr=[[m["auroc"]-lo],[hi-m["auroc"]]],fmt="o",color=C[i],capsize=4,ms=8)
    ax.text(i,hi+0.008,f"{m['auroc']:.3f}",ha="center",fontsize=9)
ax.set_xticks(list(xs)); ax.set_xticklabels([t for _,t in tg]); ax.set_ylabel("AUROC (test)")
ax.axhline(0.5,ls=":",c="gray"); ax.set_ylim(0.5,0.9); ax.set_title("Tabular ECG baseline (machine-measurements + demographics)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_baseline_auroc.png"); plt.close()

# Fig 2: mortality feature-set incremental
fig,ax=plt.subplots(figsize=(5.2,3.2))
order=["demo_only","echo_only","ecg_mm_only","echo+ecg","echo+ecg+demo"]
lab=["Demo","Echo","ECG","Echo+ECG","Echo+ECG+Demo"]
vals=[an["mortality_1yr"]["auroc"][k] for k in order]
bars=ax.bar(lab,vals,color=["#999","#2f6f9f","#3b8a6e","#c0603a","#7a5aa0"])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.004,f"{v:.3f}",ha="center",fontsize=9)
ax.set_ylim(0.5,0.75); ax.set_ylabel("1-yr mortality AUROC (5-fold CV)")
ax.set_title(f"Incremental prognostic value (NRI +{an['mortality_1yr']['continuous_NRI_ecg_added_to_echo']:.2f} adding ECG to echo)")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(f"{FIG}/fig2_mortality_incremental.png"); plt.close()

# Fig 3: LVEF distribution
fig,ax=plt.subplots(figsize=(5,3))
v=idx["lvef_value"].dropna()
ax.hist(v,bins=40,color="#2f6f9f",alpha=.85)
ax.axvline(40,ls="--",c="#c0603a",lw=2); ax.text(41,ax.get_ylim()[1]*.9,"HFrEF ≤ 40",color="#c0603a")
ax.set_xlabel("LVEF (%)"); ax.set_ylabel("patients"); ax.set_title(f"Echo LVEF distribution (n={len(v):,})")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_lvef_dist.png"); plt.close()

# Fig 4: label prevalence
fig,ax=plt.subplots(figsize=(4.6,3))
labs=["HFrEF","Severe AS","LVH","30d death","1y death"]
pv=[100*idx.hfref_le40.mean(),100*idx.as_severe.mean(),100*idx.lvh.mean(),100*idx.dead_30d.mean(),100*idx.dead_365d.mean()]
ax.barh(labs[::-1],pv[::-1],color="#3b8a6e")
for i,v in enumerate(pv[::-1]): ax.text(v+0.3,i,f"{v:.1f}%",va="center",fontsize=9)
ax.set_xlabel("prevalence (%)"); ax.set_title(f"Label prevalence (n={len(idx):,} patients)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_prevalence.png"); plt.close()
print("figures written to", FIG)
import os; print("\n".join(sorted(os.listdir(FIG))))
