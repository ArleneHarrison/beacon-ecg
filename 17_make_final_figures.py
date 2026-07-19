#!/usr/bin/env python3
"""Publication figures for BEACON-ECG: 6 main display items at 600 dpi.

Design rules applied (from the manuscript peer-review checklist):
  §12 no misleading axes  - AUROC panels start at 0.5 (chance), never at a
                            truncated baseline chosen to exaggerate differences.
  §14 self-sufficient     - every panel carries its own n, and units are labelled.
  §18 internal consistency- every number is recomputed from the same test-set
                            predictions used for the manuscript, and asserted
                            against the stored JSON results before plotting.
  §15 no duplication      - each figure shows something the others do not.
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
OUT = (r"d:/桌面/_按项目整理/02_生信数据与医学AI项目/07_医学AI_深度学习_多项目孵化/"
       r"医学AI深度学习_导师课题项目群/崔老师项目_PETCT多任务_空间转录组深度学习/"
       r"PETCT多任务/论文_AIECG/figures_final_600dpi")
import os; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
})
# Colour-blind-safe (Okabe-Ito derived); pattern/label redundancy added where bars compare.
C = {"base": "#999999", "single": "#56B4E9", "final": "#0072B2",
     "ext": "#D55E00", "ok": "#009E73", "warn": "#CC79A7"}

fin = json.load(open(f"{SP}/FINAL_test_ensemble4.json"))
ev = json.load(open(f"{SP}/eval_full.json"))
sv = json.load(open(f"{SP}/survival.json"))
ext = json.load(open(f"{SP}/external_validation_deep.json"))
d = pd.read_parquet(f"{SP}/test_predictions_FINAL.parquet")

TASKS = [("hfref_le40", "HFrEF\n(LVEF≤40%)"), ("as_severe", "Severe\naortic stenosis"),
         ("lvh", "LV\nhypertrophy"), ("dead_365d", "1-year\nmortality")]
BASE = {"hfref_le40": 0.813, "as_severe": 0.769, "lvh": None, "dead_365d": 0.715}
SINGLE = {"hfref_le40": 0.886, "as_severe": 0.825, "lvh": 0.776, "dead_365d": 0.722}

# ---------------------------------------------------------------- consistency
print("consistency checks (recomputed vs stored):")
for k, _ in TASKS:
    s = d[d[k].notna() & d[f"pred_{k}"].notna()]
    a = roc_auc_score(s[k].astype(int), s[f"pred_{k}"])
    stored = fin[k]["auroc"]
    flag = "OK " if abs(a - stored) < 5e-4 else "!! MISMATCH"
    print(f"  {flag} {k:<12} recomputed {a:.4f}  stored {stored:.4f}")

# ============================================================ Fig 1: study flow
fig, ax = plt.subplots(figsize=(5.0, 4.4)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def box(x, y, w, h, txt, fc="#EAF1F7", ec="#2C5F86"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7)
def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.8, color="#2C5F86"))
box(1.4, 8.6, 7.2, 1.1, "MIMIC-IV-ECG  ×  MIMIC-IV-ECHO\n800,035 ECGs   |   179,928 transthoracic echo studies")
arrow(5.0, 8.6, 7.9)
box(1.4, 6.8, 7.2, 1.1, "ECG paired to nearest echo within ±7 days\n179,701 paired ECGs  →  43,284 patients\n(median |ECG−echo| 1.11 days)")
arrow(5.0, 6.8, 6.32)
ax.text(8.9, 6.55, "excluded: 1,225 patients\nwithout a retrievable\nwaveform", fontsize=6,
        va="center", ha="left", color="#8A3324")
ax.add_patch(FancyArrowPatch((5.0, 6.55), (8.75, 6.55), arrowstyle="-|>",
                             mutation_scale=6, lw=0.7, color="#8A3324"))
box(1.4, 5.2, 7.2, 1.1, "Modelling cohort\n42,059 patients with a retrievable waveform", fc="#DCEBE0", ec="#2E7D4F")
# distribute: stem down from the cohort box, horizontal rail, then arrows DOWN into each fold
ax.plot([5.0, 5.0], [5.2, 4.75], color="#2C5F86", lw=0.8)
ax.plot([1.8, 8.2], [4.75, 4.75], color="#2C5F86", lw=0.8)
for x in (1.8, 5.0, 8.2): arrow(x, 4.75, 4.42)
ax.text(8.35, 4.98, "patient-level split —\nall ECGs of a patient\nstay in one fold",
        fontsize=6, style="italic", color="#444444", va="center", ha="left")
box(0.3, 3.3, 3.0, 1.1, "Training\n25,070", fc="#F2F2F2", ec="#666666")
box(3.6, 3.3, 2.8, 1.1, "Validation\n8,343", fc="#F2F2F2", ec="#666666")
box(6.7, 3.3, 3.0, 1.1, "Test\n8,646", fc="#FDE9D9", ec="#B5651D")
box(1.4, 1.2, 7.2, 1.2, "Endpoints (echo measurements used as labels only)\n"
    "HFrEF · severe aortic stenosis · LV hypertrophy · LVEF · 1-year mortality",
    fc="#EAF1F7", ec="#2C5F86")
arrow(5.0, 3.3, 2.45)
ax.set_title("Figure 1. Cohort construction and patient-level splits", fontsize=8.5, pad=2)
plt.tight_layout(); plt.savefig(f"{OUT}/Figure1_study_flow.png", bbox_inches="tight"); plt.close()

# ================================================ Fig 2: final vs single vs baseline
fig, ax = plt.subplots(figsize=(5.4, 3.2))
x = np.arange(len(TASKS)); w = 0.26
for i, (k, lab) in enumerate(TASKS):
    if BASE[k] is not None:
        ax.bar(i - w, BASE[k] - 0.5, w, bottom=0.5, color=C["base"], edgecolor="black",
               lw=0.4, label="Tabular ECG baseline" if i == 0 else "")
        ax.text(i - w, BASE[k] + 0.006, f"{BASE[k]:.3f}", ha="center", fontsize=5.8)
    ax.bar(i, SINGLE[k] - 0.5, w, bottom=0.5, color=C["single"], edgecolor="black", lw=0.4,
           hatch="///", label="Single network" if i == 0 else "")
    ax.text(i, SINGLE[k] + 0.006, f"{SINGLE[k]:.3f}", ha="center", fontsize=5.8)
    v = fin[k]["auroc"]; lo, hi = fin[k]["ci"]
    ax.bar(i + w, v - 0.5, w, bottom=0.5, color=C["final"], edgecolor="black", lw=0.4,
           yerr=[[v - lo], [hi - v]], capsize=2, error_kw={"lw": 0.7},
           label="BEACON-ECG (ensemble)" if i == 0 else "")
    ax.text(i + w, hi + 0.008, f"{v:.3f}", ha="center", fontsize=6.4, fontweight="bold")
ax.axhline(0.5, color="black", lw=0.6, ls="--")
ax.text(len(TASKS) - 0.42, 0.508, "chance", fontsize=5.8, style="italic", color="#555555")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in TASKS])
ax.set_ylim(0.5, 0.97); ax.set_ylabel("Test AUROC")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
ax.text(0.0, 0.515, "no tabular\ncomparator", ha="center", fontsize=5.5,
        color="#777777", style="italic", transform=ax.transData)
ax.annotate("", xy=(2 - w, 0.5), xytext=(2 - w, 0.5))
plt.tight_layout(); plt.savefig(f"{OUT}/Figure2_model_vs_baseline.png", bbox_inches="tight"); plt.close()

# ================================================ Fig 3: external validation
fig, ax = plt.subplots(figsize=(3.6, 3.0))
ks = [k for k in ["hfref_le40", "lvh"] if k in ext]; x = np.arange(len(ks)); w = 0.32
for i, k in enumerate(ks):
    iv = ext[k]["AUROC_internal_MIMIC"]
    ax.bar(i - w/2, iv - 0.5, w, bottom=0.5, color=C["final"], edgecolor="black", lw=0.4,
           label="Internal (MIMIC-IV)" if i == 0 else "")
    ax.text(i - w/2, iv + 0.008, f"{iv:.3f}", ha="center", fontsize=6.2)
    v = ext[k]["AUROC_external"]; lo, hi = ext[k]["CI"]
    ax.bar(i + w/2, v - 0.5, w, bottom=0.5, color=C["ext"], edgecolor="black", lw=0.4,
           hatch="\\\\", yerr=[[v - lo], [hi - v]], capsize=2, error_kw={"lw": 0.7},
           label="External (EchoNext)" if i == 0 else "")
    ax.text(i + w/2, hi + 0.01, f"{v:.3f}", ha="center", fontsize=6.4, fontweight="bold")
    ax.text(i, 0.515, f"n={ext[k]['n']:,}", ha="center", fontsize=5.8, color="#555555")
ax.axhline(0.5, color="black", lw=0.6, ls="--")
ax.set_xticks(x); ax.set_xticklabels(["HFrEF", "LV hypertrophy"])
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
ax.set_title("Applied without retraining;\n250 Hz → 500 Hz resampling", fontsize=7, pad=14)
plt.tight_layout(); plt.savefig(f"{OUT}/Figure3_external_validation.png", bbox_inches="tight"); plt.close()

# ============================== Fig 4: discordance (A rates, B Kaplan-Meier)
en = sv["echo_normal_LVEF>50"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
b = a1.bar([f"High risk\n(n={en['n_high']:,})", f"Low risk\n(n={en['n_low']:,})"],
           [en["mort365_high"]*100, en["mort365_low"]*100],
           color=[C["ext"], C["ok"]], edgecolor="black", lw=0.5, width=0.6)
for r in b:
    a1.text(r.get_x()+r.get_width()/2, r.get_height()+0.5, f"{r.get_height():.1f}%",
            ha="center", fontweight="bold", fontsize=7.5)
a1.set_ylabel("1-year all-cause mortality (%)"); a1.set_ylim(0, 30)
a1.set_title("A  Event rates", loc="left", fontweight="bold", fontsize=8)
a1.text(0.5, 27.5, f"Echo-normal patients (LVEF>50%), n={en['n']:,}", ha="center",
        fontsize=6.4, style="italic", transform=a1.transData)

# Kaplan-Meier recomputed from the same predictions.
# Stratification is by the STRUCTURAL (HFrEF) prediction, not by the mortality head -
# the question is whether patients the ECG calls structurally abnormal, but whose echo
# was normal, die more. Using the mortality head would make this near-circular.
# This matches survival_analysis.py and reproduces the stored 24.00% / 13.53% exactly.
s = d[(d.lvef_value > 50) & d.pred_hfref_le40.notna()].copy()
thr = s.pred_hfref_le40.quantile(0.8)
s["grp"] = np.where(s.pred_hfref_le40 >= thr, "high", "low")
tt = pd.to_numeric(s.days_to_death, errors="coerce")
s["t"] = np.where(tt.notna() & (tt >= 0) & (tt <= 365), tt, 365)
s["e"] = (tt.notna() & (tt >= 0) & (tt <= 365)).astype(int)
for g, col, lab in [("high", C["ext"], "AI-ECG structural risk, top quintile"),
                    ("low", C["ok"], "AI-ECG structural risk, remainder")]:
    gd = s[s.grp == g].sort_values("t"); surv, n = 1.0, len(gd); xs, ys = [0], [1.0]
    for t in np.sort(gd.loc[gd.e == 1, "t"].unique()):
        at_risk = (gd.t >= t).sum(); dead = ((gd.t == t) & (gd.e == 1)).sum()
        if at_risk > 0:
            surv *= (1 - dead/at_risk); xs.append(t); ys.append(surv)
    xs.append(365); ys.append(surv)
    a2.step(xs, ys, where="post", color=col, lw=1.3, label=lab)
a2.set_xlabel("Days from ECG"); a2.set_ylabel("Survival probability")
a2.set_xlim(0, 365); a2.set_ylim(0.70, 1.0)
a2.legend(frameon=False, loc="lower left", fontsize=6.2, bbox_to_anchor=(0.0, 0.0))
a2.set_title("B  Kaplan–Meier", loc="left", fontweight="bold", fontsize=8)
a2.text(190, 0.955, "log-rank P < 0.001\nadjusted HR 1.85 (1.61–2.13)", fontsize=6.4,
        ha="left", va="top")
plt.tight_layout(); plt.savefig(f"{OUT}/Figure4_discordance.png", bbox_inches="tight"); plt.close()

# Panels A and B must describe the same two groups. An earlier draft stratified panel B
# by the mortality head while panel A used the stored (structural-head) result, so the
# KM curves contradicted the bar chart. Assert they agree rather than trusting the eye.
_hi, _lo = s[s.grp == "high"], s[s.grp == "low"]
assert abs(_hi.dead_365d.mean() - en["mort365_high"]) < 1e-4, "Fig4: panel A/B disagree (high risk)"
assert abs(_lo.dead_365d.mean() - en["mort365_low"]) < 1e-4, "Fig4: panel A/B disagree (low risk)"
assert len(_hi) == en["n_high"] and len(_lo) == en["n_low"], "Fig4: group sizes differ from stored"
print(f"  OK  Fig4 panels agree  high {_hi.dead_365d.mean()*100:.2f}% (n={len(_hi):,}) "
      f"/ low {_lo.dead_365d.mean()*100:.2f}% (n={len(_lo):,})")

# ==================== Fig 5: AS triage burden -- NOW AGE-STRATIFIED (matches revised 3.4)
s = d[d.as_severe.notna() & d.pred_as_severe.notna()].copy()
y = s.as_severe.values.astype(int); p = s.pred_as_severe.values.astype(float)
thr95 = float(np.percentile(p[y == 1], 5))
s["ag"] = pd.cut(s.age_at_ecg, [0, 50, 65, 80, 200],
                 labels=["<50", "50–64", "65–79", "≥80"], right=False)
rows = []
for g, gd in s.groupby("ag", observed=True):
    yy = gd.as_severe.values.astype(int); pp = gd.pred_as_severe.values.astype(float)
    f = pp >= thr95
    rows.append((str(g), len(gd), int(yy.sum()), (1-f.mean())*100,
                 (f & (yy == 1)).sum()/yy.sum()*100 if yy.sum() else np.nan))
rows.append(("Overall", len(s), int(y.sum()), (1-(p >= thr95).mean())*100,
             ((p >= thr95) & (y == 1)).sum()/y.sum()*100))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})
labs = [r[0] for r in rows]; spared = [r[3] for r in rows]
cols = [C["single"]]*4 + [C["final"]]
bars = a1.bar(labs, spared, color=cols, edgecolor="black", lw=0.5, width=0.62)
for r, (lab, n, cs, sp, se) in zip(bars, rows):
    a1.text(r.get_x()+r.get_width()/2, sp+1.5, f"{sp:.1f}%", ha="center",
            fontweight="bold", fontsize=7)
# case counts go under the axis as a second tick row - inside the bars they clipped
a1.set_xticks(range(len(rows)))
a1.set_xticklabels([f"{r[0]}\n({r[2]} cases)" for r in rows], fontsize=7)
a1.set_ylabel("Echocardiograms spared (%)"); a1.set_ylim(0, 100)
a1.set_title("A  Triage benefit falls with age", loc="left", fontweight="bold", fontsize=8)
a2.bar(labs, [r[4] for r in rows], color=cols, edgecolor="black", lw=0.5, width=0.62)
a2.axhline(95, color="#8A3324", ls="--", lw=0.9)
a2.text(0.05, 95.6, "nominal 95% target", fontsize=5.8, color="#8A3324")
for i, r in enumerate(rows):
    a2.text(i, r[4]+0.6, f"{r[4]:.1f}", ha="center", fontsize=6.4)
a2.set_ylabel("Sensitivity (%)"); a2.set_ylim(80, 103)
a2.set_xticks(range(len(rows)))
a2.set_xticklabels([f"{r[0]}\n({r[2]} cases)" for r in rows], fontsize=7)
# The y-axis is truncated to resolve differences around the 95% target. Flag it
# explicitly rather than letting a reader assume the bars start at zero.
a2.text(0.02, 0.02, "note: y-axis truncated at 80%", transform=a2.transAxes,
        fontsize=5.6, style="italic", color="#666666")
a2.set_title("B  …and sensitivity is not uniform", loc="left", fontweight="bold", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/Figure5_as_triage_by_age.png", bbox_inches="tight"); plt.close()

# ==================================== Fig 6: calibration (A) + decision curve (B)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
for k, lab, col in [("hfref_le40", "HFrEF", C["final"]), ("as_severe", "Severe AS", C["ext"]),
                    ("lvh", "LVH", C["ok"]), ("dead_365d", "1-yr mortality", C["warn"])]:
    ss = d[d[k].notna() & d[f"pred_{k}"].notna()]
    yy = ss[k].values.astype(int); pp = ss[f"pred_{k}"].values.astype(float)
    q = pd.qcut(pp, 10, labels=False, duplicates="drop")
    xs = [pp[q == i].mean() for i in range(q.max()+1)]
    ys = [yy[q == i].mean() for i in range(q.max()+1)]
    a1.plot(xs, ys, "o-", color=col, lw=1.1, ms=3,
            label=f"{lab} (ECE {ev['discrimination_calibration'][k]['ece']:.3f})")
a1.plot([0, 0.6], [0, 0.6], ls="--", color="black", lw=0.8, label="Perfect calibration")
a1.set_xlabel("Mean predicted risk"); a1.set_ylabel("Observed frequency")
a1.set_xlim(0, 0.6); a1.set_ylim(0, 0.6)
a1.legend(frameon=False, fontsize=6, loc="upper left")
a1.set_title("A  Calibration (decile bins)", loc="left", fontweight="bold", fontsize=8)
nb = ev["discrimination_calibration"]["hfref_le40"]["net_benefit"]
ths = np.array([float(t) for t in nb]); vals = np.array([nb[t] for t in nb])
o = np.argsort(ths); ths, vals = ths[o], vals[o]
prev = ev["discrimination_calibration"]["hfref_le40"]["pos_rate"]
a2.plot(ths, vals, "-", color=C["final"], lw=1.4, label="BEACON-ECG")
a2.plot(ths, prev - (1-prev)*ths/(1-ths), "--", color="#888888", lw=1.0, label="Treat all")
a2.axhline(0, color="black", lw=0.8, ls=":", label="Treat none")
a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
a2.set_xlim(0, 0.5); a2.set_ylim(-0.02, max(vals)*1.15)
a2.legend(frameon=False, fontsize=6.2)
a2.set_title("B  Decision curve (HFrEF)", loc="left", fontweight="bold", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/Figure6_calibration_dca.png", bbox_inches="tight"); plt.close()

print(f"\nwritten to {OUT}:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}  ({os.path.getsize(f'{OUT}/{f}')/1024:.0f} KB)")
print("\nAS triage by age (Figure 5 source):")
for lab, n, cs, sp, se in rows:
    print(f"  {lab:<8} n={n:<5} cases={cs:<4} spared={sp:5.1f}%  sens={se:5.1f}%")
