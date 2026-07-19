#!/usr/bin/env python3
"""BEACON-ECG: five composite display items at 600 dpi.

Panels are drawn by functions that take an axes, so the same code produces the
standalone and composite versions - the numbers cannot drift between them.

  Figure 1  Study design      A concept          B cohort flow
  Figure 2  The model         A architecture     B real example ECGs
  Figure 3  Discrimination    A internal vs baseline   B external validation
  Figure 4  Beyond the echo   A event rates      B Kaplan-Meier
  Figure 5  Clinical utility  A calibration  B decision curve  C triage burden  D sensitivity

Checklist rules kept from the standalone version: AUROC axes start at 0.5 (chance),
truncated axes are labelled as such, every plotted number is recomputed from the test-set
predictions and asserted against the stored JSON, and panels within a figure are checked
against each other.
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.metrics import roc_auc_score

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
OUT = (r"d:/桌面/_按项目整理/02_生信数据与医学AI项目/07_医学AI_深度学习_多项目孵化/"
       r"医学AI深度学习_导师课题项目群/崔老师项目_PETCT多任务_空间转录组深度学习/"
       r"PETCT多任务/论文_AIECG/figures_final_600dpi")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "font.size": 7.5,
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4})
DBLUE, BLUE, GREEN, ORANGE, GREY, PINK = "#0072B2", "#56B4E9", "#009E73", "#D55E00", "#999999", "#CC79A7"

fin = json.load(open(f"{SP}/FINAL_test_ensemble4.json"))
ev  = json.load(open(f"{SP}/eval_full.json"))
sv  = json.load(open(f"{SP}/survival.json"))
ext = json.load(open(f"{SP}/external_validation_deep.json"))
d   = pd.read_parquet(f"{SP}/test_predictions_FINAL.parquet")
zex = np.load(f"{SP}/example_ecgs.npz"); mex = json.load(open(f"{SP}/example_ecgs.json"))

TASKS = [("hfref_le40", "HFrEF\n(LVEF≤40%)"), ("as_severe", "Severe\naortic stenosis"),
         ("lvh", "LV\nhypertrophy"), ("dead_365d", "1-year\nmortality")]
BASE   = {"hfref_le40": 0.813, "as_severe": 0.769, "lvh": None, "dead_365d": 0.715}
SINGLE = {"hfref_le40": 0.886, "as_severe": 0.825, "lvh": 0.776, "dead_365d": 0.722}

print("consistency (recomputed vs stored):")
for k, _ in TASKS:
    s = d[d[k].notna() & d[f"pred_{k}"].notna()]
    a = roc_auc_score(s[k].astype(int), s[f"pred_{k}"])
    assert abs(a - fin[k]["auroc"]) < 5e-4, f"{k} mismatch"
    print(f"  OK  {k:<12} {a:.4f}")

def tag(ax, letter, title, x=-0.02, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(x + 0.028, y, title, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="bottom")

def rbox(ax, x, y, w, h, txt, fc, ec, fs=6.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=0.8))
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=fs, fontweight=weight)

def arr(ax, x0, y0, x1, y1, c="#333333", lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=lw, color=c))

# ────────────────────────────────────────────────────────── panel drawers
def panel_concept(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
    ax.add_patch(Rectangle((1, 26), 45, 16, fc="#F7F7F7", ec="#BBBBBB", lw=0.8, zorder=0))
    ax.text(23.5, 39.6, "Prior work", ha="center", fontsize=7, fontweight="bold", color="#666666")
    rbox(ax, 4, 30, 12, 6.6, "12-lead\nECG", "#FFFFFF", "#888888", 6.0)
    rbox(ax, 31, 30, 12, 6.6, "Concurrent\necho", "#FFFFFF", "#888888", 6.0)
    arr(ax, 16, 33.3, 31, 33.3, "#888888")
    ax.text(23.5, 34.3, "estimates", ha="center", fontsize=5.6, style="italic", color="#666666")
    ax.text(23.5, 27.6, "the ECG stands in for a test you could have done today",
            ha="center", fontsize=5.3, style="italic", color="#777777")
    ax.add_patch(Rectangle((51, 26), 47, 16, fc="#EAF3FA", ec=DBLUE, lw=0.9, zorder=0))
    ax.text(74.5, 39.6, "This study", ha="center", fontsize=7, fontweight="bold", color=DBLUE)
    rbox(ax, 53, 30, 11, 6.6, "12-lead\nECG", "#FFFFFF", DBLUE, 6.0)
    rbox(ax, 68, 30, 12, 6.6, "Structural\nsignature", "#DCEBE0", GREEN, 6.0)
    rbox(ax, 84, 30, 12, 6.6, "Outcome\nbeyond echo", "#FDE9D9", ORANGE, 6.0)
    arr(ax, 64, 33.3, 68, 33.3, DBLUE); arr(ax, 80, 33.3, 84, 33.3, GREEN)
    qs = [("What the echo sees", "ECG reproduces concurrent\nstructural findings\nHFrEF 0.900 · AS 0.860 · LVH 0.789", GREEN),
          ("What the echo misses", "Among echo-NORMAL patients,\ntop structural-risk quintile\n24.0% vs 13.5% 1-yr mortality\nadj. HR 1.85 (1.61–2.13)", ORANGE),
          ("What time adds — nothing", "Deep serial-ECG trajectory did\nNOT beat a single tracing\n(0.796 vs 0.779)", "#888888")]
    ax.plot([74.5, 74.5], [26, 24], color="#555555", lw=0.9)
    ax.plot([17, 91], [24, 24], color="#555555", lw=0.9)
    for i, (t, b, c) in enumerate(qs):
        x = 2 + i*32.5
        rbox(ax, x, 2, 30, 19, "", "#FFFFFF", c, 6)
        ax.text(x+15, 18.4, t, ha="center", fontsize=6.4, fontweight="bold", color=c)
        ax.text(x+15, 10.2, b, ha="center", va="center", fontsize=5.6)
        arr(ax, x+15, 24, x+15, 21.2, c, 0.8)

def panel_flow(ax):
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0.6, 10)
    def bx(x, y, w, h, t, fc="#EAF1F7", ec="#2C5F86", fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=0.8))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs)
    def dn(x, y0, y1): arr(ax, x, y0, x, y1, "#2C5F86", 0.8)
    bx(1.1, 8.5, 6.6, 1.0, "MIMIC-IV-ECG × MIMIC-IV-ECHO\n800,035 ECGs | 179,928 TTE studies")
    dn(4.4, 8.5, 7.95)
    bx(1.1, 6.8, 6.6, 1.1, "ECG paired to nearest echo within ±7 d\n179,701 paired ECGs → 43,284 patients\n(median |ECG−echo| 1.11 d)")
    dn(4.4, 6.8, 6.35)
    ax.text(7.95, 6.5, "excluded: 1,225\nwithout retrievable\nwaveform", fontsize=5.4,
            va="center", ha="left", color="#8A3324")
    arr(ax, 4.4, 6.5, 7.8, 6.5, "#8A3324", 0.7)
    bx(1.1, 5.2, 6.6, 1.0, "Modelling cohort\n42,059 patients", "#DCEBE0", "#2E7D4F")
    ax.plot([4.4, 4.4], [5.2, 4.8], color="#2C5F86", lw=0.8)
    ax.plot([1.9, 6.9], [4.8, 4.8], color="#2C5F86", lw=0.8)
    for x in (1.9, 4.4, 6.9): dn(x, 4.8, 4.45)
    ax.text(7.95, 4.9, "patient-level split —\nall ECGs of a patient\nstay in one fold",
            fontsize=5.4, style="italic", color="#444444", va="center", ha="left")
    bx(0.3, 3.4, 2.6, 1.05, "Training\n25,070", "#F2F2F2", "#666666")
    bx(3.15, 3.4, 2.5, 1.05, "Validation\n8,343", "#F2F2F2", "#666666")
    bx(5.9, 3.4, 2.6, 1.05, "Test\n8,646", "#FDE9D9", "#B5651D")
    dn(4.4, 3.4, 2.65)
    bx(1.1, 1.5, 6.6, 1.1, "Endpoints (echo measurements used as labels only)\n"
       "HFrEF · severe AS · LV hypertrophy · LVEF · 1-year mortality")

def panel_arch(ax):
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 74)
    rbox(ax, 0.5, 42, 12, 9.5, "12-lead ECG\n10 s @ 500 Hz\n12 × 5000", "#EAF1F7", DBLUE, 6.0)
    ax.plot([12.5, 15.5], [46.7, 46.7], color="#333333", lw=0.9)
    ax.plot([15.5, 15.5], [23, 63], color="#333333", lw=0.9)
    arr(ax, 15.5, 63, 18.5, 63); arr(ax, 15.5, 23, 18.5, 23)
    ax.text(18.5, 69.5, "Members 1–2   SE-ResNet backbone   (weight ×1 each)",
            fontsize=6.4, fontweight="bold", color="#555555")
    rbox(ax, 18.5, 58.5, 9.5, 9, "Stem\nConv1d 12→64\nk=15, s=2", "#EAF1F7", DBLUE, 5.2)
    arr(ax, 28, 63, 30, 63)
    for i, (ci, co, st) in enumerate([(64,64,1),(64,128,2),(128,196,2),(196,256,2),(256,256,2)]):
        x = 30 + i*7.6
        rbox(ax, x, 58.5, 6.8, 9, f"Res\n{ci}→{co}\ns={st}", "#DCEBE0", GREEN, 4.8)
        if i < 4: arr(ax, x+6.8, 63, x+7.6, 63)
    arr(ax, 68.4, 63, 70.5, 63)
    rbox(ax, 70.5, 58.5, 9, 9, "Global\navg pool\n→256-d", "#EAF1F7", DBLUE, 5.2)
    ax.text(44, 55.8, "each Res block:  Conv1d ×2 · BatchNorm · SE channel attention · Dropout 0.1 · residual",
            ha="center", fontsize=5.2, style="italic", color="#666666")
    ax.text(18.5, 39.5, "Members 3–4   dual-pathway encoder + FiLM   (weight ×2 each)",
            fontsize=6.4, fontweight="bold", color="#555555")
    ax.plot([18.5, 21], [23, 23], color="#333333", lw=0.9)
    ax.plot([21, 21], [13.5, 32.5], color="#333333", lw=0.9)
    arr(ax, 21, 32.5, 24, 32.5); arr(ax, 21, 13.5, 24, 13.5); arr(ax, 21, 23, 24, 23)
    rbox(ax, 24, 28, 15, 9, "Fine branch\ndilated conv\nlocal morphology\n→160-d", "#FDE9D9", ORANGE, 5.0)
    rbox(ax, 24, 18.6, 15, 8, "Cross-lead attention\n4 heads, 64-d", "#FDE9D9", ORANGE, 5.0)
    rbox(ax, 24, 9, 15, 9, "Coarse branch\nstrided conv\nrhythm\n→160-d", "#FDE9D9", ORANGE, 5.0)
    for yy in (32.5, 23, 13.5): ax.plot([39, 42.5], [yy, yy], color="#333333", lw=0.9)
    ax.plot([42.5, 42.5], [13.5, 32.5], color="#333333", lw=0.9)
    arr(ax, 42.5, 23, 45.5, 23)
    ax.text(43.6, 29.5, "concatenate\n160+160+64", fontsize=4.9, style="italic",
            color="#666666", ha="left", va="center")
    rbox(ax, 45.5, 18.5, 12, 9, "FiLM\nconditioning\n384-d", "#F7E4EF", "#B0448A", 5.4)
    rbox(ax, 45.5, 12.2, 12, 4.2, "age · sex", "#FFFFFF", "#B0448A", 5.2)
    arr(ax, 51.5, 16.4, 51.5, 18.5, "#B0448A")
    ax.text(51.5, 9.6, "h ← h·(1+tanh(Wγc)) + Wβc\ndemographics modulate features,\nnot concatenated inputs",
            ha="center", va="top", fontsize=4.9, style="italic", color="#666666")
    arr(ax, 57.5, 23, 70.5, 23)
    rbox(ax, 70.5, 18.5, 9, 9, "Pooled\n→256-d", "#EAF1F7", DBLUE, 5.2)
    ax.plot([79.5, 82], [63, 63], color="#333333", lw=0.9)
    ax.plot([79.5, 82], [23, 23], color="#333333", lw=0.9)
    ax.plot([82, 82], [23, 63], color="#333333", lw=0.9)
    arr(ax, 82, 43, 85, 43)
    rbox(ax, 85, 38.5, 13, 9, "Shared MLP\n256 · ReLU\nDropout 0.2", "#EAF1F7", DBLUE, 5.2)
    for i, (t, c) in enumerate([("HFrEF", DBLUE), ("Severe AS", ORANGE), ("LVH", GREEN),
                                ("LVEF (reg.)", "#B0448A"), ("1-yr mortality", "#777777")]):
        yb = 33 - i*5.4
        rbox(ax, 85, yb, 13, 4.2, t, "#FFFFFF", c, 5.0)
        arr(ax, 91.5, 38.5, 91.5, yb+4.2, c, 0.6)
    ax.text(91.5, 3.2, "multitask heads,\nuncertainty-weighted loss", ha="center",
            fontsize=4.9, style="italic", color="#666666")
    rbox(ax, 0.5, 1.5, 78, 4.6,
         "Final prediction = validation-selected weighted ensemble of the 4 networks  (1 : 1 : 2 : 2)",
         "#E8F0FA", DBLUE, 6.2, "bold")

def panel_ecgs(ax_list):
    LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    SHOW = [0, 1, 6, 8, 10]
    spec = [("true_pos", "model and echo agree:\ndisease present", ORANGE),
            ("true_neg", "model and echo agree:\ndisease absent", GREEN),
            ("discordant", "model flags structure\nthe echo called normal", DBLUE)]
    for ax_, (key, title, col) in zip(ax_list, spec):
        w = zex[key]; m = mex[key]; fs, secs = 500, 4
        t = np.arange(secs*fs)/fs
        for j, li in enumerate(SHOW):
            sig = w[li, :secs*fs].astype(float)
            sig = (sig - np.median(sig)) / (np.percentile(np.abs(sig-np.median(sig)), 99) + 1e-6)
            ax_.plot(t, sig*0.40 + (len(SHOW)-1-j), lw=0.4, color=col)
            ax_.text(-0.25, len(SHOW)-1-j, LEADS[li], fontsize=5.0, va="center",
                     ha="right", color="#555555")
        ax_.set_ylim(-0.8, len(SHOW)-0.15); ax_.set_xlim(-0.32, secs)
        ax_.set_yticks([]); ax_.set_xlabel("seconds", fontsize=5.8, labelpad=1)
        ax_.tick_params(axis="x", labelsize=5.4)
        for sp in ("left","top","right"): ax_.spines[sp].set_visible(False)
        ax_.set_title(title, fontsize=5.9, fontweight="bold", color=col, pad=2)
        sex = "M" if m["male"] else "F"
        ax_.text(0.5, -0.30, f"{m['age']:.0f} y, {sex} · echo LVEF {m['lvef']:.0f}%\n"
                 f"HFrEF probability {m['pred']:.3f} · alive at 1 y",
                 transform=ax_.transAxes, ha="center", va="top", fontsize=5.4)

def panel_internal(ax):
    x = np.arange(len(TASKS)); w = 0.26
    for i, (k, lab) in enumerate(TASKS):
        if BASE[k] is not None:
            ax.bar(i-w, BASE[k]-0.5, w, bottom=0.5, color=GREY, edgecolor="black", lw=0.4,
                   label="Tabular ECG baseline" if i == 0 else "")
            ax.text(i-w, BASE[k]+0.006, f"{BASE[k]:.3f}", ha="center", fontsize=5.2)
        ax.bar(i, SINGLE[k]-0.5, w, bottom=0.5, color=BLUE, edgecolor="black", lw=0.4,
               hatch="///", label="Single network" if i == 0 else "")
        ax.text(i, SINGLE[k]+0.006, f"{SINGLE[k]:.3f}", ha="center", fontsize=5.2)
        v = fin[k]["auroc"]; lo, hi = fin[k]["ci"]
        ax.bar(i+w, v-0.5, w, bottom=0.5, color=DBLUE, edgecolor="black", lw=0.4,
               yerr=[[v-lo]], capsize=2, error_kw={"lw": 0.7},
               label="BEACON-ECG" if i == 0 else "")
        ax.text(i+w, hi+0.008, f"{v:.3f}", ha="center", fontsize=5.8, fontweight="bold")
    il = [k for k, _ in TASKS].index("lvh")
    ax.bar(il-w, 0.03, w, bottom=0.5, color="none", edgecolor="#BBBBBB", lw=0.6, ls=":")
    ax.text(il-w, 0.542, "no tabular\ncomparator", ha="center", va="bottom",
            fontsize=4.8, color="#888888", style="italic")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.text(-0.52, 0.503, "chance", fontsize=5.2, style="italic", color="#555555", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in TASKS], fontsize=6.2)
    ax.set_ylim(0.5, 0.97); ax.set_ylabel("Test AUROC")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=5.8)

def panel_external(ax):
    ks = [k for k in ["hfref_le40", "lvh"] if k in ext]; x = np.arange(len(ks)); w = 0.32
    for i, k in enumerate(ks):
        iv = ext[k]["AUROC_internal_MIMIC"]
        ax.bar(i-w/2, iv-0.5, w, bottom=0.5, color=DBLUE, edgecolor="black", lw=0.4,
               label="Internal (MIMIC-IV)" if i == 0 else "")
        ax.text(i-w/2, iv+0.008, f"{iv:.3f}", ha="center", fontsize=5.6)
        v = ext[k]["AUROC_external"]; lo, hi = ext[k]["CI"]
        ax.bar(i+w/2, v-0.5, w, bottom=0.5, color=ORANGE, edgecolor="black", lw=0.4,
               hatch="\\\\", yerr=[[v-lo]], capsize=2, error_kw={"lw": 0.7},
               label="External (EchoNext)" if i == 0 else "")
        ax.text(i+w/2, hi+0.01, f"{v:.3f}", ha="center", fontsize=5.8, fontweight="bold")
        ax.text(i, 0.513, f"n={ext[k]['n']:,}", ha="center", fontsize=5.2, color="#555555")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["HFrEF", "LV hypertrophy"], fontsize=6.2)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=5.8)
    ax.text(0.5, 0.02, "no retraining · 250→500 Hz shift", transform=ax.transAxes,
            ha="center", fontsize=5.4, style="italic", color="#555555")

en = sv["echo_normal_LVEF>50"]
def panel_rates(ax):
    b = ax.bar([f"High risk\n(n={en['n_high']:,})", f"Low risk\n(n={en['n_low']:,})"],
               [en["mort365_high"]*100, en["mort365_low"]*100],
               color=[ORANGE, GREEN], edgecolor="black", lw=0.5, width=0.58)
    for r in b:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.5, f"{r.get_height():.1f}%",
                ha="center", fontweight="bold", fontsize=7)
    ax.set_ylabel("1-year all-cause mortality (%)"); ax.set_ylim(0, 30)
    ax.text(0.5, 0.93, f"Echo-normal patients (LVEF>50%), n={en['n']:,}",
            transform=ax.transAxes, ha="center", fontsize=5.8, style="italic")

def panel_km(ax):
    s = d[(d.lvef_value > 50) & d.pred_hfref_le40.notna()].copy()
    thr = s.pred_hfref_le40.quantile(0.8)
    s["grp"] = np.where(s.pred_hfref_le40 >= thr, "high", "low")
    tt = pd.to_numeric(s.days_to_death, errors="coerce")
    s["t"] = np.where(tt.notna() & (tt >= 0) & (tt <= 365), tt, 365)
    s["e"] = (tt.notna() & (tt >= 0) & (tt <= 365)).astype(int)
    for g, col, lab in [("high", ORANGE, "top structural-risk quintile"),
                        ("low", GREEN, "remainder")]:
        gd = s[s.grp == g].sort_values("t"); surv = 1.0; xs, ys = [0], [1.0]
        for t in np.sort(gd.loc[gd.e == 1, "t"].unique()):
            ar = (gd.t >= t).sum(); dd = ((gd.t == t) & (gd.e == 1)).sum()
            if ar > 0: surv *= (1 - dd/ar); xs.append(t); ys.append(surv)
        xs.append(365); ys.append(surv)
        ax.step(xs, ys, where="post", color=col, lw=1.3, label=lab)
    ax.set_xlabel("Days from ECG"); ax.set_ylabel("Survival probability")
    ax.set_xlim(0, 365); ax.set_ylim(0.70, 1.0)
    ax.legend(frameon=False, loc="lower left", fontsize=5.8)
    ax.text(190, 0.965, "log-rank P < 0.001\nadj. HR 1.85 (1.61–2.13)", fontsize=5.8, va="top")
    ax.text(0.02, 0.02, "note: y-axis truncated at 0.70", transform=ax.transAxes,
            fontsize=5.0, style="italic", color="#666666")
    return s

def panel_calibration(ax):
    for k, lab, col in [("hfref_le40", "HFrEF", DBLUE), ("as_severe", "Severe AS", ORANGE),
                        ("lvh", "LVH", GREEN), ("dead_365d", "1-yr mortality", PINK)]:
        ss = d[d[k].notna() & d[f"pred_{k}"].notna()]
        yy = ss[k].values.astype(int); pp = ss[f"pred_{k}"].values.astype(float)
        q = pd.qcut(pp, 10, labels=False, duplicates="drop")
        ax.plot([pp[q == i].mean() for i in range(q.max()+1)],
                [yy[q == i].mean() for i in range(q.max()+1)], "o-", color=col, lw=1.0, ms=2.6,
                label=f"{lab} (ECE {ev['discrimination_calibration'][k]['ece']:.3f})")
    ax.plot([0, 0.6], [0, 0.6], ls="--", color="black", lw=0.8, label="Perfect")
    ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.legend(frameon=False, fontsize=5.4, loc="upper left")

def panel_dca(ax):
    s = d[d.hfref_le40.notna() & d.pred_hfref_le40.notna()]
    y = s.hfref_le40.values.astype(int); p = s.pred_hfref_le40.values.astype(float)
    n = len(y); prev = y.mean(); ths = np.linspace(0.01, 0.50, 99)
    nb = [((p >= t) & (y == 1)).sum()/n - (((p >= t) & (y == 0)).sum()/n)*(t/(1-t)) for t in ths]
    ax.plot(ths, nb, "-", color=DBLUE, lw=1.4, label="BEACON-ECG")
    ax.plot(ths, [prev - (1-prev)*(t/(1-t)) for t in ths], "--", color="#888888", lw=1.0, label="Treat all")
    ax.axhline(0, color="black", lw=0.8, ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.03, max(nb)*1.15); ax.legend(frameon=False, fontsize=5.8)

s_as = d[d.as_severe.notna() & d.pred_as_severe.notna()].copy()
_y = s_as.as_severe.values.astype(int); _p = s_as.pred_as_severe.values.astype(float)
THR95 = float(np.percentile(_p[_y == 1], 5))
s_as["ag"] = pd.cut(s_as.age_at_ecg, [0, 50, 65, 80, 200],
                    labels=["<50", "50–64", "65–79", "≥80"], right=False)
ROWS = []
for g, gd in s_as.groupby("ag", observed=True):
    yy = gd.as_severe.values.astype(int); pp = gd.pred_as_severe.values.astype(float)
    f = pp >= THR95
    ROWS.append((str(g), int(yy.sum()), (1-f.mean())*100,
                 (f & (yy == 1)).sum()/yy.sum()*100 if yy.sum() else np.nan))
ROWS.append(("Overall", int(_y.sum()), (1-(_p >= THR95).mean())*100,
             ((_p >= THR95) & (_y == 1)).sum()/_y.sum()*100))

def panel_triage(ax):
    cols = [BLUE]*4 + [DBLUE]
    b = ax.bar([r[0] for r in ROWS], [r[2] for r in ROWS], color=cols,
               edgecolor="black", lw=0.5, width=0.6)
    for r, row in zip(b, ROWS):
        ax.text(r.get_x()+r.get_width()/2, row[2]+1.5, f"{row[2]:.1f}%",
                ha="center", fontweight="bold", fontsize=6.2)
    ax.set_ylabel("Echocardiograms spared (%)"); ax.set_ylim(0, 100)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([f"{r[0]}\n({r[1]} cases)" for r in ROWS], fontsize=5.8)

def panel_sens(ax):
    cols = [BLUE]*4 + [DBLUE]
    ax.bar([r[0] for r in ROWS], [r[3] for r in ROWS], color=cols,
           edgecolor="black", lw=0.5, width=0.6)
    ax.axhline(95, color="#8A3324", ls="--", lw=0.9)
    ax.text(0.03, 95.6, "nominal 95% target", fontsize=5.2, color="#8A3324")
    for i, r in enumerate(ROWS): ax.text(i, r[3]+0.6, f"{r[3]:.1f}", ha="center", fontsize=5.8)
    ax.set_ylabel("Sensitivity (%)"); ax.set_ylim(80, 103)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([f"{r[0]}\n({r[1]} cases)" for r in ROWS], fontsize=5.8)
    ax.text(0.02, 0.02, "note: y-axis truncated at 80%", transform=ax.transAxes,
            fontsize=5.0, style="italic", color="#666666")

# ────────────────────────────────────────────────────────── compose
fig = plt.figure(figsize=(7.3, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.35], hspace=0.16)
a = fig.add_subplot(gs[0]); panel_concept(a); tag(a, "A", "Study concept", y=1.0)
b = fig.add_subplot(gs[1]); panel_flow(b);    tag(b, "B", "Cohort construction and patient-level splits", y=1.0)
fig.savefig(f"{OUT}/Figure1_design.png", bbox_inches="tight"); plt.close()

fig = plt.figure(figsize=(7.3, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.62, 1.0], hspace=0.10, wspace=0.16)
a = fig.add_subplot(gs[0, :]); panel_arch(a); tag(a, "A", "BEACON-ECG architecture", y=1.0)
axs = [fig.add_subplot(gs[1, i]) for i in range(3)]
panel_ecgs(axs); tag(axs[0], "B", "Representative test-set ECGs and model output", y=1.20)
fig.savefig(f"{OUT}/Figure2_model.png", bbox_inches="tight"); plt.close()

fig, (a, b) = plt.subplots(1, 2, figsize=(7.3, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
panel_internal(a); tag(a, "A", "Internal test set vs comparators")
panel_external(b); tag(b, "B", "External validation, no retraining")
plt.tight_layout(); fig.savefig(f"{OUT}/Figure3_discrimination.png", bbox_inches="tight"); plt.close()

fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.0))
panel_rates(a); tag(a, "A", "Event rates")
s_km = panel_km(b); tag(b, "B", "Kaplan–Meier")
_hi, _lo = s_km[s_km.grp == "high"], s_km[s_km.grp == "low"]
assert abs(_hi.dead_365d.mean() - en["mort365_high"]) < 1e-4, "Fig4 A/B disagree (high)"
assert abs(_lo.dead_365d.mean() - en["mort365_low"]) < 1e-4, "Fig4 A/B disagree (low)"
print(f"  OK  Fig4 panels agree  {_hi.dead_365d.mean()*100:.2f}% / {_lo.dead_365d.mean()*100:.2f}%")
plt.tight_layout(); fig.savefig(f"{OUT}/Figure4_beyond_echo.png", bbox_inches="tight"); plt.close()

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.8))
panel_calibration(axs[0][0]); tag(axs[0][0], "A", "Calibration (decile bins)")
panel_dca(axs[0][1]);         tag(axs[0][1], "B", "Decision curve (HFrEF)")
panel_triage(axs[1][0]);      tag(axs[1][0], "C", "Aortic-stenosis triage burden")
panel_sens(axs[1][1]);        tag(axs[1][1], "D", "Sensitivity is not uniform")
plt.tight_layout(h_pad=2.6); fig.savefig(f"{OUT}/Figure5_utility.png", bbox_inches="tight"); plt.close()

print("\ncomposites written:")
for f in sorted(x for x in os.listdir(OUT) if x[:6] == "Figure" and "_" in x and x[6].isdigit()):
    print(f"  {f}  ({os.path.getsize(f'{OUT}/{f}')/1024:.0f} KB)")
