#!/usr/bin/env python3
"""Schematic + example figures for BEACON-ECG.

  A  model architecture   - transcribed from train_waveform.py / train_cognition.py,
                            not from memory; layer widths and kernel sizes are the real ones.
  B  concept / mechanism  - the reframing the paper argues for.
  C  real example ECGs    - three actual test-set tracings with the model's outputs.

Honesty note for panel C: all three patients were alive at 1 year, and that is stated on
the figure. The discordance result (24.0% vs 13.5%) is a group-level statistic; a single
case cannot demonstrate it, and the figure must not imply otherwise.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
OUT = (r"d:/桌面/_按项目整理/02_生信数据与医学AI项目/07_医学AI_深度学习_多项目孵化/"
       r"医学AI深度学习_导师课题项目群/崔老师项目_PETCT多任务_空间转录组深度学习/"
       r"PETCT多任务/论文_AIECG/figures_final_600dpi")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "font.size": 7.5,
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, DBLUE, GREEN, ORANGE, GREY = "#56B4E9", "#0072B2", "#009E73", "#D55E00", "#999999"

def rbox(ax, x, y, w, h, txt, fc, ec, fs=6.5, weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tc)

def arr(ax, x0, y0, x1, y1, c="#333333", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=7, lw=lw, color=c))

# ══════════════════════════════════════════════════ A. Architecture
fig, ax = plt.subplots(figsize=(7.4, 5.4)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 78)
ax.text(50, 76, "BEACON-ECG architecture", ha="center", fontsize=9.5, fontweight="bold")

# ---- shared input
rbox(ax, 1, 45, 12, 10, "12-lead ECG\n10 s @ 500 Hz\n12 × 5000", "#EAF1F7", DBLUE, 6.3)
ax.plot([13, 16], [50, 50], color="#333333", lw=0.9)
ax.plot([16, 16], [26, 66], color="#333333", lw=0.9)
arr(ax, 16, 66, 19, 66); arr(ax, 16, 26, 19, 26)

# ---- Members 1-2 : SE-ResNet
ax.text(19, 72.2, "Members 1–2   SE-ResNet backbone   (ensemble weight ×1 each)",
        fontsize=6.8, fontweight="bold", color="#555555")
rbox(ax, 19, 61.5, 9.5, 9, "Stem\nConv1d 12→64\nk=15, s=2", "#EAF1F7", DBLUE, 5.5)
arr(ax, 28.5, 66, 30.5, 66)
for i, (cin, cout, s) in enumerate([(64, 64, 1), (64, 128, 2), (128, 196, 2),
                                    (196, 256, 2), (256, 256, 2)]):
    x = 30.5 + i*7.6
    rbox(ax, x, 61.5, 6.8, 9, f"Res\n{cin}→{cout}\ns={s}", "#DCEBE0", GREEN, 5.0)
    if i < 4: arr(ax, x+6.8, 66, x+7.6, 66)
arr(ax, 68.9, 66, 71, 66)
rbox(ax, 71, 61.5, 9, 9, "Global\navg pool\n→256-d", "#EAF1F7", DBLUE, 5.5)
ax.text(44, 58.6, "each Res block:  Conv1d ×2 · BatchNorm · SE channel attention · Dropout 0.1 · residual",
        ha="center", fontsize=5.5, style="italic", color="#666666")

# ---- Members 3-4 : dual-pathway, branches genuinely PARALLEL then fused
ax.text(19, 42.5, "Members 3–4   dual-pathway encoder + demographic conditioning   (weight ×2 each)",
        fontsize=6.8, fontweight="bold", color="#555555")
ax.plot([19, 22], [26, 26], color="#333333", lw=0.9)
ax.plot([22, 22], [16.5, 35.5], color="#333333", lw=0.9)
arr(ax, 22, 35.5, 25, 35.5); arr(ax, 22, 16.5, 25, 16.5)
rbox(ax, 25, 31, 15, 9, "Fine branch\ndilated conv\nlocal morphology\n→160-d", "#FDE9D9", ORANGE, 5.3)
rbox(ax, 25, 12, 15, 9, "Coarse branch\nstrided conv\nrhythm\n→160-d", "#FDE9D9", ORANGE, 5.3)
rbox(ax, 25, 21.6, 15, 8, "Cross-lead attention\n4 heads, 64-d", "#FDE9D9", ORANGE, 5.3)
ax.plot([22, 22], [25.6, 26.4], color="#333333", lw=0.9)
arr(ax, 22, 25.6, 25, 25.6)
# fuse the three parallel streams
for yy in (35.5, 25.6, 16.5): ax.plot([40, 43.5], [yy, yy], color="#333333", lw=0.9)
ax.plot([43.5, 43.5], [16.5, 35.5], color="#333333", lw=0.9)
arr(ax, 43.5, 26, 46.5, 26)
ax.text(44.6, 32.4, "concatenate\n160+160+64", fontsize=5.2, style="italic",
        color="#666666", ha="left", va="center")
rbox(ax, 46.5, 21.5, 12, 9, "FiLM\nconditioning\n384-d", "#F7E4EF", "#B0448A", 5.8)
rbox(ax, 46.5, 15.0, 12, 4.4, "age · sex", "#FFFFFF", "#B0448A", 5.6)
arr(ax, 52.5, 19.4, 52.5, 21.5, "#B0448A")
ax.text(52.5, 12.2, "h ← h·(1+tanh(Wγc)) + Wβc\ndemographics modulate features,\nthey are not concatenated inputs",
        ha="center", va="top", fontsize=5.1, style="italic", color="#666666")
arr(ax, 58.5, 26, 71, 26)
rbox(ax, 71, 21.5, 9, 9, "Pooled\n→256-d", "#EAF1F7", DBLUE, 5.5)

# ---- shared head + task heads (own column, no overlap)
ax.plot([80, 83], [66, 66], color="#333333", lw=0.9)
ax.plot([80, 83], [26, 26], color="#333333", lw=0.9)
ax.plot([83, 83], [26, 66], color="#333333", lw=0.9)
arr(ax, 83, 46, 86, 46)
rbox(ax, 86, 41.5, 12, 9, "Shared MLP\n256 · ReLU\nDropout 0.2", "#EAF1F7", DBLUE, 5.5)
heads = [("HFrEF", DBLUE), ("Severe AS", ORANGE), ("LVH", GREEN),
         ("LVEF (reg.)", "#B0448A"), ("1-yr mortality", "#777777")]
for i, (t, c) in enumerate(heads):
    yb = 35.5 - i*5.6
    rbox(ax, 86, yb, 12, 4.4, t, "#FFFFFF", c, 5.3)
    arr(ax, 92, 41.5, 92, yb+4.4, c, 0.6)
ax.text(92, 3.4, "multitask heads,\nuncertainty-weighted loss\n(learned log σ² per task)",
        ha="center", fontsize=5.1, style="italic", color="#666666")

# ---- ensemble bar
rbox(ax, 1, 0.5, 78, 5.2,
     "Final prediction = validation-selected weighted ensemble of the 4 networks   (1 : 1 : 2 : 2)",
     "#E8F0FA", DBLUE, 6.6, "bold")
plt.savefig(f"{OUT}/FigureA_architecture.png", bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════ B. Concept / mechanism
fig, ax = plt.subplots(figsize=(6.8, 3.6)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 52)
ax.text(50, 50, "From “ECG as a surrogate for the echo” to “ECG as a complementary readout”",
        ha="center", fontsize=8.5, fontweight="bold")

# prior framing
ax.add_patch(Rectangle((2, 28), 45, 17, fc="#F7F7F7", ec="#BBBBBB", lw=0.8, zorder=0))
ax.text(24.5, 42.8, "Prior work", ha="center", fontsize=7.2, fontweight="bold", color="#666666")
rbox(ax, 5, 32, 12, 7, "12-lead\nECG", "#FFFFFF", "#888888", 6.2)
rbox(ax, 32, 32, 12, 7, "Concurrent\necho", "#FFFFFF", "#888888", 6.2)
arr(ax, 17, 35.5, 32, 35.5, "#888888")
ax.text(24.5, 36.6, "estimates", ha="center", fontsize=5.8, style="italic", color="#666666")
ax.text(24.5, 30.2, "the ECG stands in for a test you could have done today",
        ha="center", fontsize=5.5, style="italic", color="#777777")

# our framing
ax.add_patch(Rectangle((52, 28), 46, 17, fc="#EAF3FA", ec=DBLUE, lw=0.9, zorder=0))
ax.text(75, 42.8, "This study", ha="center", fontsize=7.2, fontweight="bold", color=DBLUE)
rbox(ax, 54, 32, 11, 7, "12-lead\nECG", "#FFFFFF", DBLUE, 6.2)
rbox(ax, 69, 32, 12, 7, "Structural\nsignature", "#DCEBE0", GREEN, 6.2)
rbox(ax, 85, 32, 11, 7, "Outcome\nbeyond echo", "#FDE9D9", ORANGE, 6.2)
arr(ax, 65, 35.5, 69, 35.5, DBLUE); arr(ax, 81, 35.5, 85, 35.5, GREEN)

# the three questions
qs = [("What the echo sees",
       "ECG reproduces concurrent\nstructural findings\nHFrEF 0.900 · AS 0.860 · LVH 0.789", GREEN),
      ("What the echo misses",
       "Among echo-NORMAL patients,\ntop structural-risk quintile\n24.0% vs 13.5% 1-yr mortality\nadj. HR 1.85 (1.61–2.13)", ORANGE),
      ("What time adds — nothing",
       "Deep serial-ECG trajectory did\nNOT beat a single tracing\n(0.796 vs 0.779)", "#888888")]
# All three findings belong to THIS study, so every connector descends from the
# "This study" panel. An earlier draft ran the leftmost arrow down from "Prior work",
# which wrongly attributed our own concurrent-detection result to previous authors.
ax.plot([75, 75], [28, 26], color="#555555", lw=0.9)
ax.plot([18, 93], [26, 26], color="#555555", lw=0.9)
for i, (title, body, col) in enumerate(qs):
    x = 3 + i*32.5
    rbox(ax, x, 3, 30, 20, "", "#FFFFFF", col, 6)
    ax.text(x+15, 20.2, title, ha="center", fontsize=6.8, fontweight="bold", color=col)
    ax.text(x+15, 11.5, body, ha="center", va="center", fontsize=5.9)
    arr(ax, x+15, 26, x+15, 23.2, col, 0.8)
plt.savefig(f"{OUT}/FigureB_concept.png", bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════ C. Real example ECGs
z = np.load(f"{SP}/example_ecgs.npz"); meta = json.load(open(f"{SP}/example_ecgs.json"))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SHOW = [0, 1, 6, 8, 10]          # I, II, V1, V3, V5 - enough to read morphology
panels = [("true_pos", "Model agrees with echo — disease present", ORANGE),
          ("true_neg", "Model agrees with echo — disease absent", GREEN),
          ("discordant", "Model flags structure the echo called normal", DBLUE)]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9))
for ax_, (key, title, col) in zip(axes, panels):
    w = z[key]; m = meta[key]
    fs, secs = 500, 4
    t = np.arange(secs*fs)/fs
    for j, li in enumerate(SHOW):
        sig = w[li, :secs*fs].astype(float)
        sig = (sig - np.median(sig)) / (np.percentile(np.abs(sig - np.median(sig)), 99) + 1e-6)
        ax_.plot(t, sig*0.42 + (len(SHOW)-1-j), lw=0.45, color=col)
        ax_.text(-0.28, len(SHOW)-1-j, LEADS[li], fontsize=5.6, va="center", ha="right",
                 color="#555555")
    ax_.set_ylim(-0.9, len(SHOW)-0.1); ax_.set_xlim(-0.35, secs)
    ax_.set_yticks([]); ax_.set_xlabel("seconds", fontsize=6.2, labelpad=1)
    ax_.tick_params(axis="x", labelsize=5.8)
    for sp in ("left", "top", "right"): ax_.spines[sp].set_visible(False)
    ax_.set_title(title, fontsize=6.6, fontweight="bold", color=col, pad=3)
    sex = "M" if m["male"] else "F"
    ax_.text(0.5, -0.215, f"{m['age']:.0f} y, {sex}   |   echo LVEF {m['lvef']:.0f}%",
             transform=ax_.transAxes, ha="center", fontsize=6.0)
    ax_.text(0.5, -0.305, f"BEACON-ECG HFrEF probability  {m['pred']:.3f}",
             transform=ax_.transAxes, ha="center", fontsize=6.4, fontweight="bold", color=col)
    ax_.text(0.5, -0.385, "alive at 1 year", transform=ax_.transAxes, ha="center",
             fontsize=5.8, style="italic", color="#666666")
fig.suptitle("Real test-set ECGs and the model's output", fontsize=8.5, fontweight="bold", y=1.0)
fig.text(0.5, -0.055,
         "All three patients were alive at 1 year. The discordance result (24.0% vs 13.5%) is a group-level "
         "statistic;\nindividual cases illustrate the model's inputs and outputs and cannot demonstrate it.",
         ha="center", fontsize=5.8, style="italic", color="#555555")
plt.tight_layout(); plt.savefig(f"{OUT}/FigureC_ecg_examples.png", bbox_inches="tight"); plt.close()

print("written:")
for f in ["FigureA_architecture.png", "FigureB_concept.png", "FigureC_ecg_examples.png"]:
    print(f"  {f}  ({os.path.getsize(f'{OUT}/{f}')/1024:.0f} KB)")
