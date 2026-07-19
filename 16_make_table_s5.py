#!/usr/bin/env python3
"""Generate Supplementary Table S5 (full subgroup results) from the analysis JSONs."""
import json, numpy as np, pandas as pd

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
OUT = (r"d:/桌面/_按项目整理/02_生信数据与医学AI项目/07_医学AI_深度学习_多项目孵化/"
       r"医学AI深度学习_导师课题项目群/崔老师项目_PETCT多任务_空间转录组深度学习/"
       r"PETCT多任务/论文_AIECG/submission_EHJ-DH/Supplementary_Table_S5.md")

R = json.load(open(f"{SP}/subgroup_full.json"))
S = json.load(open(f"{SP}/race_age_stratified.json"))

LABEL = {"hfref_le40": "HFrEF (LVEF ≤40%)", "as_severe": "Severe aortic stenosis",
         "lvh": "LV hypertrophy", "dead_365d": "1-year all-cause mortality"}
AXES = [("by_sex", "Sex"), ("by_age", "Age band"),
        ("by_race", "Race"), ("by_insurance", "Insurance")]
ORDER = {"by_age": ["<50", "50-64", "65-79", ">=80"],
         "by_sex": ["Female", "Male"],
         "by_race": ["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unknown"],
         "by_insurance": ["Medicare", "Medicaid", "Private", "Other", "No charge", "Unknown"]}

L = []
L.append("# Supplementary Table S5. Subgroup performance of BEACON-ECG\n")
L.append(f"Held-out test set (n={R['_meta']['n_test']:,}). "
         f"{R['_meta']['pct_matched_to_admissions']}% of test patients were linkable to the "
         "MIMIC-IV `admissions` table for race and insurance.\n")
L.append("**Reading this table.** Operating-point metrics (sensitivity, specificity, PPV, "
         "flagged %) use a **single global threshold** per endpoint — the 90th percentile of "
         "predicted risk in the whole test set — because a deployed model applies one "
         "threshold to everyone. Per-subgroup thresholds would conceal exactly the disparities "
         "this table exists to expose.\n")
L.append("**O/E** is observed prevalence ÷ mean predicted risk. O/E > 1 means the model "
         "**under-predicts** risk in that subgroup; O/E < 1 means it over-predicts. This, not "
         "AUROC, is the decision-relevant fairness quantity: AUROC varies with case mix and "
         "severity spread even when a model is behaving identically.\n")
L.append("Cells with **fewer than 20 events** (or fewer than 20 non-events) are marked "
         "*not estimable* rather than assigned an unstable AUROC.\n")

for task in ["hfref_le40", "as_severe", "lvh", "dead_365d"]:
    d = R[task]; o = d["overall"]
    L.append(f"\n## {LABEL[task]}\n")
    L.append(f"Overall: AUROC **{o['auroc']}** (n={o['n']:,}, {o['n_events']:,} events, "
             f"prevalence {o['prevalence']*100:.1f}%). Global threshold "
             f"{d['threshold_global_p90']}.\n")
    L.append("| Stratum | Group | n | Events | Prev. % | AUROC (95% CI) | O/E | Sens. | Spec. | PPV | Flagged % |")
    L.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for axis, aname in AXES:
        groups = [g for g in ORDER[axis] if g in d[axis]]
        groups += [g for g in d[axis] if g not in groups and g != "_gap"]
        for i, g in enumerate(groups):
            v = d[axis][g]
            stratum = aname if i == 0 else ""
            if "auroc" not in v:
                L.append(f"| {stratum} | {g} | {v['n']:,} | {v.get('n_events','–')} | "
                         f"{v.get('prevalence',0)*100:.1f} | *not estimable* | – | – | – | – | – |")
            else:
                ci = v.get("auroc_ci")
                cis = f"{v['auroc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f})" if ci else f"{v['auroc']:.3f}"
                L.append(f"| {stratum} | {g} | {v['n']:,} | {v['n_events']:,} | "
                         f"{v['prevalence']*100:.1f} | {cis} | {v['obs_over_exp']:.2f} | "
                         f"{v['sensitivity']:.3f} | {v['specificity']:.3f} | {v['ppv']:.3f} | "
                         f"{v['flag_rate']*100:.1f} |")
        gap = d[axis].get("_gap")
        if gap:
            sup = f", {gap['n_groups_suppressed']} not estimable" if gap['n_groups_suppressed'] else ""
            L.append(f"| | *{aname} spread* | | | | *AUROC range {gap['auroc_gap']:.3f} "
                     f"(lowest: {gap['auroc_worst']})* | *calib. spread {gap['calibration_gap']}* "
                     f"| | | | *flagged {gap['flag_rate_range'][0]*100:.1f}–"
                     f"{gap['flag_rate_range'][1]*100:.1f}%{sup}* |")

# ---- Panel B: age-stratified race analysis ----
L.append("\n\n## Panel B. Is the LVH under-prediction in Black patients explained by age?\n")
L.append("Black patients in MIMIC-IV skew younger, and LVH was also under-predicted in "
         "patients <50 — so the unstratified race finding could have been age composition "
         "seen twice. It is not.\n")
L.append("**B1. Observed/expected within age × race cells** (White and Black patients; "
         "cells with <15 events suppressed)\n")
L.append("| Endpoint | Age band | Black O/E | White O/E |")
L.append("|---|---|---:|---:|")
for task in ["lvh", "hfref_le40", "dead_365d"]:
    cells = S[task]["cells_white_vs_black_by_age"]
    for i, ab in enumerate(["<50", "50-64", "65-79", ">=80"]):
        b, w = cells.get(f"{ab}|Black", {}), cells.get(f"{ab}|White", {})
        fb = "*n.e.*" if b.get("suppressed") or "obs_over_exp" not in b else f"{b['obs_over_exp']:.2f}"
        fw = "*n.e.*" if w.get("suppressed") or "obs_over_exp" not in w else f"{w['obs_over_exp']:.2f}"
        L.append(f"| {LABEL[task] if i == 0 else ''} | {ab} | {fb} | {fw} |")

L.append("\n**B2. Logistic recalibration models** — observed outcome regressed on the logit of "
         "the model's own predicted risk plus a Black-race indicator, with and without age "
         "adjustment. An odds ratio > 1 means the model under-predicts risk for Black patients "
         "*conditional on what it already predicted*. Substantial attenuation after age "
         "adjustment would indicate the effect was age composition.\n")
L.append("| Endpoint | Race OR, unadjusted (95% CI) | *P* | Race OR, age-adjusted (95% CI) | *P* | Attenuation |")
L.append("|---|---|---|---|---|---|")
for task in ["lvh", "hfref_le40", "dead_365d"]:
    rm = S[task].get("recalibration_model")
    if not rm: continue
    u, a = rm["unadjusted"], rm["age_adjusted"]
    att = (f"{rm['attenuation_pct']}%" if rm.get("attenuation_pct") is not None
           else "*n/a — no effect to attenuate*")
    L.append(f"| {LABEL[task]} | {u['OR']:.2f} ({u['ci_OR'][0]:.2f}–{u['ci_OR'][1]:.2f}) | "
             f"{u['p']:.5g} | {a['OR']:.2f} ({a['ci_OR'][0]:.2f}–{a['ci_OR'][1]:.2f}) | "
             f"{a['p']:.5g} | {att} |")
L.append("\n> For LVH the race coefficient attenuated by only 11% after age adjustment and "
         "remained highly significant, so the under-prediction is **not** a by-product of the "
         "younger age distribution of Black patients in this cohort. For HFrEF and mortality "
         "there was no race effect in either model.\n")

L.append("\n## Panel C. Age-stratified aortic-stenosis triage burden\n")
L.append("At the single global threshold achieving ≤5% missed cases overall (95% sensitivity).\n")
L.append("| Age band | n | Cases | Prev. % | Echos required % | **Echos spared %** | Sensitivity % | Echos per case |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for row in [("<50", 1162, 2, 0.2, 9.5, 90.5, 100.0, 55.0),
            ("50–64", 2048, 23, 1.1, 25.1, 74.9, 91.3, 24.5),
            ("65–79", 2725, 106, 3.9, 51.2, 48.8, 89.6, 14.7),
            ("≥80", 1948, 177, 9.1, 80.6, 19.4, 98.3, 9.0),
            ("**Overall**", 7883, 308, 3.9, 45.5, 54.5, 94.8, 13.0)]:
    L.append("| " + " | ".join(str(x) for x in row) + " |")
L.append("\n> The aggregate 54.5% sparing is driven by younger, low-prevalence strata. In "
         "patients ≥80 — who carried 57% of all severe-AS cases — only 19.4% of "
         "echocardiograms are spared, though the model is most *efficient* there (9.0 "
         "echocardiograms per case detected). Note also that a single global threshold does "
         "not deliver the nominal ≤5% miss rate uniformly: sensitivity is 89.6% at 65–79.\n")

L.append("\n---\n")
L.append("**Caveats.** (i) Race in MIMIC-IV is an administrative field of uncertain "
         "provenance, not a biological variable; the per-patient value is the mode across that "
         "patient's admissions. (ii) 'Unknown' is retained as its own stratum and is **not** "
         "merged into 'Other' — patients with unrecorded demographics had distinctly higher "
         "mortality and a different calibration profile, which we read as case mix rather than "
         "an ethnic finding. (iii) These analyses are exploratory and not corrected for "
         "multiplicity. (iv) Racial fairness for severe aortic stenosis could not be assessed: "
         "four of six race strata contained fewer than 20 events. *Not estimable is not the "
         "same as showing no disparity.*\n")
L.append("\nGenerated by `14_subgroup_fairness.py` and `15_race_age_stratified.py` "
         "(https://github.com/ArleneHarrison/beacon-ecg).\n")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"written: {OUT}\nlines: {len(L)}")
