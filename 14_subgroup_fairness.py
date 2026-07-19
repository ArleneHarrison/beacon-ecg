#!/usr/bin/env python3
"""Race / insurance subgroup analysis (TRIPOD+AI item 15), plus a corrected re-run
of the age/sex axes with a minimum-event-count guard.

The earlier version reported AUROC for any subgroup with n>=50 and >=1 event of each
class. That produced a meaningless AUROC of 0.975 for severe AS in patients <50, where
there were only ~2 events. Cells with fewer than MIN_EVENTS are now suppressed and
labelled, rather than reported.
"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
TASKS = ["hfref_le40", "as_severe", "lvh", "dead_365d"]
MIN_EVENTS = 20        # below this, AUROC is not reported
MIN_N = 100
RNG = np.random.default_rng(0)


def collapse_race(r):
    if not isinstance(r, str): return "Unknown"
    r = r.upper()
    if r.startswith("WHITE"): return "White"
    if r.startswith("BLACK"): return "Black"
    if r.startswith("HISPANIC"): return "Hispanic/Latino"
    if r.startswith("ASIAN"): return "Asian"
    if any(k in r for k in ["UNKNOWN", "UNABLE TO OBTAIN", "DECLINED"]): return "Unknown"
    return "Other"


def boot_ci(y, p, n=1000):
    a = []
    for _ in range(n):
        b = RNG.choice(len(y), len(y), True)
        if len(np.unique(y[b])) > 1:
            a.append(roc_auc_score(y[b], p[b]))
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None


def ece(y, p, bins=10):
    e, n = 0.0, len(y); edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i+1] if i == bins-1 else p < edges[i+1])
        if m.sum(): e += (m.sum()/n) * abs(y[m].mean() - p[m].mean())
    return round(float(e), 4)


def metrics(y, p, thr):
    n, ev = len(y), int(y.sum())
    if n < MIN_N:
        return {"n": n, "n_events": ev, "suppressed": f"n<{MIN_N}"}
    if ev < MIN_EVENTS or (n - ev) < MIN_EVENTS:
        return {"n": n, "n_events": ev, "prevalence": round(float(y.mean()), 4),
                "suppressed": f"fewer than {MIN_EVENTS} events (or non-events); "
                              "AUROC not estimable with useful precision"}
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    obs, exp = float(y.mean()), float(p.mean())
    return {"n": n, "n_events": ev, "prevalence": round(obs, 4),
            "auroc": round(float(roc_auc_score(y, p)), 4), "auroc_ci": boot_ci(y, p),
            "ece": ece(y, p), "obs_over_exp": round(obs/exp, 3) if exp > 0 else None,
            "sensitivity": round(tp/(tp+fn), 4) if tp+fn else None,
            "specificity": round(tn/(tn+fp), 4) if tn+fp else None,
            "ppv": round(tp/(tp+fp), 4) if tp+fp else None,
            "flag_rate": round(float(pred.mean()), 4)}


def main():
    d = pd.read_parquet(f"{SP}/test_predictions_FINAL.parquet")
    adm = pd.read_csv(f"{SP}/admissions.csv",
                      usecols=["subject_id", "race", "insurance"], low_memory=False)

    # One record per patient: the modal (most frequent) value across their admissions.
    # Race is recorded per-admission in MIMIC and can differ between admissions for the
    # same patient; the mode is the standard reconciliation and is stated in the paper.
    mode = (adm.groupby("subject_id")
               .agg(race=("race", lambda s: s.mode().iat[0] if len(s.mode()) else None),
                    insurance=("insurance", lambda s: s.mode().iat[0] if len(s.mode()) else None))
               .reset_index())
    d = d.merge(mode, on="subject_id", how="left")
    d["race_group"] = d.race.map(collapse_race)
    d["insurance_group"] = d.insurance.fillna("Unknown")
    d["sex_group"] = np.where(d.sex_male == 1, "Male", "Female")
    d["age_group"] = pd.cut(d.age_at_ecg, [0, 50, 65, 80, 200],
                            labels=["<50", "50-64", "65-79", ">=80"], right=False)

    matched = d.race.notna().mean()
    print(f"test n={len(d)}, matched to admissions: {matched*100:.1f}%")
    print("\nrace:\n", d.race_group.value_counts().to_string())
    print("\ninsurance:\n", d.insurance_group.value_counts().to_string())

    R = {"_meta": {
        "n_test": int(len(d)),
        "pct_matched_to_admissions": round(float(matched)*100, 1),
        "min_events_for_auroc": MIN_EVENTS,
        "race_source": "MIMIC-IV hosp admissions.race, collapsed to major categories; "
                       "per-patient value = mode across that patient's admissions",
        "note": "Operating-point metrics use a single GLOBAL threshold (90th percentile of "
                "predicted risk overall), matching deployment. Per-subgroup thresholds are "
                "deliberately not used.",
        "caveat": "Subgroup AUROC differences reflect case-mix and severity spread as well as "
                  "model behaviour. Calibration (obs_over_exp) and flag_rate are the "
                  "decision-relevant fairness quantities.",
        "race_caveat": "Race in MIMIC-IV is an administrative field of uncertain provenance, "
                       "not a biological variable; 'Unknown' is reported as its own stratum "
                       "and is not merged into 'Other'.",
    }}

    for task in TASKS:
        pc = f"pred_{task}"
        sub = d[d[task].notna() & d[pc].notna()]
        if len(sub) < MIN_N: continue
        y = sub[task].values.astype(int); p = sub[pc].values.astype(float)
        thr = float(np.percentile(p, 90))
        R[task] = {"threshold_global_p90": round(thr, 4), "overall": metrics(y, p, thr)}
        for axis, col in [("by_sex", "sex_group"), ("by_age", "age_group"),
                          ("by_race", "race_group"), ("by_insurance", "insurance_group")]:
            R[task][axis] = {}
            for g, gd in sub.groupby(col, observed=True):
                R[task][axis][str(g)] = metrics(gd[task].values.astype(int),
                                                gd[pc].values.astype(float), thr)
            rep = {k: v for k, v in R[task][axis].items() if "auroc" in v}
            if len(rep) > 1:
                au = {k: v["auroc"] for k, v in rep.items()}
                oe = {k: v["obs_over_exp"] for k, v in rep.items() if v.get("obs_over_exp")}
                fl = {k: v["flag_rate"] for k, v in rep.items()}
                R[task][axis]["_gap"] = {
                    "n_groups_reported": len(rep),
                    "n_groups_suppressed": len(R[task][axis]) - len(rep),
                    "auroc_gap": round(max(au.values()) - min(au.values()), 4),
                    "auroc_worst": min(au, key=au.get), "auroc_best": max(au, key=au.get),
                    "calibration_gap": round(max(oe.values()) - min(oe.values()), 3) if oe else None,
                    "flag_rate_range": [min(fl.values()), max(fl.values())],
                }

    json.dump(R, open(f"{SP}/subgroup_full.json", "w"), indent=2)
    print(f"\nwritten: {SP}/subgroup_full.json")


if __name__ == "__main__":
    main()
