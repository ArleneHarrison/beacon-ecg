#!/usr/bin/env python3
"""Is the LVH under-prediction in Black patients independent of age?

The unstratified analysis found observed/expected = 1.68 for LVH in Black patients.
It also found O/E = 1.68 for LVH in patients <50. Black patients in MIMIC-IV skew
younger, so these may be the same effect seen twice. This script separates them.

Two complementary approaches:

  (A) Descriptive  - O/E within age x race cells (small cells suppressed).
  (B) Inferential  - logistic recalibration model:
                        observed ~ logit(predicted risk) + race          [model 1]
                        observed ~ logit(predicted risk) + race + age    [model 2]
      A non-zero race coefficient means the model is miscalibrated for that group
      CONDITIONAL on what it already predicted. If the coefficient shrinks toward zero
      when age enters, the apparent race effect was age composition.
"""
import json, numpy as np, pandas as pd
import statsmodels.api as sm

SP = r"C:/Users/63415/AppData/Local/Temp/claude/d------------02--------AI---07---AI--------------AI-------------------PETCT--------------PETCT---/dce30790-0a57-4f27-8b12-bcb735535fe0/scratchpad"
MIN_EV = 15


def collapse_race(r):
    if not isinstance(r, str): return "Unknown"
    r = r.upper()
    if r.startswith("WHITE"): return "White"
    if r.startswith("BLACK"): return "Black"
    if r.startswith("HISPANIC"): return "Hispanic/Latino"
    if r.startswith("ASIAN"): return "Asian"
    if any(k in r for k in ["UNKNOWN", "UNABLE TO OBTAIN", "DECLINED"]): return "Unknown"
    return "Other"


d = pd.read_parquet(f"{SP}/test_predictions_FINAL.parquet")
adm = pd.read_csv(f"{SP}/admissions.csv", usecols=["subject_id", "race"], low_memory=False)
mode = (adm.groupby("subject_id").race
           .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None).reset_index())
d = d.merge(mode, on="subject_id", how="left")
d["race_group"] = d.race.map(collapse_race)
d["age_group"] = pd.cut(d.age_at_ecg, [0, 50, 65, 80, 200],
                        labels=["<50", "50-64", "65-79", ">=80"], right=False)

OUT = {"_meta": {
    "question": "Is the LVH under-prediction in Black patients independent of age?",
    "min_events_per_cell": MIN_EV,
    "race_caveat": "Race in MIMIC-IV is an administrative field, not a biological variable.",
}}

# ---- age composition by race: the potential confounder, shown explicitly ----
comp = pd.crosstab(d.race_group, d.age_group, normalize="index").round(3)
OUT["age_composition_by_race"] = json.loads(comp.to_json(orient="index"))
print("Age composition by race (row %):\n", comp.to_string(), "\n")

for task in ["lvh", "hfref_le40", "dead_365d"]:
    pc = f"pred_{task}"
    s = d[d[task].notna() & d[pc].notna()].copy()
    s["y"] = s[task].astype(int)
    s["p"] = s[pc].astype(float).clip(1e-6, 1 - 1e-6)
    s["lp"] = np.log(s.p / (1 - s.p))          # logit of predicted risk

    T = {"overall_n": int(len(s))}

    # ---- (A) descriptive: O/E within age x race ----
    cells = {}
    for (ag, rc), g in s.groupby(["age_group", "race_group"], observed=True):
        if rc not in ("White", "Black"): continue
        ev = int(g.y.sum())
        key = f"{ag}|{rc}"
        if ev < MIN_EV or len(g) < 80:
            cells[key] = {"n": int(len(g)), "events": ev, "suppressed": True}
        else:
            cells[key] = {"n": int(len(g)), "events": ev,
                          "observed": round(float(g.y.mean()), 4),
                          "predicted": round(float(g.p.mean()), 4),
                          "obs_over_exp": round(float(g.y.mean() / g.p.mean()), 3)}
    T["cells_white_vs_black_by_age"] = cells

    # ---- (B) inferential: race coefficient before and after age adjustment ----
    m = s[s.race_group.isin(["White", "Black"])].copy()
    m["black"] = (m.race_group == "Black").astype(int)
    if m.black.sum() >= 30 and m.y.sum() >= 30:
        r1 = sm.Logit(m.y, sm.add_constant(m[["lp", "black"]])).fit(disp=0)
        m["age_z"] = (m.age_at_ecg - m.age_at_ecg.mean()) / m.age_at_ecg.std()
        r2 = sm.Logit(m.y, sm.add_constant(m[["lp", "black", "age_z"]])).fit(disp=0)
        b1, b2 = float(r1.params["black"]), float(r2.params["black"])
        T["recalibration_model"] = {
            "n": int(len(m)), "n_black": int(m.black.sum()),
            "unadjusted": {"race_coef": round(b1, 4), "OR": round(float(np.exp(b1)), 3),
                           "p": round(float(r1.pvalues["black"]), 5),
                           "ci_OR": [round(float(np.exp(v)), 3)
                                     for v in r1.conf_int().loc["black"].tolist()]},
            "age_adjusted": {"race_coef": round(b2, 4), "OR": round(float(np.exp(b2)), 3),
                             "p": round(float(r2.pvalues["black"]), 5),
                             "ci_OR": [round(float(np.exp(v)), 3)
                                       for v in r2.conf_int().loc["black"].tolist()]},
            # Attenuation is only meaningful when there is an effect to attenuate.
            # With b1 near zero the ratio explodes (an earlier version reported -803%
            # for mortality, where b1 was 0.004 - an artefact, not a finding).
            "attenuation_pct": (round(100 * (1 - abs(b2) / abs(b1)), 1)
                                if abs(b1) > 0.05 else None),
            "attenuation_note": (None if abs(b1) > 0.05 else
                                 "unadjusted race coefficient is ~0; there is no effect "
                                 "to attenuate and the percentage is not defined"),
            "reading": "race_coef > 0 means the model UNDER-predicts risk for Black patients "
                       "conditional on its own prediction. If the coefficient attenuates "
                       "substantially after age adjustment, the effect was age composition.",
        }
    OUT[task] = T

json.dump(OUT, open(f"{SP}/race_age_stratified.json", "w"), indent=2)

for task in ["lvh", "hfref_le40", "dead_365d"]:
    print(f"\n{'='*70}\n{task.upper()}")
    for k, v in OUT[task]["cells_white_vs_black_by_age"].items():
        if v.get("suppressed"):
            print(f"  {k:<18} n={v['n']:<5} ev={v['events']:<4} suppressed")
        else:
            print(f"  {k:<18} n={v['n']:<5} ev={v['events']:<4} obs={v['observed']:.3f} "
                  f"pred={v['predicted']:.3f}  O/E={v['obs_over_exp']}")
    rm = OUT[task].get("recalibration_model")
    if rm:
        u, a = rm["unadjusted"], rm["age_adjusted"]
        print(f"  race OR unadjusted   : {u['OR']}  CI {u['ci_OR']}  p={u['p']}")
        print(f"  race OR age-adjusted : {a['OR']}  CI {a['ci_OR']}  p={a['p']}")
        print(f"  attenuation after age adjustment: {rm['attenuation_pct']}%")
print(f"\nwritten: {SP}/race_age_stratified.json")
