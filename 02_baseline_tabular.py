#!/usr/bin/env python3
"""
Strong tabular baseline = 'the line the waveform model must beat'.
Features: ECG machine measurements (intervals/axes) + age + sex.
Targets: HFrEF (LVEF<=40), severe AS, 1-year mortality.
Models: L2 Logistic Regression and HistGradientBoosting.
Patient-level train/test split from the cohort table. Bootstrap 95% CI on test AUROC.
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

OUT = "/data/s01011/cardio_m3t/aiecg/outputs"
df = pd.read_parquet(f"{OUT}/cohort_labels.parquet")
df = df[df["is_index_ecg"] == 1].copy()          # one ECG per patient

FEATS = ["rr_interval","pr_interval","qrs_duration","qt_interval","qtc","heart_rate",
         "p_axis","qrs_axis","t_axis","age_at_ecg","sex_male"]
TARGETS = ["hfref_le40","as_severe","dead_365d"]

def boot_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); aucs=[]
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        aucs.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

results = {}
for tgt in TARGETS:
    d = df.dropna(subset=[tgt])
    tr, te = d[d.split=="train"], d[d.split=="test"]
    Xtr, ytr = tr[FEATS].astype(float), tr[tgt].astype(int).values
    Xte, yte = te[FEATS].astype(float), te[tgt].astype(int).values
    results[tgt] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                    "pos_rate_test": float(yte.mean())}
    # Logistic
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr); p_lr = lr.predict_proba(Xte)[:,1]
    # HGB
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    hgb.fit(Xtr, ytr); p_hgb = hgb.predict_proba(Xte)[:,1]
    for name, p in [("logreg", p_lr), ("histgb", p_hgb)]:
        auc = roc_auc_score(yte, p); lo, hi = boot_auc(yte, p)
        results[tgt][name] = {"auroc": float(auc), "auroc_ci": [lo, hi],
                              "auprc": float(average_precision_score(yte, p)),
                              "brier": float(brier_score_loss(yte, p))}

print(json.dumps(results, indent=2))
with open(f"{OUT}/baseline_metrics.json","w") as f:
    json.dump(results, f, indent=2)

print("\n================ BASELINE (test set) ================")
print(f"{'target':14}{'n_test':>8}{'pos%':>7}   {'model':8}{'AUROC (95% CI)':>22}{'AUPRC':>8}")
for tgt in TARGETS:
    r = results[tgt]
    for mdl in ["logreg","histgb"]:
        m = r[mdl]
        ci = f"{m['auroc']:.3f} ({m['auroc_ci'][0]:.3f}-{m['auroc_ci'][1]:.3f})"
        print(f"{tgt:14}{r['n_test']:>8}{100*r['pos_rate_test']:>6.1f}   {mdl:8}{ci:>22}{m['auprc']:>8.3f}")
print("\nsaved:", f"{OUT}/baseline_metrics.json")
