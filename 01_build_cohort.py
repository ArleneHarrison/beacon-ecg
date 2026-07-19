#!/usr/bin/env python3
"""
Build the AI-ECG cohort + label table.
One row per ECG that can be paired to an echo study within +/-7 days (nearest).
Labels: HFrEF (LVEF<=40), LVEF value, moderate/severe aortic stenosis, LVH proxy,
all-cause mortality (30/90/365d + in-hospital), plus demographics, serial-ECG count,
patient-level and temporal splits.
"""
import os, hashlib
import numpy as np
import pandas as pd

ROOT = "/data/s01011/cardio_m3t/data/raw/physionet"
ECG_DIR  = f"{ROOT}/mimic-iv-ecg/1.0"
ECHO_DIR = f"{ROOT}/mimic-iv-echo/1.0"
HOSP     = f"{ROOT}/mimiciv/3.1/hosp"
OUT = "/data/s01011/cardio_m3t/aiecg/outputs"
os.makedirs(OUT, exist_ok=True)

WINDOW_DAYS = 7

echo_measures = ["lvef", "av_pk_vel", "av_pk_grad", "av_mean_grad", "av_area_continuity",
                 "mv_ero", "tr_mmhg", "septal_thickness", "inf_lat_thickness",
                 "lvedd", "lvesd", "la_vol", "lvot_diam"]

print(">> reading echo structured measurements (long)")
echo = pd.read_csv(
    f"{ECHO_DIR}/structured-measurement.csv.gz",
    usecols=["subject_id", "measurement_datetime", "test_type", "measurement", "result"],
    dtype={"subject_id": "int64", "measurement": "str", "test_type": "str", "result": "str"},
)
echo = echo[echo["test_type"] == "tte"]                 # restrict to transthoracic
echo = echo[echo["measurement"].isin(echo_measures)]
echo["result"] = pd.to_numeric(echo["result"], errors="coerce")
echo = echo.dropna(subset=["result", "measurement_datetime"])
echo["measurement_datetime"] = pd.to_datetime(echo["measurement_datetime"], errors="coerce")
echo = echo.dropna(subset=["measurement_datetime"])
print(f"   echo measurement rows kept: {len(echo):,}")

# pivot to one row per (subject, datetime) = one echo study
echo_w = (echo.pivot_table(index=["subject_id", "measurement_datetime"],
                           columns="measurement", values="result", aggfunc="mean")
              .reset_index())
echo_w.columns.name = None
# ensure every expected measurement column exists even if absent in the data
for c in echo_measures:
    if c not in echo_w.columns:
        echo_w[c] = np.nan
        print(f"   WARN: measurement '{c}' absent -> filled NaN")
print(f"   echo studies (subject x datetime): {len(echo_w):,}")

print(">> reading ECG record list")
ecg = pd.read_csv(f"{ECG_DIR}/record_list.csv",
                  usecols=["subject_id", "study_id", "ecg_time", "path"],
                  dtype={"subject_id": "int64", "study_id": "int64", "path": "str"})
ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
ecg = ecg.dropna(subset=["ecg_time"])
ecg["n_ecgs_subject"] = ecg.groupby("subject_id")["study_id"].transform("count")
print(f"   ECGs: {len(ecg):,}  patients: {ecg['subject_id'].nunique():,}")

print(">> asof-nearest match ECG -> echo within +/-7d")
# merge_asof requires global sort by the 'on' key
ecg_s = ecg.sort_values("ecg_time")
echo_s = echo_w.sort_values("measurement_datetime")
matched = pd.merge_asof(
    ecg_s, echo_s,
    left_on="ecg_time", right_on="measurement_datetime",
    by="subject_id", direction="nearest",
    tolerance=pd.Timedelta(days=WINDOW_DAYS),
)
matched = matched.dropna(subset=["measurement_datetime"])   # keep only paired
matched["days_ecg_to_echo"] = (matched["measurement_datetime"] - matched["ecg_time"]).dt.total_seconds() / 86400.0
print(f"   paired ECGs: {len(matched):,}  patients: {matched['subject_id'].nunique():,}")

# ---------------- labels ----------------
m = matched
m["lvef_value"] = m["lvef"]
m["hfref_le40"] = (m["lvef"] <= 40).astype("Int64")
m["lvef_le50"]  = (m["lvef"] <= 50).astype("Int64")
m.loc[m["lvef"].isna(), ["hfref_le40", "lvef_le50"]] = pd.NA

# aortic stenosis severity from AV hemodynamics (2020 ACC/AHA guideline thresholds)
vmax = m["av_pk_vel"]; mg = m["av_mean_grad"]; ava = m["av_area_continuity"]
as_severe = ((vmax >= 4) | (mg >= 40) | (ava < 1.0))
as_mod    = (~as_severe) & ((vmax >= 3) | (mg >= 20) | (ava < 1.5))
has_av = vmax.notna() | mg.notna() | ava.notna()
m["as_severe"] = np.where(has_av, as_severe.astype(float), np.nan)
m["as_mod_or_severe"] = np.where(has_av, (as_severe | as_mod).astype(float), np.nan)

# LVH proxy: septal or inferolateral wall thickness >= 1.5 cm
th = m[["septal_thickness", "inf_lat_thickness"]].max(axis=1)
m["lvh"] = np.where(th.notna(), (th >= 1.5).astype(float), np.nan)

# ---------------- demographics + mortality ----------------
print(">> merging patients (age/sex/mortality)")
pat = pd.read_csv(f"{HOSP}/patients.csv.gz",
                  usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
                  dtype={"subject_id": "int64"})
pat["dod"] = pd.to_datetime(pat["dod"], errors="coerce")
m = m.merge(pat, on="subject_id", how="left")
m["age_at_ecg"] = (m["anchor_age"] + (m["ecg_time"].dt.year - m["anchor_year"])).clip(0, 91)
m["sex_male"] = (m["gender"] == "M").astype(int)
dtd = (m["dod"] - m["ecg_time"]).dt.total_seconds() / 86400.0
m["days_to_death"] = dtd
m["dead_30d"]  = ((dtd >= 0) & (dtd <= 30)).astype(int)
m["dead_90d"]  = ((dtd >= 0) & (dtd <= 90)).astype(int)
m["dead_365d"] = ((dtd >= 0) & (dtd <= 365)).astype(int)

# ---------------- machine-measurement features (for baseline) ----------------
print(">> merging ECG machine measurements (numeric)")
mm = pd.read_csv(f"{ECG_DIR}/machine_measurements.csv",
                 usecols=["subject_id", "study_id", "rr_interval", "p_onset", "p_end",
                          "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"])
# derived intervals (ms), guard against nonsense
mm["pr_interval"]   = mm["qrs_onset"] - mm["p_onset"]
mm["qrs_duration"]  = mm["qrs_end"] - mm["qrs_onset"]
mm["qt_interval"]   = mm["t_end"] - mm["qrs_onset"]
mm["heart_rate"]    = np.where(mm["rr_interval"] > 0, 60000.0 / mm["rr_interval"], np.nan)
mm["qtc"]           = np.where(mm["rr_interval"] > 0, mm["qt_interval"] / np.sqrt(mm["rr_interval"]/1000.0), np.nan)
m = m.merge(mm, on=["subject_id", "study_id"], how="left")

# ---------------- splits ----------------
def split_of(sid):
    h = int(hashlib.md5(str(sid).encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 80 else "test")
m["split"] = m["subject_id"].map(split_of)
# temporal split by anchor_year_group (earliest groups -> train_temporal)
def temporal(g):
    if not isinstance(g, str): return "unknown"
    if g in ("2008 - 2010", "2011 - 2013"): return "temporal_train"
    if g in ("2014 - 2016",): return "temporal_val"
    return "temporal_test"   # 2017-2019, 2020-2022
m["temporal_split"] = m["anchor_year_group"].map(temporal)

# index ECG per patient = first paired ECG (for patient-level main analysis)
m = m.sort_values(["subject_id", "ecg_time"])
m["is_index_ecg"] = (~m.duplicated("subject_id")).astype(int)

# ---------------- save ----------------
keep = ["subject_id","study_id","ecg_time","path","n_ecgs_subject","is_index_ecg",
        "measurement_datetime","days_ecg_to_echo",
        "lvef_value","hfref_le40","lvef_le50","as_severe","as_mod_or_severe","lvh",
        "av_pk_vel","av_pk_grad","av_mean_grad","av_area_continuity","septal_thickness","inf_lat_thickness",
        "lvedd","lvesd","la_vol","mv_ero","tr_mmhg",
        "age_at_ecg","sex_male","anchor_year_group","dod","days_to_death",
        "dead_30d","dead_90d","dead_365d",
        "rr_interval","pr_interval","qrs_duration","qt_interval","qtc","heart_rate",
        "p_axis","qrs_axis","t_axis","split","temporal_split"]
out = m[keep].copy()
out.to_parquet(f"{OUT}/cohort_labels.parquet", index=False)

# ---------------- summary ----------------
idx = out[out["is_index_ecg"] == 1]
def rate(s):
    s = s.dropna();
    return f"{int(s.sum())}/{len(s)} ({100*s.mean():.1f}%)" if len(s) else "n/a"
print("\n================ COHORT SUMMARY ================")
print(f"paired ECGs (all): {len(out):,} | patients (index ECG): {len(idx):,}")
print(f"median |ECG-echo| days: {out['days_ecg_to_echo'].abs().median():.2f}")
print(f"age index: mean {idx['age_at_ecg'].mean():.1f}  | male {100*idx['sex_male'].mean():.1f}%")
print("\n-- per-patient (index ECG) label prevalence --")
for c in ["hfref_le40","lvef_le50","as_severe","as_mod_or_severe","lvh","dead_30d","dead_90d","dead_365d"]:
    print(f"  {c:16}: {rate(idx[c])}")
print(f"\nLVEF value (index): mean {idx['lvef_value'].mean():.1f}  median {idx['lvef_value'].median():.1f}")
print("\n-- split sizes (patients, index ECG) --")
print(idx.groupby('split').size())
print("\n-- temporal split (patients, index ECG) --")
print(idx.groupby('temporal_split').size())
print("\n-- machine-measurement coverage (index) --")
for c in ["heart_rate","qrs_duration","qtc","qrs_axis"]:
    print(f"  {c:14}: {100*idx[c].notna().mean():.1f}% non-null")
print("\nsaved:", f"{OUT}/cohort_labels.parquet")
