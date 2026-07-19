# BEACON-ECG — reproduction code

**B**eyond-**E**cho **A**ssessment of **C**ardiac structure and **O**utcome from a single 12-lead electrocardiogram.

**Final model:** a validation-selected weighted ensemble of four single-ECG networks, two with demographic (age/sex) conditioning. Internal test AUROC: HFrEF 0.900, severe AS 0.860, LVH 0.789, 1-year mortality 0.749. External validation (EchoNext, Columbia, no retraining): HFrEF 0.896, LVH 0.781.

This directory contains the complete pipeline used to produce every number in the manuscript,
from cohort construction to the final analyses. Scripts are numbered in execution order.

## 1. Data access

All data are public but **credentialed**. You must complete PhysioNet credentialing and sign the
Data Use Agreement for each dataset before running anything:

| Dataset | Version | Used for |
|---|---|---|
| MIMIC-IV | v3.1 (`hosp`) | demographics, `dod` (mortality) |
| MIMIC-IV-ECG | v1.0 | 12-lead waveforms + `machine_measurements` |
| MIMIC-IV-ECHO | v1.0 | structured echo measurements (**labels only**) |
| PTB-XL | v1.0.3 (open) | self-supervised pre-training ablation |

PTB-XL is open access and can be fetched anonymously:
`aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/ ./ptb-xl/`

MIMIC waveforms require an authenticated session; `03_download_waveforms.py` performs a
cookie-based login (PhysioNet's `/files/` endpoint no longer accepts HTTP basic auth).
**Credentials are passed as CLI arguments and are never stored in these files.**

## 2. Environment

Python ≥3.10 with: `numpy pandas pyarrow scikit-learn scipy torch wfdb lifelines matplotlib`.
A CUDA GPU is required for steps 05–07 and 11 (developed on an RTX 5090; ~16 GB VRAM is ample).

## 3. Paths

Scripts currently contain absolute paths from the development machines
(`/data/s01011/cardio_m3t/...`, `/root/autodl-tmp/aiecg/...`). **Edit the `BASE`/`OUT`
constants at the top of each script** to match your layout before running. This is the one
manual step required.

## 4. Pipeline — script → manuscript result

| # | Script | Produces | Manuscript |
|---|---|---|---|
| 01 | `01_build_cohort.py` | ECG↔echo pairing (±7 d), labels (HFrEF/AS/LVH/LVEF), mortality, patient-level splits → `cohort_labels.parquet` | §2.2–2.3, Table 1 (§3.1) |
| 02 | `02_baseline_tabular.py` | Tabular ECG machine-measurement baseline | §3.2 |
| 03 | `03_download_waveforms.py` | Directed waveform download (cookie auth, resumable) | §2.1 |
| 04 | `04_pack_waveforms.py` | Packs waveforms → `waves.npy` memmap + aligned index | — |
| 05 | `05_train_waveform.py` | Main multitask model; `--single_task` / `--no_uncertainty` give the ablations; `--pretrained` fine-tunes from step 07 | §3.3, §3.4 ablations |
| 06 | `06_train_waveform_v2_ablation.py` | Augmentation + attention pooling + cosine schedule ablation | §3.4 (null result) |
| 07 | `07_pretrain_ptbxl_simclr.py` | SimCLR/NT-Xent self-supervised pre-training on PTB-XL | §3.4 (null result) |
| 08 | `08_eval_calibration_dca.py` | Calibration (Brier/ECE), decision-curve net benefit, incremental prognosis, discordance rates | §3.4, §3.5, §3.6 |
| 09 | `09_survival_km.py` | Kaplan–Meier + log-rank for the discordance analysis | §3.6, Fig 7 |
| 10 | `10_trajectory_tabular.py` | Landmark tabular serial-ECG trajectory analysis | §3.7 |
| 11 | `11_trajectory_deep_e2e.py` | End-to-end fine-tuned deep trajectory (snapshot vs GRU) | §3.7 |
| 12 | `12_delong_cox_idi_utility.py` | **Paired DeLong** (same-test-set), **multivariable Cox**, ΔAUROC CI + **IDI**, AS operating point | §3.3, §3.5, §3.6, §3.4 |
| 13 | `13_make_figures.py` | Figures 1–4 | Figs 1–4 |

## 5. Reproducing the headline numbers

```bash
python 01_build_cohort.py                     # cohort + labels + splits
python 02_baseline_tabular.py                 # baseline: HFrEF 0.813, AS 0.769
python 03_download_waveforms.py --user U --pw P --pathlist paths.txt --out ecg_raw
python 04_pack_waveforms.py --cohort cohort_labels.parquet --files ecg_raw/files --out pack
python 05_train_waveform.py --data pack --out out/final --epochs 30 --bs 256 --lr 2e-3
python 12_delong_cox_idi_utility.py           # DeLong, Cox, IDI, AS utility
python 09_survival_km.py --pred out/final/test_predictions.parquet --out out/final
python 10_trajectory_tabular.py               # tabular trajectory
python 11_trajectory_deep_e2e.py ...          # deep trajectory (definitive null)
```

## 6. Caveats a reproducer should know

- **Per-task denominators differ** because echo measurements are unevenly recorded; labels are
  NaN-masked in the loss (see `uw_loss` in step 05). Expect slightly different `n` per endpoint.
- **The deep-trajectory pilot reported in §3.7 used outcome-balanced sampling** and is *not*
  comparable in absolute terms to the whole-cohort analyses; the definitive test is step 11 on
  the full unselected cohort. This is stated in the manuscript and repeated here deliberately.
- **Waveform download is rate-limited** by PhysioNet over long runs; step 03 re-logs-in on 401/403
  and is resumable — expect a few percent of records to fail and require a second pass.
- Bootstrap CIs use 1,000–2,000 resamples with a fixed seed; exact digits may vary in the last
  decimal place across runs.
- LVH is defined here as wall thickness ≥1.5 cm (sex-agnostic), which is **more stringent than
  guideline sex-specific thresholds** — see manuscript Limitations.
