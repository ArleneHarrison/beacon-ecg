"""Re-evaluate the tabular mortality comparator on corrected follow-up cohorts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent


def _load_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = _load_module("40_submission_correction_metrics.py", "submission_metrics")
_INCREMENTAL = _load_module("42_corrected_echo_incremental_analysis.py", "incremental")

FEATURES = [
    "rr_interval",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "qtc",
    "heart_rate",
    "p_axis",
    "qrs_axis",
    "t_axis",
    "age_at_ecg",
    "sex_male",
]
KEY_COLUMNS = ["subject_id", "study_id"]


def prepare_corrected_training_cohort(
    cohort: pd.DataFrame, followup: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_cohort = [*KEY_COLUMNS, "split", "is_index_ecg", "dead_365d"]
    required_followup = [*KEY_COLUMNS, "outcome_observed_365d", "corrected_dead_365d"]
    missing_cohort = [column for column in required_cohort if column not in cohort]
    missing_followup = [column for column in required_followup if column not in followup]
    if missing_cohort or missing_followup:
        raise ValueError(
            f"missing columns: cohort={missing_cohort}, follow-up={missing_followup}"
        )
    train = cohort.loc[
        cohort["is_index_ecg"].fillna(False).astype(bool) & cohort["split"].eq("train")
    ].copy()
    if train.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate training key")
    if followup.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate follow-up key")
    merged = train.merge(
        followup[required_followup],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} training rows lacked follow-up alignment")
    observed = merged["outcome_observed_365d"].fillna(False).astype(bool)
    prepared = merged.loc[observed].drop(columns="_merge").copy().reset_index(drop=True)
    if prepared["corrected_dead_365d"].isna().any():
        raise ValueError("observed training row contains missing corrected label")
    prepared["legacy_dead_365d"] = prepared["dead_365d"].astype(int)
    prepared["dead_365d"] = prepared["corrected_dead_365d"].astype(int)
    audit = {
        "candidate_train_rows": int(len(train)),
        "observed_train_rows": int(len(prepared)),
        "excluded_train_rows": int((~observed).sum()),
        "corrected_train_events": int(prepared["dead_365d"].sum()),
        "legacy_disagreements_among_observed": int(
            prepared["legacy_dead_365d"].ne(prepared["dead_365d"]).sum()
        ),
    }
    return prepared, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--corrected-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = pd.read_parquet(args.cohort)
    followup = pd.read_parquet(args.followup)
    test = pd.read_parquet(args.corrected_test)
    train, train_audit = prepare_corrected_training_cohort(cohort, followup)
    missing = [column for column in FEATURES if column not in train or column not in test]
    if missing:
        raise ValueError(f"missing tabular comparator features: {missing}")
    required_test = ["subject_id", "dead_365d", "pred_dead_365d"]
    test = test.dropna(subset=required_test).copy().reset_index(drop=True)
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
    )
    model.fit(train[FEATURES].astype(float), train["dead_365d"].astype(int))
    tabular = model.predict_proba(test[FEATURES].astype(float))[:, 1]
    waveform = test["pred_dead_365d"].to_numpy(dtype=float)
    label = test["dead_365d"].to_numpy(dtype=int)
    waveform_metrics = _METRICS.stratified_patient_bootstrap(
        label, waveform, n_boot=args.bootstrap, seed=args.seed
    )
    tabular_metrics = _METRICS.stratified_patient_bootstrap(
        label, tabular, n_boot=args.bootstrap, seed=args.seed + 1
    )
    paired = _INCREMENTAL.paired_stratified_bootstrap(
        label, waveform, tabular, args.bootstrap, args.seed
    )
    delong = _INCREMENTAL.paired_delong(label, waveform, tabular)
    result = {
        "analysis": "corrected_same_patient_mortality_comparator",
        "training": train_audit,
        "test_n": int(len(test)),
        "test_events": int(label.sum()),
        "features": FEATURES,
        "waveform_ensemble": waveform_metrics,
        "tabular_hist_gradient_boosting": tabular_metrics,
        "comparison": {
            "delta_auroc": delong["delta_auroc"],
            "delta_auroc_ci": paired["delta_auroc_ci"],
            "delta_auprc": float(
                waveform_metrics["auprc"]["estimate"]
                - tabular_metrics["auprc"]["estimate"]
            ),
            "delta_auprc_ci": paired["delta_auprc_ci"],
            "brier_improvement": float(
                tabular_metrics["brier"]["estimate"]
                - waveform_metrics["brier"]["estimate"]
            ),
            "brier_improvement_ci": paired["brier_improvement_ci"],
            "delong_z": delong["z"],
            "delong_p": delong["p"],
            "bootstrap_replicates": int(args.bootstrap),
        },
        "seed": int(args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
