"""Align frozen test predictions with corrected 365-day follow-up eligibility."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "40_submission_correction_metrics.py"
_SPEC = importlib.util.spec_from_file_location("submission_correction_metrics", METRICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load metrics module from {METRICS_PATH}")
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)


KEY_COLUMNS = ["subject_id", "study_id"]
FOLLOWUP_COLUMNS = [
    *KEY_COLUMNS,
    "documented_followup_days",
    "outcome_observed_365d",
    "corrected_dead_365d",
    "exclusion_reason",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def align_corrected_mortality(
    predictions: pd.DataFrame, followup: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the fully observed corrected cohort and a reconciliation audit."""
    _require_columns(predictions, [*KEY_COLUMNS, "dead_365d"], "predictions")
    _require_columns(followup, FOLLOWUP_COLUMNS, "follow-up")
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate prediction key")
    if followup.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate follow-up key")

    merged = predictions.merge(
        followup[FOLLOWUP_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} prediction rows lacked follow-up alignment")
    merged = merged.drop(columns="_merge")
    observed = merged["outcome_observed_365d"].fillna(False).astype(bool)
    corrected = merged.loc[observed].copy().reset_index(drop=True)
    if corrected["corrected_dead_365d"].isna().any():
        raise ValueError("observed follow-up rows contain missing corrected labels")
    if not corrected["corrected_dead_365d"].isin([0, 1]).all():
        raise ValueError("corrected mortality label must be binary")
    corrected["legacy_dead_365d"] = corrected["dead_365d"].astype(int)
    corrected["dead_365d"] = corrected["corrected_dead_365d"].astype(int)

    exclusions = (
        merged.loc[~observed, "exclusion_reason"]
        .fillna("unspecified")
        .replace("", "unspecified")
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    disagreement = int(
        corrected["legacy_dead_365d"].ne(corrected["dead_365d"]).sum()
    )
    audit: dict[str, object] = {
        "prediction_rows": int(len(predictions)),
        "followup_rows": int(len(followup)),
        "matched_rows": int(len(merged)),
        "unmatched_prediction_rows": unmatched,
        "observed_rows": int(len(corrected)),
        "excluded_rows": int((~observed).sum()),
        "exclusion_reasons": exclusions,
        "corrected_events": int(corrected["dead_365d"].sum()),
        "legacy_disagreements_among_observed": disagreement,
    }
    return corrected, audit


def expected_calibration_error(
    y_true: list[int] | pd.Series,
    probabilities: list[float] | pd.Series,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and probabilities must be equally sized vectors")
    if not np.isin(labels, [0, 1]).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("labels must be binary and probabilities must be in [0, 1]")
    assignments = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        members = assignments == index
        if members.any():
            error += float(members.mean()) * abs(
                float(scores[members].mean()) - float(labels[members].mean())
            )
    return error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--corrected-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    followup = pd.read_parquet(args.followup)
    corrected, audit = align_corrected_mortality(predictions, followup)
    analysis = corrected
    if "ok" in analysis:
        analysis = analysis.loc[analysis["ok"].fillna(False).astype(bool)]
    validated = _METRICS.validate_predictions(
        analysis,
        patient_col="subject_id",
        label_col="dead_365d",
        probability_col="pred_dead_365d",
    )
    metrics = _METRICS.stratified_patient_bootstrap(
        validated["dead_365d"],
        validated["pred_dead_365d"],
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    summary = {
        "analysis": "corrected_365_day_mortality_on_observed_followup",
        "alignment": audit,
        "analysis_rows": int(len(validated)),
        "analysis_events": int(validated["dead_365d"].sum()),
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "metrics": metrics,
        "expected_calibration_error_10_bins": expected_calibration_error(
            validated["dead_365d"], validated["pred_dead_365d"], bins=10
        ),
    }
    args.corrected_output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_parquet(args.corrected_output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
