"""Cox missing-data, proportional-hazards, and piecewise sensitivity analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COVARIATES = [
    "aiecg_high",
    "age_at_ecg",
    "sex_male",
    "lvef_value",
    "septal_thickness",
    "av_pk_vel",
]


def top_quantile_indicator(
    scores: pd.Series, quantile: float = 0.8
) -> tuple[pd.Series, float]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    numeric = pd.to_numeric(scores, errors="coerce")
    if numeric.isna().any():
        raise ValueError("risk score contains missing or non-numeric values")
    threshold = float(numeric.quantile(quantile))
    return (numeric >= threshold).astype(int), threshold


def prepare_echo_normal_survival(
    predictions: pd.DataFrame,
    horizon_days: int = 365,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    required = ["lvef_value", "pred_hfref_le40", "days_to_death"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = predictions.loc[
        (predictions["lvef_value"] > 50) & predictions["pred_hfref_le40"].notna()
    ].copy()
    initial_n = len(frame)
    unobserved_count = 0
    outcome_source = "legacy death-date observability"
    if {"outcome_observed_365d", "corrected_dead_365d"}.issubset(frame.columns):
        observed = frame["outcome_observed_365d"].fillna(False).astype(bool)
        unobserved_count = int((~observed).sum())
        frame = frame.loc[observed].copy()
        if frame["corrected_dead_365d"].isna().any():
            raise ValueError("observed follow-up rows contain missing corrected labels")
        outcome_source = "corrected follow-up eligibility"
    days = pd.to_numeric(frame["days_to_death"], errors="coerce")
    negative = days < 0.0
    negative_count = int(negative.sum())
    frame = frame.loc[~negative].copy().reset_index(drop=True)
    days = pd.to_numeric(frame["days_to_death"], errors="coerce")
    if outcome_source == "corrected follow-up eligibility":
        frame["event"] = frame["corrected_dead_365d"].astype(int)
        if (frame["event"].eq(1) & days.isna()).any():
            raise ValueError("corrected death event lacks days_to_death")
    else:
        frame["event"] = (
            days.notna() & days.between(0.0, float(horizon_days), inclusive="both")
        ).astype(int)
    frame["T"] = np.where(frame["event"].eq(1), days, float(horizon_days))
    frame["T"] = pd.to_numeric(frame["T"], errors="raise").clip(1.0, float(horizon_days))
    frame["aiecg_high"], threshold = top_quantile_indicator(
        frame["pred_hfref_le40"], quantile=0.8
    )
    audit: dict[str, int | float] = {
        "echo_normal_before_time_check": int(initial_n),
        "negative_death_times_excluded": negative_count,
        "unobserved_followup_excluded": unobserved_count,
        "outcome_source": outcome_source,
        "analysis_n": int(len(frame)),
        "events": int(frame["event"].sum()),
        "risk_score_80th_percentile": threshold,
        "high_risk_n": int(frame["aiecg_high"].sum()),
    }
    return frame, audit


def missingness_table(
    frame: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, int | float]]:
    total = len(frame)
    return {
        column: {
            "missing": int(frame[column].isna().sum()),
            "total": int(total),
            "percent": float(100.0 * frame[column].isna().mean()) if total else 0.0,
        }
        for column in columns
    }


def make_piecewise_risk_sets(
    frame: pd.DataFrame,
    cut_day: int = 90,
    horizon_day: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < cut_day < horizon_day:
        raise ValueError("cut_day must be between zero and horizon_day")
    early = frame.copy()
    original_time = pd.to_numeric(early["T"], errors="raise")
    original_event = early["event"].astype(int)
    early["event"] = ((original_event == 1) & (original_time <= cut_day)).astype(int)
    early["T"] = np.minimum(original_time, float(cut_day))

    at_risk = frame.loc[pd.to_numeric(frame["T"], errors="raise") > cut_day].copy()
    late_time = pd.to_numeric(at_risk["T"], errors="raise")
    late_event = at_risk["event"].astype(int)
    at_risk["event"] = (
        (late_event == 1) & (late_time <= float(horizon_day))
    ).astype(int)
    at_risk["T"] = np.minimum(late_time, float(horizon_day)) - float(cut_day)
    return early, at_risk


def _prepare_cox_table(
    frame: pd.DataFrame,
    covariates: list[str],
    imputation: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    table = frame[[*covariates, "T", "event"]].copy()
    for column in covariates:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    medians: dict[str, float] = {}
    if imputation == "median":
        for column in covariates:
            median = float(table[column].median())
            medians[column] = median
            table[column] = table[column].fillna(median)
    elif imputation == "complete_case":
        table = table.dropna()
    else:
        raise ValueError("imputation must be median or complete_case")
    return table, medians


def fit_cox_model(
    frame: pd.DataFrame,
    covariates: list[str],
    imputation: str,
) -> tuple[dict[str, object], object, pd.DataFrame]:
    from lifelines import CoxPHFitter

    table, medians = _prepare_cox_table(frame, covariates, imputation)
    fitter = CoxPHFitter()
    fitter.fit(table, duration_col="T", event_col="event", robust=True)
    row = fitter.summary.loc["aiecg_high"]
    result = {
        "n": int(len(table)),
        "events": int(table["event"].sum()),
        "imputation": imputation,
        "imputation_medians": medians,
        "high_risk_hr": float(row["exp(coef)"]),
        "high_risk_ci_lower": float(row["exp(coef) lower 95%"]),
        "high_risk_ci_upper": float(row["exp(coef) upper 95%"]),
        "high_risk_p": float(row["p"]),
        "concordance_index": float(fitter.concordance_index_),
    }
    return result, fitter, table


def proportional_hazards_summary(
    fitter: object, table: pd.DataFrame
) -> dict[str, object]:
    from lifelines.statistics import proportional_hazard_test

    test = proportional_hazard_test(fitter, table, time_transform="rank")
    p_values = {str(name): float(value) for name, value in test.summary["p"].items()}
    return {
        "method": "scaled Schoenfeld residual test with rank time transform",
        "p_values": p_values,
        "high_risk_p": p_values.get("aiecg_high"),
        "minimum_covariate_p": min(p_values.values()) if p_values else None,
        "violation_at_0_05": any(value < 0.05 for value in p_values.values()),
    }


def run_survival_sensitivity(predictions: pd.DataFrame) -> dict[str, object]:
    frame, cohort_audit = prepare_echo_normal_survival(predictions)
    covariates = [column for column in DEFAULT_COVARIATES if column in frame]
    missingness = missingness_table(frame, covariates)
    primary, fitter, primary_table = fit_cox_model(frame, covariates, "median")
    complete_case, _, _ = fit_cox_model(frame, covariates, "complete_case")
    ph = proportional_hazards_summary(fitter, primary_table)
    early, late = make_piecewise_risk_sets(frame, cut_day=90, horizon_day=365)
    early_result, _, _ = fit_cox_model(early, covariates, "median")
    late_result, _, _ = fit_cox_model(late, covariates, "median")
    return {
        "analysis": "echo-normal exploratory mortality association",
        "cohort": cohort_audit,
        "covariates": covariates,
        "missingness": missingness,
        "median_imputed_full_year": primary,
        "complete_case_full_year": complete_case,
        "proportional_hazards_test": ph,
        "piecewise_models": {
            "days_0_to_90": early_result,
            "days_91_to_365": late_result,
        },
        "interpretation_rule": (
            "If the proportional-hazards test is significant, the full-year HR is an average "
            "summary and interval-specific HRs must be reported."
        ),
        "followup_eligibility": (
            "Corrected 365-day outcome observability was applied before survival modelling."
            if cohort_audit["outcome_source"] == "corrected follow-up eligibility"
            else "Legacy death-date observability was used."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    result = run_survival_sensitivity(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
