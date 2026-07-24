"""Reconstruct observed one-year mortality outcomes from MIMIC-IV coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_mortality_followup(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    horizon_days: int = 365,
) -> pd.DataFrame:
    """Derive outcome observability using last discharge plus documented coverage."""
    required_cohort = ["subject_id", "ecg_time", "dod"]
    required_admissions = ["subject_id", "dischtime"]
    missing = [column for column in required_cohort if column not in cohort]
    missing += [column for column in required_admissions if column not in admissions]
    if missing:
        raise ValueError(f"missing required columns: {sorted(set(missing))}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    output = cohort.copy()
    output["ecg_time"] = pd.to_datetime(output["ecg_time"], errors="coerce")
    output["dod"] = pd.to_datetime(output["dod"], errors="coerce")
    if output["ecg_time"].isna().any():
        raise ValueError("ecg_time contains invalid or missing values")

    discharge = admissions[["subject_id", "dischtime"]].copy()
    discharge["dischtime"] = pd.to_datetime(discharge["dischtime"], errors="coerce")
    last_discharge = (
        discharge.dropna(subset=["dischtime"])
        .groupby("subject_id", as_index=False)["dischtime"]
        .max()
        .rename(columns={"dischtime": "last_discharge"})
    )
    output = output.merge(last_discharge, on="subject_id", how="left", validate="many_to_one")
    output["days_to_death_recomputed"] = (
        output["dod"] - output["ecg_time"]
    ).dt.total_seconds() / 86400.0
    output["negative_death_time"] = output["days_to_death_recomputed"] < 0.0
    output["coverage_end"] = output["last_discharge"] + pd.to_timedelta(
        horizon_days, unit="D"
    )
    output["documented_followup_days"] = (
        output["coverage_end"] - output["ecg_time"]
    ).dt.total_seconds() / 86400.0

    valid_death = output["days_to_death_recomputed"].notna() & ~output[
        "negative_death_time"
    ]
    event = valid_death & output["days_to_death_recomputed"].between(
        0.0, float(horizon_days), inclusive="both"
    )
    known_alive_past_horizon = valid_death & (
        output["days_to_death_recomputed"] > float(horizon_days)
    )
    coverage_sufficient = output["documented_followup_days"] >= float(horizon_days)
    observed = ~output["negative_death_time"] & (
        event | known_alive_past_horizon | coverage_sufficient.fillna(False)
    )

    corrected = pd.Series(pd.NA, index=output.index, dtype="Int64")
    corrected.loc[observed] = 0
    corrected.loc[event] = 1
    output["outcome_observed_365d"] = observed.astype(bool)
    output["corrected_dead_365d"] = corrected

    reason = pd.Series("", index=output.index, dtype="object")
    reason.loc[output["negative_death_time"]] = "negative_death_time"
    no_coverage = ~observed & ~output["negative_death_time"] & output[
        "last_discharge"
    ].isna()
    reason.loc[no_coverage] = "no_followup_coverage"
    insufficient = ~observed & ~output["negative_death_time"] & output[
        "last_discharge"
    ].notna()
    reason.loc[insufficient] = "insufficient_followup"
    output["exclusion_reason"] = reason
    return output


def _summarize_group(frame: pd.DataFrame) -> dict[str, object]:
    observed = frame["outcome_observed_365d"].astype(bool)
    exclusions = frame.loc[~observed, "exclusion_reason"].value_counts().to_dict()
    corrected = frame.loc[observed, "corrected_dead_365d"].astype(int)
    legacy_disagreement = None
    if "dead_365d" in frame:
        legacy = frame.loc[observed, "dead_365d"]
        comparable = legacy.notna()
        legacy_disagreement = int(
            (legacy.loc[comparable].astype(int) != corrected.loc[comparable]).sum()
        )
    return {
        "n_total": int(len(frame)),
        "n_observed": int(observed.sum()),
        "n_excluded": int((~observed).sum()),
        "observed_fraction": float(observed.mean()) if len(frame) else None,
        "events": int(corrected.sum()) if len(corrected) else 0,
        "event_fraction": float(corrected.mean()) if len(corrected) else None,
        "negative_death_times": int(frame["negative_death_time"].sum()),
        "exclusion_reasons": {str(key): int(value) for key, value in exclusions.items()},
        "legacy_disagreements_among_observed": legacy_disagreement,
    }


def summarize_followup(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"overall": _summarize_group(frame)}
    if "split" in frame:
        summary["by_split"] = {
            str(name): _summarize_group(group)
            for name, group in frame.groupby("split", dropna=False)
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, default=365)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = pd.read_parquet(args.cohort)
    admissions = pd.read_csv(
        args.admissions,
        compression="gzip",
        usecols=["subject_id", "dischtime"],
        parse_dates=["dischtime"],
    )
    result = build_mortality_followup(cohort, admissions, args.horizon_days)
    index_result = result
    if "is_index_ecg" in result:
        index_result = result.loc[result["is_index_ecg"] == 1].copy()
    summary = {
        "analysis": "MIMIC-IV one-year mortality follow-up completeness",
        "horizon_days": int(args.horizon_days),
        "coverage_rule": "last hospital discharge plus 365 days; known later death also confirms survival through horizon",
        "inputs": {
            "cohort_sha256": sha256_file(args.cohort),
            "admissions_sha256": sha256_file(args.admissions),
        },
        "all_ecgs": summarize_followup(result),
        "index_ecgs": summarize_followup(index_result),
    }
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_parquet, index=False)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
