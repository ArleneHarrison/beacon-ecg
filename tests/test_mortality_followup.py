from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "43_build_mortality_followup.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mortality_followup", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "study_id": [11, 22, 33, 44, 55, 66],
            "ecg_time": pd.to_datetime(["2020-01-01"] * 6),
            "dod": pd.to_datetime(
                ["2020-12-31", None, None, "2019-12-31", "2021-01-02", None]
            ),
            "dead_365d": [1, 0, 0, 0, 0, 0],
            "is_index_ecg": [1] * 6,
            "split": ["test"] * 6,
        }
    )


def make_admissions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 2, 3, 4, 5],
            "dischtime": pd.to_datetime(
                [
                    "2020-01-10",
                    "2020-01-01",
                    "2020-02-01",
                    "2019-12-31",
                    "2020-02-01",
                    "2019-12-31",
                ]
            ),
        }
    )


def test_followup_boundary_and_multiple_admissions():
    module = load_module()
    result = module.build_mortality_followup(make_cohort(), make_admissions(), horizon_days=365)
    by_subject = result.set_index("subject_id")
    assert by_subject.loc[1, "corrected_dead_365d"] == 1
    assert bool(by_subject.loc[1, "outcome_observed_365d"])
    assert by_subject.loc[2, "last_discharge"] == pd.Timestamp("2020-02-01")
    assert by_subject.loc[2, "corrected_dead_365d"] == 0
    assert bool(by_subject.loc[2, "outcome_observed_365d"])


def test_insufficient_followup_and_missing_admission_are_unobserved():
    module = load_module()
    result = module.build_mortality_followup(make_cohort(), make_admissions(), horizon_days=365)
    by_subject = result.set_index("subject_id")
    assert not bool(by_subject.loc[3, "outcome_observed_365d"])
    assert pd.isna(by_subject.loc[3, "corrected_dead_365d"])
    assert not bool(by_subject.loc[6, "outcome_observed_365d"])
    assert by_subject.loc[6, "exclusion_reason"] == "no_followup_coverage"


def test_negative_death_time_is_excluded_even_with_later_discharge():
    module = load_module()
    result = module.build_mortality_followup(make_cohort(), make_admissions(), horizon_days=365)
    row = result.set_index("subject_id").loc[4]
    assert bool(row["negative_death_time"])
    assert not bool(row["outcome_observed_365d"])
    assert row["exclusion_reason"] == "negative_death_time"


def test_known_death_after_horizon_confirms_non_event_status():
    module = load_module()
    result = module.build_mortality_followup(make_cohort(), make_admissions(), horizon_days=365)
    row = result.set_index("subject_id").loc[5]
    assert bool(row["outcome_observed_365d"])
    assert row["corrected_dead_365d"] == 0


def test_summary_reconciles_population():
    module = load_module()
    result = module.build_mortality_followup(make_cohort(), make_admissions(), horizon_days=365)
    summary = module.summarize_followup(result)
    overall = summary["overall"]
    assert overall["n_total"] == 6
    assert overall["n_observed"] + overall["n_excluded"] == 6
    assert sum(overall["exclusion_reasons"].values()) == overall["n_excluded"]
